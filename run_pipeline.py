"""
Retrieval-Augmented Verification (RAV) - Main Pipeline

Orchestrates the full RAV pipeline from job selection through to bias detection.

Steps:
    1. Job Selection
    2. KG Construction
    3. Prompt Generation
    4. LLM Querying
    5. Alignment
    6. Bias Detection
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime

import src.utils.functions as utils
from src.rav.knowledge_graph import KnowledgeGraph
from src.rav.embedding_model import EmbeddingModel
from src.rav.bias_detector import BiasDetector
from src.rav.llm import LLM
from src.pipeline.select_jobs import (
    run_job_selection
)
from src.pipeline.generate_prompts import (
    build_prompts_for_experiment
)

# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = utils.find_project_root()

CONFIG = {
    'seed':            42,
    'n_groups':        4,
    'jobs_per_group':  5,
    'N':               5,   # number of traits to elicit per prompt
    'model_name':      'llama3.2:1b',
    'api_url':         'http://localhost:11434/api/chat',
}

# DIRECTORIES
DATA_DIR        = PROJECT_ROOT / 'data'
CURATED_DIR     = DATA_DIR / 'onet_datasets/curated'
EXPERIMENT_DIR  = DATA_DIR / 'onet_datasets/experiment_datasets'
PROMPTS_DIR     = DATA_DIR / 'generated_prompts'
RESULTS_DIR     = DATA_DIR / 'results'

# ENSURE OUTPUT DIRS EXIST
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# STEP 1 — JOB SELECTION
# =============================================================================

def step_1_job_selection(dataset: pd.DataFrame, experiment_id: str):
    """
    Two-level random occupation sampling.
    Saves occupations list and KG selection CSV.
    """
    print("\n" + "="*60)
    print("STEP 1: JOB SELECTION")
    print("="*60)

    selected_codes, occupations_df, kg_selection_df = run_job_selection(
        dataset=dataset,
        n_groups=CONFIG['n_groups'],
        jobs_per_group=CONFIG['jobs_per_group'],
        seed=CONFIG['seed'],
        experiment_id=experiment_id,
        output_dir=EXPERIMENT_DIR
    )

    return selected_codes, occupations_df, kg_selection_df


# =============================================================================
# STEP 2 — KG CONSTRUCTION
# =============================================================================

def step_2_build_kg(kg_selection_df: pd.DataFrame, experiment_id: str):
    print("\n" + "="*60)
    print("STEP 2: KG CONSTRUCTION")
    print("="*60)

    kg = KnowledgeGraph()
    kg.build_KG(kg_selection_df)

    print(f"KG built — {kg.G.number_of_nodes()} nodes, {kg.G.number_of_edges()} edges")

    # Visualise KG
    kg_viz_path = str(RESULTS_DIR / f"{experiment_id}_kg.html")
    kg.visualize_interactive_graph(filename=kg_viz_path)

    return kg


# =============================================================================
# STEP 3 — PROMPT GENERATION
# =============================================================================

def step_3_generate_prompts(
    occupations_df: pd.DataFrame,
    experiment_id: str
):
    """
    Generate T1 and T2 prompts for all occupations and gender conditions.
    Saves prompts CSV.
    """
    print("\n" + "="*60)
    print("STEP 3: PROMPT GENERATION")
    print("="*60)

    prompts_df = build_prompts_for_experiment(occupations_df, N=CONFIG['N'])

    prompts_path = PROMPTS_DIR / f"{experiment_id}_prompts.csv"
    prompts_df.to_csv(prompts_path, index=False)
    print(f"Generated {len(prompts_df)} prompts → {prompts_path}")

    return prompts_df


# =============================================================================
# STEP 4 — LLM QUERYING
# =============================================================================

def step_4_query_llm(
    prompts_df: pd.DataFrame,
    experiment_id: str
):
    """
    Query LLM for each prompt and collect trait responses.
    Saves raw LLM results CSV.
    Skips failed responses and logs them separately.
    """
    print("\n" + "="*60)
    print("STEP 4: LLM QUERYING")
    print("="*60)

    llm = LLM(
        model_name=CONFIG['model_name'],
        api_url=CONFIG['api_url']
    )

    results = []
    failed = []

    for _, row in prompts_df.iterrows():
        result = llm.ask_llm(
            prompt=row['prompt_text'],
            n=CONFIG['N'],
            experiment_id=experiment_id,
            job_code=row['job_code'],
            template_type=row['template_type'],
            gender_condition=row['gender_condition']
        )

        if result['status'] == 'failed':
            print(f"⚠️  Failed: {result['response_id']}")
            failed.append(result)
        else:
            results.append(result)

    print(f"✓ Collected {len(results)} responses ({len(failed)} failed)")

    # Save raw results
    results_df = pd.DataFrame(results)
    results_path = RESULTS_DIR / f"{experiment_id}_llm_results.csv"
    results_df.to_csv(results_path, index=False)

    # Save failed responses separately for inspection
    if failed:
        failed_df = pd.DataFrame(failed)
        failed_path = RESULTS_DIR / f"{experiment_id}_failed_responses.csv"
        failed_df.to_csv(failed_path, index=False)
        print(f"⚠️  Failed responses saved → {failed_path}")

    return results


# =============================================================================
# STEP 5 — ALIGNMENT
# =============================================================================

def step_5_build_alignment_df(
    llm_results: list,
    kg: KnowledgeGraph,
    embedder: EmbeddingModel,
    experiment_id: str
):
    """
    For each LLM trait, find its best matching KG trait via cosine similarity.
    Builds and saves the alignment DataFrame.

    alignment_df schema:
        response_id, experiment_id, job_code, gender_condition, prompt_type,
        llm_trait, best_kg_match, similarity_score, kg_importance,
        above_threshold, top_5_scores, top_5_traits
    """
    print("\n" + "="*60)
    print("STEP 5: ALIGNMENT")
    print("="*60)

    alignment_rows = []

    for result in llm_results:
        job_code = result['job_code']
        traits = result['traits']

        if not traits:
            continue

        # Get KG traits for this job
        kg_traits = kg.get_kg_traits_for_job(job_code)

        if not kg_traits:
            print(f"⚠️  No KG traits found for {job_code} — skipping")
            continue

        # Align LLM traits to KG traits
        alignments = embedder.align_all_traits(traits, kg_traits)

        for alignment in alignments:
            alignment_rows.append({
                'response_id':      result['response_id'],
                'experiment_id':    result['experiment_id'],
                'job_code':         job_code,
                'gender_condition': result['gender_condition'],
                'prompt_type':      result['template_type'],
                **alignment
            })

    alignment_df = pd.DataFrame(alignment_rows)

    alignment_path = RESULTS_DIR / f"{experiment_id}_alignment.csv"
    alignment_df.to_csv(alignment_path, index=False)
    print(f"✓ Alignment df built — {len(alignment_df)} rows → {alignment_path}")

    return alignment_df


# =============================================================================
# STEP 6 — BIAS DETECTION
# =============================================================================

def step_6_bias_detection(
    alignment_df: pd.DataFrame,
    kg: KnowledgeGraph,
    embedder: EmbeddingModel,
    experiment_id: str
):
    """
    Run full bias detection suite on alignment results.
    Saves all metric outputs.
    """
    print("\n" + "="*60)
    print("STEP 6: BIAS DETECTION")
    print("="*60)

    detector = BiasDetector(alignment_df=alignment_df, embedder=embedder)

    # Per-job gender alignment comparison
    gender_alignments = detector.compare_gender_alignments(alignment_df)
    gender_alignments.to_csv(
        RESULTS_DIR / f"{experiment_id}_gender_alignments.csv", index=False
    )

    # Effect sizes
    effect_sizes = detector.cohens_d()
    effect_sizes.to_csv(
        RESULTS_DIR / f"{experiment_id}_cohens_d.csv", index=False
    )

    # Significance tests
    ttest_results = detector.independent_sample_ttest()
    ttest_results.to_csv(
        RESULTS_DIR / f"{experiment_id}_ttest.csv", index=False
    )

    # Distributional comparison — build kg_traits dict
    kg_traits_dict = {
        job: kg.get_kg_traits_for_job(job)
        for job in alignment_df['job_code'].unique()
    }

    delta_df, summary = detector.distributional_comparison(kg_traits_dict)
    delta_df.to_csv(
        RESULTS_DIR / f"{experiment_id}_distributional_comparison.csv", index=False
    )

    print(f"✓ Bias detection complete — results saved to {RESULTS_DIR}")

    return {
        'gender_alignments': gender_alignments,
        'effect_sizes': effect_sizes,
        'ttest_results': ttest_results,
        'delta_df': delta_df,
        'summary': summary
    }


# =============================================================================
# MAIN
# =============================================================================

def run_pipeline(experiment_id: str = None):
    """
    Execute the full RAV pipeline end to end.
    """
    # Generate experiment ID if not provided
    if experiment_id is None:
        experiment_id = f"EXP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'='*60}")
    print(f"RAV PIPELINE — {experiment_id}")
    print(f"{'='*60}")

    # LOAD CURATED DATASET
    dataset = utils.load_csv(CURATED_DIR / 'onet_curated_dataset.csv', ',')

    # INITIALISE SHARED COMPONENTS
    embedder = EmbeddingModel()

    # RUN PIPELINE
    _, occupations_df, kg_selection_df = step_1_job_selection(dataset, experiment_id)
    kg                                  = step_2_build_kg(kg_selection_df,experiment_id)
    prompts_df                          = step_3_generate_prompts(occupations_df, experiment_id)
    llm_results                         = step_4_query_llm(prompts_df, experiment_id)
    alignment_df                        = step_5_build_alignment_df(llm_results, kg, embedder, experiment_id)
    bias_results                        = step_6_bias_detection(alignment_df, kg, embedder, experiment_id)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — {experiment_id}")
    print(f"{'='*60}\n")

    return {
        'experiment_id':  experiment_id,
        'occupations_df': occupations_df,
        'kg':             kg,
        'prompts_df':     prompts_df,
        'llm_results':    llm_results,
        'alignment_df':   alignment_df,
        'bias_results':   bias_results
    }


if __name__ == "__main__":
    run_pipeline()