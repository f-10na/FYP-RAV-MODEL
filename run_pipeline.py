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
from src.pipeline.select_jobs import (run_job_selection)
from src.pipeline.generate_prompts import (build_prompts_for_experiment)
from src.rav.auditor import Auditor

# =============================================================================
# CONFIG
# =============================================================================


PROJECT_ROOT = utils.find_project_root()
#aim for 360 LLM queries per experiment (5 groups x 8 jobs x 3 templates x 3 gender roles)
CONFIG = {
    'seed':            42, #42,123,7
    'n_groups':        5,
    'jobs_per_group':  8,
    'N':               10,   # number of traits to elicit per prompt
    'KG_EVAL_CAP':     15,  #top-N KG traits for coverage and density metrics (1.5*CONFIG['N'])
    'model_name': 'llama3.1:8b',  #llama3.1:8b,mistral:7b
    'api_url':         'http://localhost:11434/api/chat',
}

# DIRECTORIES
DATA_DIR        = PROJECT_ROOT / 'data'
CURATED_DIR     = DATA_DIR / 'onet_datasets/curated'
EXPERIMENTS_DIR     = DATA_DIR / 'experiments'
RESULTS_DIR     = DATA_DIR / 'results'

# =============================================================================
# EXPERIMENT DIRECTORY SETUP
# =============================================================================
def _setup_experiment_dirs(experiment_id: str):
    """Create all experiment-specific subdirectories.
        Returns:
            exp_results_dir: data/results/{experiment_id}/
            exp_graphs_dir:  data/results/{experiment_id}/graphs/
            exp_prompts_dir: data/experiments/{experiment_id}/
    """
    exp_results_dir = RESULTS_DIR / experiment_id
    exp_graphs_dir  = exp_results_dir / 'graphs'
    exp_prompts_dir = EXPERIMENTS_DIR / experiment_id

    for d in [exp_results_dir, exp_graphs_dir, exp_prompts_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return exp_results_dir, exp_graphs_dir, exp_prompts_dir


# =============================================================================
# STEP 1 — JOB SELECTION
# =============================================================================

def step_1_job_selection(
        dataset: pd.DataFrame, 
        experiment_id: str,
        exp_prompts_dir: Path
):
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
        output_dir=exp_prompts_dir 
    )

    return selected_codes, occupations_df, kg_selection_df


# =============================================================================
# STEP 2 — KG CONSTRUCTION
# =============================================================================

def step_2_build_kg(
    kg_selection_df: pd.DataFrame, 
    experiment_id: str,
    exp_graphs_dir: Path
):
    """
    Build occupational Knowledge Graph from selected jobs.
    Saves full KG and sample subgraph visualisations to exp_graphs_dir.
    """
    print("\n" + "="*60)
    print("STEP 2: KG CONSTRUCTION")
    print("="*60)

    kg = KnowledgeGraph()
    kg.build_KG(kg_selection_df)

    print(f"KG built — {kg.G.number_of_nodes()} nodes, {kg.G.number_of_edges()} edges")

    # Visualise full KG
    kg_viz_path = str(exp_graphs_dir / f"{experiment_id}_kg.html")
    kg.visualize_interactive_graph(filename=kg_viz_path)

    # Visualise one job as example subgraph
    sample_job = kg_selection_df['job_code'].iloc[0]
    subgraph_path = str(exp_graphs_dir / f"{experiment_id}_{sample_job.replace('.', '_')}_subgraph.html")
    kg.visualize_job_subgraph(job_code=sample_job, filename=subgraph_path)

    return kg

# =============================================================================
# STEP 3 — PROMPT GENERATION
# =============================================================================

def step_3_generate_prompts(
    occupations_df: pd.DataFrame,
    experiment_id: str,
    exp_prompts_dir: Path
):
    """
    Generate T1 and T2 prompts for all occupations and gender conditions.
    Saves prompts CSV.
    """
    print("\n" + "="*60)
    print("STEP 3: PROMPT GENERATION")
    print("="*60)

    prompts_df = build_prompts_for_experiment(occupations_df, N=CONFIG['N'])

    prompts_path = exp_prompts_dir / f"{experiment_id}_prompts.csv"
    prompts_df.to_csv(prompts_path, index=False)
    print(f"Generated {len(prompts_df)} prompts → {prompts_path}")

    return prompts_df


# =============================================================================
# STEP 4 — LLM QUERYING
# =============================================================================

def step_4_query_llm(
    prompts_df: pd.DataFrame,
    experiment_id: str,
    exp_results_dir: Path
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
            job_title=row['job_title'],
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
    results_path = exp_results_dir / f"{experiment_id}_llm_results.csv"
    results_df.to_csv(results_path, index=False)

    # Save failed responses separately for inspection
    if failed:
        failed_df = pd.DataFrame(failed)
        failed_path = exp_results_dir / f"{experiment_id}_failed_responses.csv"
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
    experiment_id: str,
    exp_results_dir: Path
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
                'job_title':        result['job_title'],
                'gender_condition': result['gender_condition'],
                'prompt_type':      result['template_type'],
                **alignment
            })

    alignment_df = pd.DataFrame(alignment_rows)

    alignment_path = exp_results_dir / f"{experiment_id}_alignment.csv"
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
    experiment_id: str,
    exp_results_dir: Path
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
        exp_results_dir / f"{experiment_id}_gender_alignments.csv", index=False
    )

    # Effect sizes
    effect_sizes = detector.cohens_d()
    effect_sizes.to_csv(
        exp_results_dir / f"{experiment_id}_cohens_d.csv", index=False
    )

    # Significance tests
    ttest_results = detector.independent_sample_ttest()
    ttest_results.to_csv(
        exp_results_dir / f"{experiment_id}_ttest.csv", index=False
    )

    # Distributional comparison — build kg_traits dict capped to top KG_EVAL_CAP
    kg_traits_dict = {
        job: sorted(
            kg.get_kg_traits_for_job(job),
            key=lambda x: -x.get('importance', 0)
        )[:CONFIG['KG_EVAL_CAP']]
        for job in alignment_df['job_code'].unique()
    }

    delta_df, summary = detector.distributional_comparison(kg_traits_dict)
    delta_df.to_csv(
        exp_results_dir / f"{experiment_id}_distributional_comparison.csv", index=False
    )

    print(f"✓ Bias detection complete — results saved to {exp_results_dir}")

    return {
        'gender_alignments': gender_alignments,
        'effect_sizes': effect_sizes,
        'ttest_results': ttest_results,
        'delta_df': delta_df,
        'summary': summary
    }

from src.rav.visualiser import run_visualisations

# STEP 7 — VISUALISATIONS
def step_7_visualisations(
    alignment_df, 
    llm_results,
    experiment_id,
    exp_graphs_dir
):
    print("\n" + "="*60)
    print("STEP 7: VISUALISATIONS")
    print("="*60)
    run_visualisations(alignment_df, llm_results, experiment_id, exp_graphs_dir)


# STEP 8 — CORRECTIONAL LAYER
def step_8_audit(alignment_df, kg, embedder, experiment_id, exp_results_dir):
    print("\n" + "="*60)
    print("STEP 8: REPRESENTATION AUDIT")
    print("="*60)
    auditor = Auditor(alignment_df, kg, embedder)
    return auditor.run(experiment_id, exp_results_dir)

# =============================================================================
# MAIN
# =============================================================================

def run_pipeline(experiment_id: str = None):
    if experiment_id is None:
        experiment_id = f"EXP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'='*60}")
    print(f"RAV PIPELINE — {experiment_id}")
    print(f"{'='*60}")

    # Setup dirs
    exp_results_dir, exp_graphs_dir, exp_prompts_dir = _setup_experiment_dirs(experiment_id)

    # Load dataset
    dataset = utils.load_csv(CURATED_DIR / 'onet_curated_dataset.csv', ',')

    # Initialise shared components
    embedder = EmbeddingModel()

    # Run pipeline
    _, occupations_df, kg_selection_df = step_1_job_selection(dataset, experiment_id, exp_prompts_dir)
    kg                                  = step_2_build_kg(kg_selection_df, experiment_id, exp_graphs_dir)
    prompts_df                          = step_3_generate_prompts(occupations_df, experiment_id, exp_prompts_dir)
    llm_results                         = step_4_query_llm(prompts_df, experiment_id, exp_results_dir)
    alignment_df                        = step_5_build_alignment_df(llm_results, kg, embedder, experiment_id, exp_results_dir)
    # return capped traits
    original_get_kg_traits = kg.get_kg_traits_for_job
    def _capped_get_kg_traits(job_code):
        traits = original_get_kg_traits(job_code)
        return sorted(traits, key=lambda x: -x.get('importance', 0))[:CONFIG['KG_EVAL_CAP']]
    kg.get_kg_traits_for_job = _capped_get_kg_traits
    bias_results                        = step_6_bias_detection(alignment_df, kg, embedder, experiment_id, exp_results_dir)
    step_7_visualisations(alignment_df, llm_results, experiment_id, exp_graphs_dir)
    audit_df, gap_df, audit_summary, gap_summary = step_8_audit(
    alignment_df, kg, embedder, experiment_id, exp_results_dir
)

if __name__ == "__main__":
    run_pipeline()