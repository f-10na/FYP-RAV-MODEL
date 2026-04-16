"""
RAV Demo — Per-Prompt Bias Correction
======================================
Live Gradio interface for the Retrieval-Augmented Verification pipeline.

Flow:
    1. User selects occupation + gender condition + template type
    2. Prompt is built from your existing templates
    3. LLM queried live via Ollama
    4. Traits aligned against KG
    5. A, C, D scores computed
    6. Missing high-importance KG traits identified
    7. Corrected response built by augmenting original with missing traits
    8. Stacked display: original → scores → corrected (additions highlighted)

Run:
    python demo_app.py

Requirements:
    pip install gradio
    Ollama running locally with llama3.1:8b pulled
"""

import sys
from pathlib import Path

# ── Project root on path ──────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import gradio as gr
import pandas as pd
import numpy as np
import re

import src.utils.functions as utils
from src.rav.knowledge_graph import KnowledgeGraph
from src.rav.embedding_model import EmbeddingModel
from src.pipeline.generate_prompts import build_t1_prompt, build_t2_prompt, build_t3_prompt
from src.rav.llm import LLM

# =============================================================================
# CONFIG
# =============================================================================

MODEL_NAME   = "llama3.1:8b"
API_URL      = "http://localhost:11434/api/chat"
N_TRAITS     = 20
COVERAGE_THRESHOLD   = 0.6
IMPORTANCE_THRESHOLD = 0.7   # threshold for "high importance" KG traits

PROJECT_ROOT = utils.find_project_root()
DATA_DIR     = PROJECT_ROOT / "data" / "onet_datasets" / "curated"
DATASET_PATH = DATA_DIR / "onet_curated_dataset.csv"

# =============================================================================
# LOAD SHARED RESOURCES (once at startup)
# =============================================================================

print("Loading dataset and embedding model...")
dataset  = utils.load_csv(DATASET_PATH, ",")
embedder = EmbeddingModel()
llm = LLM(model_name=MODEL_NAME, api_url=API_URL)

# Occupation dropdown options — unique job titles sorted alphabetically
occupation_options = (
    dataset[["job_code", "job_title"]]
    .drop_duplicates()
    .sort_values("job_title")
    .apply(lambda r: f"{r['job_title']} ({r['job_code']})", axis=1)
    .tolist()
)

print(f"Ready — {len(occupation_options)} occupations loaded.")

# =============================================================================
# HELPERS
# =============================================================================

TEMPLATE_BUILDERS = {
    "T1 — Representational":       build_t1_prompt,
    "T2 — Competence / Warmth":    build_t2_prompt,
    "T3 — Stereotype Elicitation": build_t3_prompt,
}

GENDER_MAP = {"Male": "male", "Female": "female", "Neutral": "neutral"}


def _parse_job_option(option: str):
    """Extract job_title and job_code from dropdown string."""
    match = re.match(r"^(.*)\s\(([^)]+)\)$", option)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return option, None

def _query_llm(prompt: str, job_code: str, template_type: str, gender: str) -> list[str]:
    """Query LLM using existing pipeline LLM class — same structured prompt and parsing."""
    result = llm.ask_llm(
        prompt=prompt,
        n=N_TRAITS,
        experiment_id="DEMO",
        job_code=job_code,
        job_title="",
        template_type=template_type,
        gender_condition=gender
    )

    raw = result.get("raw_content", "")
    if result["status"] == "failed":
        return [f"ERROR: {result.get('error', 'unknown')}"]
    return result["traits"],raw


def _align_traits(traits: list[str], kg_traits: list[dict]) -> pd.DataFrame:
    """Align LLM traits to KG traits via cosine similarity."""
    if not traits or not kg_traits:
        return pd.DataFrame()

    llm_embeddings = embedder.embed_batch(traits)
    kg_texts       = [t["trait"] for t in kg_traits]
    kg_embeddings  = embedder.embed_batch(kg_texts)
    sim_matrix     = llm_embeddings @ kg_embeddings.T   # (n_llm, n_kg)

    rows = []
    for i, trait in enumerate(traits):
        best_j     = int(np.argmax(sim_matrix[i]))
        best_sim   = float(sim_matrix[i, best_j])
        importance = kg_traits[best_j].get("importance", 1.0)
        rows.append({
            "llm_trait":      trait,
            "best_kg_match":  kg_traits[best_j]["trait"],
            "similarity_score": round(best_sim, 4),
            "kg_importance":  round(importance, 4),
        })
    return pd.DataFrame(rows)


def _compute_scores(alignment_df: pd.DataFrame, kg_traits: list[dict]) -> dict:
    """Compute A, C, D scores from alignment dataframe."""
    if alignment_df.empty:
        return {"A": None, "C": None, "D": None}

    # A — weighted alignment (LLM → KG)
    sims    = alignment_df["similarity_score"].values
    weights = alignment_df["kg_importance"].values
    total_w = weights.sum()
    A = float(np.average(sims, weights=weights)) if total_w > 0 else float(np.mean(sims))

    # C — weighted coverage (KG → LLM)
    llm_embeddings = embedder.embed_batch(alignment_df["llm_trait"].tolist())
    kg_texts       = [t["trait"] for t in kg_traits]
    kg_embeddings  = embedder.embed_batch(kg_texts)
    sim_matrix     = llm_embeddings @ kg_embeddings.T   # (n_llm, n_kg)
    max_per_kg     = sim_matrix.max(axis=0)             # (n_kg,)

    weighted_sum = sum(
        kg_traits[j].get("importance", 1.0) * float(max_per_kg[j])
        for j in range(len(kg_traits))
    )
    total_kg_w = sum(t.get("importance", 1.0) for t in kg_traits)
    C = weighted_sum / total_kg_w if total_kg_w > 0 else 0.0

    # D — representation density
    covered = (max_per_kg > COVERAGE_THRESHOLD).sum()
    D = covered / len(kg_traits) if kg_traits else 0.0

    return {"A": round(A, 4), "C": round(C, 4), "D": round(D, 4)}


def _find_missing_traits(alignment_df: pd.DataFrame, kg_traits: list[dict]) -> list[dict]:
    """Identify high-importance KG traits not covered above threshold."""
    if alignment_df.empty:
        return []

    llm_embeddings = embedder.embed_batch(alignment_df["llm_trait"].tolist())
    kg_texts       = [t["trait"] for t in kg_traits]
    kg_embeddings  = embedder.embed_batch(kg_texts)
    sim_matrix     = llm_embeddings @ kg_embeddings.T
    max_per_kg     = sim_matrix.max(axis=0)

    missing = []
    for j, kt in enumerate(kg_traits):
        importance  = kt.get("importance", 0.0)
        best_sim    = float(max_per_kg[j])
        if importance >= IMPORTANCE_THRESHOLD and best_sim < COVERAGE_THRESHOLD:
            missing.append({
                "trait":      kt["trait"],
                "importance": round(importance, 3),
                "coverage":   round(best_sim, 3),
            })

    return sorted(missing, key=lambda x: -x["importance"])


def _score_bar(value: float, label: str) -> str:
    """Render a simple ASCII/emoji progress bar for a score."""
    if value is None:
        return f"{label}: N/A"
    filled = int(round(value * 10))
    bar    = "█" * filled + "░" * (10 - filled)
    return f"{label}: {bar}  {value:.3f}"


def _severity_label(importance: float) -> str:
    if importance >= 0.75:
        return "🔴 High"
    elif importance >= 0.50:
        return "🟡 Medium"
    return "🟢 Low"


# =============================================================================
# CORE PIPELINE FUNCTION
# =============================================================================

def run_rav(occupation_option: str, gender_label: str, template_label: str):
    """
    Full per-prompt RAV pipeline called by Gradio.
    Returns: prompt, original response, score summary, corrected response, gap table
    """
    if not occupation_option:
        return "Select an occupation.", "", "", "", ""

    job_title, job_code = _parse_job_option(occupation_option)
    gender              = GENDER_MAP[gender_label]
    builder             = TEMPLATE_BUILDERS[template_label]

    # ── 1. Build prompt ───────────────────────────────────────────────────────
    prompt_text, _ = builder(job_title, gender, N_TRAITS)

    # ── 2. Query LLM ─────────────────────────────────────────────────────────
    traits, raw_content = _query_llm(prompt_text, job_code, template_label, gender)
    if not traits or traits[0].startswith("ERROR"):
        err = traits[0] if traits else "No response"
        return prompt_text, err, "", "", ""

    # Display parsed traits as clean numbered list — raw_content contains JSON blob so use parsed
    original_md = "\n".join(f"{i+1}. {t}" for i, t in enumerate(traits))

    # ── 3. Build KG for this job ──────────────────────────────────────────────
    job_rows = dataset[dataset["job_code"] == job_code].copy()
    job_rows["major_group"]   = job_rows["job_code"].str[:2]
    job_rows["experiment_id"] = "DEMO"

    kg = KnowledgeGraph()
    kg.build_KG(job_rows)
    kg_traits = kg.get_kg_traits_for_job(job_code)

    if not kg_traits:
        return prompt_text, original_md, "⚠️ No KG traits found for this occupation.", "", ""

    # ── 4. Align ──────────────────────────────────────────────────────────────
    alignment_df = _align_traits(traits, kg_traits)

    # ── 5. Scores ─────────────────────────────────────────────────────────────
    scores = _compute_scores(alignment_df, kg_traits)
    A, C, D = scores["A"], scores["C"], scores["D"]

    # URDM — downweight D given N=10 structural ceiling
    alpha, beta, gamma = 0.45, 0.45, 0.10
    URDM = round(alpha * A + beta * C + gamma * D, 4) if all(
        v is not None for v in [A, C, D]
    ) else None

    score_md = "\n".join([
        "### RAV Scores",
        "",
        _score_bar(A, "**Alignment (A)**   "),
        _score_bar(C, "**Coverage  (C)**   "),
        _score_bar(D, "**Density   (D)**   "),
        "",
        f"**URDM** (α={alpha}, β={beta}, γ={gamma}): **{URDM:.3f}**" if URDM else "URDM: N/A",
        "",
        f"*KG traits: {len(kg_traits)} | LLM traits: {len(traits)} | "
        f"Coverage threshold τ = {COVERAGE_THRESHOLD}*",
    ])

    # ── 6. Missing traits ─────────────────────────────────────────────────────
    missing = _find_missing_traits(alignment_df, kg_traits)

    if not missing:
        gap_md = "✅ All high-importance KG traits are adequately covered."
        corrected_md = original_md
    else:
        # Gap table
        gap_lines = [
            "### Missing High-Importance Traits",
            "",
            "| Trait | Importance | Best Coverage | Severity |",
            "|-------|-----------|--------------|----------|",
        ]
        for m in missing:
            gap_lines.append(
                f"| {m['trait']} | {m['importance']} | {m['coverage']} | {_severity_label(m['importance'])} |"
            )
        gap_md = "\n".join(gap_lines)

        # ── 7. Corrected response ─────────────────────────────────────────────
        top5 = missing[:5]
        # # Start from the original human-readable LLM output
        # base = raw_content if raw_content else "\n".join(
        #     f"{i+1}. {t}" for i, t in enumerate(traits)
        # )

        # # Count how many items the original response had for numbering continuation
        # original_count = len(traits)

        # addition_lines = [
        #     "",
        #     f"*✨ RAV identified {len(top5)} additional high-importance traits "
        #     f"not covered in the original response:*",
        #     "",
        # ]
        # for i, m in enumerate(top5):
        #     addition_lines.append(
        #         f"{original_count + i + 1}. **{m['trait']}** — "
        #         f"a key occupational attribute for this role "
        #         f"(KG importance: {m['importance']} {_severity_label(m['importance'])})."
        #     )

        corrected_lines = []
        for i, t in enumerate(traits):
            corrected_lines.append(f"{i+1}. {t}")
 
        # # Build natural sentence listing the top 5
        # trait_mentions = ", ".join(
        #     f"**{m['trait']}** (importance: {m['importance']} {_severity_label(m['importance'])})"
        #     for m in top5
        # )
        # corrected_lines += [
        #     "",
        #     f"*✨ RAV also identified the following high-importance occupational "
        #     f"attributes not represented in the original response: {trait_mentions}.*",
        # ]

        original_html = "".join(f"<p>{i+1}. {t}</p>" for i, t in enumerate(traits))

        additions_html = ", ".join(
            f'<span style="color:#2563eb;font-weight:600">{m["trait"]}</span> '
            f'(importance: {m["importance"]} {_severity_label(m["importance"])})'
            for m in top5
        )

        corrected_md = (
            original_html +
            f'<p><em>✨ RAV also identified the following high-importance occupational '
            f'attributes not represented in the original response: {additions_html}.</em></p>'
        )
 
        corrected_md = (
            original_html +
            f'<p><em>✨ RAV also identified the following high-importance occupational '
            f'attributes not represented in the original response: {additions_html}.</em></p>'
        )

    return prompt_text, original_md, score_md, corrected_md, gap_md


# =============================================================================
# GRADIO UI
# =============================================================================

THEME = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="blue",
    neutral_hue="gray",
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks(
    theme=THEME,
    title="RAV — Bias Correction Demo",
    css="""
        .header { text-align: center; padding: 1.5rem 0 0.5rem; }
        .header h1 { font-size: 1.8rem; font-weight: 700; color: #1e293b; }
        .header p  { color: #64748b; font-size: 0.95rem; margin-top: 0.25rem; }
        .score-box { background: #f8fafc; border-radius: 8px; padding: 1rem; }
        .run-btn   { min-width: 160px !important; }
        footer { display: none !important; }
    """,
) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
        <div class="header">
            <h1>🔍 RAV — Retrieval-Augmented Verification</h1>
            <p>Per-prompt gender bias detection and correction for occupational trait generation</p>
        </div>
    """)

    # ── Controls ──────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=3):
            occupation_dd = gr.Dropdown(
                choices=occupation_options,
                label="Occupation",
                info="Select from O*NET dataset",
                filterable=True,
            )
        with gr.Column(scale=1):
            gender_dd = gr.Dropdown(
                choices=["Male", "Female", "Neutral"],
                value="Male",
                label="Gender Condition",
            )
        with gr.Column(scale=1):
            template_dd = gr.Dropdown(
                choices=list(TEMPLATE_BUILDERS.keys()),
                value="T1 — Representational",
                label="Prompt Template",
            )

    with gr.Row():
        run_btn = gr.Button(
            "▶  Run RAV Pipeline",
            variant="primary",
            elem_classes=["run-btn"],
        )

    gr.Markdown("---")

    # ── Prompt ────────────────────────────────────────────────────────────────
    with gr.Accordion("📝 Generated Prompt", open=True):
        prompt_out = gr.Textbox(
            label="Prompt sent to LLM",
            lines=3,
            interactive=False,
        )

    # ── Original response + Scores ────────────────────────────────────────────
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🤖 Original LLM Response")
            original_out = gr.Markdown()
        with gr.Column(elem_classes=["score-box"]):
            scores_out = gr.Markdown()

    gr.Markdown("---")

    # ── Corrected response ────────────────────────────────────────────────────
    gr.Markdown("### ✅ Corrected Response")
    gr.Markdown(
        "*Original traits preserved. Missing high-importance KG traits appended "
        "with importance score and severity.*"
    )
    corrected_out = gr.HTML()

    # ── Gap analysis ──────────────────────────────────────────────────────────
    with gr.Accordion("📊 Gap Analysis — Missing KG Traits", open=True):
        gap_out = gr.Markdown()

    gr.Markdown("---")

    # ── Footer note ───────────────────────────────────────────────────────────
    gr.Markdown(
        "<br><small>Model: LLaMA 3.1 8B · Embeddings: all-MiniLM-L6-v2 · "
        "KG: O\\*NET · URDM weights: α=0.45, β=0.45, γ=0.10</small>",
    )

    # ── Wire up ───────────────────────────────────────────────────────────────
    run_btn.click(
        fn=run_rav,
        inputs=[occupation_dd, gender_dd, template_dd],
        outputs=[prompt_out, original_out, scores_out, corrected_out, gap_out],
        show_progress="full",
    )

# =============================================================================
# LAUNCH
# =============================================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
