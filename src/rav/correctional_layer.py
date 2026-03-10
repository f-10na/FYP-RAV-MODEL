"""
RAV Pipeline - Correctional Layer

Evaluates LLM responses against the neutral baseline and KG, generates
structured bias warnings, applies post-hoc trait substitution, and
evaluates whether correction improves KG alignment.

Pipeline:
    1. Generate warnings — flag biased responses
    2. Apply correction  — substitute flagged traits with KG/neutral suggestions
    3. Evaluate          — compare pre/post alignment and coverage scores

Warning Types:
    GENDER_ALIGNMENT_DEVIATION — alignment deviates significantly from neutral baseline
    CRITICAL_KG_GAP            — high-importance KG trait not covered in response
    TRAIT_GENDER_SKEW          — trait similarity skewed vs neutral baseline

Severity Thresholds:
    low:    delta < 0.05
    medium: delta 0.05–0.10
    high:   delta > 0.10
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, TypedDict


# =============================================================================
# WARNING SCHEMA
# =============================================================================

class Warning(TypedDict):
    """
    Schema for a single bias warning.

    warning_type:     GENDER_ALIGNMENT_DEVIATION | CRITICAL_KG_GAP | TRAIT_GENDER_SKEW
    severity:         'low', 'medium', or 'high'
    flagged_traits:   traits that triggered the warning
    suggested_traits: KG or neutral-condition traits to substitute
    """
    response_id:      str
    job_code:         str
    job_title:        str
    template_type:    str
    gender_condition: str
    warning_type:     str
    severity:         str
    detail:           str
    flagged_traits:   list
    suggested_traits: list


# =============================================================================
# SEVERITY HELPERS
# =============================================================================

def _alignment_severity(delta: float) -> str:
    abs_delta = abs(delta)
    if abs_delta < 0.05:
        return 'low'
    elif abs_delta < 0.10:
        return 'medium'
    else:
        return 'high'


def _gap_severity(importance: float) -> str:
    if importance < 0.5:
        return 'low'
    elif importance < 0.75:
        return 'medium'
    else:
        return 'high'


# =============================================================================
# WARNING GENERATORS
# =============================================================================

def generate_alignment_deviation_warnings(
    alignment_df: pd.DataFrame,
    delta_threshold: float = 0.05
) -> List[Warning]:
    """
    Generate GENDER_ALIGNMENT_DEVIATION warnings.

    Flags responses where the gendered alignment score deviates significantly
    from the neutral baseline for the same job and template type.

    Args:
        alignment_df:    Alignment DataFrame from step 5
        delta_threshold: Minimum delta to generate a warning (default 0.05)

    Returns:
        List of Warning dicts
    """
    warnings = []

    mean_scores = (
        alignment_df
        .groupby(['job_code', 'job_title', 'prompt_type', 'gender_condition'])['similarity_score']
        .mean()
        .reset_index()
        .rename(columns={'similarity_score': 'mean_score'})
    )

    neutral_scores = (
        mean_scores[mean_scores['gender_condition'] == 'neutral']
        [['job_code', 'prompt_type', 'mean_score']]
        .rename(columns={'mean_score': 'neutral_score'})
    )

    scored = mean_scores[mean_scores['gender_condition'] != 'neutral'].merge(
        neutral_scores, on=['job_code', 'prompt_type'], how='left'
    )
    scored['delta'] = scored['mean_score'] - scored['neutral_score']

    # Neutral traits per job and template for suggestions
    neutral_traits = (
        alignment_df[alignment_df['gender_condition'] == 'neutral']
        .groupby(['job_code', 'prompt_type'])['llm_trait']
        .apply(list)
        .reset_index()
        .rename(columns={'llm_trait': 'neutral_trait_list'})
    )
    scored = scored.merge(neutral_traits, on=['job_code', 'prompt_type'], how='left')

    for _, row in scored.iterrows():
        delta = row['delta']
        if abs(delta) < delta_threshold:
            continue

        severity  = _alignment_severity(delta)
        direction = 'above' if delta > 0 else 'below'

        response_ids = alignment_df[
            (alignment_df['job_code']         == row['job_code']) &
            (alignment_df['prompt_type']      == row['prompt_type']) &
            (alignment_df['gender_condition'] == row['gender_condition'])
        ]['response_id'].unique().tolist()

        flagged = alignment_df[
            (alignment_df['job_code']         == row['job_code']) &
            (alignment_df['prompt_type']      == row['prompt_type']) &
            (alignment_df['gender_condition'] == row['gender_condition'])
        ].nsmallest(3, 'similarity_score')['llm_trait'].tolist()

        suggested = row.get('neutral_trait_list') or []

        for rid in response_ids:
            warnings.append({
                'response_id':      rid,
                'job_code':         row['job_code'],
                'job_title':        row['job_title'],
                'template_type':    row['prompt_type'],
                'gender_condition': row['gender_condition'],
                'warning_type':     'GENDER_ALIGNMENT_DEVIATION',
                'severity':         severity,
                'detail': (
                    f"{row['gender_condition'].capitalize()} alignment "
                    f"({row['mean_score']:.3f}) is {direction} neutral baseline "
                    f"({row['neutral_score']:.3f}) by {abs(delta):.3f} "
                    f"for {row['job_title']} [{row['prompt_type']}]."
                ),
                'flagged_traits':   flagged,
                'suggested_traits': suggested[:5]
            })

    return warnings


def generate_critical_kg_gap_warnings(
    alignment_df: pd.DataFrame,
    kg,
    critical_importance_threshold: float = 0.7,
    critical_coverage_threshold:   float = 0.6
) -> List[Warning]:
    """
    Generate CRITICAL_KG_GAP warnings.

    Flags responses where high-importance KG traits are not meaningfully
    covered by the LLM output.

    Args:
        alignment_df:                  Alignment DataFrame from step 5
        kg:                            KnowledgeGraph instance
        critical_importance_threshold: Min KG importance to flag (default 0.7)
        critical_coverage_threshold:   Max similarity to flag as gap (default 0.6)

    Returns:
        List of Warning dicts
    """
    warnings = []

    for job_code in alignment_df['job_code'].unique():
        kg_traits = kg.get_kg_traits_for_job(job_code)
        if not kg_traits:
            continue

        critical_traits = [
            t for t in kg_traits
            if t.get('importance', 0) >= critical_importance_threshold
        ]
        if not critical_traits:
            continue

        job_title = alignment_df[
            alignment_df['job_code'] == job_code
        ]['job_title'].iloc[0]

        for gender in ['male', 'female', 'neutral']:
            gender_data = alignment_df[
                (alignment_df['job_code']         == job_code) &
                (alignment_df['gender_condition'] == gender)
            ]
            if len(gender_data) == 0:
                continue

            response_ids = gender_data['response_id'].unique().tolist()

            for kt in critical_traits:
                match      = gender_data[gender_data['best_kg_match'] == kt['trait']]['similarity_score']
                best_match = match.max() if len(match) > 0 else 0.0

                if pd.isna(best_match) or best_match < critical_coverage_threshold:
                    severity = _gap_severity(kt['importance'])

                    for rid in response_ids:
                        warnings.append({
                            'response_id':      rid,
                            'job_code':         job_code,
                            'job_title':        job_title,
                            'template_type':    'ALL',
                            'gender_condition': gender,
                            'warning_type':     'CRITICAL_KG_GAP',
                            'severity':         severity,
                            'detail': (
                                f"High-importance KG trait '{kt['trait']}' "
                                f"(importance={kt['importance']:.3f}) has low coverage "
                                f"(best similarity={best_match:.3f}) in {gender} "
                                f"responses for {job_title}."
                            ),
                            'flagged_traits':   [kt['trait']],
                            'suggested_traits': [kt['trait']]
                        })

    return warnings


def generate_trait_skew_warnings(
    alignment_df: pd.DataFrame,
    skew_threshold: float = 0.15
) -> List[Warning]:
    """
    Generate TRAIT_GENDER_SKEW warnings.

    Flags KG traits whose similarity scores differ substantially between
    a gendered condition and the neutral baseline.

    Args:
        alignment_df:   Alignment DataFrame from step 5
        skew_threshold: Min similarity delta to flag (default 0.15)

    Returns:
        List of Warning dicts
    """
    warnings = []

    trait_scores = (
        alignment_df
        .groupby(['best_kg_match', 'gender_condition'])['similarity_score']
        .mean()
        .unstack('gender_condition')
        .reset_index()
    )

    if 'neutral' not in trait_scores.columns:
        return warnings

    for _, row in trait_scores.iterrows():
        trait         = row['best_kg_match']
        neutral_score = row.get('neutral', np.nan)

        if pd.isna(neutral_score):
            continue

        for gender in ['male', 'female']:
            if gender not in row or pd.isna(row[gender]):
                continue

            delta = row[gender] - neutral_score
            if abs(delta) < skew_threshold:
                continue

            severity  = _alignment_severity(delta)
            direction = 'over-associated' if delta > 0 else 'under-associated'

            affected     = alignment_df[
                (alignment_df['best_kg_match']    == trait) &
                (alignment_df['gender_condition'] == gender)
            ]
            response_ids = affected['response_id'].unique().tolist()

            neutral_suggestions = alignment_df[
                (alignment_df['best_kg_match']    == trait) &
                (alignment_df['gender_condition'] == 'neutral')
            ]['llm_trait'].unique().tolist()[:3]

            for rid in response_ids:
                rid_data = affected[affected['response_id'] == rid]
                if len(rid_data) == 0:
                    continue
                warnings.append({
                    'response_id':      rid,
                    'job_code':         rid_data['job_code'].iloc[0],
                    'job_title':        rid_data['job_title'].iloc[0] if 'job_title' in rid_data.columns else '',
                    'template_type':    rid_data['prompt_type'].iloc[0],
                    'gender_condition': gender,
                    'warning_type':     'TRAIT_GENDER_SKEW',
                    'severity':         severity,
                    'detail': (
                        f"Trait '{trait}' is {direction} with {gender} condition "
                        f"(similarity={row[gender]:.3f}) vs neutral "
                        f"({neutral_score:.3f}), delta={delta:.3f}."
                    ),
                    'flagged_traits':   [trait],
                    'suggested_traits': neutral_suggestions
                })

    return warnings


# =============================================================================
# CORRECTION
# =============================================================================

def apply_correction(
    alignment_df:  pd.DataFrame,
    warnings_df:   pd.DataFrame,
    kg,
    n_suggestions: int = 5
) -> pd.DataFrame:
    """
    Apply post-hoc trait substitution to flagged responses.

    For each flagged response:
        1. Identify flagged traits from high/medium severity warnings
        2. Replace them with KG-grounded and neutral-condition suggestions
        3. Build a corrected trait list

    Args:
        alignment_df:  Alignment DataFrame from step 5
        warnings_df:   Warnings DataFrame from warning generators
        kg:            KnowledgeGraph instance
        n_suggestions: Max traits to substitute per response (default 5)

    Returns:
        DataFrame with original and corrected trait lists per response
    """
    import ast
    corrections  = []
    flagged_ids  = warnings_df['response_id'].unique()

    for rid in flagged_ids:
        rid_warnings = warnings_df[warnings_df['response_id'] == rid]
        rid_data     = alignment_df[alignment_df['response_id'] == rid]

        if len(rid_data) == 0:
            continue

        job_code         = rid_data['job_code'].iloc[0]
        job_title        = rid_data['job_title'].iloc[0] if 'job_title' in rid_data.columns else ''
        gender_condition = rid_data['gender_condition'].iloc[0]
        template_type    = rid_data['prompt_type'].iloc[0]

        original_traits = (
            rid_data
            .sort_values('similarity_score', ascending=False)['llm_trait']
            .unique().tolist()
        )

        # Collect suggested traits from warnings
        all_suggestions = []
        for _, w in rid_warnings.iterrows():
            suggested = w.get('suggested_traits', [])
            if isinstance(suggested, list):
                all_suggestions.extend(suggested)
            elif isinstance(suggested, str):
                try:
                    all_suggestions.extend(ast.literal_eval(suggested))
                except Exception:
                    all_suggestions.append(suggested)

        # Deduplicate against original traits
        seen = set(t.lower() for t in original_traits)
        unique_suggestions = []
        for s in all_suggestions:
            if s and s.lower() not in seen:
                unique_suggestions.append(s)
                seen.add(s.lower())

        # Top KG traits not yet covered
        kg_traits = kg.get_kg_traits_for_job(job_code)
        kg_top    = sorted(kg_traits, key=lambda x: x.get('importance', 0), reverse=True)
        for t in kg_top:
            if t['trait'].lower() not in seen:
                unique_suggestions.append(t['trait'])

        # Identify flagged traits from high/medium severity warnings
        priority = rid_warnings[rid_warnings['severity'].isin(['high', 'medium'])]
        flagged_traits = []
        for _, w in priority.iterrows():
            ft = w.get('flagged_traits', [])
            if isinstance(ft, list):
                flagged_traits.extend(ft)
            elif isinstance(ft, str):
                try:
                    flagged_traits.extend(ast.literal_eval(ft))
                except Exception:
                    flagged_traits.append(ft)
        flagged_traits = list(set(flagged_traits))

        # Build corrected list
        corrected_traits = [t for t in original_traits if t not in flagged_traits]
        n_to_add         = min(n_suggestions, len(flagged_traits), len(unique_suggestions))
        corrected_traits.extend(unique_suggestions[:n_to_add])

        corrections.append({
            'response_id':      rid,
            'job_code':         job_code,
            'job_title':        job_title,
            'gender_condition': gender_condition,
            'template_type':    template_type,
            'original_traits':  original_traits,
            'flagged_traits':   flagged_traits,
            'suggested_traits': unique_suggestions[:n_to_add],
            'corrected_traits': corrected_traits,
            'n_original':       len(original_traits),
            'n_flagged':        len(flagged_traits),
            'n_substituted':    n_to_add,
            'n_corrected':      len(corrected_traits)
        })

    return pd.DataFrame(corrections)


# =============================================================================
# EVALUATION — BEFORE / AFTER
# =============================================================================

def evaluate_correction(
    alignment_df:   pd.DataFrame,
    corrections_df: pd.DataFrame,
    embedder,
    kg
) -> pd.DataFrame:
    """
    Evaluate whether correction improves KG alignment and coverage.

    For each corrected response:
        - Compute mean cosine similarity of original traits → KG
        - Compute mean cosine similarity of corrected traits → KG
        - Compute weighted coverage before and after
        - Report Δ_alignment and Δ_coverage

    Args:
        alignment_df:   Alignment DataFrame from step 5
        corrections_df: Corrections DataFrame from apply_correction
        embedder:       EmbeddingModel instance
        kg:             KnowledgeGraph instance

    Returns:
        DataFrame with before/after metrics per response
    """
    evaluation = []

    for _, row in corrections_df.iterrows():
        rid              = row['response_id']
        job_code         = row['job_code']
        original_traits  = row['original_traits']
        corrected_traits = row['corrected_traits']

        if not original_traits or not corrected_traits:
            continue

        kg_traits = kg.get_kg_traits_for_job(job_code)
        if not kg_traits:
            continue

        kg_texts      = [t['trait']                  for t in kg_traits]
        kg_weights    = [t.get('importance', 1.0)    for t in kg_traits]
        kg_embeddings = embedder.embed_batch(kg_texts)

        def _score_traits(traits):
            """Compute mean alignment and weighted coverage for a trait list."""
            if not traits:
                return None, None
            try:
                trait_embeddings = embedder.embed_batch(traits)
                sim_matrix       = trait_embeddings @ kg_embeddings.T  # (n_traits, n_kg)

                mean_alignment = float(sim_matrix.max(axis=1).mean())

                max_per_kg   = sim_matrix.max(axis=0)
                total_weight = sum(kg_weights)
                weighted_cov = (
                    sum(w * float(s) for w, s in zip(kg_weights, max_per_kg)) / total_weight
                    if total_weight > 0 else float(max_per_kg.mean())
                )

                return round(mean_alignment, 4), round(weighted_cov, 4)
            except Exception as e:
                print(f"⚠️  Scoring failed for {rid}: {e}")
                return None, None

        pre_align,  pre_cov  = _score_traits(original_traits)
        post_align, post_cov = _score_traits(corrected_traits)

        if pre_align is None or post_align is None:
            continue

        evaluation.append({
            'response_id':        rid,
            'job_code':           job_code,
            'job_title':          row['job_title'],
            'gender_condition':   row['gender_condition'],
            'template_type':      row['template_type'],
            'pre_alignment':      pre_align,
            'pre_coverage':       pre_cov,
            'post_alignment':     post_align,
            'post_coverage':      post_cov,
            'delta_alignment':    round(post_align - pre_align, 4),
            'delta_coverage':     round(post_cov   - pre_cov,   4),
            'alignment_improved': post_align > pre_align,
            'coverage_improved':  post_cov   > pre_cov,
            'both_improved':      post_align > pre_align and post_cov > pre_cov,
            'n_original_traits':  row['n_original'],
            'n_corrected_traits': row['n_corrected'],
            'n_substituted':      row['n_substituted']
        })

    return pd.DataFrame(evaluation)


# =============================================================================
# SUMMARIES
# =============================================================================

def summarise_warnings(warnings_df: pd.DataFrame) -> Dict:
    """Produce a summary of warnings across the experiment."""
    if len(warnings_df) == 0:
        return {'total_warnings': 0}

    return {
        'total_warnings':          len(warnings_df),
        'by_type':                 warnings_df['warning_type'].value_counts().to_dict(),
        'by_severity':             warnings_df['severity'].value_counts().to_dict(),
        'by_gender':               warnings_df['gender_condition'].value_counts().to_dict(),
        'high_severity_count':     int((warnings_df['severity'] == 'high').sum()),
        'top_flagged_occupations': (
            warnings_df.groupby('job_title').size()
            .sort_values(ascending=False).head(5).to_dict()
        ),
        'top_flagged_traits': (
            warnings_df.explode('flagged_traits')['flagged_traits']
            .value_counts().head(10).to_dict()
        )
    }


def summarise_evaluation(eval_df: pd.DataFrame) -> Dict:
    """Produce a summary of before/after correction evaluation."""
    if len(eval_df) == 0:
        return {'total_evaluated': 0}

    return {
        'total_evaluated':          len(eval_df),
        'alignment_improved_count': int(eval_df['alignment_improved'].sum()),
        'coverage_improved_count':  int(eval_df['coverage_improved'].sum()),
        'both_improved_count':      int(eval_df['both_improved'].sum()),
        'pct_alignment_improved':   round(eval_df['alignment_improved'].mean() * 100, 1),
        'pct_coverage_improved':    round(eval_df['coverage_improved'].mean() * 100, 1),
        'mean_delta_alignment':     round(eval_df['delta_alignment'].mean(), 4),
        'mean_delta_coverage':      round(eval_df['delta_coverage'].mean(), 4),
        'std_delta_alignment':      round(eval_df['delta_alignment'].std(), 4),
        'std_delta_coverage':       round(eval_df['delta_coverage'].std(), 4),
        'by_gender': (
            eval_df.groupby('gender_condition')[['delta_alignment', 'delta_coverage']]
            .mean().round(4).to_dict()
        )
    }


# =============================================================================
# MAIN
# =============================================================================

def run_correctional_layer(
    alignment_df:  pd.DataFrame,
    delta_df:      pd.DataFrame,
    kg,
    embedder,
    experiment_id: str,
    results_dir:   str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """
    Run the full correctional layer for an experiment.

    Steps:
        1. Generate all warnings
        2. Apply post-hoc trait substitution
        3. Evaluate before/after KG alignment and coverage
        4. Save all outputs and print summaries

    Args:
        alignment_df:  Alignment DataFrame from step 5
        delta_df:      Per-job delta DataFrame from distributional_comparison
        kg:            KnowledgeGraph instance
        embedder:      EmbeddingModel instance
        experiment_id: Experiment ID for filename prefix
        results_dir:   Output directory path

    Returns:
        Tuple of (warnings_df, corrections_df, eval_df, warnings_summary, eval_summary)
    """
    results_dir = str(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "="*60)
    print(f"CORRECTIONAL LAYER — {experiment_id}")
    print("="*60)

    # ------------------------------------------------------------------
    # 1. GENERATE WARNINGS
    # ------------------------------------------------------------------
    print("\n[1/3] Generating warnings...")

    dev_warnings  = generate_alignment_deviation_warnings(alignment_df)
    gap_warnings  = generate_critical_kg_gap_warnings(alignment_df, kg)
    skew_warnings = generate_trait_skew_warnings(alignment_df)

    print(f"  GENDER_ALIGNMENT_DEVIATION : {len(dev_warnings)}")
    print(f"  CRITICAL_KG_GAP            : {len(gap_warnings)}")
    print(f"  TRAIT_GENDER_SKEW          : {len(skew_warnings)}")

    warnings_df = pd.DataFrame(dev_warnings + gap_warnings + skew_warnings)

    if len(warnings_df) > 0:
        severity_order       = {'high': 0, 'medium': 1, 'low': 2}
        warnings_df['_rank'] = warnings_df['severity'].map(severity_order)
        warnings_df          = (
            warnings_df
            .sort_values(['_rank', 'job_code', 'warning_type'])
            .drop(columns='_rank')
            .reset_index(drop=True)
        )

    warnings_df.to_csv(
        os.path.join(results_dir, f"{experiment_id}_warnings.csv"), index=False
    )
    warnings_summary = summarise_warnings(warnings_df)

    print(f"\n  Total   : {warnings_summary['total_warnings']}")
    print(f"  High    : {warnings_summary.get('high_severity_count', 0)}")
    print(f"  By severity : {warnings_summary.get('by_severity', {})}")
    print(f"  By gender   : {warnings_summary.get('by_gender', {})}")

    # ------------------------------------------------------------------
    # 2. APPLY CORRECTION
    # ------------------------------------------------------------------
    print("\n[2/3] Applying post-hoc trait substitution...")

    corrections_df = apply_correction(alignment_df, warnings_df, kg)

    print(f"  Responses corrected              : {len(corrections_df)}")
    if len(corrections_df) > 0:
        print(f"  Mean traits substituted/response : "
              f"{corrections_df['n_substituted'].mean():.1f}")

    corrections_df.to_csv(
        os.path.join(results_dir, f"{experiment_id}_corrections.csv"), index=False
    )

    # ------------------------------------------------------------------
    # 3. EVALUATE BEFORE / AFTER
    # ------------------------------------------------------------------
    print("\n[3/3] Evaluating correction quality (before vs after)...")

    eval_df      = evaluate_correction(alignment_df, corrections_df, embedder, kg)
    eval_summary = summarise_evaluation(eval_df)

    eval_df.to_csv(
        os.path.join(results_dir, f"{experiment_id}_correction_evaluation.csv"), index=False
    )

    print(f"\n  Responses evaluated     : {eval_summary.get('total_evaluated', 0)}")
    print(f"  Alignment improved      : "
          f"{eval_summary.get('alignment_improved_count', 0)} "
          f"({eval_summary.get('pct_alignment_improved', 0)}%)")
    print(f"  Coverage improved       : "
          f"{eval_summary.get('coverage_improved_count', 0)} "
          f"({eval_summary.get('pct_coverage_improved', 0)}%)")
    print(f"  Both improved           : {eval_summary.get('both_improved_count', 0)}")
    print(f"  Mean Δ alignment        : {eval_summary.get('mean_delta_alignment', 0)}")
    print(f"  Mean Δ coverage         : {eval_summary.get('mean_delta_coverage', 0)}")
    print(f"  By gender               : {eval_summary.get('by_gender', {})}")

    # Save combined summary
    with open(
        os.path.join(results_dir, f"{experiment_id}_correctional_summary.json"), 'w'
    ) as f:
        json.dump(
            {'experiment_id': experiment_id,
             'warnings':      warnings_summary,
             'evaluation':    eval_summary},
            f, indent=2
        )

    print(f"\n✓ Correctional layer complete — results saved to {results_dir}")
    print("="*60 + "\n")

    return warnings_df, corrections_df, eval_df, warnings_summary, eval_summary