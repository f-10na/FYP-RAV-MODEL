"""
RAV Pipeline - Representation Auditor

KG-guided representation auditing of LLM-generated occupational trait responses.

The Auditor does NOT modify LLM outputs. Instead it diagnoses representation
gaps by comparing LLM responses against the KG ground truth and neutral baseline,
surfacing what is missing, skewed, or underrepresented.

Audit Report (per response):
    - Which high-importance KG traits are missing or poorly covered?
    - Which traits are over/under-represented vs the neutral baseline?
    - What traits would a complete, unbiased response include?

Gap Scores (per response):
    - representation_gap:  distance from full KG coverage
    - alignment_gap:       distance from neutral baseline alignment
    - overall_gap:         combined gap score

Audit Types:
    REPRESENTATION_GAP   — high-importance KG trait missing or poorly covered
    ALIGNMENT_DEVIATION  — alignment score deviates from neutral baseline
    TRAIT_SKEW           — trait similarity skewed vs neutral baseline

Severity:
    low:    delta < 0.05  | importance < 0.50
    medium: delta 0.05–0.10 | importance 0.50–0.75
    high:   delta > 0.10  | importance > 0.75
"""

import os
import ast
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, TypedDict


# =============================================================================
# AUDIT FINDING SCHEMA
# =============================================================================

class AuditFinding(TypedDict):
    """
    Schema for a single audit finding.

    audit_type:           REPRESENTATION_GAP | ALIGNMENT_DEVIATION | TRAIT_SKEW
    severity:             'low', 'medium', or 'high'
    missing_traits:       KG traits absent or poorly covered in this response
    recommended_traits:   traits a complete unbiased response should include
    """
    response_id:          str
    job_code:             str
    job_title:            str
    template_type:        str
    gender_condition:     str
    audit_type:           str
    severity:             str
    detail:               str
    missing_traits:       list
    recommended_traits:   list


# =============================================================================
# AUDITOR CLASS
# =============================================================================

class Auditor:
    """
    KG-guided representation auditor for LLM-generated occupational trait responses.

    Diagnoses representation gaps, alignment deviations, and trait skew
    without modifying the original LLM output.

    Usage:
        auditor = Auditor(alignment_df, kg, embedder)
        audit_df, gap_scores_df, summary = auditor.run(experiment_id, results_dir)
    """

    # Severity thresholds
    DELTA_LOW    = 0.05
    DELTA_MEDIUM = 0.10
    GAP_LOW      = 0.50
    GAP_MEDIUM   = 0.75
    SKEW_THRESHOLD         = 0.15
    COVERAGE_THRESHOLD     = 0.6
    IMPORTANCE_THRESHOLD   = 0.7

    def __init__(self, alignment_df: pd.DataFrame, kg, embedder):
        """
        Args:
            alignment_df: Alignment DataFrame from pipeline step 5
            kg:           KnowledgeGraph instance
            embedder:     EmbeddingModel instance
        """
        self.alignment_df = alignment_df
        self.kg           = kg
        self.embedder     = embedder

    # -------------------------------------------------------------------------
    # PRIVATE HELPERS
    # -------------------------------------------------------------------------

    def _severity_from_delta(self, delta: float) -> str:
        abs_d = abs(delta)
        if abs_d < self.DELTA_LOW:
            return 'low'
        elif abs_d < self.DELTA_MEDIUM:
            return 'medium'
        else:
            return 'high'

    def _severity_from_importance(self, importance: float) -> str:
        if importance < self.GAP_LOW:
            return 'low'
        elif importance < self.GAP_MEDIUM:
            return 'medium'
        else:
            return 'high'

    def _parse_list(self, value) -> list:
        """Safely parse a list field that may have been serialised as string."""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except Exception:
                return [value]
        return []

    # -------------------------------------------------------------------------
    # AUDIT GENERATORS
    # -------------------------------------------------------------------------

    def audit_representation_gaps(self) -> List[AuditFinding]:
        """
        Audit REPRESENTATION_GAP findings.

        For each response, identifies high-importance KG traits that are
        absent or poorly covered. These are the traits a complete occupational
        profile should include but the LLM failed to generate.

        Returns:
            List of AuditFinding dicts
        """
        findings = []
        df       = self.alignment_df

        for job_code in df['job_code'].unique():
            kg_traits = self.kg.get_kg_traits_for_job(job_code)
            if not kg_traits:
                continue

            # Only audit high-importance KG traits
            critical_traits = [
                t for t in kg_traits
                if t.get('importance', 0) >= self.IMPORTANCE_THRESHOLD
            ]
            if not critical_traits:
                continue

            job_title = df[df['job_code'] == job_code]['job_title'].iloc[0]

            for gender in ['male', 'female', 'neutral']:
                gender_data = df[
                    (df['job_code']         == job_code) &
                    (df['gender_condition'] == gender)
                ]
                if len(gender_data) == 0:
                    continue

                response_ids = gender_data['response_id'].unique().tolist()

                # Collect all missing critical traits for this gender/job
                missing     = []
                recommended = []

                for kt in critical_traits:
                    match      = gender_data[gender_data['best_kg_match'] == kt['trait']]['similarity_score']
                    best_match = float(match.max()) if len(match) > 0 else 0.0

                    if pd.isna(best_match) or best_match < self.COVERAGE_THRESHOLD:
                        missing.append({
                            'trait':       kt['trait'],
                            'importance':  round(kt['importance'], 3),
                            'best_coverage': round(best_match, 3)
                        })
                        recommended.append(kt['trait'])

                if not missing:
                    continue

                # Severity based on highest importance missing trait
                max_importance = max(m['importance'] for m in missing)
                severity       = self._severity_from_importance(max_importance)

                missing_names = [m['trait'] for m in missing]
                detail = (
                    f"{len(missing)} high-importance KG trait(s) poorly covered "
                    f"in {gender} responses for {job_title}: "
                    f"{', '.join(missing_names[:3])}"
                    f"{'...' if len(missing_names) > 3 else ''}."
                )

                for rid in response_ids:
                    findings.append({
                        'response_id':        rid,
                        'job_code':           job_code,
                        'job_title':          job_title,
                        'template_type':      'ALL',
                        'gender_condition':   gender,
                        'audit_type':         'REPRESENTATION_GAP',
                        'severity':           severity,
                        'detail':             detail,
                        'missing_traits':     missing_names,
                        'recommended_traits': recommended
                    })

        return findings

    def audit_alignment_deviations(
        self,
        delta_threshold: float = 0.05
    ) -> List[AuditFinding]:
        """
        Audit ALIGNMENT_DEVIATION findings.

        For each response, identifies where the gendered alignment score
        deviates significantly from the neutral baseline. Surfaces which
        traits are present in the gendered response but absent from the
        neutral condition — the likely source of the deviation.

        Args:
            delta_threshold: Minimum delta to generate a finding (default 0.05)

        Returns:
            List of AuditFinding dicts
        """
        findings = []
        df       = self.alignment_df

        mean_scores = (
            df
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

        # Neutral traits per job/template — used to identify what's missing
        neutral_trait_map = (
            df[df['gender_condition'] == 'neutral']
            .groupby(['job_code', 'prompt_type'])['llm_trait']
            .apply(set)
            .to_dict()
        )

        # Gendered traits per job/template
        gendered_trait_map = (
            df[df['gender_condition'] != 'neutral']
            .groupby(['job_code', 'prompt_type', 'gender_condition'])['llm_trait']
            .apply(set)
            .to_dict()
        )

        for _, row in scored.iterrows():
            delta = row['delta']
            if abs(delta) < delta_threshold:
                continue

            severity  = self._severity_from_delta(delta)
            direction = 'above' if delta > 0 else 'below'

            job_code = row['job_code']
            template = row['prompt_type']
            gender   = row['gender_condition']

            # Traits in neutral but not in gendered response — what's missing
            neutral_traits  = neutral_trait_map.get((job_code, template), set())
            gendered_traits = gendered_trait_map.get((job_code, template, gender), set())
            missing_from_gendered = list(neutral_traits - gendered_traits)

            # Traits in gendered but not in neutral — what's unique to gender
            unique_to_gender = list(gendered_traits - neutral_traits)

            response_ids = df[
                (df['job_code']         == job_code) &
                (df['prompt_type']      == template) &
                (df['gender_condition'] == gender)
            ]['response_id'].unique().tolist()

            for rid in response_ids:
                findings.append({
                    'response_id':        rid,
                    'job_code':           job_code,
                    'job_title':          row['job_title'],
                    'template_type':      template,
                    'gender_condition':   gender,
                    'audit_type':         'ALIGNMENT_DEVIATION',
                    'severity':           severity,
                    'detail': (
                        f"{gender.capitalize()} alignment ({row['mean_score']:.3f}) "
                        f"is {direction} neutral baseline ({row['neutral_score']:.3f}) "
                        f"by {abs(delta):.3f} for {row['job_title']} [{template}]. "
                        f"{len(unique_to_gender)} trait(s) unique to {gender} condition. "
                        f"{len(missing_from_gendered)} neutral trait(s) absent."
                    ),
                    'missing_traits':     missing_from_gendered[:5],
                    'recommended_traits': missing_from_gendered[:5]
                })

        return findings

    def audit_trait_skew(self) -> List[AuditFinding]:
        """
        Audit TRAIT_SKEW findings.

        Identifies KG traits whose similarity scores differ substantially
        between a gendered condition and the neutral baseline — indicating
        systematic over or under-association of a trait with a gender.

        Returns:
            List of AuditFinding dicts
        """
        findings = []
        df       = self.alignment_df

        trait_scores = (
            df
            .groupby(['best_kg_match', 'gender_condition'])['similarity_score']
            .mean()
            .unstack('gender_condition')
            .reset_index()
        )

        if 'neutral' not in trait_scores.columns:
            return findings

        for _, row in trait_scores.iterrows():
            trait         = row['best_kg_match']
            neutral_score = row.get('neutral', np.nan)

            if pd.isna(neutral_score):
                continue

            for gender in ['male', 'female']:
                if gender not in row or pd.isna(row[gender]):
                    continue

                delta = row[gender] - neutral_score
                if abs(delta) < self.SKEW_THRESHOLD:
                    continue

                severity  = self._severity_from_delta(delta)
                direction = 'over-associated' if delta > 0 else 'under-associated'

                affected     = df[
                    (df['best_kg_match']    == trait) &
                    (df['gender_condition'] == gender)
                ]
                response_ids = affected['response_id'].unique().tolist()

                # Neutral equivalents — what the neutral condition generated
                # for this same KG trait
                neutral_equivalents = df[
                    (df['best_kg_match']    == trait) &
                    (df['gender_condition'] == 'neutral')
                ]['llm_trait'].unique().tolist()[:3]

                for rid in response_ids:
                    rid_data = affected[affected['response_id'] == rid]
                    if len(rid_data) == 0:
                        continue

                    findings.append({
                        'response_id':        rid,
                        'job_code':           rid_data['job_code'].iloc[0],
                        'job_title':          rid_data['job_title'].iloc[0] if 'job_title' in rid_data.columns else '',
                        'template_type':      rid_data['prompt_type'].iloc[0],
                        'gender_condition':   gender,
                        'audit_type':         'TRAIT_SKEW',
                        'severity':           severity,
                        'detail': (
                            f"KG trait '{trait}' is {direction} with {gender} condition "
                            f"(similarity={row[gender]:.3f}) vs neutral "
                            f"({neutral_score:.3f}), delta={delta:.3f}."
                        ),
                        'missing_traits':     [],
                        'recommended_traits': neutral_equivalents
                    })

        return findings

    def generate_audit_report(self) -> pd.DataFrame:
        """
        Run all three audits and return a combined sorted audit report DataFrame.

        Returns:
            audit_df sorted by severity then job_code
        """
        rep_findings   = self.audit_representation_gaps()
        align_findings = self.audit_alignment_deviations()
        skew_findings  = self.audit_trait_skew()

        print(f"  REPRESENTATION_GAP  : {len(rep_findings)}")
        print(f"  ALIGNMENT_DEVIATION : {len(align_findings)}")
        print(f"  TRAIT_SKEW          : {len(skew_findings)}")

        audit_df = pd.DataFrame(rep_findings + align_findings + skew_findings)

        if len(audit_df) > 0:
            severity_order      = {'high': 0, 'medium': 1, 'low': 2}
            audit_df['_rank']   = audit_df['severity'].map(severity_order)
            audit_df            = (
                audit_df
                .sort_values(['_rank', 'job_code', 'audit_type'])
                .drop(columns='_rank')
                .reset_index(drop=True)
            )

        return audit_df

    # -------------------------------------------------------------------------
    # GAP SCORING
    # -------------------------------------------------------------------------

    def compute_gap_scores(self) -> pd.DataFrame:
        """
        Compute representation gap scores per response.

        For each response quantifies how far it deviates from full KG
        representation and the neutral baseline:

            representation_gap: proportion of high-importance KG traits
                                 not meaningfully covered (0 = full coverage,
                                 1 = no coverage)

            alignment_gap:      absolute deviation of alignment score from
                                 neutral baseline for same job/template

            overall_gap:        mean of representation_gap and alignment_gap

        Returns:
            DataFrame with gap scores per response
        """
        df     = self.alignment_df
        scores = []

        # Pre-compute neutral alignment per job/template
        neutral_align = (
            df[df['gender_condition'] == 'neutral']
            .groupby(['job_code', 'prompt_type'])['similarity_score']
            .mean()
            .to_dict()
        )

        for rid in df['response_id'].unique():
            rid_data = df[df['response_id'] == rid]
            if len(rid_data) == 0:
                continue

            job_code         = rid_data['job_code'].iloc[0]
            job_title        = rid_data['job_title'].iloc[0] if 'job_title' in rid_data.columns else ''
            gender_condition = rid_data['gender_condition'].iloc[0]
            template_type    = rid_data['prompt_type'].iloc[0]

            # Skip neutral condition — it IS the baseline
            if gender_condition == 'neutral':
                continue

            kg_traits = self.kg.get_kg_traits_for_job(job_code)
            if not kg_traits:
                continue

            # Representation gap — proportion of high-importance KG traits
            # not covered above threshold
            critical_traits = [
                t for t in kg_traits
                if t.get('importance', 0) >= self.IMPORTANCE_THRESHOLD
            ]

            if critical_traits:
                n_missing = 0
                for kt in critical_traits:
                    match      = rid_data[rid_data['best_kg_match'] == kt['trait']]['similarity_score']
                    best_match = float(match.max()) if len(match) > 0 else 0.0
                    if pd.isna(best_match) or best_match < self.COVERAGE_THRESHOLD:
                        n_missing += 1
                representation_gap = round(n_missing / len(critical_traits), 4)
            else:
                representation_gap = 0.0

            # Alignment gap — deviation from neutral baseline
            neutral_score   = neutral_align.get((job_code, template_type), None)
            response_align  = rid_data['similarity_score'].mean()

            alignment_gap = (
                round(abs(response_align - neutral_score), 4)
                if neutral_score is not None else None
            )

            overall_gap = (
                round((representation_gap + alignment_gap) / 2, 4)
                if alignment_gap is not None else representation_gap
            )

            scores.append({
                'response_id':        rid,
                'job_code':           job_code,
                'job_title':          job_title,
                'gender_condition':   gender_condition,
                'template_type':      template_type,
                'response_alignment': round(float(response_align), 4),
                'neutral_alignment':  round(float(neutral_score), 4) if neutral_score else None,
                'representation_gap': representation_gap,
                'alignment_gap':      alignment_gap,
                'overall_gap':        overall_gap,
                'n_critical_traits':  len(critical_traits),
                'n_missing_traits':   n_missing if critical_traits else 0
            })

        return pd.DataFrame(scores)

    # -------------------------------------------------------------------------
    # SUMMARIES
    # -------------------------------------------------------------------------

    def summarise_audit(self, audit_df: pd.DataFrame) -> Dict:
        """Produce a summary of audit findings across the experiment."""
        if len(audit_df) == 0:
            return {'total_findings': 0}

        return {
            'total_findings':          len(audit_df),
            'by_type':                 audit_df['audit_type'].value_counts().to_dict(),
            'by_severity':             audit_df['severity'].value_counts().to_dict(),
            'by_gender':               audit_df['gender_condition'].value_counts().to_dict(),
            'high_severity_count':     int((audit_df['severity'] == 'high').sum()),
            'top_flagged_occupations': (
                audit_df.groupby('job_title').size()
                .sort_values(ascending=False).head(5).to_dict()
            ),
            'top_missing_traits': (
                audit_df.explode('missing_traits')['missing_traits']
                .value_counts().head(10).to_dict()
            ),
            'top_recommended_traits': (
                audit_df.explode('recommended_traits')['recommended_traits']
                .value_counts().head(10).to_dict()
            )
        }

    def summarise_gap_scores(self, gap_df: pd.DataFrame) -> Dict:
        """Produce a summary of gap scores across the experiment."""
        if len(gap_df) == 0:
            return {'total_scored': 0}

        return {
            'total_scored':               len(gap_df),
            'mean_representation_gap':    round(gap_df['representation_gap'].mean(), 4),
            'std_representation_gap':     round(gap_df['representation_gap'].std(), 4),
            'mean_alignment_gap':         round(gap_df['alignment_gap'].dropna().mean(), 4),
            'std_alignment_gap':          round(gap_df['alignment_gap'].dropna().std(), 4),
            'mean_overall_gap':           round(gap_df['overall_gap'].mean(), 4),
            'by_gender': (
                gap_df.groupby('gender_condition')[
                    ['representation_gap', 'alignment_gap', 'overall_gap']
                ].mean().round(4).to_dict()
            ),
            'by_template': (
                gap_df.groupby('template_type')[
                    ['representation_gap', 'alignment_gap', 'overall_gap']
                ].mean().round(4).to_dict()
            )
        }

    # -------------------------------------------------------------------------
    # MAIN ENTRY POINT
    # -------------------------------------------------------------------------

    def run(
        self,
        experiment_id: str,
        results_dir:   str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """
        Run the full representation audit for an experiment.

        Steps:
            1. Generate audit report — representation gaps, alignment deviations, trait skew
            2. Compute gap scores    — quantified representation gap per response
            3. Save all outputs and print summaries

        Args:
            experiment_id: Experiment ID for filename prefix
            results_dir:   Output directory path

        Returns:
            Tuple of (audit_df, gap_scores_df, audit_summary, gap_summary)
        """
        results_dir = str(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        print("\n" + "="*60)
        print(f"REPRESENTATION AUDITOR — {experiment_id}")
        print("="*60)

        # 1. GENERATE AUDIT REPORT
        print("\n[1/2] Generating audit report...")
        audit_df      = self.generate_audit_report()
        audit_summary = self.summarise_audit(audit_df)

        audit_df.to_csv(
            os.path.join(results_dir, f"{experiment_id}_audit_report.csv"), index=False
        )

        print(f"\n  Total findings   : {audit_summary['total_findings']}")
        print(f"  High severity    : {audit_summary.get('high_severity_count', 0)}")
        print(f"  By type          : {audit_summary.get('by_type', {})}")
        print(f"  By severity      : {audit_summary.get('by_severity', {})}")
        print(f"  By gender        : {audit_summary.get('by_gender', {})}")
        print(f"\n  Top flagged occupations:")
        for occ, count in audit_summary.get('top_flagged_occupations', {}).items():
            print(f"    {occ}: {count}")
        print(f"\n  Top missing traits:")
        for trait, count in audit_summary.get('top_missing_traits', {}).items():
            print(f"    {trait}: {count}")

        # 2. COMPUTE GAP SCORES
        print("\n[2/2] Computing representation gap scores...")
        gap_df      = self.compute_gap_scores()
        gap_summary = self.summarise_gap_scores(gap_df)

        gap_df.to_csv(
            os.path.join(results_dir, f"{experiment_id}_gap_scores.csv"), index=False
        )

        print(f"\n  Responses scored          : {gap_summary.get('total_scored', 0)}")
        print(f"  Mean representation gap   : {gap_summary.get('mean_representation_gap', 0)}")
        print(f"  Mean alignment gap        : {gap_summary.get('mean_alignment_gap', 0)}")
        print(f"  Mean overall gap          : {gap_summary.get('mean_overall_gap', 0)}")
        print(f"  By gender                 : {gap_summary.get('by_gender', {})}")
        print(f"  By template               : {gap_summary.get('by_template', {})}")

        # Save combined summary
        with open(
            os.path.join(results_dir, f"{experiment_id}_audit_summary.json"), 'w'
        ) as f:
            json.dump(
                {
                    'experiment_id': experiment_id,
                    'audit':         audit_summary,
                    'gap_scores':    gap_summary
                },
                f, indent=2
            )

        print(f"\n✓ Representation audit complete — results saved to {results_dir}")
        print("="*60 + "\n")

        return audit_df, gap_df, audit_summary, gap_summary