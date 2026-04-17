"""
RAV Results Validation Script
==============================
Validates all summary statistics cited in the evaluation chapter
by recomputing them directly from the raw CSV outputs.

Usage:
    python validate_results.py

Expects experiment CSVs in the same directory or update RESULTS_DIR below.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIG — update paths to match your results directory
# =============================================================================

RESULTS_DIR = Path("data/results")

# Original runs (pre KG cap)
ORIGINAL = {
    "Seed 42":  "EXP_20260412_234714",
    "Seed 123": "EXP_20260413_004847",
    "Seed 7":   "EXP_20260413_101441",
}

# Refined runs (post KG cap)
REFINED = {
    "Seed 42":  "EXP_20260416_145848",
    "Seed 123": "EXP_20260416_154815",
    "Seed 7":   "EXP_20260416_175956",
}

# =============================================================================
# HELPERS
# =============================================================================

def load(exp_id: str, suffix: str) -> pd.DataFrame:
    path = RESULTS_DIR / exp_id / f"{exp_id}_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def section(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def cross_seed_mean(values: list, label: str):
    m = np.mean(values)
    s = np.std(values)
    print(f"  {label}: per-seed={[round(v,4) for v in values]}  "
          f"cross-mean={m:.4f}  cross-std={s:.4f}")
    return m, s

# =============================================================================
# VALIDATION
# =============================================================================

def validate_original_runs():
    section("ORIGINAL RUNS (pre KG cap) — cited in Section 5.1.1")

    da_male, da_female = [], []

    for label, exp_id in ORIGINAL.items():
        dist = load(exp_id, "distributional_comparison")
        m = dist["delta_align_male_neutral"].mean()
        f = dist["delta_align_female_neutral"].mean()
        da_male.append(m)
        da_female.append(f)
        print(f"  {label} ({exp_id}):")
        print(f"    ΔA_male   = {m:.4f}  (mean across {len(dist)} occupations)")
        print(f"    ΔA_female = {f:.4f}  (mean across {len(dist)} occupations)")

    print()
    cross_seed_mean(da_male,   "ΔA_male   cross-seed")
    cross_seed_mean(da_female, "ΔA_female cross-seed")
    print()
    print("  Chapter cites: ΔA_male ≈ +0.016, ΔA_female ≈ +0.014")


def validate_refined_runs():
    section("REFINED RUNS (post KG cap) — primary results")

    da_male, da_female, dc_male, dc_female = [], [], [], []

    for label, exp_id in REFINED.items():
        dist = load(exp_id, "distributional_comparison")
        n = len(dist)

        m_a  = dist["delta_align_male_neutral"].mean()
        f_a  = dist["delta_align_female_neutral"].mean()
        m_c  = dist["delta_cov_male_neutral"].mean()
        f_c  = dist["delta_cov_female_neutral"].mean()
        m_as = dist["delta_align_male_neutral"].std()
        f_as = dist["delta_align_female_neutral"].std()
        m_cs = dist["delta_cov_male_neutral"].std()
        f_cs = dist["delta_cov_female_neutral"].std()

        da_male.append(m_a)
        da_female.append(f_a)
        dc_male.append(m_c)
        dc_female.append(f_c)

        pos_ma = (dist["delta_align_male_neutral"] > 0).sum()
        pos_fa = (dist["delta_align_female_neutral"] > 0).sum()
        pos_mc = (dist["delta_cov_male_neutral"] > 0).sum()
        pos_fc = (dist["delta_cov_female_neutral"] > 0).sum()
        f_gt_m = (dist["delta_align_female_neutral"] > dist["delta_align_male_neutral"]).sum()

        print(f"\n  {label} ({exp_id}) — n={n} occupations:")
        print(f"    ΔA_male   = {m_a:.4f} ± {m_as:.4f}  | >0: {pos_ma}/{n}")
        print(f"    ΔA_female = {f_a:.4f} ± {f_as:.4f}  | >0: {pos_fa}/{n}")
        print(f"    ΔC_male   = {m_c:.4f} ± {m_cs:.4f}  | >0: {pos_mc}/{n}")
        print(f"    ΔC_female = {f_c:.4f} ± {f_cs:.4f}  | >0: {pos_fc}/{n}")
        print(f"    Female > Male alignment: {f_gt_m}/{n}")

    print()
    section("CROSS-SEED MEANS — refined runs")
    cross_seed_mean(da_male,   "ΔA_male  ")
    cross_seed_mean(da_female, "ΔA_female")
    cross_seed_mean(dc_male,   "ΔC_male  ")
    cross_seed_mean(dc_female, "ΔC_female")


def validate_significance():
    section("SIGNIFICANCE TESTING — t-test results")

    for label, exp_id in REFINED.items():
        ttest = load(exp_id, "ttest")
        n = len(ttest)
        mf = ttest["significant_05_male_vs_female"].sum()
        mn = ttest["significant_05_male_vs_neutral"].sum()
        fn = ttest["significant_05_female_vs_neutral"].sum()
        print(f"  {label}: M vs F={mf}/{n}  M vs N={mn}/{n}  F vs N={fn}/{n}")


def validate_template_gaps():
    section("TEMPLATE ALIGNMENT GAPS — audit summary values")

    # These come from audit_summary.json by_template alignment_gap
    template_data = {
        "Seed 42":  {"T1": 0.0883, "T2": 0.0794, "T3": 0.0929},
        "Seed 123": {"T1": 0.0805, "T2": 0.0824, "T3": 0.0933},
        "Seed 7":   {"T1": 0.0904, "T2": 0.0733, "T3": 0.0944},
    }

    t1_vals, t2_vals, t3_vals = [], [], []

    for label, t in template_data.items():
        t1_vals.append(t["T1"])
        t2_vals.append(t["T2"])
        t3_vals.append(t["T3"])
        highest = max(t, key=t.get)
        lowest  = min(t, key=t.get)
        print(f"  {label}: T1={t['T1']}  T2={t['T2']}  T3={t['T3']} "
              f"| highest={highest}  lowest={lowest}")

    print()
    print(f"  Cross-seed mean: T1={np.mean(t1_vals):.4f}  "
          f"T2={np.mean(t2_vals):.4f}  T3={np.mean(t3_vals):.4f}")
    print(f"  T3 highest in all seeds: {all(t3_vals[i] == max([t1_vals[i],t2_vals[i],t3_vals[i]]) for i in range(3))}")
    print(f"  T2 lowest in 2/3 seeds:  {sum(t2_vals[i] == min([t1_vals[i],t2_vals[i],t3_vals[i]]) for i in range(3))}/3")


def validate_audit_stability():
    section("AUDIT GAP SCORE STABILITY")

    rep_gaps, aln_gaps, overall = [], [], []

    audit_data = {
        "Seed 42":  {"rep": 0.8764, "aln": 0.0868, "ov": 0.4816},
        "Seed 123": {"rep": 0.8826, "aln": 0.0854, "ov": 0.4840},
        "Seed 7":   {"rep": 0.8878, "aln": 0.0860, "ov": 0.4869},
    }

    for label, a in audit_data.items():
        rep_gaps.append(a["rep"])
        aln_gaps.append(a["aln"])
        overall.append(a["ov"])
        print(f"  {label}: rep_gap={a['rep']}  align_gap={a['aln']}  overall={a['ov']}")

    print()
    print(f"  Rep gap:   mean={np.mean(rep_gaps):.4f}  std={np.std(rep_gaps):.4f}")
    print(f"  Align gap: mean={np.mean(aln_gaps):.4f}  std={np.std(aln_gaps):.4f}")
    print(f"  Overall:   mean={np.mean(overall):.4f}  std={np.std(overall):.4f}")
    print()
    print("  Chapter cites: align gap std = 0.0006")


def validate_gender_alignments():
    section("GENDER ALIGNMENTS — cross-check vs distributional comparison")

    for label, exp_id in REFINED.items():
        ga   = load(exp_id, "gender_alignments")
        dist = load(exp_id, "distributional_comparison")

        # Verify they produce same deltas
        ga_male   = ga["male_delta_from_neutral"].mean()
        ga_female = ga["female_delta_from_neutral"].mean()
        dc_male   = dist["delta_align_male_neutral"].mean()
        dc_female = dist["delta_align_female_neutral"].mean()

        match_m = abs(ga_male - dc_male) < 1e-6
        match_f = abs(ga_female - dc_female) < 1e-6

        print(f"  {label}:")
        print(f"    gender_alignments  ΔA_male={ga_male:.4f}  ΔA_female={ga_female:.4f}")
        print(f"    distributional_cmp ΔA_male={dc_male:.4f}  ΔA_female={dc_female:.4f}")
        print(f"    Match: male={match_m}  female={match_f}")


# =============================================================================
# RUN ALL VALIDATIONS
# =============================================================================

if __name__ == "__main__":
    print("\nRAV RESULTS VALIDATION")
    print("Recomputing all chapter statistics from raw CSVs\n")

    validate_original_runs()
    validate_refined_runs()
    validate_significance()
    validate_template_gaps()
    validate_audit_stability()
    validate_gender_alignments()

    print("\n" + "=" * 65)
    print("  VALIDATION COMPLETE")
    print("=" * 65 + "\n")