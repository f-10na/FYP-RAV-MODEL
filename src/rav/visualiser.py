"""
RAV Pipeline - Bias Visualiser

Generates all visualisations for bias analysis results.

Visualisations:
    1. Gender Alignment Heatmap       — alignment scores per occupation × gender
    2. Alignment by Template          — grouped bar chart per template type
    3. Gender Alignment Distribution  — box plots per gender condition
    4. Per-Occupation Gender Gap      — diverging bar chart (delta from neutral)
    5. Trait Word Clouds              — frequency-sized trait clouds per gender
    6. Trait Similarity Heatmap       — cosine similarity per trait × gender

All outputs saved as PNG to results_dir with experiment_id prefix.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from collections import Counter
from wordcloud import WordCloud
import seaborn as sns


# =============================================================================
# PALETTE
# =============================================================================

PALETTE = {
    'male':    '#3da4ff',
    'female':  '#ff6eb4',
    'neutral': '#a0e4a0',
    'bg':      '#1a1a2e',
    'surface': '#16213e',
    'text':    '#e0e0e0',
    'grid':    '#2a2a4a',
    'accent':  '#f5a623',
}

GENDER_ORDER = ['male', 'female', 'neutral']


def _apply_base_style(fig, ax_list):
    """Apply consistent dark theme to figure and axes."""
    fig.patch.set_facecolor(PALETTE['bg'])
    for ax in ax_list:
        ax.set_facecolor(PALETTE['surface'])
        ax.tick_params(colors=PALETTE['text'], labelsize=9)
        ax.xaxis.label.set_color(PALETTE['text'])
        ax.yaxis.label.set_color(PALETTE['text'])
        ax.title.set_color(PALETTE['text'])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE['grid'])


def _save(fig, path):
    """Save figure with consistent settings."""
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved → {path}")


# =============================================================================
# 1. GENDER ALIGNMENT HEATMAP
# =============================================================================

def plot_gender_alignment_heatmap(
    alignment_df: pd.DataFrame,
    experiment_id: str,
    results_dir: str
):
    """
    Heatmap of mean alignment scores — occupations × gender conditions.

    Immediately surfaces which occupations have the largest gender gaps
    and which gender condition produces highest/lowest alignment overall.

    Args:
        alignment_df:  Alignment DataFrame from step 5
        experiment_id: Experiment ID for filename prefix
        results_dir:   Output directory path
    """
    print("\n[1/6] Gender Alignment Heatmap...")

    pivot = (
        alignment_df
        .groupby(['job_code', 'gender_condition'])['similarity_score']
        .mean()
        .unstack('gender_condition')
        .reindex(columns=GENDER_ORDER)
    )

    # Add job titles as labels
    job_labels = (
        alignment_df[['job_code', 'job_title']]
        .drop_duplicates()
        .set_index('job_code')['job_title']
    )
    pivot.index = pivot.index.map(job_labels)

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.45)))
    _apply_base_style(fig, [ax])

    cmap = plt.cm.RdYlGn
    im = ax.imshow(pivot.values, aspect='auto', cmap=cmap, vmin=0.3, vmax=0.8)

    # Axis labels
    ax.set_xticks(range(len(GENDER_ORDER)))
    ax.set_xticklabels([g.capitalize() for g in GENDER_ORDER], fontsize=11)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    # Cell annotations
    for i in range(len(pivot)):
        for j in range(len(GENDER_ORDER)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color='black' if 0.4 < val < 0.7 else 'white',
                        fontweight='bold')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=PALETTE['text'])
    cbar.ax.tick_params(labelsize=8, colors=PALETTE['text'])
    cbar.set_label('Mean Alignment Score', color=PALETTE['text'], fontsize=9)

    ax.set_title(f'Gender Alignment by Occupation\n{experiment_id}',
                 fontsize=13, fontweight='bold', pad=12)

    path = os.path.join(results_dir, f"{experiment_id}_heatmap_alignment.png")
    _save(fig, path)


# =============================================================================
# 2. ALIGNMENT BY TEMPLATE TYPE — GROUPED BAR CHART
# =============================================================================

def plot_alignment_by_template(
    alignment_df: pd.DataFrame,
    experiment_id: str,
    results_dir: str
):
    """
    Grouped bar chart of mean alignment scores per template type × gender.

    Shows which template (T1/T2/T3) surfaces the most bias and whether
    different gender conditions respond differently to each framing.

    Args:
        alignment_df:  Alignment DataFrame from step 5
        experiment_id: Experiment ID for filename prefix
        results_dir:   Output directory path
    """
    print("\n[2/6] Alignment by Template Type...")

    summary = (
        alignment_df
        .groupby(['prompt_type', 'gender_condition'])['similarity_score']
        .agg(['mean', 'sem'])
        .reset_index()
    )

    templates = sorted(summary['prompt_type'].unique())
    x = np.arange(len(templates))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_base_style(fig, [ax])

    for i, gender in enumerate(GENDER_ORDER):
        gdata = summary[summary['gender_condition'] == gender]
        gdata = gdata.set_index('prompt_type').reindex(templates)
        bars = ax.bar(
            x + (i - 1) * width,
            gdata['mean'],
            width=width,
            color=PALETTE[gender],
            alpha=0.85,
            label=gender.capitalize(),
            yerr=gdata['sem'],
            capsize=4,
            error_kw={'ecolor': PALETTE['text'], 'alpha': 0.6}
        )

    ax.set_xticks(x)
    ax.set_xticklabels(templates, fontsize=11)
    ax.set_xlabel('Template Type', fontsize=11)
    ax.set_ylabel('Mean Alignment Score', fontsize=11)
    ax.set_title(f'Alignment Score by Template Type\n{experiment_id}',
                 fontsize=13, fontweight='bold')
    ax.yaxis.grid(True, color=PALETTE['grid'], linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    legend = ax.legend(fontsize=10, facecolor=PALETTE['surface'],
                       edgecolor=PALETTE['grid'], labelcolor=PALETTE['text'])

    path = os.path.join(results_dir, f"{experiment_id}_bar_template_alignment.png")
    _save(fig, path)


# =============================================================================
# 3. GENDER ALIGNMENT DISTRIBUTION — BOX PLOTS
# =============================================================================

def plot_gender_distribution(
    alignment_df: pd.DataFrame,
    experiment_id: str,
    results_dir: str
):
    """
    Box plots of alignment score distributions per gender condition.

    Shows spread and outliers — pairs with Cohen's d and t-test results
    to give an intuitive picture of distributional differences.

    Args:
        alignment_df:  Alignment DataFrame from step 5
        experiment_id: Experiment ID for filename prefix
        results_dir:   Output directory path
    """
    print("\n[3/6] Gender Alignment Distribution...")

    data_by_gender = [
        alignment_df[alignment_df['gender_condition'] == g]['similarity_score'].dropna().values
        for g in GENDER_ORDER
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    _apply_base_style(fig, [ax])

    bp = ax.boxplot(
        data_by_gender,
        patch_artist=True,
        notch=True,
        widths=0.45,
        medianprops=dict(color=PALETTE['accent'], linewidth=2.5),
        whiskerprops=dict(color=PALETTE['text'], linewidth=1.2),
        capprops=dict(color=PALETTE['text'], linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor=PALETTE['accent'],
                        markersize=4, alpha=0.5, linestyle='none')
    )

    for patch, gender in zip(bp['boxes'], GENDER_ORDER):
        patch.set_facecolor(PALETTE[gender])
        patch.set_alpha(0.75)

    # Overlay jittered points
    for i, (data, gender) in enumerate(zip(data_by_gender, GENDER_ORDER), start=1):
        jitter = np.random.normal(0, 0.06, size=len(data))
        ax.scatter(i + jitter, data, alpha=0.25, s=12,
                   color=PALETTE[gender], zorder=3)

    ax.set_xticks(range(1, len(GENDER_ORDER) + 1))
    ax.set_xticklabels([g.capitalize() for g in GENDER_ORDER], fontsize=11)
    ax.set_ylabel('Alignment Score', fontsize=11)
    ax.set_title(f'Alignment Score Distribution by Gender\n{experiment_id}',
                 fontsize=13, fontweight='bold')
    ax.yaxis.grid(True, color=PALETTE['grid'], linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    path = os.path.join(results_dir, f"{experiment_id}_boxplot_distribution.png")
    _save(fig, path)


# =============================================================================
# 4. PER-OCCUPATION GENDER GAP — DIVERGING BAR CHART
# =============================================================================

def plot_gender_gap_diverging(
    alignment_df: pd.DataFrame,
    experiment_id: str,
    results_dir: str
):
    """
    Diverging bar chart of alignment delta from neutral per occupation.

    Male delta = male_score - neutral_score
    Female delta = female_score - neutral_score

    Bars left of zero = underalignment vs neutral baseline.
    Bars right of zero = overalignment vs neutral baseline.

    Args:
        alignment_df:  Alignment DataFrame from step 5
        experiment_id: Experiment ID for filename prefix
        results_dir:   Output directory path
    """
    print("\n[4/6] Per-Occupation Gender Gap...")

    mean_scores = (
        alignment_df
        .groupby(['job_code', 'gender_condition'])['similarity_score']
        .mean()
        .unstack('gender_condition')
        .reindex(columns=GENDER_ORDER)
    )

    job_labels = (
        alignment_df[['job_code', 'job_title']]
        .drop_duplicates()
        .set_index('job_code')['job_title']
    )
    mean_scores.index = mean_scores.index.map(job_labels)

    # Compute deltas from neutral
    delta_male   = mean_scores['male']   - mean_scores['neutral']
    delta_female = mean_scores['female'] - mean_scores['neutral']

    # Sort by male delta
    order = delta_male.sort_values().index
    delta_male   = delta_male.reindex(order)
    delta_female = delta_female.reindex(order)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(order) * 0.4)),
                             sharey=True)
    _apply_base_style(fig, axes)

    for ax, delta, gender, label in zip(
        axes,
        [delta_male, delta_female],
        ['male', 'female'],
        ['Male − Neutral', 'Female − Neutral']
    ):
        colors = [PALETTE[gender] if v >= 0 else '#ff4444' for v in delta.values]
        ax.barh(range(len(delta)), delta.values, color=colors, alpha=0.8, height=0.65)
        ax.axvline(0, color=PALETTE['text'], linewidth=1.2, linestyle='--', alpha=0.6)
        ax.set_xlabel(label, fontsize=10)
        ax.xaxis.grid(True, color=PALETTE['grid'], linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)

    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=8)

    fig.suptitle(f'Alignment Gap from Neutral Baseline\n{experiment_id}',
                 fontsize=13, fontweight='bold', color=PALETTE['text'], y=1.01)

    path = os.path.join(results_dir, f"{experiment_id}_diverging_gap.png")
    _save(fig, path)


# =============================================================================
# 5. TRAIT WORD CLOUDS
# =============================================================================

def plot_trait_wordclouds(
    llm_results: list,
    experiment_id: str,
    results_dir: str
):
    """
    Word clouds of LLM-generated traits per gender condition.

    Trait size proportional to frequency across all responses for that
    gender condition. Useful for spotting competence vs warmth vocabulary
    differences visually.

    Args:
        llm_results:   Raw LLM results list from step 4
        experiment_id: Experiment ID for filename prefix
        results_dir:   Output directory path
    """
    print("\n[5/6] Trait Word Clouds...")

    # Aggregate traits per gender
    traits_by_gender = {g: [] for g in GENDER_ORDER}
    for result in llm_results:
        gender = result.get('gender_condition')
        traits = result.get('traits', [])
        if gender in traits_by_gender and traits:
            traits_by_gender[gender].extend([t.lower().strip() for t in traits])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(PALETTE['bg'])

    wc_bg = '#0d0d1a'

    for ax, gender in zip(axes, GENDER_ORDER):
        ax.set_facecolor(wc_bg)
        ax.axis('off')

        traits = traits_by_gender[gender]
        if not traits:
            ax.set_title(f'{gender.capitalize()}\n(no data)',
                         color=PALETTE['text'], fontsize=12)
            continue

        freq = Counter(traits)

        wc = WordCloud(
            width=500,
            height=400,
            background_color=wc_bg,
            colormap=None,
            color_func=lambda *args, **kwargs: PALETTE[gender],
            max_words=60,
            prefer_horizontal=0.85,
            collocations=False,
            min_font_size=9
        ).generate_from_frequencies(freq)

        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(gender.capitalize(), color=PALETTE[gender],
                     fontsize=13, fontweight='bold', pad=10)

    fig.suptitle(f'Trait Word Clouds by Gender Condition\n{experiment_id}',
                 fontsize=13, fontweight='bold', color=PALETTE['text'])

    path = os.path.join(results_dir, f"{experiment_id}_wordclouds.png")
    _save(fig, path)


# =============================================================================
# 6. TRAIT SIMILARITY HEATMAP
# =============================================================================

def plot_trait_similarity_heatmap(
    alignment_df: pd.DataFrame,
    experiment_id: str,
    results_dir: str,
    top_n: int = 30
):
    """
    Heatmap of mean cosine similarity per KG trait × gender condition.

    Rows = top N most frequently matched KG traits
    Columns = gender conditions

    Shows whether certain trait clusters are consistently associated with
    one gender — key signal for systematic stereotyping.

    Args:
        alignment_df:  Alignment DataFrame from step 5
        experiment_id: Experiment ID for filename prefix
        results_dir:   Output directory path
        top_n:         Number of top traits to include
    """
    print("\n[6/6] Trait Similarity Heatmap...")

    # Get top N most frequently matched KG traits
    top_traits = (
        alignment_df['best_kg_match']
        .value_counts()
        .head(top_n)
        .index.tolist()
    )

    pivot = (
        alignment_df[alignment_df['best_kg_match'].isin(top_traits)]
        .groupby(['best_kg_match', 'gender_condition'])['similarity_score']
        .mean()
        .unstack('gender_condition')
        .reindex(columns=GENDER_ORDER)
        .reindex(top_traits)
    )

    fig, ax = plt.subplots(figsize=(9, max(8, len(pivot) * 0.35)))
    _apply_base_style(fig, [ax])

    cmap = plt.cm.coolwarm
    im = ax.imshow(pivot.values, aspect='auto', cmap=cmap, vmin=0.3, vmax=0.9)

    ax.set_xticks(range(len(GENDER_ORDER)))
    ax.set_xticklabels([g.capitalize() for g in GENDER_ORDER], fontsize=11)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    # Cell annotations
    for i in range(len(pivot)):
        for j in range(len(GENDER_ORDER)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7.5, color='white', fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.tick_params(labelsize=8, colors=PALETTE['text'])
    cbar.set_label('Mean Cosine Similarity', color=PALETTE['text'], fontsize=9)

    ax.set_title(f'Trait Similarity by Gender Condition (Top {top_n} Traits)\n{experiment_id}',
                 fontsize=12, fontweight='bold', pad=12)

    path = os.path.join(results_dir, f"{experiment_id}_heatmap_traits.png")
    _save(fig, path)


# =============================================================================
# MAIN — RUN ALL VISUALISATIONS
# =============================================================================

def run_visualisations(
    alignment_df: pd.DataFrame,
    llm_results: list,
    experiment_id: str,
    results_dir: str
):
    """
    Run all RAV visualisations for an experiment.

    Args:
        alignment_df:  Alignment DataFrame from step 5
        llm_results:   Raw LLM results list from step 4
        experiment_id: Experiment ID for filename prefix
        results_dir:   Output directory path (str or Path)
    """
    results_dir = str(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"VISUALISATIONS — {experiment_id}")
    print(f"{'='*60}")

    plot_gender_alignment_heatmap(alignment_df, experiment_id, results_dir)
    plot_alignment_by_template(alignment_df, experiment_id, results_dir)
    plot_gender_distribution(alignment_df, experiment_id, results_dir)
    plot_gender_gap_diverging(alignment_df, experiment_id, results_dir)
    plot_trait_wordclouds(llm_results, experiment_id, results_dir)
    plot_trait_similarity_heatmap(alignment_df, experiment_id, results_dir)

    print(f"\n✓ All visualisations saved to {results_dir}")
