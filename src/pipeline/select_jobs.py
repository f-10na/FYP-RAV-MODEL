import pandas as pd
import random as rnd
from pathlib import Path
from src.utils.functions import generate_experiment_id

def get_k_major_groups(df, k, seed=42, soc_col='job_code'):
    """
    Randomly samples K major group prefixes from those present in the dataset.
    Fixed seed ensures reproducibility.
    """
    prefixes = sorted(df[soc_col].str[:2].unique(), key=lambda x: int(x))
    
    rnd.seed(seed)
    selected_groups = rnd.sample(list(prefixes), k)
    
    return selected_groups

def get_k_random_soc_codes(df, k, major_group=None, seed=42):
    """
    Returns a list of K unique detailed SOC codes.
    If major_group is provided, sampling is restricted to that 2-digit group.
    """

    rnd.seed(seed)
    
    # Use a single, consistent SOC column
    soc_col = "job_code"

    # Optional major group filter
    if major_group is not None:
        mask = df[soc_col].str.startswith(str(major_group))
        candidate_codes = df[mask][soc_col].unique()
    else:
        candidate_codes = df[soc_col].unique()

    # Safety check
    if k > len(candidate_codes):
        print(
            f"Warning: Requested {k} codes, but only {len(candidate_codes)} available."
        )
        k = len(candidate_codes)

    # Random sampling
    sampled_codes = rnd.sample(list(candidate_codes), k)

    return sampled_codes

def select_experiment_jobs(df, n_groups, jobs_per_group, seed=42):
    """
    Randomly selects n_groups major SOC groups then samples
    jobs_per_group occupations within each for diversity across industries.
    """
    major_groups = get_k_major_groups(df, k=n_groups, seed=seed)
    print(f"Selected major groups: {major_groups}")
    
    selected_codes = []
    for group in major_groups:
        codes = get_k_random_soc_codes(df, k=jobs_per_group, major_group=group, seed=seed)
        selected_codes.extend(codes)
    
    return selected_codes

def run_job_selection(
    dataset: pd.DataFrame,
    n_groups: int,
    jobs_per_group: int,
    seed: int,
    output_dir: Path,
    experiment_id: str = None, # auto-generated if not provided
) -> tuple:
    """
    Two-level occupation sampling pipeline.
    
    1. Randomly selects n_groups major SOC groups
    2. Samples jobs_per_group occupations within each group
    3. Saves occupations list for prompt generation
    4. Saves full KG selection for KG construction
    
    Args:
        dataset:        Curated O*NET dataset (snake_case columns)
        n_groups:       Number of major SOC groups to sample
        jobs_per_group: Number of occupations to sample per group
        seed:           Random seed for reproducibility
        experiment_id:  Unique experiment identifier
        output_dir:     Directory to save output CSVs
    
    Returns:
        Tuple of (selected_codes, occupations_df, kg_selection_df)
    """
    # Auto-generate experiment ID if not provided
    if experiment_id is None:
        experiment_id = generate_experiment_id()
    
    print(f"Experiment ID: {experiment_id}")
    
    # 1. TWO-LEVEL SAMPLING
    selected_codes = select_experiment_jobs(
        dataset,
        n_groups=n_groups,
        jobs_per_group=jobs_per_group,
        seed=seed
    )
    print(f"Selected {len(selected_codes)} jobs across {n_groups} major groups")
    print(f"Selected codes: {selected_codes}")

    # 2. OCCUPATIONS LIST — derive directly from curated dataset
    occupations_df = (
        dataset[dataset['job_code'].isin(selected_codes)]
        [['job_code', 'job_title']]
        .drop_duplicates()
        .copy()
    )

    occupations_df['major_group'] = occupations_df['job_code'].str[:2]
    occupations_df['selection_seed'] = seed
    occupations_df['selection_strategy'] = 'two_level_random_sample'
    occupations_df['experiment_id'] = experiment_id

    assert occupations_df['job_code'].is_unique, "Duplicate job codes in occupations list"

    occupations_path = output_dir / f"{experiment_id}_occupations_list.csv"
    occupations_df.to_csv(occupations_path, index=False)
    print(f"Saved occupations list ({len(occupations_df)} jobs) → {occupations_path}")

    # 3. KG SELECTION — full trait rows for KG construction
    kg_selection_df = dataset[dataset['job_code'].isin(selected_codes)].copy()
    kg_selection_df['major_group'] = kg_selection_df['job_code'].str[:2]
    kg_selection_df['selection_seed'] = seed
    kg_selection_df['selection_strategy'] = 'two_level_random_sample'
    kg_selection_df['experiment_id'] = experiment_id

    kg_path = output_dir / f"{experiment_id}_kg_selection.csv"
    kg_selection_df.to_csv(kg_path, index=False)
    print(f"Saved KG selection ({len(kg_selection_df)} rows) → {kg_path}")

    return selected_codes, occupations_df, kg_selection_df