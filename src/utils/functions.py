#--- FIND PROJECT ROOT ---
from pathlib import Path

def find_project_root(marker="README.md"):
    path = Path.cwd().resolve()
    for parent in [path] + list(path.parents):
        if (parent / marker).exists():
            return parent
    raise RuntimeError("Project root not found")

#--- LOAD CSV UTILITY FUNCTION ---
import pandas as pd
def load_csv(filepath,separator):
    return pd.read_csv(
        filepath,
        sep=separator,
        encoding="utf-8",
        low_memory=False
    )

#Generate a unique experiment ID based on current timestamp
from datetime import datetime
def generate_experiment_id() -> str:
    """Generate a unique experiment ID based on current timestamp."""
    return f"EXP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

#----save trait results utility functions---
import json
def save_trait_results(results_list, filepath):
    """
    Save LLM trait extraction results to CSV.
    
    Args:
        results_list: List of dicts with trait extraction results
        filepath: Where to save the CSV
        experiment_id: Optional experiment identifier
        
    Expected result dict format:
    {
        'experiment_id': str,
        'job_code': str,
        'job_title': str,
        'prompt_type': 'T1' or 'T2',
        'gender_condition': 'male', 'female',
        'traits': list of trait strings,
        'raw_response': str (optional, for debugging),
        'parse_status': 'success' or 'failed',
        'timestamp': datetime
    }
    """
    df = pd.DataFrame(results_list)
    
    # Convert traits list to JSON string for CSV storage
    if 'parse_status' in df.columns and df['parse_status'].eq('failed').any():
        failed_rows = df[df['parse_status'] == 'failed']
        print("Failed rows:")
        for _, row in failed_rows.iterrows():
            print(f"Job Code: {row['job_code']}, Raw Response: {row['raw_response']}")
    # if 'traits' in df.columns:
    #     df['traits_json'] = df['traits'].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    
    df.to_csv(filepath, index=False)
    print(f"✓ Saved {len(df)} results to {filepath}")
    return df


def load_trait_results(filepath):
    """
    Load trait results from CSV and parse JSON traits back to lists.
    
    Returns: DataFrame with traits as lists
    """
    df = pd.read_csv(filepath)
    
    # Convert JSON strings back to lists
    if 'traits_json' in df.columns:
        df['traits'] = df['traits_json'].apply(lambda x: json.loads(x) if pd.notna(x) else [])
    
    return df


def create_experiment_results_structure(job_codes, prompt_types=['T1', 'T2'], 
                                        gender_conditions=['male', 'female']):
    """
    Create empty results structure for an experiment run.
    
    Returns: List of dicts ready to be populated with traits
    """
    results = []
    experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for job_code in job_codes:
        for prompt_type in prompt_types:
            for gender in gender_conditions:
                results.append({
                    'experiment_id': experiment_id,
                    'job_code': job_code,
                    'job_title': None,  # Fill in when you have KG data
                    'prompt_type': prompt_type,
                    'gender_condition': gender,
                    'traits': None,
                    'raw_response': None,
                    'parse_status': 'pending',
                    'timestamp': None
                })
    
    return results, experiment_id