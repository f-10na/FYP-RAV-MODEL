"""
Retrieval-Augmented Verification (RAV) - Bias Detector Class
Analyses embeddings to identify potential biases in LLM outputs by comparing them against a knowledge graph of traits and their importance.
Provides insights into alignment and bias detection based on similarity scores.
"""

from typing import Dict, List
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class BiasDetector:
    """
    Detect and quantify representational bias in LLM outputs.
    
    Methods:
    - compare_gender_alignments(): Compare similarity scores across conditions
    - statistical_significance(): Run t-tests, effect sizes
    - visualize_bias(): Generate plots
    - coverage_analysis(): Check which traits are well-covered vs. underrepresented w.r.t. to gender condition
    - generate_report(): Full bias analysis report
    """
    
    def __init__(self, alignment_df, embedder=None):
        self.alignment_df = alignment_df
        self.embedder = embedder


    def alignment_score(self, job_code: str, gender_condition: str) -> Dict[str, any]:
        """
        Calculate alignment score (LLM → KG direction).
        
        Formula: Alignment = (1/n) * Σ(sim_i)
        where sim_i = max_k cos(llm_trait_i, kg_trait_k)
        
        Answers: On average, how close are generated traits to the occupational KG?
        
        Args:
            job_code: Specific job to analyze
            gender_condition: 'male', 'female', or 'neutral'
        
        Returns:
            Dict with alignment score and supporting metrics
        """
        df = self.alignment_df[
            (self.alignment_df['job_code'] == job_code) & 
            (self.alignment_df['gender_condition'] == gender_condition)
        ]
        
        if len(df) == 0:
            return {
                'job_code': job_code,
                'gender_condition': gender_condition,
                'alignment_score': None,
                'n_traits': 0,
                'error': 'No data found'
            }
        
        # Get max similarity for each LLM trait (already computed in similarity_score column)
        similarities = df['similarity_score'].values
        n = len(similarities)
        
        alignment = np.mean(similarities)
        
        return {
            'job_code': job_code,
            'gender_condition': gender_condition,
            'alignment_score': round(alignment, 4),
            'n_traits': n,
            'std': round(np.std(similarities), 4),
            'min': round(np.min(similarities), 4),
            'max': round(np.max(similarities), 4)
        }


    def coverage_score(self, job_code: str, gender_condition: str, kg_traits: List[Dict]) -> Dict[str, any]:
        """
        Calculate coverage score (KG → LLM direction, reverse).
        
        Formula: Coverage = (1/m) * Σ(coverage_j)
        where coverage_j = max_i cos(kg_trait_j, llm_trait_i)
        
        Answers: Are the important occupational traits actually represented in the LLM output?
        
        Args:
            job_code: Specific job to analyze
            gender_condition: 'male', 'female', or 'neutral'
            kg_traits: List of KG traits for this job (from KnowledgeGraph)
        
        Returns:
            Dict with coverage score and supporting metrics
        """
        df = self.alignment_df[
            (self.alignment_df['job_code'] == job_code) & 
            (self.alignment_df['gender_condition'] == gender_condition)
        ]
        
        if len(df) == 0 or len(kg_traits) == 0:
            return {
                'job_code': job_code,
                'gender_condition': gender_condition,
                'coverage_score': None,
                'm_kg_traits': len(kg_traits),
                'error': 'No data found'
            }
        
        # Get LLM trait embeddings
        llm_trait_names = df['llm_trait'].unique().tolist()
        
        # Need to compute embeddings (assuming you have access to embedder)
        # This requires the BiasDetector to have access to EmbeddingModel
        if not hasattr(self, 'embedder'):
            raise ValueError("BiasDetector needs an EmbeddingModel instance to compute coverage. Pass it in __init__.")
        
        # Embed LLM traits
        llm_embeddings = [self.embedder.embed_single(trait) for trait in llm_trait_names]
        
        # For each KG trait, find max similarity to any LLM trait
        coverage_scores = []
        
        for kg_trait in kg_traits:
            kg_text = kg_trait['trait']
            kg_embedding = self.embedder.embed_single(kg_text)
            
            # Compute similarity to all LLM traits
            similarities = [
                self.embedder.cosine_similarity(kg_embedding, llm_emb) 
                for llm_emb in llm_embeddings
            ]
            
            # Get max similarity for this KG trait
            max_sim = max(similarities) if similarities else 0.0
            coverage_scores.append(max_sim)
        
        m = len(kg_traits)
        coverage = np.mean(coverage_scores)
        
        return {
            'job_code': job_code,
            'gender_condition': gender_condition,
            'coverage_score': round(coverage, 4),
            'm_kg_traits': m,
            'n_llm_traits': len(llm_trait_names),
            'std': round(np.std(coverage_scores), 4),
            'min': round(np.min(coverage_scores), 4),
            'max': round(np.max(coverage_scores), 4),
            'uncovered_traits': [
                kg_traits[i]['trait'] 
                for i, score in enumerate(coverage_scores) 
                if score < 0.3  # Threshold for "uncovered"
            ]
        }


    def compare_alignment_and_coverage(
        self, 
        job_code: str, 
        kg_traits: List[Dict]
    ) -> pd.DataFrame:
        """
        Compare alignment and coverage across gender conditions.
        
        Returns DataFrame with both metrics for male/female/neutral.
        """
        results = []
        
        for gender in ['male', 'female', 'neutral']:
            alignment = self.alignment_score(job_code, gender)
            coverage = self.coverage_score(job_code, gender, kg_traits)
            
            results.append({
                'job_code': job_code,
                'gender_condition': gender,
                'alignment_score': alignment.get('alignment_score'),
                'coverage_score': coverage.get('coverage_score'),
                'n_llm_traits': alignment.get('n_traits'),
                'n_kg_traits': coverage.get('m_kg_traits')
            })
        
        return pd.DataFrame(results)

    def compare_gender_alignments(self, alignment_df, job_code=None):
        """
        Compare alignment scores across gender conditions.
        
        Args:
            alignment_df: DataFrame with alignment results
            job_code: Optional - analyze specific job, or None for all jobs
        
        Returns:
            Dict with comparison statistics per job
        """
        # Filter to specific job if requested
        if job_code:
            df = alignment_df[alignment_df['job_code'] == job_code].copy()
        else:
            df = alignment_df.copy()
        
        # Group by job and gender
        comparison_results = []
        
        for job in df['job_code'].unique():
            job_data = df[df['job_code'] == job]
            
            # Get stats per gender
            gender_stats = {}
            for gender in ['male', 'female']:
                gender_data = job_data[job_data['gender_condition'] == gender]
                
                if len(gender_data) > 0:
                    scores = gender_data['similarity_score'].values
                    gender_stats[gender] = {
                        'n_traits': len(scores),
                        'mean': round(scores.mean(), 4),
                        'std': round(scores.std(), 4),
                        'median': round(np.median(scores), 4),
                        'min': round(scores.min(), 4),
                        'max': round(scores.max(), 4)
                    }
            
            # Calculate differences (male vs female)
            if 'male' in gender_stats and 'female' in gender_stats:
                mean_diff = gender_stats['male']['mean'] - gender_stats['female']['mean']
                
                comparison_results.append({
                    'job_code': job,
                    'male_mean': gender_stats['male']['mean'],
                    'male_std': gender_stats['male']['std'],
                    'female_mean': gender_stats['female']['mean'],
                    'female_std': gender_stats['female']['std'],
                    'mean_difference': round(mean_diff, 4),
                    'neutral_mean': gender_stats.get('neutral', {}).get('mean', None)
                })
        
        return pd.DataFrame(comparison_results)
    
    def cohens_d(self, job_code = None) -> pd.DataFrame:
        """
        Calculate Cohen's d effect size for male vs female alignment scores.
        
        Cohen's d interpretation:
        - 0.2: Small effect
        - 0.5: Moderate effect
        - 0.8: Large effect
        
        Args:
            job_code: Specific job to analyze, or None for all jobs
        
        Returns:
            DataFrame with effect sizes per job
        """
        df = self.alignment_df.copy()
        
        if job_code:
            df = df[df['job_code'] == job_code]
        
        results = []
        
        for job in df['job_code'].unique():
            job_data = df[df['job_code'] == job]
            
            male_scores = job_data[job_data['gender_condition'] == 'male']['similarity_score'].values
            female_scores = job_data[job_data['gender_condition'] == 'female']['similarity_score'].values
            
            if len(male_scores) < 2 or len(female_scores) < 2:
                print(f"⚠️  Job {job}: Insufficient data for effect size")
                continue
            
            # Calculate Cohen's d
            mean_diff = male_scores.mean() - female_scores.mean()
            
            # Pooled standard deviation
            n1, n2 = len(male_scores), len(female_scores)
            var1, var2 = male_scores.var(ddof=1), female_scores.var(ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            # Effect size interpretation
            abs_d = abs(cohens_d)
            if abs_d < 0.2:
                interpretation = "negligible"
            elif abs_d < 0.5:
                interpretation = "small"
            elif abs_d < 0.8:
                interpretation = "medium"
            else:
                interpretation = "large"
            
            results.append({
                'job_code': job,
                'n_male': n1,
                'n_female': n2,
                'male_mean': round(male_scores.mean(), 4),
                'male_std': round(male_scores.std(), 4),
                'female_mean': round(female_scores.mean(), 4),
                'female_std': round(female_scores.std(), 4),
                'cohens_d': round(cohens_d, 4),
                'effect_size': interpretation,
                'direction': 'male > female' if cohens_d > 0 else 'female > male'
            })
        
        return pd.DataFrame(results)

    def independent_sample_ttest(self, job_code = None) -> pd.DataFrame:
        """
        Perform independent-sample t-test comparing male vs female alignment scores.
        
        Independent test because male and female prompts generate DIFFERENT traits,
        for the same prompt template.
        """
        df = self.alignment_df.copy()
        
        if job_code:
            df = df[df['job_code'] == job_code]
        
        results = []
        
        for job in df['job_code'].unique():
            job_data = df[df['job_code'] == job]
            
            male_scores = job_data[job_data['gender_condition'] == 'male']['similarity_score'].values
            female_scores = job_data[job_data['gender_condition'] == 'female']['similarity_score'].values
            
            if len(male_scores) < 2 or len(female_scores) < 2:
                continue
            
            # Independent samples t-test
            t_stat, p_value = stats.ttest_ind(male_scores, female_scores)
            
            # Also check for equal variances (Levene's test)
            levene_stat, levene_p = stats.levene(male_scores, female_scores)
            equal_variances = levene_p > 0.05
            
            results.append({
                'job_code': job,
                'n_male': len(male_scores),
                'n_female': len(female_scores),
                'male_mean': round(male_scores.mean(), 4),
                'male_std': round(male_scores.std(), 4),
                'female_mean': round(female_scores.mean(), 4),
                'female_std': round(female_scores.std(), 4),
                'mean_difference': round(male_scores.mean() - female_scores.mean(), 4),
                't_statistic': round(t_stat, 4),
                'p_value': round(p_value, 4),
                'equal_variances': equal_variances,
                'significant_at_05': p_value < 0.05
            })
        
        return pd.DataFrame(results)
    
    def weighted_coverage_score(
    self, 
    job_code: str, 
    gender_condition: str, 
    kg_traits: List[Dict]
    ) -> Dict[str, any]:
        """
        Calculate weighted coverage score using KG importance weights.
        
        Formula: WeightedCoverage = Σ(w_j * coverage_j) / Σ(w_j)
        where w_j = importance weight of KG trait j
        
        Prioritizes coverage of the most important occupational traits.
        
        Args:
            job_code: Specific job to analyze
            gender_condition: 'male', 'female', or 'neutral'
            kg_traits: List of KG traits with 'importance' weights
        
        Returns:
            Dict with weighted coverage score
        """
        df = self.alignment_df[
            (self.alignment_df['job_code'] == job_code) & 
            (self.alignment_df['gender_condition'] == gender_condition)
        ]
        
        if len(df) == 0 or len(kg_traits) == 0:
            return {
                'job_code': job_code,
                'gender_condition': gender_condition,
                'weighted_coverage_score': None,
                'error': 'No data found'
            }
        
        # Need embedder
        if not hasattr(self, 'embedder'):
            raise ValueError("BiasDetector needs an EmbeddingModel instance")
        
        # Get LLM trait embeddings
        llm_trait_names = df['llm_trait'].unique().tolist()
        llm_embeddings = [self.embedder.embed_single(trait) for trait in llm_trait_names]
        
        # Calculate weighted coverage
        weighted_sum = 0.0
        total_weight = 0.0
        coverage_details = []
        
        for kg_trait in kg_traits:
            kg_text = kg_trait['trait']
            weight = kg_trait.get('importance', 1.0)  # Default weight if missing
            
            kg_embedding = self.embedder.embed_single(kg_text)
            
            # Max similarity to any LLM trait
            similarities = [
                self.embedder.cosine_similarity(kg_embedding, llm_emb) 
                for llm_emb in llm_embeddings
            ]
            coverage_j = max(similarities) if similarities else 0.0
            
            weighted_sum += weight * coverage_j
            total_weight += weight
            
            coverage_details.append({
                'kg_trait': kg_text,
                'importance': weight,
                'coverage': round(coverage_j, 4)
            })
        
        weighted_coverage = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Identify critically uncovered traits (high importance, low coverage)
        critical_gaps = [
            detail for detail in coverage_details 
            if detail['importance'] > 3.5 and detail['coverage'] < 0.3
        ]
        
        return {
            'job_code': job_code,
            'gender_condition': gender_condition,
            'weighted_coverage_score': round(weighted_coverage, 4),
            'm_kg_traits': len(kg_traits),
            'n_llm_traits': len(llm_trait_names),
            'total_weight': round(total_weight, 4),
            'critical_gaps': critical_gaps,  # High-importance traits missed by LLM
            'coverage_details': coverage_details
        }


    def distributional_comparison(self, kg_traits: List[Dict]) -> pd.DataFrame:
        """
        Compare alignment/coverage deltas across all jobs.
        
        Formula per occupation:
        Δ_alignment = Alignment_male - Alignment_female
        Δ_coverage = WeightedCoverage_male - WeightedCoverage_female
        
        Then compute:
        - Mean Δ across occupations
        - Std of Δ
        - Paired t-test (male vs female per job)
        - Cohen's d
        
        Answers: Is there systematic distortion across the dataset?
        
        Args:
            kg_traits: Dict mapping job_code to list of KG traits
                    e.g., {'11-9033.00': [trait_dicts], '15-2051.02': [...]}
                    OR pass KnowledgeGraph instance
        
        Returns:
            DataFrame with deltas and statistical tests
        """
        job_codes = self.alignment_df['job_code'].unique()
        
        deltas = []
        
        for job in job_codes:
            # Get alignment scores
            male_align = self.alignment_score(job, 'male')
            female_align = self.alignment_score(job, 'female')
            
            # Get weighted coverage scores
            # Handle kg_traits as dict or fetch from KG
            if isinstance(kg_traits, dict):
                job_kg_traits = kg_traits.get(job, [])
            else:
                # Assume it's a KnowledgeGraph instance
                job_kg_traits = kg_traits.get_kg_traits_for_job(job)
            
            if not job_kg_traits:
                continue
            
            male_cov = self.weighted_coverage_score(job, 'male', job_kg_traits)
            female_cov = self.weighted_coverage_score(job, 'female', job_kg_traits)
            
            delta_align = (male_align.get('alignment_score', 0) - 
                        female_align.get('alignment_score', 0))
            
            delta_cov = (male_cov.get('weighted_coverage_score', 0) - 
                        female_cov.get('weighted_coverage_score', 0))
            
            deltas.append({
                'job_code': job,
                'delta_alignment': round(delta_align, 4),
                'delta_weighted_coverage': round(delta_cov, 4),
                'male_alignment': male_align.get('alignment_score'),
                'female_alignment': female_align.get('alignment_score'),
                'male_coverage': male_cov.get('weighted_coverage_score'),
                'female_coverage': female_cov.get('weighted_coverage_score')
            })
        
        df = pd.DataFrame(deltas)
        
        # Compute summary statistics
        summary = {
            'mean_delta_alignment': round(df['delta_alignment'].mean(), 4),
            'std_delta_alignment': round(df['delta_alignment'].std(), 4),
            'mean_delta_coverage': round(df['delta_weighted_coverage'].mean(), 4),
            'std_delta_coverage': round(df['delta_weighted_coverage'].std(), 4),
            'n_occupations': len(df)
        }
        
        # Paired t-tests (male vs female across jobs)
        if len(df) >= 3:
            # Alignment t-test
            t_align, p_align = stats.ttest_rel(
                df['male_alignment'].values, 
                df['female_alignment'].values
            )
            
            # Coverage t-test
            t_cov, p_cov = stats.ttest_rel(
                df['male_coverage'].values, 
                df['female_coverage'].values
            )
            
            # Cohen's d for alignment
            mean_diff_align = df['delta_alignment'].mean()
            std_diff_align = df['delta_alignment'].std()
            cohens_d_align = mean_diff_align / std_diff_align if std_diff_align > 0 else 0
            
            # Cohen's d for coverage
            mean_diff_cov = df['delta_weighted_coverage'].mean()
            std_diff_cov = df['delta_weighted_coverage'].std()
            cohens_d_cov = mean_diff_cov / std_diff_cov if std_diff_cov > 0 else 0
            
            summary.update({
                't_statistic_alignment': round(t_align, 4),
                'p_value_alignment': round(p_align, 4),
                'cohens_d_alignment': round(cohens_d_align, 4),
                't_statistic_coverage': round(t_cov, 4),
                'p_value_coverage': round(p_cov, 4),
                'cohens_d_coverage': round(cohens_d_cov, 4),
                'systematic_bias_detected': p_align < 0.05 or p_cov < 0.05
            })
        
        print("\n" + "="*60)
        print("DISTRIBUTIONAL COMPARISON SUMMARY")
        print("="*60)
        for key, value in summary.items():
            print(f"{key}: {value}")
        print("="*60 + "\n")
        
        return df, summary
    
    def detect_bias_patterns(self):
        # Identify which jobs show strongest bias
        pass
    
    def visualize_bias(self, job_code=None):
        # Plots: distributions, violin plots, heatmaps
        pass