"""
Retrieval-Augmented Verification (RAV) - Embedding Model
Semantic alignment measurement for LLM-generated traits vs. KG traits
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import json

class EmbeddingModel:
    """
    Handles semantic embedding and alignment scoring for RAV pipeline.
    
    Features:
    - Loads model once, reuses across all operations
    - Batch processing for efficiency
    - Caching for KG embeddings
    - Similarity computation
        - Cosine similarity for semantic alignment
        - Manhattan distance as an alternative metric
        - Euclidean distance as another alternative
        - Dot product similarity for unnormalized vectors
    - Best-match alignment with metadata
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.3,
        cache_kg_embeddings: bool = True
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model identifier
            similarity_threshold: Minimum score to consider a match valid
            cache_kg_embeddings: Whether to cache KG trait embeddings
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.cache_enabled = cache_kg_embeddings
        self.kg_embedding_cache = {}  # {trait_string: vector}
        print("✓ Model loaded successfully")
    
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single trait string into a vector.
        
        Args:
            text: Trait description string
            
        Returns:
            Normalized embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty string")
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        # Normalize to unit length for cosine similarity
        return embedding / np.linalg.norm(embedding)
    
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple traits efficiently in batch.
        
        Args:
            texts: List of trait description strings
            
        Returns:
            Array of normalized embedding vectors (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Filter empty strings
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("All input texts are empty")
        
        # Batch encode
        embeddings = self.model.encode(valid_texts, convert_to_numpy=True)
        
        # Normalize each vector
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        
        return normalized
    
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1, vec2: Normalized embedding vectors
            
        Returns:
            Similarity score in range [-1, 1] (typically [0, 1] for semantic similarity)
        """
        return float(np.dot(vec1, vec2))
    
    
    def embed_kg_traits(
        self, 
        kg_traits: List[Dict[str, any]], 
        use_cache: bool = True
    ) -> Tuple[List[str], np.ndarray, List[Dict]]:
        """
        Embed KG traits with optional caching.
        
        Args:
            kg_traits: List of dicts with 'trait' and 'importance' keys
            use_cache: Whether to use/update the cache
            
        Returns:
            Tuple of (trait_strings, embeddings, metadata)
        """
        trait_strings = []
        embeddings_list = []
        metadata = []
        
        for kg_trait in kg_traits:
            trait_text = kg_trait['trait']
            importance = kg_trait.get('importance', None)
            
            # Check cache first
            if use_cache and self.cache_enabled and trait_text in self.kg_embedding_cache:
                embedding = self.kg_embedding_cache[trait_text]
            else:
                # Compute embedding
                embedding = self.embed_single(trait_text)
                
                # Store in cache
                if use_cache and self.cache_enabled:
                    self.kg_embedding_cache[trait_text] = embedding
            
            trait_strings.append(trait_text)
            embeddings_list.append(embedding)
            metadata.append({
                'trait': trait_text,
                'importance': importance
            })
        
        embeddings_array = np.array(embeddings_list)
        return trait_strings, embeddings_array, metadata
    
    
    def find_best_alignment(
        self,
        llm_trait: str,
        kg_traits: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """
        Find the best KG trait match for a single LLM trait.
        
        Args:
            llm_trait: Trait extracted from LLM output
            kg_traits: List of KG traits with metadata
            
        Returns:
            Dict with alignment results and metadata
        """
        if not llm_trait or not llm_trait.strip():
            return {
                'llm_trait': llm_trait,
                'best_kg_match': None,
                'similarity_score': 0.0,
                'kg_importance': None,
                'above_threshold': False,
                'error': 'Empty LLM trait'
            }
        
        if not kg_traits:
            return {
                'llm_trait': llm_trait,
                'best_kg_match': None,
                'similarity_score': 0.0,
                'kg_importance': None,
                'above_threshold': False,
                'error': 'No KG traits provided'
            }
        
        # Embed LLM trait
        llm_embedding = self.embed_single(llm_trait)
        
        # Embed KG traits (with caching)
        kg_strings, kg_embeddings, kg_metadata = self.embed_kg_traits(kg_traits)
        
        # Compute similarities
        similarities = [
            self.cosine_similarity(llm_embedding, kg_emb) 
            for kg_emb in kg_embeddings
        ]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_kg_trait = kg_metadata[best_idx]
        
        # Get top 5 matches
        top_5_indices = np.argsort(similarities)[-5:][::-1]  # Top 5, descending
        
        return {
            'llm_trait': llm_trait,
            'best_kg_match': best_kg_trait['trait'],
            'similarity_score': round(best_score, 4),
            'kg_importance': best_kg_trait['importance'],
            'above_threshold': best_score >= self.similarity_threshold,
            'top_5_scores': [round(similarities[i], 4) for i in top_5_indices],
            'top_5_traits': [kg_metadata[i]['trait'] for i in top_5_indices],
            #include all scores if 10 or fewer KG traits for transparency
            #good way to see if kg had adequate information on job to fairly evaluate llm trait alignment
            'all_scores': [round(s, 4) for s in similarities] if len(similarities) <= 10 else None
        }
    
    
    def align_all_traits(
        self,
        llm_traits: List[str],
        kg_traits: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """
        Align all LLM traits to KG traits in batch.
        
        Args:
            llm_traits: List of traits from LLM output
            kg_traits: List of KG trait dictionaries
            
        Returns:
            List of alignment results for each LLM trait
        """
        results = []
        
        for llm_trait in llm_traits:
            alignment = self.find_best_alignment(llm_trait, kg_traits)
            results.append(alignment)
        
        return results
    
    
    def get_alignment_summary(self, alignment_results: List[Dict]) -> Dict[str, any]:
        """
        Generate summary statistics from alignment results.
        Args:
            alignment_results: Output from align_all_traits() 
        Returns:
            Summary statistics
        """
        valid_results = [r for r in alignment_results if 'error' not in r]
        
        if not valid_results:
            return {
                'total_traits': len(alignment_results),
                'valid_alignments': 0,
                'above_threshold': 0,
                'mean_similarity': 0.0,
                'median_similarity': 0.0,
                'min_similarity': 0.0,
                'max_similarity': 0.0
            }
        
        scores = [r['similarity_score'] for r in valid_results]
        above_threshold = sum(1 for r in valid_results if r['above_threshold'])
        
        return {
            'total_traits': len(alignment_results),
            'valid_alignments': len(valid_results),
            'above_threshold': above_threshold,
            'below_threshold': len(valid_results) - above_threshold,
            'mean_similarity': round(np.mean(scores), 4),
            'median_similarity': round(np.median(scores), 4),
            'min_similarity': round(min(scores), 4),
            'max_similarity': round(max(scores), 4),
            'std_similarity': round(np.std(scores), 4)
        }
    
    
    def save_cache(self, filepath: str):
        """Save KG embedding cache to disk."""
        if not self.kg_embedding_cache:
            print("Cache is empty, nothing to save")
            return
        
        # Convert numpy arrays to lists for JSON serialization
        cache_serializable = {
            trait: embedding.tolist() 
            for trait, embedding in self.kg_embedding_cache.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(cache_serializable, f, indent=2)
        
        print(f"✓ Saved {len(cache_serializable)} cached embeddings to {filepath}")
    
    
    def load_cache(self, filepath: str):
        """Load KG embedding cache from disk."""
        try:
            with open(filepath, 'r') as f:
                cache_serializable = json.load(f)
            
            # Convert lists back to numpy arrays
            self.kg_embedding_cache = {
                trait: np.array(embedding)
                for trait, embedding in cache_serializable.items()
            }
            
            print(f"✓ Loaded {len(self.kg_embedding_cache)} cached embeddings from {filepath}")
        except FileNotFoundError:
            print(f"Cache file not found: {filepath}")
        except Exception as e:
            print(f"Error loading cache: {e}")