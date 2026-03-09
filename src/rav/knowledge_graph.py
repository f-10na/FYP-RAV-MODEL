'''
Class for the KG that covers core functionality
    • Initialization and loading of the KG
    • Creating and maintaining the KG structure
    • Integration with the embedding model
    • Methods for retrieving and updating KG entries
'''

import networkx as nx
import pandas as pd
import random as rnd
from pyvis.network import Network

class KnowledgeGraph:
    """
    Knowledge Graph (KG) class for managing entities and relationships.
    
    Responsibilities:
    - Initialize and load the KG from a file or create a new one
    - Create and maintain the KG structure (entities, relationships, metadata)
    - Integrate with the embedding model for semantic representation
    - Provide methods for retrieving and updating KG entries
    - Ensure efficient storage and retrieval of KG data
    - Support for querying the KG based on various criteria (e.g., entity type, relationship type)
    - Handle updates to the KG as new information is added or existing information is modified
    """
    
    def __init__(self, kg_file: str = None):
        """
        Initialize the Knowledge Graph.
        
        Args:
            kg_file (str): Optional path to a file containing the KG data. If None, initializes an empty KG.
        """
        self.G = None  # Will hold the NetworkX graph
        self.kg_file = kg_file

    def build_KG(self, df):
        """
        Forms the basis for building the knowledge graph from the O*NET dataset to be 
        used as static authoritative source of information about jobs and their associated 
        skills/abilities. Normalises weights before construction.
        """
        # STANDARDISE COLUMN CASING
        df = df.copy()
        df.columns = df.columns.str.lower()

        # NORMALISE BEFORE CONSTRUCTION
        df = self.normalise_weights(df)

        G = nx.Graph()

        # 1. ADD JOB NODES
        job_metadata = df[['job_code', 'job_title', 'major_group']].drop_duplicates()
        for _, row in job_metadata.iterrows():
            G.add_node(row['job_code'], 
                    name=row['job_title'], 
                    type='job', 
                    major_group=row['major_group'])

        # 2. ADD TRAIT NODES
        trait_metadata = df[['attribute_name', 'trait_type']].drop_duplicates()
        for _, row in trait_metadata.iterrows():
            G.add_node(row['attribute_name'], 
                    type='attribute', 
                    category=row['trait_type'])

        # 3. ADD EDGES WITH NORMALISED WEIGHTS
        for _, row in df.iterrows():
            G.add_edge(row['job_code'], 
                    row['attribute_name'], 
                    weight=float(row['importance_score']),
                    experiment_id=row['experiment_id'])

        self.G = G
        return G
    
    #----- normalise importnace scores for traits -----
    def normalise_weights(self, df):
        """
        Normalises importance scores before KG construction:
        1. Validates trait types
        2. Drops negative WI (Work Styles Impact) scores
        3. Rescales each scale type to 0-1
        4. Per-occupation min-max normalisation across all traits
        """
        VALID_trait_typeS = {'Skills', 'Abilities', 'Work_Styles'}

        df = df.copy()

        # 1. VALIDATE TRAIT TYPES
        unknown = df[~df['trait_type'].isin(VALID_trait_typeS)]
        if not unknown.empty:
            raise ValueError(f"Unknown trait_types found: {unknown['trait_type'].unique()}")

        # 2. DROP NEGATIVE WORK STYLES IMPACT SCORES
        negative_mask = (df['trait_type'] == 'Work_Styles') & (df['importance_score'] < 0)
        dropped = negative_mask.sum()
        df = df[~negative_mask].copy()
        if dropped > 0:
            print(f"[normalise_weights] Dropped {dropped} negative Work_Styles edges")

        # 3. RESCALE TO 0-1 PER SCALE TYPE
        def rescale(row):
            if row['trait_type'] == 'Work_Styles':  # WI: -3 to 3
                return (row['importance_score'] + 3) / 6
            else:  # IM: 1-5 (Skills, Abilities)
                return (row['importance_score'] - 1) / 4

        df['importance_score'] = df.apply(rescale, axis=1)

        # 4. PER-OCCUPATION MIN-MAX NORMALISATION
        def minmax_per_job(group):
            min_score = group['importance_score'].min()
            max_score = group['importance_score'].max()
            if max_score - min_score == 0:  # flat scores edge case
                group['importance_score'] = 1.0
            else:
                group['importance_score'] = (
                    (group['importance_score'] - min_score) / (max_score - min_score)
                )
            return group

        df = df.groupby('job_code', group_keys=False).apply(minmax_per_job)

        return df

    def get_kg_traits_for_job(self, job_code):
        """
        Extract traits for a job in embedding-ready format.
        
        Returns list of dicts: [{'trait': str, 'importance': float, 'category': str}, ...]
        """
        if self.G is None:
            raise ValueError("KG not built yet. Call build_KG() first.")
        
        kg_traits = []
        
        for neighbor in self.G.neighbors(job_code):
            node_data = self.G.nodes[neighbor]
            
            if node_data.get('type') == 'attribute':
                edge_data = self.G.edges[job_code, neighbor]
                
                kg_traits.append({
                    'trait': neighbor,  # Attribute_Name
                    'importance': edge_data['weight'],  # Importance_Score
                    'category': node_data.get('category')  # trait_type
                })
        
        return kg_traits

    def get_all_kg_traits(self, job_codes):
        """
        Get traits for multiple jobs at once.
        Returns dict: {job_code: [trait_dicts]}
        """
        all_traits = {}
        for job_code in job_codes:
            all_traits[job_code] = self.get_kg_traits_for_job(job_code)
        return all_traits
    
    def visualize_interactive_graph(self, filename="job_graph.html"):
        """
        Create an interactive visualization of the graph.
        """
        if self.G is None:
            raise ValueError("KG not built yet. Call build_KG() first.")
        
        # Create a pyvis network
        net = Network(height="750px", width="100%", notebook=True, bgcolor="#222222", font_color="white")
        
        # Load the NetworkX graph into pyvis
        net.from_nx(self.G)
        
        # Customizing the look based on your attributes
        for node in net.nodes:
            if node.get('type') == 'job':
                node['color'] = '#3da4ff'  # Blue for Jobs
                node['size'] = 25
            else:
                node['color'] = '#ffa500'  # Orange for Skills/Abilities
                node['size'] = 15
                
        # Use physics so the nodes don't overlap
        net.toggle_physics(True)
        return net.show(filename)