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
        used as static authoritative source of information about jobs and their associated skills/abilities
        """
        G = nx.Graph()

        # 1. ADD JOB NODES (Unique job metadata)
        job_metadata = df[['Job_Code', 'Job_Title', 'Major_Group']].drop_duplicates()
        for _, row in job_metadata.iterrows():
            G.add_node(row['Job_Code'], 
                    name=row['Job_Title'], 
                    type='job', 
                    major_group=row['Major_Group'])

        # 2. ADD SKILL/ATTRIBUTE NODES
        skill_metadata = df[['Attribute_Name', 'Trait_Type']].drop_duplicates()
        for _, row in skill_metadata.iterrows():
            G.add_node(row['Attribute_Name'], 
                    type='attribute', 
                    category=row['Trait_Type'])

        # 3. ADD EDGES (Connecting Jobs to Skills with Weight)
        for _, row in df.iterrows():
            G.add_edge(row['Job_Code'], 
                    row['Attribute_Name'], 
                    weight=float(row['Importance_Score']),
                    experiment_id=row['Experiment_ID'])
        
        self.G = G  # Store the graph in the instance
        return G

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
                    'category': node_data.get('category')  # Trait_Type
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