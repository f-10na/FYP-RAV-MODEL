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
        pass  # Implementation goes here

    #-- FUNCTION TO BUILD ONET KNOWLEDGE GRAPH --#
    '''
    forms the basis for building the knowledge graph from the O*NET dataset to be 
    used as static authoritative source of information about jobs and their associated skills/abilities
    '''

    def build_KG(df):
        G = nx.Graph()

        # 1. ADD JOB NODES (Unique job metadata)
        # Drop duplicates to ensure only process each job's metadata once
        job_metadata = df[['Job_Code', 'Job_Title', 'Major_Group']].drop_duplicates()
        for _, row in job_metadata.iterrows():
            G.add_node(row['Job_Code'], 
                    name=row['Job_Title'], 
                    type='job', 
                    major_group=row['Major_Group'])

        # 2. ADD SKILL/ATTRIBUTE NODES
        # get unique trait names
        skill_metadata = df[['Attribute_Name', 'Trait_Type']].drop_duplicates()
        for _, row in skill_metadata.iterrows():
            G.add_node(row['Attribute_Name'], 
                    type='attribute', 
                    category=row['Trait_Type'])

        # 3. ADD EDGES (Connecting Jobs to Skills with Weight)
        # do this in one batch for performance
        for _, row in df.iterrows():
            G.add_edge(row['Job_Code'], 
                    row['Attribute_Name'], 
                    weight=float(row['Importance_Score']),
                    experiment_id=row['Experiment_ID'])
        return G
    

    def visualize_interactive_graph(G, filename="job_graph.html"):
        # Create a pyvis network
        net = Network(height="750px", width="100%", notebook=True, bgcolor="#222222", font_color="white")
        
        # Load the NetworkX graph into pyvis
        net.from_nx(G)
        
        # Customizing the look based on your attributes
        for node in net.nodes:
            if node['type'] == 'job':
                node['color'] = '#3da4ff' # Blue for Jobs
                node['size'] = 25
            else:
                node['color'] = '#ffa500' # Orange for Skills/Abilities
                node['size'] = 15
                
        # Use physics so the nodes don't overlap
        net.toggle_physics(True)
        return net.show(filename)