'''
Class for the KG that covers core functionality
    • Initialization and loading of the KG
    • Creating and maintaining the KG structure
    • Integration with the embedding model
    • Methods for retrieving and updating KG entries
'''

# MANUAL call
# kg.visualize_job_subgraph('17-2031.00', filename='civil_engineer.html')
import networkx as nx
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
        print("After normalise:", df.columns.tolist())
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
        for job in df['job_code'].unique():
            mask = df['job_code'] == job
            scores = df.loc[mask, 'importance_score']
            min_score = scores.min()
            max_score = scores.max()
            if max_score - min_score == 0:
                df.loc[mask, 'importance_score'] = 1.0
            else:
                df.loc[mask, 'importance_score'] = (
                    (scores - min_score) / (max_score - min_score)
                )

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
        Create an interactive visualisation of the Knowledge Graph.
        
        Job nodes rendered in blue, trait nodes in orange.
        Edge thickness reflects normalised importance weight.
        
        Args:
            filename: Output HTML file path
        """
        if self.G is None:
            raise ValueError("KG not built yet. Call build_KG() first.")

        net = Network(
            height="750px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
            notebook=False  # must be False for script context
        )

        # ADD NODES MANUALLY to control styling per type
        for node_id, node_data in self.G.nodes(data=True):
            if node_data.get('type') == 'job':
                net.add_node(
                    node_id,
                    label=node_data.get('name', node_id),
                    color='#3da4ff',
                    size=25,
                    title=f"Job: {node_data.get('name', node_id)}\nMajor Group: {node_data.get('major_group', '')}"
                )
            else:
                net.add_node(
                    node_id,
                    label=node_id,
                    color='#ffa500',
                    size=10,
                    title=f"Trait: {node_id}\nCategory: {node_data.get('category', '')}"
                )

        # ADD EDGES with weight-based thickness
        for source, target, edge_data in self.G.edges(data=True):
            weight = edge_data.get('weight', 0.5)
            net.add_edge(
                source,
                target,
                value=weight,       # controls thickness
                title=f"Importance: {round(weight, 3)}"
            )

        net.set_options("""
        {
            "physics": {
                "enabled": true
            }
        }
        """)
        net.save_graph(filename)
        print(f"KG visualisation saved → {filename}")


    def visualize_job_subgraph(self, job_code, filename=None):
        """
        Visualise a single job and its direct trait connections.

        Much cleaner than the full graph — shows only the ego network
        for one occupation (job node + all its trait nodes).

        Args:
            job_code: SOC code of the job to visualise
            filename: Output HTML path. Defaults to '{job_code}_subgraph.html'
        """
        if self.G is None:
            raise ValueError("KG not built yet. Call build_KG() first.")

        if job_code not in self.G.nodes:
            raise ValueError(f"Job code {job_code} not found in KG.")

        if filename is None:
            filename = f"{job_code.replace('.', '_')}_subgraph.html"

        # Extract ego subgraph — job node + all direct neighbours
        subgraph = nx.ego_graph(self.G, job_code, radius=1)

        #job_title = self.G.nodes[job_code].get('name', job_code)

        net = Network(
            height="750px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
            notebook=False
        )

        # ADD NODES
        for node_id, node_data in subgraph.nodes(data=True):
            if node_data.get('type') == 'job':
                net.add_node(
                    node_id,
                    label=node_data.get('name', node_id),
                    color='#3da4ff',
                    size=35,
                    shape='star',
                    title=f"Job: {node_data.get('name', node_id)}"
                )
            else:
                # Scale trait node size by importance
                edge_data = self.G.edges[job_code, node_id]
                weight = edge_data.get('weight', 0.5)
                net.add_node(
                    node_id,
                    label=node_id,
                    color='#ffa500',
                    size=8 + (weight * 20),  # bigger = more important
                    title=(
                        f"Trait: {node_id}\n"
                        f"Category: {node_data.get('category', '')}\n"
                        f"Importance: {round(weight, 3)}"
                    )
                )

        # ADD EDGES
        for source, target, edge_data in subgraph.edges(data=True):
            weight = edge_data.get('weight', 0.5)
            net.add_edge(
                source,
                target,
                value=weight,
                title=f"Importance: {round(weight, 3)}"
            )

        # Tighter physics for ego graph
        net.set_options("""
        {
        "physics": {
            "barnesHut": {
            "gravitationalConstant": -5000,
            "centralGravity": 0.3,
            "springLength": 150,
            "springConstant": 0.05
            }
        }
        }
        """)

        net.save_graph(filename)
        print(f"Subgraph visualisation saved → {filename}")