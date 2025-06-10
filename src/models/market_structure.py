"""
Market structure analysis using network theory.
"""
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from node2vec import Node2Vec
from sklearn.cluster import DBSCAN
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class MarketStructureModel:
    """Market structure analysis using network theory."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.graph = nx.Graph()
        self.embeddings = None
        self.clusters = None
        self.centrality = None
    
    def build_graph(self, returns: pd.DataFrame, threshold: float = 0.5) -> None:
        """
        Build market structure graph from returns.
        
        Args:
            returns: DataFrame of asset returns
            threshold: Correlation threshold for edge creation
        """
        # Compute correlation matrix
        corr_matrix = returns.corr()
        
        # Create edges
        edges = []
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                if abs(corr_matrix.iloc[i,j]) > threshold:
                    edges.append((
                        corr_matrix.index[i],
                        corr_matrix.index[j],
                        {'weight': corr_matrix.iloc[i,j]}
                    ))
        
        # Build graph
        self.graph.add_edges_from(edges)
        logger.info(f"Built graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
    
    def compute_embeddings(self, dimensions: int = 64, walk_length: int = 30,
                         num_walks: int = 200) -> np.ndarray:
        """
        Compute node embeddings using node2vec.
        
        Args:
            dimensions: Embedding dimensions
            walk_length: Length of random walks
            num_walks: Number of random walks per node
            
        Returns:
            Node embeddings
        """
        # Initialize node2vec
        node2vec = Node2Vec(
            self.graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=4
        )
        
        # Fit and get embeddings
        model = node2vec.fit(window=5, min_count=1)
        self.embeddings = np.array([model.wv[str(node)] for node in self.graph.nodes])
        
        return self.embeddings
    
    def detect_clusters(self, eps: float = 0.5, min_samples: int = 5) -> Dict:
        """
        Detect market clusters using DBSCAN.
        
        Args:
            eps: Maximum distance between samples
            min_samples: Minimum samples in a cluster
            
        Returns:
            Dictionary of clusters
        """
        if self.embeddings is None:
            raise ValueError("Must compute embeddings before clustering")
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(self.embeddings)
        
        # Organize clusters
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(list(self.graph.nodes)[i])
        
        self.clusters = clusters
        return clusters
    
    def compute_centrality(self) -> Dict[str, pd.Series]:
        """
        Compute various centrality measures.
        
        Returns:
            Dictionary of centrality measures
        """
        # Degree centrality
        degree_centrality = nx.degree_centrality(self.graph)
        
        # Betweenness centrality
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        
        # Eigenvector centrality
        eigenvector_centrality = nx.eigenvector_centrality(self.graph)
        
        # PageRank
        pagerank = nx.pagerank(self.graph)
        
        # Combine measures
        self.centrality = {
            'degree': pd.Series(degree_centrality),
            'betweenness': pd.Series(betweenness_centrality),
            'eigenvector': pd.Series(eigenvector_centrality),
            'pagerank': pd.Series(pagerank)
        }
        
        return self.centrality
    
    def analyze_risk_propagation(self, shock_size: float = 0.1,
                               max_steps: int = 10) -> Dict:
        """
        Analyze risk propagation through the network.
        
        Args:
            shock_size: Initial shock size
            max_steps: Maximum propagation steps
            
        Returns:
            Risk propagation analysis
        """
        if self.centrality is None:
            self.compute_centrality()
        
        # Initialize shock
        shock = {node: 0.0 for node in self.graph.nodes}
        most_central = self.centrality['eigenvector'].idxmax()
        shock[most_central] = shock_size
        
        # Propagate shock
        propagation = {0: shock.copy()}
        for step in range(1, max_steps + 1):
            new_shock = {node: 0.0 for node in self.graph.nodes}
            
            for node in self.graph.nodes:
                # Get neighbors
                neighbors = list(self.graph.neighbors(node))
                
                # Propagate shock
                for neighbor in neighbors:
                    weight = self.graph[node][neighbor]['weight']
                    new_shock[node] += shock[neighbor] * weight
            
            # Update shock
            shock = new_shock
            propagation[step] = shock.copy()
        
        return propagation
    
    def get_portfolio_weights(self, method: str = 'eigenvector',
                            top_n: int = 10) -> pd.Series:
        """
        Get portfolio weights based on centrality.
        
        Args:
            method: Centrality measure to use
            top_n: Number of top assets to include
            
        Returns:
            Portfolio weights
        """
        if self.centrality is None:
            self.compute_centrality()
        
        # Get top assets
        top_assets = self.centrality[method].nlargest(top_n)
        
        # Normalize weights
        weights = top_assets / top_assets.sum()
        
        return weights 