"""Network-based metrics for semantic analysis."""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import networkx as nx
from collections import defaultdict

from .base_metric import BaseMetric
from ..core.types import Word, Timestamp

class SemanticCentrality(BaseMetric):
    """Measures centrality of words in semantic networks"""
    
    def compute(self, graph: nx.Graph, 
                centrality_type: str = 'eigenvector') -> Dict[Word, float]:
        """
        Compute various centrality measures
        """
        if graph.number_of_nodes() == 0:
            return {}
        
        try:
            if centrality_type == 'eigenvector':
                return nx.eigenvector_centrality(graph, weight='weight', max_iter=1000)
            elif centrality_type == 'betweenness':
                return nx.betweenness_centrality(graph, weight='weight')
            elif centrality_type == 'closeness':
                return nx.closeness_centrality(graph, distance='weight')
            elif centrality_type == 'pagerank':
                return nx.pagerank(graph, weight='weight')
            elif centrality_type == 'degree':
                return nx.degree_centrality(graph)
            else:
                raise ValueError(f"Unknown centrality type: {centrality_type}")
        except:
            # Fallback to degree centrality
            return nx.degree_centrality(graph)

class ClusteringEvolution(BaseMetric):
    """Tracks evolution of clustering patterns"""
    
    def compute(self, graph_t1: nx.Graph, 
                graph_t2: nx.Graph,
                method: str = 'local') -> Dict[str, float]:
        """
        Measure changes in clustering structure
        """
        if method == 'local':
            return self._local_clustering_change(graph_t1, graph_t2)
        elif method == 'global':
            return self._global_clustering_change(graph_t1, graph_t2)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _local_clustering_change(self, graph_t1: nx.Graph, 
                                graph_t2: nx.Graph) -> Dict[str, float]:
        """Compute local clustering coefficient changes"""
        # Get common nodes
        common_nodes = set(graph_t1.nodes()) & set(graph_t2.nodes())
        
        if not common_nodes:
            return {'mean_change': 0.0, 'std_change': 0.0, 'max_change': 0.0}
        
        # Compute clustering coefficients
        clustering_t1 = nx.clustering(graph_t1, weight='weight')
        clustering_t2 = nx.clustering(graph_t2, weight='weight')
        
        # Compute changes
        changes = []
        for node in common_nodes:
            c1 = clustering_t1.get(node, 0.0)
            c2 = clustering_t2.get(node, 0.0)
            changes.append(abs(c2 - c1))
        
        if not changes:
            return {'mean_change': 0.0, 'std_change': 0.0, 'max_change': 0.0}
        
        return {
            'mean_change': np.mean(changes),
            'std_change': np.std(changes),
            'max_change': max(changes)
        }
    
    def _global_clustering_change(self, graph_t1: nx.Graph,
                                 graph_t2: nx.Graph) -> Dict[str, float]:
        """Compute global clustering changes"""
        try:
            global_c1 = nx.average_clustering(graph_t1, weight='weight')
            global_c2 = nx.average_clustering(graph_t2, weight='weight')
            
            # Transitivity (different from clustering)
            trans_c1 = nx.transitivity(graph_t1)
            trans_c2 = nx.transitivity(graph_t2)
            
            return {
                'clustering_change': abs(global_c2 - global_c1),
                'transitivity_change': abs(trans_c2 - trans_c1),
                'relative_clustering_change': abs(global_c2 - global_c1) / max(global_c1, 1e-6)
            }
        except:
            return {'clustering_change': 0.0, 'transitivity_change': 0.0, 'relative_clustering_change': 0.0}

class BridgeStrength(BaseMetric):
    """Measures strength of semantic bridges between communities"""
    
    def compute(self, graph: nx.Graph,
                communities: List[Set[Word]]) -> Dict[str, float]:
        """
        Compute bridge strength metrics
        """
        if len(communities) < 2:
            return {'total_bridge_strength': 0.0, 'normalized_bridge_strength': 0.0}
        
        # Find inter-community edges (bridges)
        bridge_weights = []
        total_edge_weight = 0.0
        
        for edge in graph.edges(data=True):
            u, v, data = edge
            weight = data.get('weight', 1.0)
            total_edge_weight += weight
            
            # Check if edge is a bridge (connects different communities)
            u_community = None
            v_community = None
            
            for i, community in enumerate(communities):
                if u in community:
                    u_community = i
                if v in community:
                    v_community = i
            
            if u_community is not None and v_community is not None and u_community != v_community:
                bridge_weights.append(weight)
        
        total_bridge_strength = sum(bridge_weights)
        normalized_bridge_strength = total_bridge_strength / total_edge_weight if total_edge_weight > 0 else 0.0
        
        return {
            'total_bridge_strength': total_bridge_strength,
            'normalized_bridge_strength': normalized_bridge_strength,
            'bridge_count': len(bridge_weights),
            'average_bridge_strength': np.mean(bridge_weights) if bridge_weights else 0.0
        }

class PathLengthEvolution(BaseMetric):
    """Tracks evolution of path lengths in semantic networks"""
    
    def compute(self, graph_t1: nx.Graph,
                graph_t2: nx.Graph,
                sample_size: int = 1000) -> Dict[str, float]:
        """
        Measure changes in characteristic path length
        """
        # Get common nodes
        common_nodes = list(set(graph_t1.nodes()) & set(graph_t2.nodes()))
        
        if len(common_nodes) < 10:
            return {'path_length_change': 0.0, 'efficiency_change': 0.0}
        
        # Sample node pairs for efficiency
        if len(common_nodes) > sample_size:
            sample_nodes = np.random.choice(common_nodes, sample_size, replace=False)
        else:
            sample_nodes = common_nodes
        
        # Compute path lengths
        paths_t1 = []
        paths_t2 = []
        
        for i, node1 in enumerate(sample_nodes):
            for node2 in sample_nodes[i+1:]:
                try:
                    path_t1 = nx.shortest_path_length(graph_t1, node1, node2, weight='weight')
                    path_t2 = nx.shortest_path_length(graph_t2, node1, node2, weight='weight')
                    paths_t1.append(path_t1)
                    paths_t2.append(path_t2)
                except:
                    # Nodes not connected
                    continue
        
        if not paths_t1:
            return {'path_length_change': 0.0, 'efficiency_change': 0.0}
        
        # Compute statistics
        avg_path_t1 = np.mean(paths_t1)
        avg_path_t2 = np.mean(paths_t2)
        
        # Efficiency (1/path_length)
        efficiency_t1 = np.mean([1/p for p in paths_t1 if p > 0])
        efficiency_t2 = np.mean([1/p for p in paths_t2 if p > 0])
        
        return {
            'path_length_change': avg_path_t2 - avg_path_t1,
            'relative_path_change': (avg_path_t2 - avg_path_t1) / avg_path_t1 if avg_path_t1 > 0 else 0.0,
            'efficiency_change': efficiency_t2 - efficiency_t1
        }

class ModularityEvolution(BaseMetric):
    """Tracks changes in network modularity"""
    
    def compute(self, graph_t1: nx.Graph,
                graph_t2: nx.Graph,
                resolution: float = 1.0) -> Dict[str, float]:
        """
        Measure modularity changes over time
        """
        try:
            # Detect communities
            communities_t1 = nx.community.louvain_communities(graph_t1, weight='weight', resolution=resolution)
            communities_t2 = nx.community.louvain_communities(graph_t2, weight='weight', resolution=resolution)
            
            # Compute modularity
            mod_t1 = nx.community.modularity(graph_t1, communities_t1, weight='weight')
            mod_t2 = nx.community.modularity(graph_t2, communities_t2, weight='weight')
            
            # Community overlap (how much communities changed)
            overlap = self._compute_community_overlap(communities_t1, communities_t2)
            
            return {
                'modularity_change': mod_t2 - mod_t1,
                'modularity_t1': mod_t1,
                'modularity_t2': mod_t2,
                'community_count_t1': len(communities_t1),
                'community_count_t2': len(communities_t2),
                'community_overlap': overlap
            }
            
        except:
            return {
                'modularity_change': 0.0,
                'modularity_t1': 0.0,
                'modularity_t2': 0.0,
                'community_count_t1': 0,
                'community_count_t2': 0,
                'community_overlap': 0.0
            }
    
    def _compute_community_overlap(self, communities_t1: List[Set[Word]],
                                  communities_t2: List[Set[Word]]) -> float:
        """Compute overlap between community structures"""
        if not communities_t1 or not communities_t2:
            return 0.0
        
        # For each community in t1, find best matching community in t2
        max_overlaps = []
        
        for c1 in communities_t1:
            best_overlap = 0.0
            for c2 in communities_t2:
                overlap = len(c1 & c2) / len(c1 | c2) if len(c1 | c2) > 0 else 0.0
                best_overlap = max(best_overlap, overlap)
            max_overlaps.append(best_overlap)
        
        return np.mean(max_overlaps) if max_overlaps else 0.0

class NetworkResilience(BaseMetric):
    """Measures resilience of semantic networks to perturbations"""
    
    def compute(self, graph: nx.Graph,
                perturbation_type: str = 'random',
                perturbation_fraction: float = 0.1) -> Dict[str, float]:
        """
        Measure network resilience to node/edge removal
        """
        if graph.number_of_nodes() < 10:
            return {'resilience_score': 0.0, 'critical_threshold': 0.0}
        
        original_connectivity = self._compute_connectivity(graph)
        
        if perturbation_type == 'random':
            resilience = self._random_resilience(graph, perturbation_fraction)
        elif perturbation_type == 'targeted':
            resilience = self._targeted_resilience(graph, perturbation_fraction)
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")
        
        return resilience
    
    def _compute_connectivity(self, graph: nx.Graph) -> float:
        """Compute network connectivity measure"""
        if graph.number_of_nodes() <= 1:
            return 0.0
        
        try:
            # Use efficiency as connectivity measure
            efficiency = nx.global_efficiency(graph)
            return efficiency
        except:
            # Fallback: fraction of possible edges
            actual_edges = graph.number_of_edges()
            possible_edges = graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2
            return actual_edges / possible_edges if possible_edges > 0 else 0.0
    
    def _random_resilience(self, graph: nx.Graph,
                          perturbation_fraction: float) -> Dict[str, float]:
        """Measure resilience to random node removal"""
        original_connectivity = self._compute_connectivity(graph)
        
        # Remove random nodes
        nodes_to_remove = int(perturbation_fraction * graph.number_of_nodes())
        random_nodes = np.random.choice(list(graph.nodes()), 
                                      size=min(nodes_to_remove, graph.number_of_nodes() - 1),
                                      replace=False)
        
        perturbed_graph = graph.copy()
        perturbed_graph.remove_nodes_from(random_nodes)
        
        perturbed_connectivity = self._compute_connectivity(perturbed_graph)
        
        resilience_score = perturbed_connectivity / original_connectivity if original_connectivity > 0 else 0.0
        
        return {
            'resilience_score': resilience_score,
            'nodes_removed': len(random_nodes),
            'connectivity_loss': original_connectivity - perturbed_connectivity
        }
    
    def _targeted_resilience(self, graph: nx.Graph,
                           perturbation_fraction: float) -> Dict[str, float]:
        """Measure resilience to targeted high-centrality node removal"""
        original_connectivity = self._compute_connectivity(graph)
        
        # Compute centrality and target highest-centrality nodes
        try:
            centrality = nx.betweenness_centrality(graph)
        except:
            centrality = nx.degree_centrality(graph)
        
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        nodes_to_remove = int(perturbation_fraction * graph.number_of_nodes())
        target_nodes = [node for node, _ in sorted_nodes[:nodes_to_remove]]
        
        perturbed_graph = graph.copy()
        perturbed_graph.remove_nodes_from(target_nodes)
        
        perturbed_connectivity = self._compute_connectivity(perturbed_graph)
        
        resilience_score = perturbed_connectivity / original_connectivity if original_connectivity > 0 else 0.0
        
        return {
            'resilience_score': resilience_score,
            'nodes_removed': len(target_nodes),
            'connectivity_loss': original_connectivity - perturbed_connectivity
        }

class DegreeDistributionEvolution(BaseMetric):
    """Tracks evolution of degree distributions"""
    
    def compute(self, graph_t1: nx.Graph,
                graph_t2: nx.Graph) -> Dict[str, float]:
        """
        Measure changes in degree distribution
        """
        # Get degree sequences
        degrees_t1 = [d for n, d in graph_t1.degree()]
        degrees_t2 = [d for n, d in graph_t2.degree()]
        
        if not degrees_t1 or not degrees_t2:
            return {'ks_statistic': 0.0, 'mean_degree_change': 0.0}
        
        # Statistical comparison
        from scipy.stats import ks_2samp
        
        try:
            ks_stat, p_value = ks_2samp(degrees_t1, degrees_t2)
        except:
            ks_stat, p_value = 0.0, 1.0
        
        # Mean degree change
        mean_degree_change = np.mean(degrees_t2) - np.mean(degrees_t1)
        
        # Variance change
        var_change = np.var(degrees_t2) - np.var(degrees_t1)
        
        return {
            'ks_statistic': ks_stat,
            'ks_p_value': p_value,
            'mean_degree_change': mean_degree_change,
            'variance_change': var_change,
            'max_degree_t1': max(degrees_t1) if degrees_t1 else 0,
            'max_degree_t2': max(degrees_t2) if degrees_t2 else 0
        }