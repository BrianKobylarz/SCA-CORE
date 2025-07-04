"""Semantic cascade analysis and contagion dynamics."""

import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cosine

from ..core.types import Word, Timestamp, FlowEvent, FlowEventType
from ..core.base import BaseAnalyzer
from ..metrics.complexity_metrics import CascadeRisk, NetworkContagion

class SemanticCascadeAnalyzer(BaseAnalyzer):
    """Analyzes cascade dynamics and contagion potential"""
    
    def __init__(self, embeddings_store, flow_tracker, config):
        super().__init__("SemanticCascadeAnalyzer")
        self.embeddings = embeddings_store
        self.flow_tracker = flow_tracker
        self.config = config
        self.cascade_risk_metric = CascadeRisk()
        self.contagion_metric = NetworkContagion()
    
    def compute_cascade_risk(self, influence_matrix: np.ndarray) -> float:
        """Compute cascade risk from influence matrix."""
        if influence_matrix.size == 0:
            return 0.0
        return self.cascade_risk_metric.compute(influence_matrix)
    
    def compute_r0(self, influence_matrix: np.ndarray) -> float:
        """Compute basic reproduction number R0."""
        if influence_matrix.size == 0:
            return 0.0
        
        # R0 is the spectral radius (largest eigenvalue) of the influence matrix
        try:
            eigenvalues = np.linalg.eigvals(influence_matrix)
            return float(np.max(np.abs(eigenvalues)))
        except:
            return 0.0
    
    def analyze(self, timestamp=None, **kwargs):
        """Implement BaseAnalyzer interface."""
        timestamps = [timestamp] if timestamp else self.embeddings.get_timestamps()
        
        all_results = {}
        total_risk = 0
        
        for ts in timestamps:
            ts_results = self.analyze_cascade_potential(ts)
            all_results[ts] = ts_results
            total_risk += ts_results.get('cascade_risk', 0)
        
        avg_risk = total_risk / len(timestamps) if timestamps else 0
        
        self.results = {
            'timestamp_results': all_results,
            'summary': {
                'total_timestamps': len(timestamps),
                'average_cascade_risk': avg_risk,
                'high_risk_timestamps': [
                    ts for ts, result in all_results.items() 
                    if result.get('cascade_risk', 0) > self.config.cascade_risk_threshold
                ]
            }
        }
        
        return self.results
    
    def analyze_cascade_potential(self, timestamp: Timestamp) -> Dict:
        """Comprehensive cascade analysis at a time point"""
        # Build network
        network = self._build_semantic_network(timestamp)
        
        # Compute cascade risk (R₀)
        adjacency = nx.adjacency_matrix(network).toarray()
        cascade_risk = self.cascade_risk_metric.compute(adjacency)
        
        # Identify superspreader words
        superspreaders = self._identify_superspreaders(network)
        
        # Find vulnerable clusters
        vulnerable_clusters = self._find_vulnerable_clusters(network)
        
        # Compute contagion paths
        contagion_paths = self._compute_contagion_paths(network)
        
        # Predict cascade scenarios
        scenarios = self._predict_cascade_scenarios(
            network, cascade_risk, superspreaders
        )
        
        return {
            'cascade_risk': cascade_risk,
            'risk_interpretation': self._interpret_cascade_risk(cascade_risk),
            'superspreaders': superspreaders,
            'vulnerable_clusters': vulnerable_clusters,
            'contagion_paths': contagion_paths,
            'scenarios': scenarios,
            'network_stats': {
                'nodes': network.number_of_nodes(),
                'edges': network.number_of_edges(),
                'density': nx.density(network),
                'clustering': nx.average_clustering(network, weight='weight')
            }
        }
    
    def track_cascade_evolution(self) -> Dict[Timestamp, Dict]:
        """Track how cascade potential evolves over time"""
        evolution = {}
        timestamps = self.embeddings.get_timestamps()
        
        for timestamp in timestamps:
            evolution[timestamp] = self.analyze_cascade_potential(timestamp)
        
        # Add trend analysis
        evolution['trends'] = self._analyze_cascade_trends(evolution)
        
        return evolution
    
    def simulate_semantic_contagion(self, 
                                   timestamp: Timestamp,
                                   seed_words: List[Word],
                                   steps: int = 10) -> List[Set[Word]]:
        """Simulate how semantic change might spread"""
        network = self._build_semantic_network(timestamp)
        
        # Initialize with seed words
        infected = set(seed_words)
        susceptible = set(network.nodes()) - infected
        
        infection_history = [infected.copy()]
        
        for step in range(steps):
            newly_infected = set()
            
            for word in infected:
                # Check each neighbor
                for neighbor in network.neighbors(word):
                    if neighbor in susceptible:
                        # Probability of infection based on edge weight
                        edge_weight = network[word][neighbor]['weight']
                        infection_prob = self._compute_infection_probability(
                            edge_weight, len(infected), len(susceptible)
                        )
                        
                        if np.random.random() < infection_prob:
                            newly_infected.add(neighbor)
            
            # Update sets
            infected.update(newly_infected)
            susceptible -= newly_infected
            
            infection_history.append(infected.copy())
            
            # Stop if no new infections
            if not newly_infected:
                break
        
        return infection_history
    
    def _build_semantic_network(self, timestamp: Timestamp) -> nx.Graph:
        """Build weighted semantic network"""
        G = nx.Graph()
        
        vocabulary = self.embeddings.get_vocabulary(timestamp)
        
        # Convert to list for indexing
        vocab_list = list(vocabulary)
        
        # Add nodes
        G.add_nodes_from(vocab_list)
        
        # Add edges based on similarity
        threshold = self.config.flow_similarity_threshold
        
        for i, w1 in enumerate(vocab_list):
            emb1 = self.embeddings.get_embedding(w1, timestamp)
            if emb1 is None:
                continue
            
            for w2 in vocab_list[i+1:]:
                emb2 = self.embeddings.get_embedding(w2, timestamp)
                if emb2 is not None:
                    similarity = 1 - cosine(emb1, emb2)
                    
                    if similarity > threshold:
                        G.add_edge(w1, w2, weight=similarity)
        
        return G
    
    def _identify_superspreaders(self, network: nx.Graph, 
                                top_k: int = 10) -> List[Tuple[Word, float]]:
        """Identify words with highest cascade potential"""
        # Compute eigenvector centrality (considers network effects)
        try:
            centrality = nx.eigenvector_centrality(network, weight='weight')
        except:
            # Fallback to degree centrality
            centrality = nx.degree_centrality(network)
        
        # Sort by centrality
        sorted_words = sorted(centrality.items(), 
                            key=lambda x: x[1], reverse=True)
        
        return sorted_words[:top_k]
    
    def _find_vulnerable_clusters(self, network: nx.Graph) -> List[Set[Word]]:
        """Find clusters susceptible to cascade"""
        # Detect communities
        communities = nx.community.louvain_communities(network, weight='weight')
        
        vulnerable = []
        
        for community in communities:
            # Compute internal cascade risk
            subgraph = network.subgraph(community)
            
            if subgraph.number_of_nodes() > 2:
                adjacency = nx.adjacency_matrix(subgraph).toarray()
                internal_risk = self.cascade_risk_metric.compute(adjacency)
                
                if internal_risk > self.config.cascade_risk_threshold:
                    vulnerable.append(set(community))
        
        return vulnerable
    
    def _compute_contagion_paths(self, network: nx.Graph) -> Dict:
        """Compute likely paths for semantic contagion"""
        paths = {
            'highways': [],  # High-weight paths
            'bridges': [],   # Connect communities
            'cycles': []     # Feedback loops
        }
        
        # Find high-weight paths (semantic highways)
        edge_weights = nx.get_edge_attributes(network, 'weight')
        sorted_edges = sorted(edge_weights.items(), 
                            key=lambda x: x[1], reverse=True)
        
        paths['highways'] = [edge for edge, weight in sorted_edges[:20]]
        
        # Find bridges between communities
        communities = nx.community.louvain_communities(network, weight='weight')
        
        for i, c1 in enumerate(communities):
            for c2 in communities[i+1:]:
                # Find edges between communities
                bridge_edges = [
                    (u, v) for u in c1 for v in c2 
                    if network.has_edge(u, v)
                ]
                
                if bridge_edges:
                    # Keep strongest bridge
                    best_bridge = max(
                        bridge_edges, 
                        key=lambda e: network[e[0]][e[1]]['weight']
                    )
                    paths['bridges'].append(best_bridge)
        
        # Find cycles (feedback potential)
        try:
            cycles = nx.cycle_basis(network)
            paths['cycles'] = cycles[:10]  # Keep shortest cycles
        except:
            paths['cycles'] = []
        
        return paths
    
    def _predict_cascade_scenarios(self, network: nx.Graph,
                                  cascade_risk: float,
                                  superspreaders: List[Tuple[Word, float]]) -> List[Dict]:
        """Predict likely cascade scenarios"""
        scenarios = []
        
        if cascade_risk > 1.5:  # High risk
            scenarios.append({
                'type': 'system_wide_cascade',
                'probability': 'high',
                'trigger': 'Any superspreader word shifting',
                'expected_impact': 'Global semantic reorganization',
                'timeline': '2-3 time steps'
            })
        
        elif cascade_risk > 1.0:  # Moderate risk
            scenarios.append({
                'type': 'localized_cascades',
                'probability': 'moderate',
                'trigger': 'Multiple words in vulnerable cluster',
                'expected_impact': 'Community-level changes',
                'timeline': '3-5 time steps'
            })
        
        else:  # Low risk
            scenarios.append({
                'type': 'isolated_changes',
                'probability': 'low',
                'trigger': 'Requires coordinated shift',
                'expected_impact': 'Individual word changes only',
                'timeline': 'Gradual evolution'
            })
        
        return scenarios
    
    def _compute_infection_probability(self, edge_weight: float,
                                     infected_count: int,
                                     susceptible_count: int) -> float:
        """Compute probability of semantic contagion"""
        # Base probability from edge weight
        base_prob = edge_weight
        
        # Adjust for system state (more infected → higher pressure)
        system_pressure = infected_count / (infected_count + susceptible_count)
        
        # Final probability
        return base_prob * (1 + system_pressure) / 2
    
    def _interpret_cascade_risk(self, risk: float) -> str:
        """Interpret cascade risk value"""
        if risk > 1.5:
            return "CRITICAL: System highly susceptible to cascading changes"
        elif risk > 1.2:
            return "HIGH: Elevated risk of semantic cascades"
        elif risk > 1.0:
            return "MODERATE: System at cascade threshold"
        elif risk > 0.8:
            return "LOW: Cascades possible but unlikely"
        else:
            return "MINIMAL: System resistant to cascades"
    
    def _analyze_cascade_trends(self, evolution: Dict) -> Dict:
        """Analyze trends in cascade potential"""
        timestamps = sorted([t for t in evolution.keys() if t != 'trends'])
        
        risks = [evolution[t]['cascade_risk'] for t in timestamps]
        
        # Compute trend statistics
        trend = np.polyfit(range(len(risks)), risks, 1)[0]
        
        # Find critical transitions
        transitions = []
        for i in range(1, len(risks)):
            if risks[i-1] < 1.0 and risks[i] >= 1.0:
                transitions.append({
                    'timestamp': timestamps[i],
                    'type': 'subcritical_to_critical',
                    'risk_before': risks[i-1],
                    'risk_after': risks[i]
                })
        
        return {
            'overall_trend': 'increasing' if trend > 0 else 'decreasing',
            'trend_magnitude': abs(trend),
            'critical_transitions': transitions,
            'current_regime': 'critical' if risks[-1] > 1.0 else 'subcritical'
        }