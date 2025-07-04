"""Phase transition detection in semantic systems."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import entropy
import networkx as nx

from ..core.types import Word, Timestamp, PhaseTransition
from ..core.base import BaseAnalyzer
from ..metrics.complexity_metrics import CriticalityMeasure

class PhaseTransitionDetector(BaseAnalyzer):
    """Detects phase transitions in semantic systems"""
    
    def __init__(self, embeddings_store, config):
        super().__init__("PhaseTransitionDetector")
        self.embeddings = embeddings_store
        self.config = config
        self.criticality_metric = CriticalityMeasure()
        
        # Order parameters for different transition types
        self.order_parameters = {
            'modularity': self._compute_modularity_order,
            'clustering': self._compute_clustering_order,
            'connectivity': self._compute_connectivity_order,
            'entropy': self._compute_entropy_order,
            'synchronization': self._compute_synchronization_order
        }
    
    def analyze(self) -> Dict[str, any]:
        """Detect phase transitions across timeline"""
        timestamps = self.embeddings.get_timestamps()
        
        # Compute order parameters over time
        order_parameter_timeseries = self._compute_order_parameters(timestamps)
        
        # Detect transitions
        transitions = self._detect_transitions(
            order_parameter_timeseries, timestamps
        )
        
        # Analyze criticality indicators
        criticality_analysis = self._analyze_criticality(
            order_parameter_timeseries, timestamps
        )
        
        return {
            'order_parameters': order_parameter_timeseries,
            'phase_transitions': [self._serialize_transition(t) for t in transitions],
            'criticality_analysis': criticality_analysis,
            'phase_diagram': self._construct_phase_diagram(order_parameter_timeseries),
            'summary': self._generate_summary(transitions, criticality_analysis)
        }
    
    def _compute_order_parameters(self, timestamps: List[Timestamp]) -> Dict[str, List[float]]:
        """Compute order parameters across all timestamps"""
        timeseries = {param: [] for param in self.order_parameters.keys()}
        
        for timestamp in timestamps:
            for param_name, param_func in self.order_parameters.items():
                try:
                    value = param_func(timestamp)
                    timeseries[param_name].append(value)
                except Exception:
                    timeseries[param_name].append(0.0)
        
        return timeseries
    
    def _compute_modularity_order(self, timestamp: Timestamp) -> float:
        """Modularity as order parameter"""
        network = self._build_network(timestamp)
        
        if network.number_of_nodes() < 3:
            return 0.0
        
        try:
            communities = nx.community.louvain_communities(network, weight='weight')
            modularity = nx.community.modularity(network, communities, weight='weight')
            return modularity
        except:
            return 0.0
    
    def _compute_clustering_order(self, timestamp: Timestamp) -> float:
        """Average clustering as order parameter"""
        network = self._build_network(timestamp)
        
        if network.number_of_nodes() < 3:
            return 0.0
        
        try:
            return nx.average_clustering(network, weight='weight')
        except:
            return 0.0
    
    def _compute_connectivity_order(self, timestamp: Timestamp) -> float:
        """Global efficiency as connectivity order parameter"""
        network = self._build_network(timestamp)
        
        if network.number_of_nodes() < 2:
            return 0.0
        
        try:
            return nx.global_efficiency(network)
        except:
            # Fallback: density
            return nx.density(network)
    
    def _compute_entropy_order(self, timestamp: Timestamp) -> float:
        """Entropy-based order parameter"""
        network = self._build_network(timestamp)
        
        if network.number_of_nodes() == 0:
            return 0.0
        
        # Degree distribution entropy
        degrees = [d for n, d in network.degree()]
        
        if not degrees or max(degrees) == 0:
            return 0.0
        
        # Normalize degree distribution
        degree_counts = {}
        for d in degrees:
            degree_counts[d] = degree_counts.get(d, 0) + 1
        
        total = len(degrees)
        probabilities = [count / total for count in degree_counts.values()]
        
        return entropy(probabilities, base=2)
    
    def _compute_synchronization_order(self, timestamp: Timestamp) -> float:
        """Synchronization order parameter"""
        # This would require temporal data - simplified version
        vocabulary = self.embeddings.get_vocabulary(timestamp)
        
        if len(vocabulary) < 10:
            return 0.0
        
        # Sample words and compute pairwise similarities
        sample_words = list(vocabulary)[:100]  # Sample for efficiency
        similarities = []
        
        for i, w1 in enumerate(sample_words):
            emb1 = self.embeddings.get_embedding(w1, timestamp)
            if emb1 is None:
                continue
            
            for w2 in sample_words[i+1:]:
                emb2 = self.embeddings.get_embedding(w2, timestamp)
                if emb2 is not None:
                    from scipy.spatial.distance import cosine
                    sim = 1 - cosine(emb1, emb2)
                    similarities.append(sim)
        
        if not similarities:
            return 0.0
        
        # Synchronization ~ variance in similarities (lower = more synchronized)
        return 1.0 - np.var(similarities)
    
    def _build_network(self, timestamp: Timestamp) -> nx.Graph:
        """Build semantic network for order parameter computation"""
        from scipy.spatial.distance import cosine
        
        G = nx.Graph()
        vocabulary = list(self.embeddings.get_vocabulary(timestamp))
        
        if len(vocabulary) < 2:
            return G
        
        # Add nodes
        G.add_nodes_from(vocabulary)
        
        # Add edges based on similarity
        threshold = 0.3
        
        for i, w1 in enumerate(vocabulary):
            emb1 = self.embeddings.get_embedding(w1, timestamp)
            if emb1 is None:
                continue
            
            for w2 in vocabulary[i+1:]:
                emb2 = self.embeddings.get_embedding(w2, timestamp)
                if emb2 is not None:
                    similarity = 1 - cosine(emb1, emb2)
                    if similarity > threshold:
                        G.add_edge(w1, w2, weight=similarity)
        
        return G
    
    def _detect_transitions(self, order_parameters: Dict[str, List[float]], 
                           timestamps: List[Timestamp]) -> List[PhaseTransition]:
        """Detect phase transitions from order parameter changes"""
        transitions = []
        window_size = getattr(self.config, 'phase_detection_window', 5)
        
        for param_name, values in order_parameters.items():
            if len(values) < window_size * 2:
                continue
            
            # Sliding window to detect abrupt changes
            for i in range(window_size, len(values) - window_size):
                before_window = values[i-window_size:i]
                after_window = values[i:i+window_size]
                
                # Test for significant change
                before_mean = np.mean(before_window)
                after_mean = np.mean(after_window)
                
                # Combined variance
                combined_var = (np.var(before_window) + np.var(after_window)) / 2
                
                if combined_var > 0:
                    # Normalized change
                    change_magnitude = abs(after_mean - before_mean) / np.sqrt(combined_var)
                    
                    if change_magnitude > 2.0:  # Threshold for significant change
                        # Determine transition type
                        if after_mean > before_mean:
                            transition_type = f"{param_name}_increase"
                        else:
                            transition_type = f"{param_name}_decrease"
                        
                        # Get affected communities (simplified)
                        affected_communities = self._identify_affected_communities(
                            timestamps[i]
                        )
                        
                        transition = PhaseTransition(
                            timestamp=timestamps[i],
                            transition_type=transition_type,
                            order_parameter_before=before_mean,
                            order_parameter_after=after_mean,
                            criticality_indicators={
                                'change_magnitude': change_magnitude,
                                'parameter_name': param_name
                            },
                            affected_communities=affected_communities
                        )
                        transitions.append(transition)
        
        return transitions
    
    def _identify_affected_communities(self, timestamp: Timestamp) -> List[set]:
        """Identify communities affected by transition"""
        network = self._build_network(timestamp)
        
        try:
            communities = nx.community.louvain_communities(network, weight='weight')
            return [set(c) for c in communities]
        except:
            return []
    
    def _analyze_criticality(self, order_parameters: Dict[str, List[float]], 
                           timestamps: List[Timestamp]) -> Dict[str, any]:
        """Analyze criticality indicators"""
        criticality_results = {}
        
        for param_name, values in order_parameters.items():
            if len(values) < 10:
                continue
            
            # Apply criticality analysis
            criticality = self.criticality_metric.compute(values)
            
            # Early warning signals
            variance_trend = self._compute_trend(values, metric='variance')
            autocorr_trend = self._compute_trend(values, metric='autocorrelation')
            
            criticality_results[param_name] = {
                'criticality_measures': criticality,
                'variance_trend': variance_trend,
                'autocorrelation_trend': autocorr_trend,
                'early_warning_score': (
                    criticality.get('variance', 0) + 
                    criticality.get('autocorrelation', 0)
                ) / 2
            }
        
        return criticality_results
    
    def _compute_trend(self, timeseries: List[float], metric: str) -> float:
        """Compute trend in a specific metric"""
        if len(timeseries) < 5:
            return 0.0
        
        window_size = 5
        metric_values = []
        
        for i in range(len(timeseries) - window_size + 1):
            window = timeseries[i:i + window_size]
            
            if metric == 'variance':
                metric_values.append(np.var(window))
            elif metric == 'autocorrelation':
                if len(window) > 1:
                    autocorr = np.corrcoef(window[:-1], window[1:])[0, 1]
                    metric_values.append(autocorr if not np.isnan(autocorr) else 0.0)
                else:
                    metric_values.append(0.0)
        
        if len(metric_values) < 2:
            return 0.0
        
        # Linear trend
        x = np.arange(len(metric_values))
        trend_slope = np.polyfit(x, metric_values, 1)[0]
        
        return trend_slope
    
    def _construct_phase_diagram(self, order_parameters: Dict[str, List[float]]) -> Dict[str, any]:
        """Construct phase diagram from order parameters"""
        if len(order_parameters) < 2:
            return {}
        
        # Use first two parameters for 2D phase diagram
        param_names = list(order_parameters.keys())[:2]
        x_values = order_parameters[param_names[0]]
        y_values = order_parameters[param_names[1]]
        
        # Identify phase regions using clustering
        try:
            from sklearn.cluster import KMeans
            
            if len(x_values) >= 3:
                # Combine parameters
                data = np.array([x_values, y_values]).T
                
                # Determine optimal number of clusters (phases)
                n_phases = min(4, len(x_values) // 3)  # Max 4 phases
                
                if n_phases >= 2:
                    kmeans = KMeans(n_clusters=n_phases, random_state=42)
                    phase_labels = kmeans.fit_predict(data)
                    
                    return {
                        'x_parameter': param_names[0],
                        'y_parameter': param_names[1],
                        'x_values': x_values,
                        'y_values': y_values,
                        'phase_labels': phase_labels.tolist(),
                        'phase_centers': kmeans.cluster_centers_.tolist(),
                        'n_phases': n_phases
                    }
        except ImportError:
            pass
        
        return {
            'x_parameter': param_names[0] if param_names else None,
            'y_parameter': param_names[1] if len(param_names) > 1 else None,
            'x_values': x_values if param_names else [],
            'y_values': y_values if len(param_names) > 1 else []
        }
    
    def _generate_summary(self, transitions: List[PhaseTransition], 
                         criticality_analysis: Dict) -> Dict[str, any]:
        """Generate summary of phase transition analysis"""
        total_transitions = len(transitions)
        
        # Count transition types
        transition_types = {}
        for transition in transitions:
            t_type = transition.transition_type
            transition_types[t_type] = transition_types.get(t_type, 0) + 1
        
        # Average criticality scores
        avg_criticality = {}
        for param, analysis in criticality_analysis.items():
            avg_criticality[param] = analysis.get('early_warning_score', 0.0)
        
        overall_criticality = np.mean(list(avg_criticality.values())) if avg_criticality else 0.0
        
        return {
            'total_transitions': total_transitions,
            'transition_types': transition_types,
            'average_criticality_by_parameter': avg_criticality,
            'overall_criticality_score': overall_criticality,
            'system_stability': 'unstable' if total_transitions > 3 else 'stable',
            'critical_warning': overall_criticality > 0.5
        }
    
    def _serialize_transition(self, transition: PhaseTransition) -> Dict:
        """Serialize transition for export"""
        return {
            'timestamp': transition.timestamp,
            'transition_type': transition.transition_type,
            'order_parameter_before': transition.order_parameter_before,
            'order_parameter_after': transition.order_parameter_after,
            'criticality_indicators': transition.criticality_indicators,
            'affected_communities': [list(c) for c in transition.affected_communities]
        }