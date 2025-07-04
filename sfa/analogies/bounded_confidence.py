"""Bounded confidence opinion dynamics analogy for semantic flow analysis."""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from scipy.spatial.distance import pdist, squareform
import networkx as nx

from ..core.types import Word, Timestamp, SemanticFlow
from .base_analogy import BaseAnalogy

class BoundedConfidenceAnalogy(BaseAnalogy):
    """Models semantic consensus using bounded confidence opinion dynamics"""
    
    def __init__(self, embeddings_store, config):
        super().__init__("BoundedConfidenceAnalogy", embeddings_store, config)
        
        # Bounded confidence parameters
        self.analogy_parameters = {
            'confidence_threshold': 0.3,     # ε: confidence bound
            'convergence_rate': 0.1,         # α: rate of opinion adjustment
            'stubbornness': 0.05,            # σ: resistance to change
            'noise_level': 0.02,             # η: random opinion fluctuations
            'influence_decay': 0.9,          # decay of influence with distance
            'consensus_threshold': 0.1,      # threshold for consensus detection
            'fragmentation_threshold': 0.5,  # threshold for opinion fragmentation
            'zealot_fraction': 0.05,         # fraction of zealot agents
            'homophily_strength': 0.7,       # tendency to interact with similar others
            'model_type': 'hegselmann_krause' # hegselmann_krause, deffuant, etc.
        }
        
        # Opinion dynamics structures
        self.opinion_states = {}          # timestamp -> {word: opinion_vector}
        self.confidence_networks = {}    # timestamp -> confidence network
        self.consensus_evolution = {}    # timestamp -> consensus metrics
        self.opinion_clusters = {}       # timestamp -> opinion clusters
        
    def fit_model(self, flows: List[SemanticFlow]) -> Dict[str, Any]:
        """Fit bounded confidence model to semantic flows"""
        if not flows:
            return {'parameters': self.analogy_parameters, 'fit_quality': 0.0}
        
        # Organize flows by timestamp
        flow_timeline = self._organize_flows_by_time(flows)
        
        # Map semantic states to opinion dynamics
        self._map_semantic_to_opinions(flow_timeline)
        
        # Build confidence networks
        self._build_confidence_networks()
        
        # Analyze consensus evolution
        self._analyze_consensus_evolution()
        
        # Detect opinion clusters
        self._detect_opinion_clusters()
        
        # Fit model parameters
        fitted_parameters = self._fit_opinion_parameters(flow_timeline)
        
        # Analyze convergence dynamics
        convergence_analysis = self._analyze_convergence_dynamics(fitted_parameters)
        
        # Detect opinion events
        opinion_events = self._detect_opinion_events()
        
        return {
            'parameters': fitted_parameters,
            'opinion_states': self.opinion_states,
            'confidence_networks': self.confidence_networks,
            'consensus_evolution': self.consensus_evolution,
            'opinion_clusters': self.opinion_clusters,
            'convergence_analysis': convergence_analysis,
            'opinion_events': opinion_events,
            'regime': self._determine_opinion_regime(fitted_parameters),
            'polarization_analysis': self._analyze_polarization(),
            'interpretation': self._interpret_opinion_dynamics(fitted_parameters)
        }
    
    def _organize_flows_by_time(self, flows: List[SemanticFlow]) -> Dict[Timestamp, List[SemanticFlow]]:
        """Organize flows by timestamp"""
        flow_timeline = {}
        for flow in flows:
            timestamp = flow.timestamp
            if timestamp not in flow_timeline:
                flow_timeline[timestamp] = []
            flow_timeline[timestamp].append(flow)
        return flow_timeline
    
    def _map_semantic_to_opinions(self, flow_timeline: Dict[Timestamp, List[SemanticFlow]]) -> None:
        """Map semantic embeddings to opinion vectors"""
        for timestamp, flows in flow_timeline.items():
            vocabulary = list(self.embeddings.get_vocabulary(timestamp))
            opinion_state = {}
            
            for word in vocabulary:
                embedding = self.embeddings.get_embedding(word, timestamp)
                if embedding is not None:
                    # Map high-dimensional embedding to opinion space
                    # Use PCA or simple projection for dimensionality reduction
                    opinion_vector = self._embedding_to_opinion(embedding)
                    opinion_state[word] = opinion_vector
            
            self.opinion_states[timestamp] = opinion_state
    
    def _embedding_to_opinion(self, embedding: np.ndarray) -> np.ndarray:
        """Convert semantic embedding to opinion vector"""
        # Simple approach: use first few dimensions or PCA
        if len(embedding) >= 2:
            # Use first two dimensions normalized to [-1, 1]
            opinion = embedding[:2] / (np.linalg.norm(embedding[:2]) + 1e-8)
            return opinion
        else:
            return np.array([0.0, 0.0])
    
    def _build_confidence_networks(self) -> None:
        """Build confidence networks based on opinion similarity"""
        for timestamp, opinion_state in self.opinion_states.items():
            G = nx.Graph()
            words = list(opinion_state.keys())
            
            # Add nodes
            G.add_nodes_from(words)
            
            # Add edges based on confidence threshold
            epsilon = self.analogy_parameters['confidence_threshold']
            
            for i, word1 in enumerate(words):
                opinion1 = opinion_state[word1]
                
                for word2 in words[i+1:]:
                    opinion2 = opinion_state[word2]
                    
                    # Opinion distance
                    distance = np.linalg.norm(opinion1 - opinion2)
                    
                    # Add edge if within confidence bound
                    if distance <= epsilon:
                        confidence = 1.0 - distance / epsilon  # Confidence weight
                        G.add_edge(word1, word2, weight=confidence, distance=distance)
            
            self.confidence_networks[timestamp] = G
    
    def _analyze_consensus_evolution(self) -> None:
        """Analyze evolution of consensus in opinion space"""
        for timestamp, opinion_state in self.opinion_states.items():
            if not opinion_state:
                continue
            
            opinions = list(opinion_state.values())
            opinions_array = np.array(opinions)
            
            # Consensus metrics
            consensus_metrics = {
                'mean_opinion': np.mean(opinions_array, axis=0),
                'opinion_variance': np.var(opinions_array, axis=0),
                'total_variance': np.sum(np.var(opinions_array, axis=0)),
                'opinion_range': np.max(opinions_array, axis=0) - np.min(opinions_array, axis=0),
                'consensus_strength': self._compute_consensus_strength(opinions_array),
                'fragmentation_level': self._compute_fragmentation_level(opinions_array),
                'polarization_index': self._compute_polarization_index(opinions_array)
            }
            
            self.consensus_evolution[timestamp] = consensus_metrics
    
    def _compute_consensus_strength(self, opinions: np.ndarray) -> float:
        """Compute consensus strength (inverse of disagreement)"""
        if len(opinions) < 2:
            return 1.0
        
        # Average pairwise disagreement
        pairwise_distances = pdist(opinions, metric='euclidean')
        mean_disagreement = np.mean(pairwise_distances)
        
        # Convert to consensus (0 = no consensus, 1 = perfect consensus)
        max_possible_distance = 2 * np.sqrt(2)  # Maximum distance in [-1,1]^2 space
        consensus = 1.0 - (mean_disagreement / max_possible_distance)
        
        return max(0.0, consensus)
    
    def _compute_fragmentation_level(self, opinions: np.ndarray) -> float:
        """Compute opinion fragmentation level"""
        if len(opinions) < 3:
            return 0.0
        
        # Use clustering to detect fragments
        try:
            from sklearn.cluster import DBSCAN
            
            # DBSCAN clustering to find opinion clusters
            epsilon = self.analogy_parameters['confidence_threshold']
            clustering = DBSCAN(eps=epsilon, min_samples=2).fit(opinions)
            
            n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            
            # Fragmentation = number of clusters / maximum possible clusters
            max_clusters = len(opinions)
            fragmentation = n_clusters / max_clusters if max_clusters > 0 else 0.0
            
            return fragmentation
        except ImportError:
            # Fallback: use variance as proxy
            total_variance = np.sum(np.var(opinions, axis=0))
            max_variance = 2.0  # Maximum variance in [-1,1]^2 space
            return min(1.0, total_variance / max_variance)
    
    def _compute_polarization_index(self, opinions: np.ndarray) -> float:
        """Compute opinion polarization index"""
        if len(opinions) < 2:
            return 0.0
        
        # Distance from centroid
        centroid = np.mean(opinions, axis=0)
        distances_from_center = [np.linalg.norm(op - centroid) for op in opinions]
        
        # Polarization = variance of distances from center
        polarization = np.var(distances_from_center)
        
        # Normalize to [0, 1]
        max_polarization = 1.0  # Maximum possible variance
        return min(1.0, polarization / max_polarization)
    
    def _detect_opinion_clusters(self) -> None:
        """Detect opinion clusters at each timestamp"""
        for timestamp, opinion_state in self.opinion_states.items():
            if not opinion_state:
                continue
            
            words = list(opinion_state.keys())
            opinions = np.array([opinion_state[word] for word in words])
            
            if len(opinions) < 2:
                continue
            
            # Use DBSCAN clustering
            try:
                from sklearn.cluster import DBSCAN
                
                epsilon = self.analogy_parameters['confidence_threshold']
                clustering = DBSCAN(eps=epsilon, min_samples=2).fit(opinions)
                
                # Organize clusters
                clusters = {}
                for i, label in enumerate(clustering.labels_):
                    if label != -1:  # Not noise
                        if label not in clusters:
                            clusters[label] = []
                        clusters[label].append(words[i])
                
                # Compute cluster properties
                cluster_info = {}
                for label, cluster_words in clusters.items():
                    cluster_opinions = [opinion_state[word] for word in cluster_words]
                    cluster_centroid = np.mean(cluster_opinions, axis=0)
                    cluster_size = len(cluster_words)
                    cluster_coherence = 1.0 - np.std([np.linalg.norm(op - cluster_centroid) 
                                                     for op in cluster_opinions])
                    
                    cluster_info[label] = {
                        'words': cluster_words,
                        'centroid': cluster_centroid,
                        'size': cluster_size,
                        'coherence': max(0.0, cluster_coherence)
                    }
                
                self.opinion_clusters[timestamp] = cluster_info
                
            except ImportError:
                # Fallback: simple distance-based clustering
                self.opinion_clusters[timestamp] = self._simple_clustering(words, opinions)
    
    def _simple_clustering(self, words: List[str], opinions: np.ndarray) -> Dict[int, Dict]:
        """Simple distance-based clustering fallback"""
        epsilon = self.analogy_parameters['confidence_threshold']
        clusters = {}
        used = set()
        cluster_id = 0
        
        for i, word in enumerate(words):
            if word in used:
                continue
            
            # Start new cluster
            cluster = [word]
            used.add(word)
            
            # Find all words within confidence bound
            for j, other_word in enumerate(words):
                if other_word not in used:
                    distance = np.linalg.norm(opinions[i] - opinions[j])
                    if distance <= epsilon:
                        cluster.append(other_word)
                        used.add(other_word)
            
            if len(cluster) >= 2:  # Minimum cluster size
                cluster_opinions = [opinions[words.index(w)] for w in cluster]
                centroid = np.mean(cluster_opinions, axis=0)
                
                clusters[cluster_id] = {
                    'words': cluster,
                    'centroid': centroid,
                    'size': len(cluster),
                    'coherence': 0.5  # Default coherence
                }
                cluster_id += 1
        
        return clusters
    
    def _fit_opinion_parameters(self, flow_timeline: Dict[Timestamp, List[SemanticFlow]]) -> Dict[str, Any]:
        """Fit bounded confidence parameters from data"""
        fitted_params = self.analogy_parameters.copy()
        
        # Estimate confidence threshold from network connectivity
        if self.confidence_networks:
            edge_distances = []
            for network in self.confidence_networks.values():
                for _, _, data in network.edges(data=True):
                    edge_distances.append(data.get('distance', 0.0))
            
            if edge_distances:
                # Use 75th percentile as confidence threshold
                fitted_params['confidence_threshold'] = np.percentile(edge_distances, 75)
        
        # Estimate convergence rate from opinion variance changes
        timestamps = sorted(self.consensus_evolution.keys())
        if len(timestamps) > 1:
            variance_changes = []
            for i in range(1, len(timestamps)):
                prev_var = self.consensus_evolution[timestamps[i-1]]['total_variance']
                curr_var = self.consensus_evolution[timestamps[i]]['total_variance']
                
                if prev_var > 0:
                    change_rate = abs(curr_var - prev_var) / prev_var
                    variance_changes.append(change_rate)
            
            if variance_changes:
                fitted_params['convergence_rate'] = np.mean(variance_changes)
        
        # Estimate fragmentation threshold
        if self.consensus_evolution:
            fragmentation_levels = [
                metrics['fragmentation_level'] 
                for metrics in self.consensus_evolution.values()
            ]
            if fragmentation_levels:
                fitted_params['fragmentation_threshold'] = np.mean(fragmentation_levels)
        
        return fitted_params
    
    def _analyze_convergence_dynamics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze convergence dynamics of opinion system"""
        timestamps = sorted(self.consensus_evolution.keys())
        
        if len(timestamps) < 2:
            return {}
        
        # Convergence trends
        consensus_trend = []
        fragmentation_trend = []
        polarization_trend = []
        
        for timestamp in timestamps:
            metrics = self.consensus_evolution[timestamp]
            consensus_trend.append(metrics['consensus_strength'])
            fragmentation_trend.append(metrics['fragmentation_level'])
            polarization_trend.append(metrics['polarization_index'])
        
        # Analyze trends
        convergence_analysis = {
            'consensus_trend': self._analyze_trend(consensus_trend),
            'fragmentation_trend': self._analyze_trend(fragmentation_trend),
            'polarization_trend': self._analyze_trend(polarization_trend),
            'convergence_speed': self._estimate_convergence_speed(consensus_trend),
            'equilibrium_prediction': self._predict_equilibrium(consensus_trend),
            'stability_analysis': self._analyze_stability(consensus_trend)
        }
        
        return convergence_analysis
    
    def _analyze_trend(self, values: List[float]) -> Dict[str, Any]:
        """Analyze trend in time series"""
        if len(values) < 2:
            return {'direction': 'stable', 'magnitude': 0.0}
        
        # Linear trend
        x = np.arange(len(values))
        trend_slope = np.polyfit(x, values, 1)[0]
        
        # Determine direction
        if trend_slope > 0.01:
            direction = 'increasing'
        elif trend_slope < -0.01:
            direction = 'decreasing'
        else:
            direction = 'stable'
        
        return {
            'direction': direction,
            'magnitude': abs(trend_slope),
            'slope': trend_slope,
            'r_squared': self._compute_trend_r_squared(x, values)
        }
    
    def _compute_trend_r_squared(self, x: np.ndarray, y: List[float]) -> float:
        """Compute R-squared for trend"""
        if len(y) < 2:
            return 0.0
        
        y_array = np.array(y)
        y_mean = np.mean(y_array)
        
        # Linear fit
        slope, intercept = np.polyfit(x, y_array, 1)
        y_pred = slope * x + intercept
        
        # R-squared
        ss_res = np.sum((y_array - y_pred) ** 2)
        ss_tot = np.sum((y_array - y_mean) ** 2)
        
        if ss_tot > 0:
            return 1 - (ss_res / ss_tot)
        else:
            return 0.0
    
    def _estimate_convergence_speed(self, consensus_trend: List[float]) -> float:
        """Estimate speed of convergence"""
        if len(consensus_trend) < 3:
            return 0.0
        
        # Rate of change in consensus
        changes = np.diff(consensus_trend)
        
        # Average absolute change per time step
        convergence_speed = np.mean(np.abs(changes))
        
        return convergence_speed
    
    def _predict_equilibrium(self, consensus_trend: List[float]) -> Dict[str, Any]:
        """Predict equilibrium state"""
        if len(consensus_trend) < 3:
            return {'equilibrium_consensus': 0.5, 'confidence': 0.0}
        
        # Fit exponential or linear trend to predict equilibrium
        x = np.arange(len(consensus_trend))
        y = np.array(consensus_trend)
        
        # Simple linear extrapolation
        slope, intercept = np.polyfit(x, y, 1)
        
        # Extrapolate to future
        future_steps = 10
        future_x = len(consensus_trend) + future_steps
        predicted_equilibrium = slope * future_x + intercept
        
        # Bound to [0, 1]
        predicted_equilibrium = max(0.0, min(1.0, predicted_equilibrium))
        
        # Confidence based on trend consistency
        trend_r_squared = self._compute_trend_r_squared(x, consensus_trend)
        
        return {
            'equilibrium_consensus': predicted_equilibrium,
            'confidence': trend_r_squared,
            'time_to_equilibrium': future_steps
        }
    
    def _analyze_stability(self, consensus_trend: List[float]) -> Dict[str, Any]:
        """Analyze stability of opinion dynamics"""
        if len(consensus_trend) < 3:
            return {'stability': 'unknown', 'volatility': 0.0}
        
        # Volatility
        volatility = np.std(consensus_trend)
        
        # Determine stability
        if volatility < 0.1:
            stability = 'stable'
        elif volatility < 0.3:
            stability = 'moderately_stable'
        else:
            stability = 'unstable'
        
        # Look for oscillations
        oscillations = self._detect_oscillations(consensus_trend)
        
        return {
            'stability': stability,
            'volatility': volatility,
            'oscillations': oscillations,
            'mean_consensus': np.mean(consensus_trend),
            'consensus_range': max(consensus_trend) - min(consensus_trend)
        }
    
    def _detect_oscillations(self, values: List[float]) -> Dict[str, Any]:
        """Detect oscillatory behavior"""
        if len(values) < 4:
            return {'present': False}
        
        # Simple peak detection
        peaks = 0
        troughs = 0
        
        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                peaks += 1
            elif values[i] < values[i-1] and values[i] < values[i+1]:
                troughs += 1
        
        oscillation_count = peaks + troughs
        oscillation_frequency = oscillation_count / len(values)
        
        return {
            'present': oscillation_frequency > 0.2,
            'frequency': oscillation_frequency,
            'peaks': peaks,
            'troughs': troughs
        }
    
    def _detect_opinion_events(self) -> List[Dict[str, Any]]:
        """Detect significant events in opinion dynamics"""
        events = []
        timestamps = sorted(self.consensus_evolution.keys())
        
        for i in range(1, len(timestamps)):
            prev_metrics = self.consensus_evolution[timestamps[i-1]]
            curr_metrics = self.consensus_evolution[timestamps[i]]
            
            # Consensus formation
            consensus_change = curr_metrics['consensus_strength'] - prev_metrics['consensus_strength']
            if consensus_change > 0.3:
                events.append({
                    'type': 'consensus_formation',
                    'timestamp': timestamps[i],
                    'magnitude': consensus_change,
                    'description': f'Consensus increased by {consensus_change:.3f}'
                })
            
            # Consensus breakdown
            elif consensus_change < -0.3:
                events.append({
                    'type': 'consensus_breakdown',
                    'timestamp': timestamps[i],
                    'magnitude': abs(consensus_change),
                    'description': f'Consensus decreased by {abs(consensus_change):.3f}'
                })
            
            # Polarization events
            polarization_change = curr_metrics['polarization_index'] - prev_metrics['polarization_index']
            if polarization_change > 0.3:
                events.append({
                    'type': 'polarization_event',
                    'timestamp': timestamps[i],
                    'magnitude': polarization_change,
                    'description': f'Polarization increased by {polarization_change:.3f}'
                })
            
            # Fragmentation events
            fragmentation_change = curr_metrics['fragmentation_level'] - prev_metrics['fragmentation_level']
            if fragmentation_change > 0.3:
                events.append({
                    'type': 'fragmentation_event',
                    'timestamp': timestamps[i],
                    'magnitude': fragmentation_change,
                    'description': f'Fragmentation increased by {fragmentation_change:.3f}'
                })
        
        return events
    
    def _determine_opinion_regime(self, parameters: Dict[str, Any]) -> str:
        """Determine opinion dynamics regime"""
        epsilon = parameters['confidence_threshold']
        convergence_rate = parameters['convergence_rate']
        fragmentation = parameters['fragmentation_threshold']
        
        if epsilon < 0.2 and fragmentation > 0.5:
            return 'fragmented'
        elif epsilon > 0.5 and convergence_rate > 0.1:
            return 'consensus_forming'
        elif fragmentation < 0.2:
            return 'consensual'
        else:
            return 'mixed_dynamics'
    
    def _analyze_polarization(self) -> Dict[str, Any]:
        """Analyze polarization patterns"""
        if not self.consensus_evolution:
            return {}
        
        polarization_values = [
            metrics['polarization_index'] 
            for metrics in self.consensus_evolution.values()
        ]
        
        return {
            'mean_polarization': np.mean(polarization_values),
            'max_polarization': max(polarization_values),
            'polarization_variance': np.var(polarization_values),
            'polarization_trend': self._analyze_trend(polarization_values),
            'polarization_stability': 'stable' if np.var(polarization_values) < 0.1 else 'unstable'
        }
    
    def _interpret_opinion_dynamics(self, parameters: Dict[str, Any]) -> Dict[str, str]:
        """Interpret opinion dynamics in semantic terms"""
        regime = self._determine_opinion_regime(parameters)
        epsilon = parameters['confidence_threshold']
        
        interpretations = {
            'bounded_confidence_analogy': 'Semantic consensus follows bounded confidence dynamics',
            'opinion_meaning': 'Word embeddings represent semantic opinions or positions',
            'confidence_bound': f'Words interact only if semantic distance < {epsilon:.3f}',
            'consensus_meaning': 'Semantic agreement emerges through local interactions',
            'fragmentation_meaning': 'Semantic communities form around similar meanings',
            'regime_meaning': f'System exhibits {regime} opinion dynamics'
        }
        
        if regime == 'consensus_forming':
            interpretations['dynamics'] = 'Semantic meanings converge toward agreement'
        elif regime == 'fragmented':
            interpretations['dynamics'] = 'Semantic meanings fragment into isolated clusters'
        elif regime == 'consensual':
            interpretations['dynamics'] = 'Strong semantic consensus maintained'
        else:
            interpretations['dynamics'] = 'Mixed semantic dynamics with partial consensus'
        
        return interpretations
    
    def predict_dynamics(self, current_state: Dict[str, Any], 
                        time_horizon: int) -> Dict[str, Any]:
        """Predict future opinion dynamics"""
        # Get current parameters
        epsilon = self.analogy_parameters['confidence_threshold']
        alpha = self.analogy_parameters['convergence_rate']
        
        # Current opinion state
        current_opinions = current_state.get('opinions', {})
        if not current_opinions:
            return {}
        
        # Simulate bounded confidence dynamics
        predictions = self._simulate_hegselmann_krause(current_opinions, epsilon, alpha, time_horizon)
        
        return {
            'opinion_trajectories': predictions['trajectories'],
            'consensus_evolution': predictions['consensus'],
            'cluster_evolution': predictions['clusters'],
            'convergence_prediction': predictions['convergence'],
            'final_configuration': predictions['final_state']
        }
    
    def _simulate_hegselmann_krause(self, initial_opinions: Dict[str, np.ndarray],
                                   epsilon: float, alpha: float, 
                                   time_horizon: int) -> Dict[str, Any]:
        """Simulate Hegselmann-Krause bounded confidence model"""
        words = list(initial_opinions.keys())
        opinions = {word: [initial_opinions[word].copy()] for word in words}
        consensus_history = []
        cluster_history = []
        
        current_opinions = {word: initial_opinions[word].copy() for word in words}
        
        for t in range(time_horizon):
            # Update opinions
            new_opinions = {}
            
            for word in words:
                # Find neighbors within confidence bound
                neighbors = []
                for other_word in words:
                    distance = np.linalg.norm(current_opinions[word] - current_opinions[other_word])
                    if distance <= epsilon:
                        neighbors.append(other_word)
                
                # Average opinions of neighbors
                if neighbors:
                    neighbor_opinions = [current_opinions[neighbor] for neighbor in neighbors]
                    mean_neighbor_opinion = np.mean(neighbor_opinions, axis=0)
                    
                    # Update toward neighbor average
                    new_opinions[word] = current_opinions[word] + alpha * (mean_neighbor_opinion - current_opinions[word])
                else:
                    # No neighbors: opinion stays the same
                    new_opinions[word] = current_opinions[word].copy()
            
            # Record trajectory
            for word in words:
                opinions[word].append(new_opinions[word].copy())
            
            current_opinions = new_opinions
            
            # Compute consensus metrics
            opinion_array = np.array(list(current_opinions.values()))
            consensus_strength = self._compute_consensus_strength(opinion_array)
            consensus_history.append(consensus_strength)
            
            # Detect clusters
            clusters = self._detect_current_clusters(current_opinions, epsilon)
            cluster_history.append(len(clusters))
        
        # Analyze convergence
        convergence_analysis = self._analyze_convergence_from_simulation(consensus_history)
        
        return {
            'trajectories': opinions,
            'consensus': consensus_history,
            'clusters': cluster_history,
            'convergence': convergence_analysis,
            'final_state': current_opinions
        }
    
    def _detect_current_clusters(self, opinions: Dict[str, np.ndarray], epsilon: float) -> List[List[str]]:
        """Detect opinion clusters at current time"""
        words = list(opinions.keys())
        clusters = []
        used = set()
        
        for word in words:
            if word in used:
                continue
            
            # Start new cluster
            cluster = [word]
            used.add(word)
            
            # Find all words within confidence bound
            for other_word in words:
                if other_word not in used:
                    distance = np.linalg.norm(opinions[word] - opinions[other_word])
                    if distance <= epsilon:
                        cluster.append(other_word)
                        used.add(other_word)
            
            clusters.append(cluster)
        
        return clusters
    
    def _analyze_convergence_from_simulation(self, consensus_history: List[float]) -> Dict[str, Any]:
        """Analyze convergence from simulation"""
        if len(consensus_history) < 5:
            return {'converged': False, 'convergence_time': None}
        
        # Check for convergence (consensus stops changing)
        recent_changes = np.abs(np.diff(consensus_history[-5:]))
        converged = np.all(recent_changes < 0.01)
        
        # Find convergence time
        convergence_time = None
        if converged:
            for i in range(len(consensus_history) - 1):
                changes = np.abs(np.diff(consensus_history[i:i+5]))
                if len(changes) > 0 and np.all(changes < 0.01):
                    convergence_time = i
                    break
        
        return {
            'converged': converged,
            'convergence_time': convergence_time,
            'final_consensus': consensus_history[-1],
            'consensus_stability': np.std(consensus_history[-5:])
        }
    
    def compute_analogy_metrics(self, flows: List[SemanticFlow]) -> Dict[str, float]:
        """Compute bounded confidence specific metrics"""
        if not flows:
            return {}
        
        metrics = {}
        
        # Consensus metrics
        if self.consensus_evolution:
            consensus_values = [m['consensus_strength'] for m in self.consensus_evolution.values()]
            metrics['mean_consensus'] = np.mean(consensus_values)
            metrics['consensus_variance'] = np.var(consensus_values)
            metrics['max_consensus'] = max(consensus_values)
        
        # Fragmentation metrics
        if self.consensus_evolution:
            fragmentation_values = [m['fragmentation_level'] for m in self.consensus_evolution.values()]
            metrics['mean_fragmentation'] = np.mean(fragmentation_values)
            metrics['fragmentation_stability'] = 1.0 - np.var(fragmentation_values)
        
        # Network metrics
        if self.confidence_networks:
            densities = []
            avg_clustering = []
            
            for network in self.confidence_networks.values():
                if network.number_of_nodes() > 0:
                    densities.append(nx.density(network))
                    if network.number_of_nodes() > 2:
                        avg_clustering.append(nx.average_clustering(network))
            
            if densities:
                metrics['network_density'] = np.mean(densities)
            if avg_clustering:
                metrics['average_clustering'] = np.mean(avg_clustering)
        
        # Polarization metrics
        polarization_analysis = self._analyze_polarization()
        metrics.update(polarization_analysis)
        
        return metrics
    
    def interpret_results(self, model_results: Dict[str, Any]) -> Dict[str, str]:
        """Interpret bounded confidence model results"""
        interpretation = model_results.get('interpretation', {})
        
        # Add bounded confidence specific interpretations
        parameters = model_results.get('parameters', {})
        regime = model_results.get('regime', 'unknown')
        
        semantic_interpretation = {
            'bounded_confidence_analogy': 'Semantic consensus emerges through bounded confidence interactions',
            'confidence_threshold': 'Words only influence each other within semantic similarity bounds',
            'opinion_dynamics': 'Word meanings evolve through local semantic interactions',
            'consensus_formation': 'Global semantic agreement emerges from local similarity-based interactions',
            'fragmentation_dynamics': 'Semantic communities form when confidence bounds are restrictive',
            'convergence_behavior': 'System converges to consensus or fragments into opinion clusters'
        }
        
        semantic_interpretation.update(interpretation)
        return semantic_interpretation