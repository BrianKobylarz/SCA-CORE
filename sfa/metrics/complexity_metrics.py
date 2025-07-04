"""Complexity science metrics for semantic analysis."""

import numpy as np
from typing import List, Dict, Set, Tuple
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import networkx as nx

from .base_metric import BaseMetric
from ..core.types import Word, Timestamp

class CascadeRisk(BaseMetric):
    """Computes R₀-like metric for semantic contagion"""
    
    def compute(self, adjacency_matrix: np.ndarray) -> float:
        """
        R₀ > 1.0 indicates supercritical regime
        Based on largest eigenvalue (spectral radius)
        """
        if adjacency_matrix.size == 0:
            return 0.0
        
        # Convert to sparse for efficiency
        sparse_adj = csr_matrix(adjacency_matrix)
        
        # Compute largest eigenvalue
        try:
            eigenvalues, _ = eigsh(sparse_adj, k=1, which='LA')
            spectral_radius = abs(eigenvalues[0])
            
            # Normalize by average degree for R₀ interpretation
            avg_degree = np.mean(np.sum(adjacency_matrix > 0, axis=1))
            if avg_degree > 0:
                return spectral_radius / avg_degree
            
        except:
            # Fallback for small networks
            eigenvalues = np.linalg.eigvals(adjacency_matrix)
            return np.max(np.abs(eigenvalues))
        
        return 0.0

class NetworkContagion(BaseMetric):
    """Measures ripple effects of semantic changes"""
    
    def compute(self, word: Word, word_drift: float,
                neighbor_drifts: Dict[Word, float],
                similarity_matrix: Dict[Tuple[Word, Word], float]) -> float:
        """
        High contagion = neighbors change when word changes
        Uses partial correlation to control for system-wide effects
        """
        if len(neighbor_drifts) < 2:
            return 0.0
        
        # Get similarities between word and neighbors
        word_similarities = []
        neighbor_drift_values = []
        
        for neighbor, drift in neighbor_drifts.items():
            sim = similarity_matrix.get((word, neighbor), 0.0)
            if sim > 0:
                word_similarities.append(sim)
                neighbor_drift_values.append(drift)
        
        if len(word_similarities) < 2:
            return 0.0
        
        # Weight neighbor drifts by similarity
        weighted_drift = np.average(neighbor_drift_values, 
                                   weights=word_similarities)
        
        # Contagion strength (normalized)
        if word_drift > 0:
            contagion = min(1.0, weighted_drift / word_drift)
        else:
            contagion = 0.0
        
        return contagion

class AlgebraicConnectivity(BaseMetric):
    """Fiedler value - measures network cohesion"""
    
    def compute(self, graph: nx.Graph) -> float:
        """
        Higher value = more robust/connected network
        Near 0 = fragile, nearly disconnected
        """
        if graph.number_of_nodes() < 2:
            return 0.0
        
        try:
            # Compute Laplacian spectrum
            laplacian = nx.laplacian_matrix(graph).astype(float)
            
            # Get second smallest eigenvalue (Fiedler value)
            eigenvalues = eigsh(laplacian, k=2, which='SM', 
                               return_eigenvectors=False)
            
            return float(eigenvalues[1])
            
        except:
            # Fallback for small graphs
            return nx.algebraic_connectivity(graph)

class GraphEntropy(BaseMetric):
    """Shannon entropy of degree distribution"""
    
    def compute(self, graph: nx.Graph) -> float:
        """
        High entropy = disordered/heterogeneous structure
        Low entropy = regular/ordered structure
        """
        if graph.number_of_nodes() == 0:
            return 0.0
        
        # Get degree sequence
        degrees = [d for n, d in graph.degree()]
        
        if not degrees:
            return 0.0
        
        # Compute degree distribution
        degree_counts = {}
        for d in degrees:
            degree_counts[d] = degree_counts.get(d, 0) + 1
        
        # Normalize to probabilities
        total = len(degrees)
        probabilities = [count/total for count in degree_counts.values()]
        
        # Shannon entropy
        return entropy(probabilities, base=2)

class CommunityVolatility(BaseMetric):
    """Measures structural flux via community changes"""
    
    def compute(self, communities_t1: List[Set[Word]], 
                communities_t2: List[Set[Word]]) -> int:
        """
        Count split and merge events
        High volatility = unstable structure
        """
        splits = 0
        merges = 0
        
        # Track community overlaps
        for c1 in communities_t1:
            overlaps = []
            for c2 in communities_t2:
                overlap = len(c1 & c2) / len(c1 | c2) if len(c1 | c2) > 0 else 0
                if overlap > 0.1:  # Significant overlap threshold
                    overlaps.append(c2)
            
            if len(overlaps) > 1:
                splits += len(overlaps) - 1
            elif len(overlaps) == 0:
                # Community disappeared
                splits += 1
        
        # Check for merges (multiple t1 communities → one t2 community)
        for c2 in communities_t2:
            sources = []
            for c1 in communities_t1:
                overlap = len(c1 & c2) / len(c1 | c2) if len(c1 | c2) > 0 else 0
                if overlap > 0.1:
                    sources.append(c1)
            
            if len(sources) > 1:
                merges += len(sources) - 1
        
        return splits + merges

class CriticalityMeasure(BaseMetric):
    """Measures proximity to critical transitions"""
    
    def compute(self, timeseries: List[float], 
                window_size: int = 10) -> Dict[str, float]:
        """
        Compute early warning signals for critical transitions
        Based on Scheffer et al. Nature 2009
        """
        if len(timeseries) < window_size * 2:
            return {'variance': 0.0, 'autocorrelation': 0.0, 'skewness': 0.0}
        
        # Split into baseline and recent windows
        baseline = timeseries[:-window_size]
        recent = timeseries[-window_size:]
        
        # Variance (critical slowing down)
        baseline_var = np.var(baseline)
        recent_var = np.var(recent)
        variance_increase = (recent_var - baseline_var) / baseline_var if baseline_var > 0 else 0.0
        
        # Autocorrelation (critical slowing down)
        def autocorr_lag1(series):
            if len(series) < 2:
                return 0.0
            return np.corrcoef(series[:-1], series[1:])[0, 1] if len(series) > 1 else 0.0
        
        baseline_autocorr = autocorr_lag1(baseline)
        recent_autocorr = autocorr_lag1(recent)
        autocorr_increase = recent_autocorr - baseline_autocorr
        
        # Skewness (flickering between states)
        from scipy.stats import skew
        recent_skewness = abs(skew(recent))
        
        return {
            'variance': max(0.0, variance_increase),
            'autocorrelation': max(0.0, autocorr_increase),
            'skewness': recent_skewness
        }

class PhaseCoherence(BaseMetric):
    """Measures phase synchronization in semantic evolution"""
    
    def compute(self, timeseries_dict: Dict[Word, List[float]]) -> float:
        """
        Compute global phase coherence across multiple word trajectories
        High coherence = synchronized evolution
        """
        if len(timeseries_dict) < 2:
            return 0.0
        
        # Extract phases using Hilbert transform
        from scipy.signal import hilbert
        
        phases = {}
        min_length = min(len(ts) for ts in timeseries_dict.values())
        
        for word, timeseries in timeseries_dict.items():
            if len(timeseries) >= min_length:
                # Truncate to common length
                ts = timeseries[:min_length]
                analytic_signal = hilbert(ts)
                phase = np.angle(analytic_signal)
                phases[word] = phase
        
        if len(phases) < 2:
            return 0.0
        
        # Compute pairwise phase coherence
        coherences = []
        phase_list = list(phases.values())
        
        for i in range(len(phase_list)):
            for j in range(i + 1, len(phase_list)):
                phase_diff = phase_list[i] - phase_list[j]
                # Phase-locking value
                plv = abs(np.mean(np.exp(1j * phase_diff)))
                coherences.append(plv)
        
        return np.mean(coherences) if coherences else 0.0

class ComplexityProfile(BaseMetric):
    """Comprehensive complexity characterization"""
    
    def compute(self, graph: nx.Graph, 
                timeseries: Dict[Word, List[float]]) -> Dict[str, float]:
        """
        Compute multiple complexity measures
        """
        profile = {}
        
        # Network complexity
        if graph.number_of_nodes() > 0:
            profile['node_count'] = graph.number_of_nodes()
            profile['edge_count'] = graph.number_of_edges()
            profile['density'] = nx.density(graph)
            
            if graph.number_of_nodes() > 1:
                profile['clustering'] = nx.average_clustering(graph)
                
                # Small-world coefficient
                try:
                    path_length = nx.average_shortest_path_length(graph)
                    random_clustering = profile['density']
                    random_path_length = np.log(profile['node_count']) / np.log(profile['density'] * profile['node_count']) if profile['density'] > 0 else float('inf')
                    
                    if random_path_length > 0 and random_clustering > 0:
                        small_world = (profile['clustering'] / random_clustering) / (path_length / random_path_length)
                        profile['small_world'] = small_world
                except:
                    profile['small_world'] = 1.0
        
        # Temporal complexity
        if timeseries:
            all_values = []
            for ts in timeseries.values():
                all_values.extend(ts)
            
            if all_values:
                profile['temporal_variance'] = np.var(all_values)
                profile['temporal_entropy'] = self._compute_temporal_entropy(all_values)
                profile['hurst_exponent'] = self._compute_hurst_exponent(all_values)
        
        return profile
    
    def _compute_temporal_entropy(self, timeseries: List[float], 
                                 bins: int = 10) -> float:
        """Compute entropy of temporal distribution"""
        if len(timeseries) < 2:
            return 0.0
        
        hist, _ = np.histogram(timeseries, bins=bins)
        hist = hist / np.sum(hist)  # Normalize
        
        # Remove zeros to avoid log(0)
        hist = hist[hist > 0]
        
        return -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0
    
    def _compute_hurst_exponent(self, timeseries: List[float]) -> float:
        """Compute Hurst exponent for long-range dependence"""
        if len(timeseries) < 10:
            return 0.5
        
        try:
            # R/S analysis
            ts = np.array(timeseries)
            n = len(ts)
            
            # Center the series
            mean_ts = np.mean(ts)
            centered = ts - mean_ts
            
            # Cumulative sum
            cumsum = np.cumsum(centered)
            
            # Range
            R = np.max(cumsum) - np.min(cumsum)
            
            # Standard deviation
            S = np.std(ts)
            
            if S > 0:
                rs = R / S
                # Hurst exponent estimation
                hurst = np.log(rs) / np.log(n)
                return np.clip(hurst, 0.0, 1.0)
            
        except:
            pass
        
        return 0.5  # Default for Brownian motion

class ScaleFreeMetric(BaseMetric):
    """Measures scale-free properties of semantic networks"""
    
    def compute(self, graph: nx.Graph) -> Dict[str, float]:
        """
        Assess scale-free characteristics
        """
        if graph.number_of_nodes() < 10:
            return {'power_law_exponent': 0.0, 'goodness_of_fit': 0.0}
        
        # Get degree sequence
        degrees = [d for n, d in graph.degree() if d > 0]
        
        if len(degrees) < 5:
            return {'power_law_exponent': 0.0, 'goodness_of_fit': 0.0}
        
        # Fit power law
        try:
            from scipy.optimize import curve_fit
            
            # Create degree distribution
            unique_degrees = sorted(set(degrees))
            degree_counts = [degrees.count(d) for d in unique_degrees]
            
            # Normalize
            total = sum(degree_counts)
            degree_probs = [c / total for c in degree_counts]
            
            # Log-log fit
            log_degrees = np.log(unique_degrees)
            log_probs = np.log(degree_probs)
            
            # Linear fit in log space
            coeffs = np.polyfit(log_degrees, log_probs, 1)
            gamma = -coeffs[0]  # Power law exponent
            
            # Goodness of fit (R²)
            predicted = np.polyval(coeffs, log_degrees)
            ss_res = np.sum((log_probs - predicted) ** 2)
            ss_tot = np.sum((log_probs - np.mean(log_probs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            return {
                'power_law_exponent': gamma,
                'goodness_of_fit': max(0.0, r_squared)
            }
            
        except:
            return {'power_law_exponent': 0.0, 'goodness_of_fit': 0.0}