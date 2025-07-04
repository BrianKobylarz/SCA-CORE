"""Flow-specific metrics for semantic analysis."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.spatial.distance import cosine
from scipy.stats import entropy

from .base_metric import BaseMetric
from ..core.types import Word, Timestamp, Embedding, FlowVector

class SemanticVelocity(BaseMetric):
    """Computes instantaneous semantic velocity"""
    
    def compute(self, word: Word, t1: Timestamp, t2: Timestamp,
                emb1: Embedding, emb2: Embedding, 
                time_delta: float = 1.0) -> float:
        """Velocity = distance / time"""
        distance = np.linalg.norm(emb2 - emb1)
        return distance / time_delta
    
class SemanticAcceleration(BaseMetric):
    """Computes semantic acceleration (change in velocity)"""
    
    def compute(self, velocities: List[float], 
                time_delta: float = 1.0) -> List[float]:
        """Acceleration = change in velocity / time"""
        if len(velocities) < 2:
            return []
        
        accelerations = []
        for i in range(1, len(velocities)):
            acc = (velocities[i] - velocities[i-1]) / time_delta
            accelerations.append(acc)
        
        return accelerations

class FlowCoherence(BaseMetric):
    """Measures how directed vs dispersed a semantic flow is"""
    
    def compute(self, flow_vectors: List[FlowVector]) -> float:
        """
        High coherence = flow in consistent direction
        Low coherence = dispersed/chaotic flow
        """
        if len(flow_vectors) < 2:
            return 1.0
        
        # Extract directional components
        directions = np.array([(v[0], v[1]) for v in flow_vectors])
        magnitudes = np.array([v[2] for v in flow_vectors])
        
        # Normalize directions
        norms = np.linalg.norm(directions, axis=1)
        unit_directions = directions / norms[:, np.newaxis]
        
        # Weighted average direction
        avg_direction = np.average(unit_directions, 
                                  weights=magnitudes, axis=0)
        
        # Coherence = how aligned individual flows are with average
        coherences = [np.dot(d, avg_direction) for d in unit_directions]
        
        return np.average(coherences, weights=magnitudes)

class SemanticStochasticity(BaseMetric):
    """Measures volatility in semantic stability"""
    
    def compute(self, word: Word, stability_history: List[float],
                window_size: int = 5, scaling_factor: float = 4.0) -> float:
        """
        High stochasticity = erratic changes
        Low stochasticity = smooth evolution
        """
        if len(stability_history) < window_size:
            return 0.0
        
        stochasticities = []
        
        for i in range(len(stability_history) - window_size + 1):
            window = stability_history[i:i + window_size]
            variance = np.var(window)
            stochasticity = min(1.0, scaling_factor * variance)
            stochasticities.append(stochasticity)
        
        return np.mean(stochasticities)

class PathDependency(BaseMetric):
    """Measures whether drift has momentum or is random"""
    
    def compute(self, drift_sequence: List[float], 
                n_permutations: int = 100) -> float:
        """
        Compare observed drift accumulation to random permutations
        """
        if len(drift_sequence) < 3:
            return 0.0
        
        # Observed cumulative product (momentum effect)
        observed = np.sum(np.cumprod(drift_sequence))
        
        # Null distribution from permutations
        null_values = []
        for _ in range(n_permutations):
            permuted = np.random.permutation(drift_sequence)
            null_val = np.sum(np.cumprod(permuted))
            null_values.append(null_val)
        
        # Normalized path dependency
        mean_null = np.mean(null_values)
        std_null = np.std(null_values)
        
        if std_null > 0:
            z_score = (observed - mean_null) / std_null
            # Normalize to [0, 1]
            return 1 / (1 + np.exp(-z_score/2))
        
        return 0.5

class FlowDivergence(BaseMetric):
    """Measures tendency for meanings to spread out"""
    
    def compute(self, word: Word, neighbor_flows: Dict[Word, FlowVector]) -> float:
        """
        High divergence = word pushing neighbors apart
        Low divergence = word pulling neighbors together
        """
        if len(neighbor_flows) < 2:
            return 0.0
        
        # Compute pairwise angles between flow vectors
        flow_vectors = list(neighbor_flows.values())
        angles = []
        
        for i in range(len(flow_vectors)):
            for j in range(i + 1, len(flow_vectors)):
                v1 = np.array([flow_vectors[i][0], flow_vectors[i][1]])
                v2 = np.array([flow_vectors[j][0], flow_vectors[j][1]])
                
                # Compute angle between vectors
                dot_product = np.dot(v1, v2)
                norms = np.linalg.norm(v1) * np.linalg.norm(v2)
                
                if norms > 0:
                    cosine_angle = dot_product / norms
                    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
                    angles.append(angle)
        
        if not angles:
            return 0.0
        
        # Average angle (higher = more divergent)
        avg_angle = np.mean(angles)
        return avg_angle / np.pi  # Normalize to [0, 1]

class FlowCurl(BaseMetric):
    """Measures rotational tendency in semantic flow"""
    
    def compute(self, central_word: Word, 
                neighbor_flows: Dict[Word, FlowVector],
                neighbor_positions: Dict[Word, np.ndarray]) -> float:
        """
        High curl = circular/rotational flow patterns
        Low curl = radial flow patterns
        """
        if len(neighbor_flows) < 3:
            return 0.0
        
        curl_contributions = []
        
        # For each neighbor, compute circulation
        for word, flow_vec in neighbor_flows.items():
            if word not in neighbor_positions:
                continue
            
            position = neighbor_positions[word]
            flow = np.array([flow_vec[0], flow_vec[1]])
            
            # Tangent vector (perpendicular to radius)
            radius = position / np.linalg.norm(position)
            tangent = np.array([-radius[1], radius[0]])
            
            # Project flow onto tangent
            circulation = np.dot(flow, tangent)
            curl_contributions.append(circulation)
        
        if not curl_contributions:
            return 0.0
        
        # RMS curl
        return np.sqrt(np.mean(np.square(curl_contributions)))

class FlowIntensity(BaseMetric):
    """Measures overall strength of semantic flow"""
    
    def compute(self, flows: Dict[Word, List[FlowVector]], 
                aggregation: str = 'rms') -> float:
        """
        Aggregate flow magnitudes across all words
        """
        all_magnitudes = []
        
        for word_flows in flows.values():
            for flow_vec in word_flows:
                all_magnitudes.append(flow_vec[2])  # magnitude component
        
        if not all_magnitudes:
            return 0.0
        
        if aggregation == 'rms':
            return np.sqrt(np.mean(np.square(all_magnitudes)))
        elif aggregation == 'mean':
            return np.mean(all_magnitudes)
        elif aggregation == 'max':
            return np.max(all_magnitudes)
        elif aggregation == 'std':
            return np.std(all_magnitudes)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

class SemanticTurbulence(BaseMetric):
    """Measures chaotic vs smooth flow patterns"""
    
    def compute(self, flows: Dict[Word, List[FlowVector]], 
                window_size: int = 3) -> float:
        """
        High turbulence = rapidly changing flow directions
        Low turbulence = smooth, consistent flows
        """
        turbulence_values = []
        
        for word, word_flows in flows.items():
            if len(word_flows) < window_size:
                continue
            
            # Compute direction changes over sliding window
            for i in range(len(word_flows) - window_size + 1):
                window_flows = word_flows[i:i + window_size]
                
                # Extract directions
                directions = []
                for flow_vec in window_flows:
                    direction = np.array([flow_vec[0], flow_vec[1]])
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        directions.append(direction / norm)
                
                if len(directions) < 2:
                    continue
                
                # Compute direction variance
                direction_matrix = np.array(directions)
                centroid = np.mean(direction_matrix, axis=0)
                
                if np.linalg.norm(centroid) > 0:
                    centroid = centroid / np.linalg.norm(centroid)
                
                # Angular deviations from centroid
                deviations = []
                for direction in directions:
                    cos_angle = np.dot(direction, centroid)
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                    deviations.append(angle)
                
                if deviations:
                    turbulence_values.append(np.var(deviations))
        
        return np.mean(turbulence_values) if turbulence_values else 0.0

class FlowPersistence(BaseMetric):
    """Measures how long flow patterns maintain direction"""
    
    def compute(self, word: Word, flow_history: List[FlowVector],
                coherence_threshold: float = 0.7) -> float:
        """
        High persistence = flow maintains direction over time
        Low persistence = flow direction changes frequently
        """
        if len(flow_history) < 2:
            return 0.0
        
        # Find runs of coherent flow
        coherent_runs = []
        current_run = 1
        
        for i in range(1, len(flow_history)):
            prev_flow = np.array([flow_history[i-1][0], flow_history[i-1][1]])
            curr_flow = np.array([flow_history[i][0], flow_history[i][1]])
            
            # Normalize vectors
            prev_norm = np.linalg.norm(prev_flow)
            curr_norm = np.linalg.norm(curr_flow)
            
            if prev_norm > 0 and curr_norm > 0:
                prev_flow = prev_flow / prev_norm
                curr_flow = curr_flow / curr_norm
                
                # Compute coherence
                coherence = np.dot(prev_flow, curr_flow)
                
                if coherence >= coherence_threshold:
                    current_run += 1
                else:
                    coherent_runs.append(current_run)
                    current_run = 1
        
        coherent_runs.append(current_run)
        
        # Return average run length as persistence measure
        return np.mean(coherent_runs) / len(flow_history)

class FlowAnisotropy(BaseMetric):
    """Measures directional bias in flow patterns"""
    
    def compute(self, flows: Dict[Word, FlowVector]) -> Tuple[float, np.ndarray]:
        """
        High anisotropy = strong directional preference
        Low anisotropy = isotropic (equal in all directions)
        
        Returns:
            Tuple of (anisotropy_strength, preferred_direction)
        """
        if not flows:
            return 0.0, np.array([0.0, 0.0])
        
        # Collect all flow directions
        directions = []
        magnitudes = []
        
        for flow_vec in flows.values():
            direction = np.array([flow_vec[0], flow_vec[1]])
            magnitude = flow_vec[2]
            
            norm = np.linalg.norm(direction)
            if norm > 0:
                directions.append(direction / norm)
                magnitudes.append(magnitude)
        
        if not directions:
            return 0.0, np.array([0.0, 0.0])
        
        directions = np.array(directions)
        magnitudes = np.array(magnitudes)
        
        # Compute weighted mean direction
        mean_direction = np.average(directions, weights=magnitudes, axis=0)
        mean_direction_norm = np.linalg.norm(mean_direction)
        
        if mean_direction_norm > 0:
            mean_direction = mean_direction / mean_direction_norm
        
        # Compute anisotropy as concentration around mean direction
        alignments = [np.dot(d, mean_direction) for d in directions]
        weighted_alignment = np.average(alignments, weights=magnitudes)
        
        # Anisotropy strength
        anisotropy = max(0.0, weighted_alignment)
        
        return anisotropy, mean_direction