"""Stability and drift metrics for semantic analysis."""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from scipy.stats import pearsonr
from collections import defaultdict

from .base_metric import BaseMetric
from ..core.types import Word, Timestamp, Embedding

class NeighborStability(BaseMetric):
    """Measures stability of semantic neighborhoods"""
    
    def compute(self, word: Word, 
                neighbors_t1: List[Tuple[Word, float]], 
                neighbors_t2: List[Tuple[Word, float]],
                top_k: int = 10) -> float:
        """
        Compute Jaccard similarity of top-k neighbors
        High stability = consistent neighborhood
        """
        # Extract top-k neighbors
        top_neighbors_t1 = set(w for w, _ in neighbors_t1[:top_k])
        top_neighbors_t2 = set(w for w, _ in neighbors_t2[:top_k])
        
        # Jaccard similarity
        intersection = len(top_neighbors_t1 & top_neighbors_t2)
        union = len(top_neighbors_t1 | top_neighbors_t2)
        
        return intersection / union if union > 0 else 0.0

class TemporalConsistency(BaseMetric):
    """Measures consistency of semantic properties over time"""
    
    def compute(self, word: Word, 
                property_timeseries: List[float],
                window_size: int = 5) -> float:
        """
        Compute temporal consistency using sliding window correlation
        High consistency = stable properties over time
        """
        if len(property_timeseries) < window_size * 2:
            return 0.0
        
        correlations = []
        
        # Sliding window correlation
        for i in range(len(property_timeseries) - window_size + 1):
            window1 = property_timeseries[i:i + window_size]
            
            # Look ahead for next window
            for j in range(i + 1, min(i + window_size + 1, 
                                    len(property_timeseries) - window_size + 1)):
                window2 = property_timeseries[j:j + window_size]
                
                # Compute correlation
                try:
                    corr, _ = pearsonr(window1, window2)
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                except:
                    continue
        
        return np.mean(correlations) if correlations else 0.0

class SemanticDrift(BaseMetric):
    """Measures long-term directional change in meaning"""
    
    def compute(self, word: Word, 
                embedding_timeline: List[Tuple[Timestamp, Embedding]],
                drift_type: str = 'linear') -> Dict[str, float]:
        """
        Compute semantic drift metrics
        """
        if len(embedding_timeline) < 3:
            return {'drift_magnitude': 0.0, 'drift_direction': 0.0, 'drift_consistency': 0.0}
        
        timestamps, embeddings = zip(*embedding_timeline)
        embeddings = np.array(embeddings)
        
        if drift_type == 'linear':
            return self._compute_linear_drift(embeddings)
        elif drift_type == 'directional':
            return self._compute_directional_drift(embeddings)
        else:
            raise ValueError(f"Unknown drift type: {drift_type}")
    
    def _compute_linear_drift(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Compute linear drift using regression"""
        n_timepoints, n_dims = embeddings.shape
        time_indices = np.arange(n_timepoints)
        
        # Fit linear regression for each dimension
        slopes = []
        r_squares = []
        
        for dim in range(n_dims):
            dim_values = embeddings[:, dim]
            
            # Linear regression
            coeffs = np.polyfit(time_indices, dim_values, 1)
            slope = coeffs[0]
            slopes.append(slope)
            
            # R-squared
            predicted = np.polyval(coeffs, time_indices)
            ss_res = np.sum((dim_values - predicted) ** 2)
            ss_tot = np.sum((dim_values - np.mean(dim_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            r_squares.append(r_squared)
        
        # Overall drift metrics
        drift_magnitude = np.linalg.norm(slopes)
        drift_direction = np.mean(r_squares)  # How linear the drift is
        
        # Consistency: how well does linear model fit
        drift_consistency = np.mean(r_squares)
        
        return {
            'drift_magnitude': drift_magnitude,
            'drift_direction': drift_direction, 
            'drift_consistency': drift_consistency
        }
    
    def _compute_directional_drift(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Compute directional drift using vector analysis"""
        if len(embeddings) < 2:
            return {'drift_magnitude': 0.0, 'drift_direction': 0.0, 'drift_consistency': 0.0}
        
        # Compute step vectors
        step_vectors = []
        for i in range(1, len(embeddings)):
            step = embeddings[i] - embeddings[i-1]
            step_vectors.append(step)
        
        step_vectors = np.array(step_vectors)
        
        # Overall displacement
        total_displacement = embeddings[-1] - embeddings[0]
        displacement_magnitude = np.linalg.norm(total_displacement)
        
        # Path length
        path_length = np.sum([np.linalg.norm(step) for step in step_vectors])
        
        # Drift consistency = displacement / path_length
        drift_consistency = displacement_magnitude / path_length if path_length > 0 else 0.0
        
        # Direction consistency
        if displacement_magnitude > 0:
            displacement_direction = total_displacement / displacement_magnitude
            
            # Measure how aligned each step is with overall direction
            alignments = []
            for step in step_vectors:
                step_magnitude = np.linalg.norm(step)
                if step_magnitude > 0:
                    step_direction = step / step_magnitude
                    alignment = np.dot(step_direction, displacement_direction)
                    alignments.append(alignment)
            
            drift_direction = np.mean(alignments) if alignments else 0.0
        else:
            drift_direction = 0.0
        
        return {
            'drift_magnitude': displacement_magnitude,
            'drift_direction': drift_direction,
            'drift_consistency': drift_consistency
        }

class LocalStability(BaseMetric):
    """Measures stability within local semantic regions"""
    
    def compute(self, embeddings_t1: Dict[Word, Embedding],
                embeddings_t2: Dict[Word, Embedding],
                region_words: Set[Word]) -> float:
        """
        Measure how much a semantic region has changed
        """
        if not region_words:
            return 0.0
        
        # Get embeddings for words in region
        region_embs_t1 = {}
        region_embs_t2 = {}
        
        for word in region_words:
            if word in embeddings_t1 and word in embeddings_t2:
                region_embs_t1[word] = embeddings_t1[word]
                region_embs_t2[word] = embeddings_t2[word]
        
        if len(region_embs_t1) < 2:
            return 0.0
        
        # Compute pairwise distances within region
        words = list(region_embs_t1.keys())
        
        distances_t1 = []
        distances_t2 = []
        
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                w1, w2 = words[i], words[j]
                
                dist_t1 = np.linalg.norm(region_embs_t1[w1] - region_embs_t1[w2])
                dist_t2 = np.linalg.norm(region_embs_t2[w1] - region_embs_t2[w2])
                
                distances_t1.append(dist_t1)
                distances_t2.append(dist_t2)
        
        # Compute correlation between distance matrices
        if len(distances_t1) > 1:
            try:
                correlation, _ = pearsonr(distances_t1, distances_t2)
                return max(0.0, correlation) if not np.isnan(correlation) else 0.0
            except:
                return 0.0
        
        return 0.0

class StructuralStability(BaseMetric):
    """Measures stability of overall semantic structure"""
    
    def compute(self, embeddings_t1: Dict[Word, Embedding],
                embeddings_t2: Dict[Word, Embedding],
                method: str = 'procrustes') -> float:
        """
        Measure structural alignment between two embedding spaces
        """
        # Get common vocabulary
        common_words = set(embeddings_t1.keys()) & set(embeddings_t2.keys())
        
        if len(common_words) < 3:
            return 0.0
        
        # Create aligned embedding matrices
        words = sorted(common_words)
        emb_matrix_t1 = np.array([embeddings_t1[w] for w in words])
        emb_matrix_t2 = np.array([embeddings_t2[w] for w in words])
        
        if method == 'procrustes':
            return self._procrustes_similarity(emb_matrix_t1, emb_matrix_t2)
        elif method == 'cka':
            return self._cka_similarity(emb_matrix_t1, emb_matrix_t2)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _procrustes_similarity(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute Procrustes similarity between embedding matrices"""
        try:
            from scipy.spatial.distance import procrustes
            
            # Procrustes analysis
            _, _, disparity = procrustes(X, Y)
            
            # Convert disparity to similarity (lower disparity = higher similarity)
            similarity = 1.0 / (1.0 + disparity)
            return similarity
            
        except:
            # Fallback: simple correlation
            X_flat = X.flatten()
            Y_flat = Y.flatten()
            
            try:
                correlation, _ = pearsonr(X_flat, Y_flat)
                return max(0.0, correlation) if not np.isnan(correlation) else 0.0
            except:
                return 0.0
    
    def _cka_similarity(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute Centered Kernel Alignment similarity"""
        try:
            # Linear CKA
            def linear_cka(X, Y):
                # Center the matrices
                n = X.shape[0]
                H = np.eye(n) - np.ones((n, n)) / n
                
                X_centered = H @ X
                Y_centered = H @ Y
                
                # Compute similarity
                numerator = np.trace(X_centered.T @ Y_centered @ Y_centered.T @ X_centered)
                denom_x = np.trace(X_centered.T @ X_centered @ X_centered.T @ X_centered)
                denom_y = np.trace(Y_centered.T @ Y_centered @ Y_centered.T @ Y_centered)
                
                if denom_x > 0 and denom_y > 0:
                    return numerator / np.sqrt(denom_x * denom_y)
                return 0.0
            
            return linear_cka(X, Y)
            
        except:
            return 0.0

class AdaptiveStability(BaseMetric):
    """Measures stability with adaptive thresholds"""
    
    def compute(self, word: Word,
                stability_history: List[float],
                adaptation_rate: float = 0.1) -> Dict[str, float]:
        """
        Compute stability with adaptive baseline
        """
        if len(stability_history) < 5:
            return {'current_stability': 0.0, 'stability_trend': 0.0, 'anomaly_score': 0.0}
        
        # Adaptive baseline using exponential moving average
        baseline = stability_history[0]
        baselines = [baseline]
        
        for stability in stability_history[1:]:
            baseline = (1 - adaptation_rate) * baseline + adaptation_rate * stability
            baselines.append(baseline)
        
        # Current stability relative to adaptive baseline
        current_stability = stability_history[-1] / baselines[-1] if baselines[-1] > 0 else 0.0
        
        # Stability trend
        if len(baselines) >= 3:
            recent_trend = np.polyfit(range(len(baselines[-5:])), baselines[-5:], 1)[0]
            stability_trend = recent_trend
        else:
            stability_trend = 0.0
        
        # Anomaly score (how much current stability deviates from expected)
        if len(stability_history) >= 10:
            recent_mean = np.mean(stability_history[-10:])
            recent_std = np.std(stability_history[-10:])
            
            if recent_std > 0:
                anomaly_score = abs(stability_history[-1] - recent_mean) / recent_std
            else:
                anomaly_score = 0.0
        else:
            anomaly_score = 0.0
        
        return {
            'current_stability': current_stability,
            'stability_trend': stability_trend,
            'anomaly_score': anomaly_score
        }