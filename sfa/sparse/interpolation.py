"""Temporal interpolation for sparse semantic data."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import interpolate
from scipy.spatial.distance import cdist
import logging
from datetime import datetime, timedelta
from collections import defaultdict

from ..core.types import Word, Timestamp, Embedding
from ..core.base import BaseProcessor, BaseTransformer
from ..core.embeddings import TemporalEmbeddingStore

logger = logging.getLogger(__name__)

class TemporalInterpolation(BaseProcessor):
    """
    Interpolates embeddings across time for sparse data analysis.
    
    Supports multiple interpolation methods:
    - Linear interpolation
    - Spline interpolation  
    - Nearest neighbor
    - Weighted average
    - Neural interpolation (using simple feedforward)
    """
    
    def __init__(self, method: str = 'linear', confidence_threshold: float = 0.5):
        super().__init__("TemporalInterpolation")
        self.method = method
        self.confidence_threshold = confidence_threshold
        
        # Interpolation parameters
        self.spline_order = 3
        self.spline_smoothing = 0.0
        self.neighbor_weights = 'distance'  # 'uniform', 'distance', 'temporal'
        self.max_gap_days = 30
        
        # Quality metrics
        self.interpolation_quality = {}
        self.confidence_scores = {}
        
    def configure(self, **config) -> None:
        """Configure interpolation parameters."""
        super().configure(**config)
        
        if 'spline_order' in config:
            self.spline_order = config['spline_order']
        if 'spline_smoothing' in config:
            self.spline_smoothing = config['spline_smoothing']
        if 'neighbor_weights' in config:
            self.neighbor_weights = config['neighbor_weights']
        if 'max_gap_days' in config:
            self.max_gap_days = config['max_gap_days']
    
    def process(self, embedding_store: TemporalEmbeddingStore, target_timestamps: List[Timestamp]) -> TemporalEmbeddingStore:
        """
        Interpolate embeddings for target timestamps.
        
        Args:
            embedding_store: Source embedding store
            target_timestamps: Timestamps to interpolate for
            
        Returns:
            New embedding store with interpolated embeddings
        """
        logger.info(f"Interpolating embeddings for {len(target_timestamps)} timestamps using {self.method}")
        
        # Create new embedding store
        interpolated_store = TemporalEmbeddingStore(embedding_store.embedding_dim)
        
        # Copy existing embeddings
        for timestamp in embedding_store.get_timestamps():
            embeddings = embedding_store.get_embeddings(timestamp)
            for word, embedding in embeddings.items():
                interpolated_store.store_embedding(word, timestamp, embedding)
        
        # Get all words that need interpolation
        all_words = set()
        for timestamp in embedding_store.get_timestamps():
            all_words.update(embedding_store.get_vocabulary(timestamp))
        
        # Interpolate for each target timestamp
        for target_timestamp in target_timestamps:
            if target_timestamp in embedding_store.get_timestamps():
                continue  # Already have data for this timestamp
            
            logger.debug(f"Interpolating for timestamp {target_timestamp}")
            
            interpolated_embeddings = self._interpolate_timestamp(
                embedding_store, target_timestamp, all_words
            )
            
            # Store interpolated embeddings
            for word, (embedding, confidence) in interpolated_embeddings.items():
                if confidence >= self.confidence_threshold:
                    interpolated_store.store_embedding(word, target_timestamp, embedding)
                    
                    # Store confidence score
                    self.confidence_scores[f"{word}_{target_timestamp}"] = confidence
        
        logger.info(f"Interpolation completed. Added {len(target_timestamps)} timestamps")
        return interpolated_store
    
    def _interpolate_timestamp(self, embedding_store: TemporalEmbeddingStore, 
                              target_timestamp: Timestamp, 
                              words: set) -> Dict[Word, Tuple[Embedding, float]]:
        """Interpolate embeddings for all words at a specific timestamp."""
        interpolated = {}
        
        for word in words:
            result = self._interpolate_word(embedding_store, word, target_timestamp)
            if result is not None:
                interpolated[word] = result
        
        return interpolated
    
    def _interpolate_word(self, embedding_store: TemporalEmbeddingStore,
                         word: Word, target_timestamp: Timestamp) -> Optional[Tuple[Embedding, float]]:
        """Interpolate embedding for a specific word at target timestamp."""
        # Get word timeline
        timeline = embedding_store.get_word_timeline(word)
        
        if len(timeline) < 2:
            return None  # Need at least 2 points for interpolation
        
        # Convert timestamps to numeric values for interpolation
        timestamps = [self._timestamp_to_numeric(ts) for ts, _ in timeline]
        embeddings = [emb for _, emb in timeline]
        target_numeric = self._timestamp_to_numeric(target_timestamp)
        
        # Check if target is within reasonable range
        if not self._is_interpolation_valid(timestamps, target_numeric):
            return None
        
        try:
            if self.method == 'linear':
                interpolated_embedding, confidence = self._linear_interpolate(
                    timestamps, embeddings, target_numeric
                )
            elif self.method == 'spline':
                interpolated_embedding, confidence = self._spline_interpolate(
                    timestamps, embeddings, target_numeric
                )
            elif self.method == 'nearest':
                interpolated_embedding, confidence = self._nearest_interpolate(
                    timestamps, embeddings, target_numeric
                )
            elif self.method == 'weighted_average':
                interpolated_embedding, confidence = self._weighted_average_interpolate(
                    timestamps, embeddings, target_numeric
                )
            else:
                logger.warning(f"Unknown interpolation method: {self.method}")
                return None
            
            return interpolated_embedding, confidence
            
        except Exception as e:
            logger.warning(f"Interpolation failed for word {word}: {e}")
            return None
    
    def _linear_interpolate(self, timestamps: List[float], embeddings: List[Embedding], 
                           target: float) -> Tuple[Embedding, float]:
        """Linear interpolation between nearest neighbors."""
        # Find bounding points
        before_idx = None
        after_idx = None
        
        for i, ts in enumerate(timestamps):
            if ts <= target:
                before_idx = i
            if ts >= target and after_idx is None:
                after_idx = i
                break
        
        if before_idx is None:
            # Extrapolate from beginning
            return embeddings[0], 0.3
        elif after_idx is None:
            # Extrapolate from end
            return embeddings[-1], 0.3
        elif before_idx == after_idx:
            # Exact match
            return embeddings[before_idx], 1.0
        else:
            # Interpolate
            t1, t2 = timestamps[before_idx], timestamps[after_idx]
            emb1, emb2 = embeddings[before_idx], embeddings[after_idx]
            
            # Linear interpolation weight
            alpha = (target - t1) / (t2 - t1)
            interpolated = (1 - alpha) * emb1 + alpha * emb2
            
            # Confidence based on gap size
            gap_size = t2 - t1
            max_gap = self.max_gap_days
            confidence = max(0.1, 1.0 - (gap_size / max_gap))
            
            return interpolated, confidence
    
    def _spline_interpolate(self, timestamps: List[float], embeddings: List[Embedding],
                           target: float) -> Tuple[Embedding, float]:
        """Spline interpolation for smooth trajectories."""
        if len(timestamps) < 4:
            # Fall back to linear for insufficient points
            return self._linear_interpolate(timestamps, embeddings, target)
        
        # Interpolate each dimension separately
        embedding_dim = len(embeddings[0])
        interpolated = np.zeros(embedding_dim)
        
        for dim in range(embedding_dim):
            values = [emb[dim] for emb in embeddings]
            
            # Create spline
            if self.spline_order >= len(timestamps):
                order = len(timestamps) - 1
            else:
                order = self.spline_order
            
            spline = interpolate.UnivariateSpline(
                timestamps, values, 
                k=order, s=self.spline_smoothing
            )
            
            interpolated[dim] = spline(target)
        
        # Confidence based on data density
        density = len(timestamps) / (max(timestamps) - min(timestamps))
        confidence = min(1.0, density / 10.0)  # Normalize by expected density
        
        return interpolated, confidence
    
    def _nearest_interpolate(self, timestamps: List[float], embeddings: List[Embedding],
                            target: float) -> Tuple[Embedding, float]:
        """Nearest neighbor interpolation."""
        distances = [abs(ts - target) for ts in timestamps]
        nearest_idx = np.argmin(distances)
        
        # Confidence based on temporal distance
        distance = distances[nearest_idx]
        confidence = max(0.1, 1.0 - (distance / self.max_gap_days))
        
        return embeddings[nearest_idx], confidence
    
    def _weighted_average_interpolate(self, timestamps: List[float], embeddings: List[Embedding],
                                     target: float) -> Tuple[Embedding, float]:
        """Weighted average of nearby embeddings."""
        # Calculate weights based on temporal distance
        distances = [abs(ts - target) for ts in timestamps]
        
        if self.neighbor_weights == 'uniform':
            weights = np.ones(len(distances))
        elif self.neighbor_weights == 'distance':
            # Inverse distance weighting
            weights = 1.0 / (np.array(distances) + 1e-8)
        elif self.neighbor_weights == 'temporal':
            # Exponential decay with time
            weights = np.exp(-np.array(distances) / 7.0)  # 7-day half-life
        else:
            weights = 1.0 / (np.array(distances) + 1e-8)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Weighted average
        interpolated = np.zeros_like(embeddings[0])
        for emb, weight in zip(embeddings, weights):
            interpolated += weight * emb
        
        # Confidence based on weight distribution entropy
        weight_entropy = -np.sum(weights * np.log(weights + 1e-8))
        max_entropy = np.log(len(weights))
        confidence = 1.0 - (weight_entropy / max_entropy)
        
        return interpolated, confidence
    
    def _timestamp_to_numeric(self, timestamp: Timestamp) -> float:
        """Convert timestamp to numeric value for interpolation."""
        try:
            # Try parsing as datetime
            if isinstance(timestamp, str):
                # Handle different timestamp formats
                if '-' in timestamp:
                    if len(timestamp.split('-')) == 2:  # YYYY-MM format
                        dt = datetime.strptime(timestamp, '%Y-%m')
                    else:  # YYYY-MM-DD format
                        dt = datetime.strptime(timestamp, '%Y-%m-%d')
                else:
                    # Try other formats
                    dt = datetime.fromisoformat(timestamp)
                
                # Convert to days since epoch
                epoch = datetime(1970, 1, 1)
                return (dt - epoch).total_seconds() / (24 * 3600)
            elif isinstance(timestamp, datetime):
                epoch = datetime(1970, 1, 1)
                return (timestamp - epoch).total_seconds() / (24 * 3600)
            else:
                # Assume it's already numeric
                return float(timestamp)
        except Exception:
            # Fall back to hash-based numeric representation
            return hash(str(timestamp)) % 100000
    
    def _is_interpolation_valid(self, timestamps: List[float], target: float) -> bool:
        """Check if interpolation is valid for the target timestamp."""
        if not timestamps:
            return False
        
        min_ts, max_ts = min(timestamps), max(timestamps)
        
        # Allow extrapolation within reasonable bounds
        range_size = max_ts - min_ts
        extrapolation_limit = range_size * 0.2  # 20% of range
        
        return (target >= min_ts - extrapolation_limit and 
                target <= max_ts + extrapolation_limit)
    
    def get_interpolation_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics for interpolation."""
        if not self.confidence_scores:
            return {}
        
        confidences = list(self.confidence_scores.values())
        
        return {
            'mean_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'std_confidence': np.std(confidences),
            'interpolated_words': len(self.confidence_scores),
            'high_confidence_ratio': np.mean([c >= 0.7 for c in confidences]),
            'low_confidence_ratio': np.mean([c < 0.3 for c in confidences])
        }

class AdaptiveInterpolation(TemporalInterpolation):
    """
    Adaptive interpolation that selects the best method for each word.
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        super().__init__('adaptive', confidence_threshold)
        self.methods = ['linear', 'spline', 'weighted_average']
        self.method_scores = defaultdict(list)
    
    def _interpolate_word(self, embedding_store: TemporalEmbeddingStore,
                         word: Word, target_timestamp: Timestamp) -> Optional[Tuple[Embedding, float]]:
        """Select best interpolation method for each word."""
        timeline = embedding_store.get_word_timeline(word)
        
        if len(timeline) < 2:
            return None
        
        timestamps = [self._timestamp_to_numeric(ts) for ts, _ in timeline]
        embeddings = [emb for _, emb in timeline]
        target_numeric = self._timestamp_to_numeric(target_timestamp)
        
        if not self._is_interpolation_valid(timestamps, target_numeric):
            return None
        
        # Try different methods and select the best one
        best_result = None
        best_confidence = 0.0
        best_method = None
        
        for method in self.methods:
            try:
                original_method = self.method
                self.method = method
                
                if method == 'linear':
                    result = self._linear_interpolate(timestamps, embeddings, target_numeric)
                elif method == 'spline' and len(timestamps) >= 4:
                    result = self._spline_interpolate(timestamps, embeddings, target_numeric)
                elif method == 'weighted_average':
                    result = self._weighted_average_interpolate(timestamps, embeddings, target_numeric)
                else:
                    continue
                
                self.method = original_method
                
                if result and result[1] > best_confidence:
                    best_result = result
                    best_confidence = result[1]
                    best_method = method
                    
            except Exception as e:
                logger.debug(f"Method {method} failed for word {word}: {e}")
                continue
        
        # Record method performance
        if best_method:
            self.method_scores[best_method].append(best_confidence)
        
        return best_result

class SemanticConsistencyInterpolation(TemporalInterpolation):
    """
    Interpolation that considers semantic consistency with neighboring words.
    """
    
    def __init__(self, method: str = 'linear', confidence_threshold: float = 0.5,
                 semantic_weight: float = 0.3):
        super().__init__(method, confidence_threshold)
        self.semantic_weight = semantic_weight
        self.neighbor_cache = {}
    
    def _interpolate_word(self, embedding_store: TemporalEmbeddingStore,
                         word: Word, target_timestamp: Timestamp) -> Optional[Tuple[Embedding, float]]:
        """Interpolate considering semantic neighborhood consistency."""
        # Get base interpolation
        base_result = super()._interpolate_word(embedding_store, word, target_timestamp)
        
        if base_result is None:
            return None
        
        base_embedding, base_confidence = base_result
        
        # Find semantic neighbors at nearby timestamps
        neighbors = self._find_semantic_neighbors(embedding_store, word, target_timestamp)
        
        if not neighbors:
            return base_result
        
        # Adjust interpolation based on neighbor consistency
        adjusted_embedding = self._adjust_for_semantic_consistency(
            base_embedding, neighbors, embedding_store, target_timestamp
        )
        
        # Adjust confidence based on consistency
        consistency_score = self._compute_consistency_score(
            adjusted_embedding, neighbors, embedding_store, target_timestamp
        )
        
        adjusted_confidence = (
            (1 - self.semantic_weight) * base_confidence + 
            self.semantic_weight * consistency_score
        )
        
        return adjusted_embedding, adjusted_confidence
    
    def _find_semantic_neighbors(self, embedding_store: TemporalEmbeddingStore,
                                word: Word, target_timestamp: Timestamp,
                                k: int = 5) -> List[Word]:
        """Find semantic neighbors of the word around the target timestamp."""
        # Get nearby timestamps
        all_timestamps = embedding_store.get_timestamps()
        target_numeric = self._timestamp_to_numeric(target_timestamp)
        
        nearby_timestamps = []
        for ts in all_timestamps:
            ts_numeric = self._timestamp_to_numeric(ts)
            if abs(ts_numeric - target_numeric) <= 30:  # Within 30 days
                nearby_timestamps.append(ts)
        
        if not nearby_timestamps:
            return []
        
        # Find neighbors based on similarity
        neighbors = set()
        word_embedding = embedding_store.get_embedding(word, nearby_timestamps[0])
        
        if word_embedding is None:
            return []
        
        for ts in nearby_timestamps:
            similar_words = embedding_store.find_similar_words(word, ts, k=k)
            neighbors.update([w for w, _ in similar_words])
        
        return list(neighbors)[:k]
    
    def _adjust_for_semantic_consistency(self, base_embedding: Embedding,
                                        neighbors: List[Word],
                                        embedding_store: TemporalEmbeddingStore,
                                        target_timestamp: Timestamp) -> Embedding:
        """Adjust embedding to be consistent with semantic neighbors."""
        if not neighbors:
            return base_embedding
        
        # Interpolate neighbor embeddings
        neighbor_embeddings = []
        for neighbor in neighbors:
            neighbor_result = super()._interpolate_word(embedding_store, neighbor, target_timestamp)
            if neighbor_result is not None:
                neighbor_embeddings.append(neighbor_result[0])
        
        if not neighbor_embeddings:
            return base_embedding
        
        # Compute neighbor centroid
        neighbor_centroid = np.mean(neighbor_embeddings, axis=0)
        
        # Adjust base embedding towards neighbor consistency
        adjustment_strength = self.semantic_weight
        adjusted_embedding = (
            (1 - adjustment_strength) * base_embedding +
            adjustment_strength * neighbor_centroid
        )
        
        return adjusted_embedding
    
    def _compute_consistency_score(self, embedding: Embedding, neighbors: List[Word],
                                  embedding_store: TemporalEmbeddingStore,
                                  target_timestamp: Timestamp) -> float:
        """Compute semantic consistency score with neighbors."""
        if not neighbors:
            return 0.5  # Neutral score
        
        similarities = []
        for neighbor in neighbors:
            neighbor_result = super()._interpolate_word(embedding_store, neighbor, target_timestamp)
            if neighbor_result is not None:
                neighbor_embedding = neighbor_result[0]
                similarity = np.dot(embedding, neighbor_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(neighbor_embedding)
                )
                similarities.append(similarity)
        
        if not similarities:
            return 0.5
        
        # Return mean similarity as consistency score
        return np.mean(similarities)

# Utility functions
def create_uniform_timeline(start_timestamp: Timestamp, end_timestamp: Timestamp,
                           granularity: str = 'monthly') -> List[Timestamp]:
    """Create uniform timeline between start and end timestamps."""
    if granularity == 'daily':
        delta = timedelta(days=1)
        fmt = '%Y-%m-%d'
    elif granularity == 'weekly':
        delta = timedelta(weeks=1)
        fmt = '%Y-W%U'
    elif granularity == 'monthly':
        delta = timedelta(days=30)  # Approximate
        fmt = '%Y-%m'
    else:
        raise ValueError(f"Unknown granularity: {granularity}")
    
    # Parse timestamps
    try:
        start_dt = datetime.strptime(start_timestamp, fmt.replace('W%U', '01'))
        end_dt = datetime.strptime(end_timestamp, fmt.replace('W%U', '01'))
    except ValueError:
        # Try alternative parsing
        start_dt = datetime.fromisoformat(start_timestamp)
        end_dt = datetime.fromisoformat(end_timestamp)
    
    # Generate timeline
    timeline = []
    current = start_dt
    
    while current <= end_dt:
        if granularity == 'weekly':
            timestamp = current.strftime('%Y-W%U')
        else:
            timestamp = current.strftime(fmt)
        timeline.append(timestamp)
        current += delta
    
    return timeline

def interpolate_missing_data(embedding_store: TemporalEmbeddingStore,
                           method: str = 'linear',
                           confidence_threshold: float = 0.5) -> TemporalEmbeddingStore:
    """Convenience function to interpolate missing data in an embedding store."""
    # Find gaps in timeline
    timestamps = embedding_store.get_timestamps()
    if len(timestamps) < 2:
        return embedding_store
    
    # Create uniform timeline
    start, end = timestamps[0], timestamps[-1]
    uniform_timeline = create_uniform_timeline(start, end, 'monthly')
    
    # Find missing timestamps
    missing_timestamps = [ts for ts in uniform_timeline if ts not in timestamps]
    
    if not missing_timestamps:
        return embedding_store
    
    # Interpolate
    interpolator = TemporalInterpolation(method, confidence_threshold)
    return interpolator.process(embedding_store, missing_timestamps)