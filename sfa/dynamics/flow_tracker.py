"""Semantic flow tracking and analysis."""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from scipy.spatial.distance import cosine

from ..core.types import (
    Word, Timestamp, Embedding, SemanticFlow, 
    FlowVector, WordTrajectory
)
from ..core.base import BaseAnalyzer
from ..metrics.flow_metrics import (
    SemanticVelocity, SemanticAcceleration, 
    FlowCoherence, SemanticStochasticity
)

class SemanticFlowTracker(BaseAnalyzer):
    """Tracks and analyzes semantic flows through time"""
    
    def __init__(self, embeddings_store, config):
        super().__init__("SemanticFlowTracker")
        self.embeddings = embeddings_store
        self.config = config
        self.velocity_metric = SemanticVelocity()
        self.acceleration_metric = SemanticAcceleration()
        self.coherence_metric = FlowCoherence()
        self.stochasticity_metric = SemanticStochasticity()
        
        # Caches
        self.flow_cache = {}
        self.trajectory_cache = {}
    
    def track_flows_for_timeline(self, focus_words: Optional[List[Word]] = None) -> List[SemanticFlow]:
        """Track semantic flows for all words across the timeline"""
        flows = []
        timestamps = self.embeddings.get_timestamps()
        
        # Get all words if no focus words specified
        if focus_words is None:
            all_words = set()
            for timestamp in timestamps:
                all_words.update(self.embeddings.get_vocabulary(timestamp))
            focus_words = list(all_words)
        
        # Track flows for each word
        for word in focus_words:
            word_flows = self.track_word_flow(word)
            for timestamp, flow in word_flows.items():
                flows.append(flow)
        
        return flows
    
    def analyze(self, focus_words: Optional[List[Word]] = None, **kwargs) -> Dict[str, Any]:
        """Implement BaseAnalyzer interface."""
        flows = self.track_flows_for_timeline(focus_words)
        
        # Compute summary statistics
        total_flows = len(flows)
        avg_magnitude = sum(flow.magnitude for flow in flows) / total_flows if total_flows > 0 else 0
        
        self.results = {
            'flows': flows,
            'summary': {
                'total_flows': total_flows,
                'average_magnitude': avg_magnitude,
                'focus_words': focus_words or []
            }
        }
        
        return self.results
    
    def track_word_flow(self, word: Word, k: int = None) -> Dict[Timestamp, SemanticFlow]:
        """Track semantic flow for a word across all timestamps"""
        k = k or self.config.default_k_neighbors
        
        flows = {}
        timestamps = self.embeddings.get_timestamps()
        
        for i in range(len(timestamps) - 1):
            t1, t2 = timestamps[i], timestamps[i+1]
            
            flow = self.compute_flow_between(word, t1, t2, k)
            if flow:
                flows[t2] = flow
        
        return flows
    
    def compute_flow_between(self, word: Word, t1: Timestamp, 
                           t2: Timestamp, k: int) -> Optional[SemanticFlow]:
        """Compute semantic flow between two timestamps"""
        try:
            # Get embeddings with proper None checking
            emb1 = self.embeddings.get_embedding(word, t1)
            emb2 = self.embeddings.get_embedding(word, t2)
            
            if emb1 is None or emb2 is None:
                return None
            
            # Validate embedding dimensions
            if len(emb1) == 0 or len(emb2) == 0:
                return None
            
            if len(emb1) != len(emb2):
                return None
        except (AttributeError, KeyError, IndexError) as e:
            # Handle embedding retrieval errors gracefully
            return None
        
        # Get neighbors at both times with error handling
        try:
            neighbors_t1 = self._get_semantic_neighbors(word, t1, k)
            neighbors_t2 = self._get_semantic_neighbors(word, t2, k)
            
            if not neighbors_t1 and not neighbors_t2:
                return None
                
        except Exception as e:
            return None
        
        # Compute flow vectors to each t2 neighbor
        flow_vectors = []
        target_words = []
        
        for neighbor, sim in neighbors_t2:
            try:
                neighbor_emb = self.embeddings.get_embedding(neighbor, t2)
                if neighbor_emb is not None and len(neighbor_emb) == len(emb2):
                    # Flow vector from word's movement toward this neighbor
                    direction = neighbor_emb - emb2
                    direction_norm = np.linalg.norm(direction)
                    
                    if direction_norm > 0:
                        direction = direction / direction_norm
                        
                        # Magnitude based on similarity and movement
                        movement = emb2 - emb1
                        flow_magnitude = np.dot(movement, direction) * sim
                        
                        if flow_magnitude > 0:  # Only positive flows
                            flow_vectors.append((
                                direction[0], 
                                direction[1], 
                                flow_magnitude
                            ))
                            target_words.append(neighbor)
            except (AttributeError, KeyError, IndexError, ValueError):
                continue
        
        if not flow_vectors:
            return None
        
        # Compute total flow magnitude and coherence
        total_magnitude = sum(v[2] for v in flow_vectors)
        coherence = self.coherence_metric.compute(flow_vectors)
        
        return SemanticFlow(
            source_word=word,
            target_words=target_words,
            flow_vectors=flow_vectors,
            total_magnitude=total_magnitude,
            coherence=coherence,
            timestamp=t2
        )
    
    def compute_word_trajectory(self, word: Word) -> Optional[WordTrajectory]:
        """Compute complete trajectory for a word"""
        if word in self.trajectory_cache:
            return self.trajectory_cache[word]
        
        timestamps = self.embeddings.get_timestamps()
        positions = []
        velocities = []
        
        # Collect positions
        for t in timestamps:
            emb = self.embeddings.get_embedding(word, t)
            if emb is not None:
                positions.append((t, emb))
        
        if len(positions) < 2:
            return None
        
        # Compute velocities
        for i in range(1, len(positions)):
            t1, emb1 = positions[i-1]
            t2, emb2 = positions[i]
            
            velocity = self.velocity_metric.compute(
                word, t1, t2, emb1, emb2
            )
            velocities.append(velocity)
        
        # Compute accelerations
        accelerations = self.acceleration_metric.compute(velocities)
        
        # Compute cumulative distance
        cumulative_distance = sum(velocities)
        
        # Compute stochasticity
        # Need to get stability history first
        stability_history = self._compute_stability_history(word)
        stochasticity = self.stochasticity_metric.compute(
            word, stability_history
        )
        
        # Compute path dependency
        from ..metrics.flow_metrics import PathDependency
        path_dep_metric = PathDependency()
        path_dependency = path_dep_metric.compute(velocities)
        
        trajectory = WordTrajectory(
            word=word,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            cumulative_distance=cumulative_distance,
            stochasticity=stochasticity,
            path_dependency=path_dependency
        )
        
        self.trajectory_cache[word] = trajectory
        return trajectory
    
    def identify_convergence_points(self, 
                                  threshold: float = 0.7) -> Dict[Timestamp, List[Tuple[Word, Word]]]:
        """Find where words are converging in semantic space"""
        convergences = defaultdict(list)
        
        timestamps = self.embeddings.get_timestamps()
        
        for i in range(len(timestamps) - 1):
            t1, t2 = timestamps[i], timestamps[i+1]
            
            # Get all words at both timestamps
            words_t1 = set(self.embeddings.get_vocabulary(t1))
            words_t2 = set(self.embeddings.get_vocabulary(t2))
            common_words = words_t1 & words_t2
            
            # Check each pair
            for w1 in common_words:
                for w2 in common_words:
                    if w1 < w2:  # Avoid duplicates
                        # Check if getting closer
                        dist_t1 = self._compute_distance(w1, w2, t1)
                        dist_t2 = self._compute_distance(w1, w2, t2)
                        
                        if dist_t1 and dist_t2:
                            if dist_t2 < dist_t1 * threshold:
                                convergences[t2].append((w1, w2))
        
        return dict(convergences)
    
    def _get_semantic_neighbors(self, word: Word, timestamp: Timestamp, 
                               k: int) -> List[Tuple[Word, float]]:
        """Get k nearest semantic neighbors with similarities"""
        try:
            word_emb = self.embeddings.get_embedding(word, timestamp)
            if word_emb is None or len(word_emb) == 0:
                return []
            
            vocabulary = self.embeddings.get_vocabulary(timestamp)
            if not vocabulary:
                return []
            
            similarities = []
            for other_word in vocabulary:
                if other_word != word:
                    try:
                        other_emb = self.embeddings.get_embedding(other_word, timestamp)
                        if other_emb is not None and len(other_emb) == len(word_emb):
                            sim = 1 - cosine(word_emb, other_emb)
                            # Handle potential NaN or infinite values
                            if np.isfinite(sim):
                                similarities.append((other_word, sim))
                    except (ValueError, TypeError):
                        continue
        except (AttributeError, KeyError):
            return []
        
        # Sort and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def _compute_distance(self, w1: Word, w2: Word, 
                         timestamp: Timestamp) -> Optional[float]:
        """Compute distance between two words"""
        emb1 = self.embeddings.get_embedding(w1, timestamp)
        emb2 = self.embeddings.get_embedding(w2, timestamp)
        
        if emb1 is None or emb2 is None:
            return None
        
        return np.linalg.norm(emb2 - emb1)
    
    def _compute_stability_history(self, word: Word) -> List[float]:
        """Compute neighbor stability history for stochasticity"""
        stabilities = []
        timestamps = self.embeddings.get_timestamps()
        
        for i in range(len(timestamps) - 1):
            t1, t2 = timestamps[i], timestamps[i+1]
            
            n1 = set(w for w, _ in self._get_semantic_neighbors(word, t1, 10))
            n2 = set(w for w, _ in self._get_semantic_neighbors(word, t2, 10))
            
            if n1 and n2:
                stability = len(n1 & n2) / len(n1 | n2)
                stabilities.append(stability)
        
        return stabilities