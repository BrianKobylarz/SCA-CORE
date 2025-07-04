"""Semantic burst detection and analysis."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from scipy import stats
from scipy.spatial.distance import cosine

from ..core.types import Word, Timestamp, FlowEvent, FlowEventType
from ..core.base import BaseAnalyzer
from ..metrics.flow_metrics import SemanticVelocity

class SemanticBurstDetector(BaseAnalyzer):
    """Detects burst events in semantic flow"""
    
    def __init__(self, embeddings_store, config):
        super().__init__("SemanticBurstDetector")
        self.embeddings = embeddings_store
        self.config = config
        self.velocity_metric = SemanticVelocity()
        
        # Cache for efficiency
        self.velocity_cache = {}
        self.burst_cache = {}
    
    def analyze(self, z_threshold=None, **kwargs):
        """Implement BaseAnalyzer interface."""
        bursts = self.detect_bursts(z_threshold)
        
        # Compute summary statistics
        total_bursts = sum(len(events) for events in bursts.values())
        burst_timestamps = list(bursts.keys())
        
        self.results = {
            'burst_events': bursts,
            'summary': {
                'total_bursts': total_bursts,
                'timestamps_with_bursts': len(burst_timestamps),
                'burst_density': total_bursts / len(self.embeddings.get_timestamps()) if self.embeddings.get_timestamps() else 0,
                'z_threshold': z_threshold or self.config.burst_z_threshold
            },
            'burst_chains': self.detect_burst_chains() if kwargs.get('detect_chains', False) else []
        }
        
        return self.results
    
    def detect_bursts(self, z_threshold: float = None) -> Dict[Timestamp, List[FlowEvent]]:
        """Detect all burst events across timeline"""
        z_threshold = z_threshold or self.config.burst_z_threshold
        
        bursts_by_time = defaultdict(list)
        
        # First, compute all velocities
        velocity_matrix = self._compute_velocity_matrix()
        
        # For each word, find burst points
        for word, velocities in velocity_matrix.items():
            word_bursts = self._detect_word_bursts(
                word, velocities, z_threshold
            )
            
            for burst in word_bursts:
                bursts_by_time[burst.timestamp].append(burst)
        
        # Detect cascade bursts (multiple words bursting together)
        cascade_bursts = self._detect_cascade_bursts(bursts_by_time)
        
        # Merge individual and cascade bursts
        all_bursts = dict(bursts_by_time)
        for timestamp, cascades in cascade_bursts.items():
            all_bursts[timestamp].extend(cascades)
        
        return all_bursts
    
    def _compute_velocity_matrix(self) -> Dict[Word, List[Tuple[Timestamp, float]]]:
        """Compute velocity for all words across time"""
        velocity_matrix = defaultdict(list)
        timestamps = self.embeddings.get_timestamps()
        
        for i in range(len(timestamps) - 1):
            t1, t2 = timestamps[i], timestamps[i+1]
            
            # Get common vocabulary
            vocab_t1 = set(self.embeddings.get_vocabulary(t1))
            vocab_t2 = set(self.embeddings.get_vocabulary(t2))
            common_vocab = vocab_t1 & vocab_t2
            
            for word in common_vocab:
                emb1 = self.embeddings.get_embedding(word, t1)
                emb2 = self.embeddings.get_embedding(word, t2)
                
                if emb1 is not None and emb2 is not None:
                    velocity = self.velocity_metric.compute(
                        word, t1, t2, emb1, emb2
                    )
                    velocity_matrix[word].append((t2, velocity))
        
        return dict(velocity_matrix)
    
    def _detect_word_bursts(self, word: Word, 
                           velocities: List[Tuple[Timestamp, float]], 
                           z_threshold: float) -> List[FlowEvent]:
        """Detect bursts for a single word"""
        if len(velocities) < 3:
            return []
        
        # Extract velocity values
        timestamps = [v[0] for v in velocities]
        velocity_values = [v[1] for v in velocities]
        
        # Compute statistics
        mean_velocity = np.mean(velocity_values)
        std_velocity = np.std(velocity_values)
        
        if std_velocity == 0:
            return []
        
        bursts = []
        
        for i, (timestamp, velocity) in enumerate(velocities):
            z_score = (velocity - mean_velocity) / std_velocity
            
            if z_score > z_threshold:
                # Determine affected radius
                affected_radius = self._compute_affected_radius(
                    word, timestamp, velocity
                )
                
                burst = FlowEvent(
                    timestamp=timestamp,
                    event_type=FlowEventType.BURST,
                    primary_words=[word],
                    magnitude=velocity,
                    affected_radius=affected_radius,
                    metadata={
                        'z_score': z_score,
                        'baseline_velocity': mean_velocity,
                        'velocity_std': std_velocity
                    }
                )
                bursts.append(burst)
        
        return bursts
    
    def _detect_cascade_bursts(self, 
                              bursts_by_time: Dict[Timestamp, List[FlowEvent]]) -> Dict[Timestamp, List[FlowEvent]]:
        """Detect when multiple words burst together"""
        cascade_bursts = defaultdict(list)
        
        for timestamp, burst_list in bursts_by_time.items():
            if len(burst_list) >= self.config.contagion_min_neighbors:
                # Check if bursts are semantically related
                burst_words = []
                total_magnitude = 0
                
                for burst in burst_list:
                    burst_words.extend(burst.primary_words)
                    total_magnitude += burst.magnitude
                
                # Compute semantic coherence of bursting words
                coherence = self._compute_burst_coherence(
                    burst_words, timestamp
                )
                
                if coherence > 0.5:  # Coherent burst cluster
                    cascade = FlowEvent(
                        timestamp=timestamp,
                        event_type=FlowEventType.CASCADE,
                        primary_words=burst_words,
                        magnitude=total_magnitude,
                        affected_radius=len(burst_words),
                        metadata={
                            'coherence': coherence,
                            'burst_count': len(burst_list),
                            'cascade_type': 'synchronized'
                        }
                    )
                    cascade_bursts[timestamp].append(cascade)
        
        return dict(cascade_bursts)
    
    def _compute_affected_radius(self, word: Word, 
                                timestamp: Timestamp, 
                                velocity: float) -> int:
        """Estimate how many neighbors are affected by burst"""
        # Get word's neighbors
        word_emb = self.embeddings.get_embedding(word, timestamp)
        if word_emb is None:
            return 0
        
        affected_count = 0
        velocity_threshold = velocity * 0.5  # 50% of burst velocity
        
        for other_word in self.embeddings.get_vocabulary(timestamp):
            if other_word != word:
                other_velocities = self.velocity_cache.get(other_word, [])
                
                # Check if neighbor also shows elevated velocity
                for t, v in other_velocities:
                    if t == timestamp and v > velocity_threshold:
                        affected_count += 1
                        break
        
        return affected_count
    
    def _compute_burst_coherence(self, words: List[Word], 
                                timestamp: Timestamp) -> float:
        """Compute semantic coherence of bursting words"""
        if len(words) < 2:
            return 1.0
        
        # Compute average pairwise similarity
        similarities = []
        
        for i, w1 in enumerate(words):
            emb1 = self.embeddings.get_embedding(w1, timestamp)
            if emb1 is None:
                continue
                
            for w2 in words[i+1:]:
                emb2 = self.embeddings.get_embedding(w2, timestamp)
                if emb2 is not None:
                    sim = 1 - cosine(emb1, emb2)
                    similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def detect_burst_chains(self, min_chain_length: int = 3) -> List[Dict]:
        """Detect chains of bursts (burst → burst → burst)"""
        all_bursts = self.detect_bursts()
        chains = []
        
        # Build burst graph
        burst_graph = defaultdict(list)
        timestamps = sorted(all_bursts.keys())
        
        for i in range(len(timestamps) - 1):
            t1, t2 = timestamps[i], timestamps[i+1]
            
            if t1 in all_bursts and t2 in all_bursts:
                # Check for connections between bursts
                for b1 in all_bursts[t1]:
                    for b2 in all_bursts[t2]:
                        # Check word overlap
                        overlap = set(b1.primary_words) & set(b2.primary_words)
                        if overlap:
                            burst_graph[b1].append(b2)
        
        # Find chains using DFS
        visited = set()
        
        def find_chain(burst, current_chain):
            if len(current_chain) >= min_chain_length:
                chains.append(current_chain.copy())
            
            for next_burst in burst_graph.get(burst, []):
                if next_burst not in visited:
                    visited.add(next_burst)
                    current_chain.append(next_burst)
                    find_chain(next_burst, current_chain)
                    current_chain.pop()
        
        # Start from each burst
        for timestamp in timestamps:
            for burst in all_bursts.get(timestamp, []):
                if burst not in visited:
                    visited.add(burst)
                    find_chain(burst, [burst])
        
        return chains