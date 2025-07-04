"""Adaptive neighbor expansion for sparse semantic data."""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from scipy.spatial.distance import cosine
from collections import defaultdict

from ..core.types import Word, Timestamp, Embedding

class AdaptiveNeighborExpansion:
    """Dynamically expands neighborhoods to handle sparsity"""
    
    def __init__(self, embeddings_store, config):
        self.embeddings = embeddings_store
        self.config = config
        self.neighbor_cache = {}
    
    def expand_neighborhood(self, 
                          word: Word, 
                          timestamp: Timestamp,
                          target_size: int = None,
                          min_similarity: float = None) -> List[Tuple[Word, float]]:
        """Adaptively expand neighborhood to reach target size"""
        target_size = target_size or self.config.default_k_neighbors
        min_similarity = min_similarity or self.config.get('min_similarity', 0.1)
        
        # Check cache
        cache_key = f"{word}:{timestamp}:{target_size}:{min_similarity}"
        if cache_key in self.neighbor_cache:
            return self.neighbor_cache[cache_key]
        
        word_emb = self.embeddings.get_embedding(word, timestamp)
        if word_emb is None:
            return []
        
        # Start with strict similarity threshold
        similarity_threshold = 0.7
        neighbors = []
        
        # Iteratively relax threshold until we have enough neighbors
        while len(neighbors) < target_size and similarity_threshold >= min_similarity:
            neighbors = self._find_neighbors_with_threshold(
                word, word_emb, timestamp, similarity_threshold
            )
            
            if len(neighbors) < target_size:
                # Relax threshold
                similarity_threshold *= 0.9
        
        # If still not enough, use temporal expansion
        if len(neighbors) < target_size:
            temporal_neighbors = self._temporal_neighbor_expansion(
                word, timestamp, target_size - len(neighbors)
            )
            neighbors.extend(temporal_neighbors)
        
        # Sort by similarity and trim to target size
        neighbors.sort(key=lambda x: x[1], reverse=True)
        neighbors = neighbors[:target_size]
        
        # Cache result
        self.neighbor_cache[cache_key] = neighbors
        
        return neighbors
    
    def _find_neighbors_with_threshold(self, 
                                     word: Word,
                                     word_emb: Embedding,
                                     timestamp: Timestamp,
                                     threshold: float) -> List[Tuple[Word, float]]:
        """Find neighbors above similarity threshold"""
        neighbors = []
        vocabulary = self.embeddings.get_vocabulary(timestamp)
        
        for other_word in vocabulary:
            if other_word != word:
                other_emb = self.embeddings.get_embedding(other_word, timestamp)
                if other_emb is not None:
                    similarity = 1 - cosine(word_emb, other_emb)
                    if similarity >= threshold:
                        neighbors.append((other_word, similarity))
        
        return neighbors
    
    def _temporal_neighbor_expansion(self, 
                                   word: Word,
                                   timestamp: Timestamp,
                                   needed_count: int) -> List[Tuple[Word, float]]:
        """Expand neighborhood using temporal information"""
        if needed_count <= 0:
            return []
        
        timestamps = self.embeddings.get_timestamps()
        current_idx = timestamps.index(timestamp) if timestamp in timestamps else -1
        
        if current_idx == -1:
            return []
        
        # Look at nearby timestamps
        candidate_neighbors = defaultdict(list)
        
        # Check previous and next timestamps
        for offset in [-1, 1, -2, 2]:
            neighbor_idx = current_idx + offset
            if 0 <= neighbor_idx < len(timestamps):
                neighbor_timestamp = timestamps[neighbor_idx]
                
                # Find neighbors at this timestamp
                temp_neighbors = self._find_temporal_neighbors(
                    word, timestamp, neighbor_timestamp
                )
                
                for neighbor_word, similarity in temp_neighbors:
                    # Weight by temporal distance
                    temporal_weight = 1.0 / (abs(offset) + 1)
                    weighted_similarity = similarity * temporal_weight
                    candidate_neighbors[neighbor_word].append(weighted_similarity)
        
        # Average similarities across timestamps
        temporal_neighbors = []
        for neighbor_word, similarities in candidate_neighbors.items():
            avg_similarity = np.mean(similarities)
            temporal_neighbors.append((neighbor_word, avg_similarity))
        
        # Sort and return top needed
        temporal_neighbors.sort(key=lambda x: x[1], reverse=True)
        return temporal_neighbors[:needed_count]
    
    def _find_temporal_neighbors(self, 
                               word: Word,
                               base_timestamp: Timestamp,
                               neighbor_timestamp: Timestamp) -> List[Tuple[Word, float]]:
        """Find neighbors by projecting word to different timestamp"""
        word_emb_base = self.embeddings.get_embedding(word, base_timestamp)
        word_emb_neighbor = self.embeddings.get_embedding(word, neighbor_timestamp)
        
        if word_emb_base is None or word_emb_neighbor is None:
            return []
        
        # Use the word's embedding at the neighbor timestamp as reference
        neighbors = []
        vocabulary = self.embeddings.get_vocabulary(neighbor_timestamp)
        
        for other_word in vocabulary:
            if other_word != word:
                other_emb = self.embeddings.get_embedding(other_word, neighbor_timestamp)
                if other_emb is not None:
                    similarity = 1 - cosine(word_emb_neighbor, other_emb)
                    if similarity > 0.3:  # Basic threshold
                        neighbors.append((other_word, similarity))
        
        return neighbors
    
    def expand_similarity_network(self, 
                                timestamp: Timestamp,
                                expansion_factor: float = 1.5) -> Dict[Tuple[Word, Word], float]:
        """Expand similarity network by relaxing connection criteria"""
        vocabulary = list(self.embeddings.get_vocabulary(timestamp))
        expanded_network = {}
        
        # Compute base similarity threshold
        all_similarities = []
        for i, w1 in enumerate(vocabulary):
            emb1 = self.embeddings.get_embedding(w1, timestamp)
            if emb1 is None:
                continue
            
            for w2 in vocabulary[i+1:]:
                emb2 = self.embeddings.get_embedding(w2, timestamp)
                if emb2 is not None:
                    similarity = 1 - cosine(emb1, emb2)
                    all_similarities.append(similarity)
        
        if not all_similarities:
            return {}
        
        # Adaptive threshold based on data density
        base_threshold = np.percentile(all_similarities, 70)  # 70th percentile
        relaxed_threshold = base_threshold / expansion_factor
        
        # Build expanded network
        for i, w1 in enumerate(vocabulary):
            emb1 = self.embeddings.get_embedding(w1, timestamp)
            if emb1 is None:
                continue
            
            for w2 in vocabulary[i+1:]:
                emb2 = self.embeddings.get_embedding(w2, timestamp)
                if emb2 is not None:
                    similarity = 1 - cosine(emb1, emb2)
                    if similarity >= relaxed_threshold:
                        expanded_network[(w1, w2)] = similarity
        
        return expanded_network
    
    def adaptive_k_selection(self, 
                           word: Word,
                           timestamp: Timestamp,
                           min_k: int = 5,
                           max_k: int = 50) -> int:
        """Adaptively select optimal k based on data availability"""
        word_emb = self.embeddings.get_embedding(word, timestamp)
        if word_emb is None:
            return min_k
        
        vocabulary = self.embeddings.get_vocabulary(timestamp)
        
        # Compute all similarities
        similarities = []
        for other_word in vocabulary:
            if other_word != word:
                other_emb = self.embeddings.get_embedding(other_word, timestamp)
                if other_emb is not None:
                    similarity = 1 - cosine(word_emb, other_emb)
                    similarities.append(similarity)
        
        if not similarities:
            return min_k
        
        similarities.sort(reverse=True)
        
        # Find elbow point in similarity curve
        optimal_k = self._find_elbow_point(similarities, min_k, max_k)
        
        return optimal_k
    
    def _find_elbow_point(self, 
                         similarities: List[float],
                         min_k: int,
                         max_k: int) -> int:
        """Find elbow point in similarity curve using second derivative"""
        if len(similarities) < min_k:
            return min(len(similarities), max_k)
        
        # Limit to reasonable range
        max_check = min(len(similarities), max_k)
        
        if max_check <= min_k:
            return max_check
        
        # Compute second derivative
        second_derivatives = []
        for i in range(min_k, max_check - 1):
            if i >= 2 and i < len(similarities) - 1:
                second_deriv = similarities[i-1] - 2*similarities[i] + similarities[i+1]
                second_derivatives.append((i, second_deriv))
        
        if not second_derivatives:
            return min_k
        
        # Find maximum second derivative (elbow point)
        elbow_k = max(second_derivatives, key=lambda x: x[1])[0]
        
        return max(min_k, min(elbow_k, max_k))
    
    def compute_neighbor_stability(self, 
                                 word: Word,
                                 timestamps: List[Timestamp],
                                 k: int = 10) -> float:
        """Compute stability of neighborhood across timestamps"""
        if len(timestamps) < 2:
            return 1.0
        
        neighborhood_sets = []
        
        for timestamp in timestamps:
            neighbors = self.expand_neighborhood(word, timestamp, k)
            neighbor_words = set(w for w, _ in neighbors)
            neighborhood_sets.append(neighbor_words)
        
        if not neighborhood_sets:
            return 0.0
        
        # Compute pairwise Jaccard similarities
        stabilities = []
        for i in range(len(neighborhood_sets) - 1):
            set1 = neighborhood_sets[i]
            set2 = neighborhood_sets[i + 1]
            
            if len(set1 | set2) > 0:
                jaccard = len(set1 & set2) / len(set1 | set2)
                stabilities.append(jaccard)
        
        return np.mean(stabilities) if stabilities else 0.0
    
    def get_expansion_statistics(self) -> Dict[str, any]:
        """Get statistics about neighborhood expansion"""
        return {
            'cache_size': len(self.neighbor_cache),
            'cache_hit_rate': getattr(self, 'cache_hits', 0) / max(getattr(self, 'cache_requests', 1), 1)
        }
    
    def clear_cache(self) -> None:
        """Clear the neighbor cache"""
        self.neighbor_cache.clear()