"""Embedding management for Semantic Flow Analyzer."""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import pickle
import h5py
from pathlib import Path
import logging

from .types import Word, Timestamp, Embedding, EmbeddingMatrix
from .base import BaseCache

logger = logging.getLogger(__name__)

class EmbeddingStore:
    """Manages diachronic embeddings with efficient storage and retrieval."""
    
    def __init__(self, cache: Optional[BaseCache] = None):
        self.embeddings: EmbeddingMatrix = {}
        self.vocabulary: Dict[Timestamp, Set[Word]] = {}
        self.timestamps: List[Timestamp] = []
        self.dimension: Optional[int] = None
        self.cache = cache
        
        # Statistics
        self.stats = {
            'total_words': 0,
            'total_embeddings': 0,
            'memory_usage': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def add_embeddings(self, timestamp: Timestamp, 
                      embeddings: Dict[Word, Embedding]) -> None:
        """Add embeddings for a specific timestamp."""
        # Validate embeddings
        if not embeddings:
            logger.warning(f"No embeddings provided for timestamp {timestamp}")
            return
        
        # Check dimension consistency
        first_emb = next(iter(embeddings.values()))
        if self.dimension is None:
            self.dimension = len(first_emb)
        elif len(first_emb) != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {len(first_emb)}")
        
        # Store embeddings
        self.embeddings[timestamp] = embeddings.copy()
        self.vocabulary[timestamp] = set(embeddings.keys())
        
        # Update timestamps
        if timestamp not in self.timestamps:
            self.timestamps.append(timestamp)
            self.timestamps.sort()
        
        # Update statistics
        self.stats['total_words'] = len(set().union(*self.vocabulary.values()))
        self.stats['total_embeddings'] += len(embeddings)
        self.stats['memory_usage'] = self._calculate_memory_usage()
        
        logger.info(f"Added {len(embeddings)} embeddings for timestamp {timestamp}")
    
    def get_embedding(self, word: Word, timestamp: Timestamp) -> Optional[Embedding]:
        """Get embedding for a word at a specific timestamp."""
        cache_key = f"emb:{word}:{timestamp}"
        
        # Check cache first
        if self.cache:
            cached_emb = self.cache.get(cache_key)
            if cached_emb is not None:
                self.stats['cache_hits'] += 1
                return cached_emb
            self.stats['cache_misses'] += 1
        
        # Get from storage
        if timestamp in self.embeddings and word in self.embeddings[timestamp]:
            embedding = self.embeddings[timestamp][word]
            
            # Cache the result
            if self.cache:
                self.cache.set(cache_key, embedding)
            
            return embedding
        
        return None
    
    def get_embeddings(self, timestamp: Timestamp) -> Dict[Word, Embedding]:
        """Get all embeddings for a timestamp."""
        return self.embeddings.get(timestamp, {}).copy()
    
    def get_vocabulary(self, timestamp: Timestamp) -> Set[Word]:
        """Get vocabulary for a timestamp."""
        return self.vocabulary.get(timestamp, set()).copy()
    
    def get_timestamps(self) -> List[Timestamp]:
        """Get all timestamps."""
        return self.timestamps.copy()
    
    def get_common_vocabulary(self, timestamps: List[Timestamp]) -> Set[Word]:
        """Get words that appear in all specified timestamps."""
        if not timestamps:
            return set()
        
        common_vocab = self.vocabulary.get(timestamps[0], set())
        for timestamp in timestamps[1:]:
            common_vocab &= self.vocabulary.get(timestamp, set())
        
        return common_vocab
    
    def get_word_timeline(self, word: Word) -> List[Tuple[Timestamp, Embedding]]:
        """Get complete timeline for a word."""
        timeline = []
        
        for timestamp in self.timestamps:
            embedding = self.get_embedding(word, timestamp)
            if embedding is not None:
                timeline.append((timestamp, embedding))
        
        return timeline
    
    def compute_similarity(self, word1: Word, word2: Word, 
                          timestamp: Timestamp, metric: str = 'cosine') -> Optional[float]:
        """Compute similarity between two words at a timestamp."""
        emb1 = self.get_embedding(word1, timestamp)
        emb2 = self.get_embedding(word2, timestamp)
        
        if emb1 is None or emb2 is None:
            return None
        
        if metric == 'cosine':
            return self._cosine_similarity(emb1, emb2)
        elif metric == 'euclidean':
            return 1.0 / (1.0 + np.linalg.norm(emb1 - emb2))
        elif metric == 'dot':
            return np.dot(emb1, emb2)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def find_similar_words(self, word: Word, timestamp: Timestamp, 
                          k: int = 10, metric: str = 'cosine') -> List[Tuple[Word, float]]:
        """Find k most similar words to a given word at a timestamp."""
        target_emb = self.get_embedding(word, timestamp)
        if target_emb is None:
            return []
        
        similarities = []
        vocab = self.get_vocabulary(timestamp)
        
        for other_word in vocab:
            if other_word != word:
                similarity = self.compute_similarity(word, other_word, timestamp, metric)
                if similarity is not None:
                    similarities.append((other_word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def compute_trajectory_distance(self, word1: Word, word2: Word) -> float:
        """Compute distance between two word trajectories."""
        timeline1 = self.get_word_timeline(word1)
        timeline2 = self.get_word_timeline(word2)
        
        if not timeline1 or not timeline2:
            return float('inf')
        
        # Find common timestamps
        times1 = {t for t, _ in timeline1}
        times2 = {t for t, _ in timeline2}
        common_times = times1 & times2
        
        if not common_times:
            return float('inf')
        
        # Compute average distance at common timestamps
        distances = []
        for timestamp in common_times:
            emb1 = self.get_embedding(word1, timestamp)
            emb2 = self.get_embedding(word2, timestamp)
            if emb1 is not None and emb2 is not None:
                distances.append(np.linalg.norm(emb1 - emb2))
        
        return np.mean(distances) if distances else float('inf')
    
    def detect_new_words(self, timestamp: Timestamp, 
                        previous_timestamps: Optional[List[Timestamp]] = None) -> Set[Word]:
        """Detect words that first appear at a given timestamp."""
        if previous_timestamps is None:
            # Use all previous timestamps
            idx = self.timestamps.index(timestamp)
            previous_timestamps = self.timestamps[:idx]
        
        if not previous_timestamps:
            return self.get_vocabulary(timestamp)
        
        current_vocab = self.get_vocabulary(timestamp)
        previous_vocab = set()
        
        for prev_timestamp in previous_timestamps:
            previous_vocab.update(self.get_vocabulary(prev_timestamp))
        
        return current_vocab - previous_vocab
    
    def detect_disappeared_words(self, timestamp: Timestamp,
                                next_timestamps: Optional[List[Timestamp]] = None) -> Set[Word]:
        """Detect words that disappear after a given timestamp."""
        if next_timestamps is None:
            # Use all next timestamps
            idx = self.timestamps.index(timestamp)
            next_timestamps = self.timestamps[idx + 1:]
        
        if not next_timestamps:
            return set()
        
        current_vocab = self.get_vocabulary(timestamp)
        next_vocab = set()
        
        for next_timestamp in next_timestamps:
            next_vocab.update(self.get_vocabulary(next_timestamp))
        
        return current_vocab - next_vocab
    
    def save(self, path: str, format: str = 'h5') -> None:
        """Save embeddings to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'h5':
            self._save_h5(path)
        elif format == 'pickle':
            self._save_pickle(path)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Saved embeddings to {path}")
    
    def load(self, path: str, format: str = 'h5') -> None:
        """Load embeddings from file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if format == 'h5':
            self._load_h5(path)
        elif format == 'pickle':
            self._load_pickle(path)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Loaded embeddings from {path}")
    
    def _save_h5(self, path: Path) -> None:
        """Save embeddings to HDF5 format."""
        with h5py.File(path, 'w') as f:
            # Metadata
            f.attrs['dimension'] = self.dimension or 0
            f.attrs['num_timestamps'] = len(self.timestamps)
            f.attrs['num_words'] = self.stats['total_words']
            
            # Timestamps
            timestamps_group = f.create_group('timestamps')
            for i, timestamp in enumerate(self.timestamps):
                timestamps_group.attrs[f'timestamp_{i}'] = timestamp
            
            # Embeddings
            embeddings_group = f.create_group('embeddings')
            for timestamp, emb_dict in self.embeddings.items():
                timestamp_group = embeddings_group.create_group(timestamp)
                
                words = list(emb_dict.keys())
                embeddings = np.array([emb_dict[word] for word in words])
                
                timestamp_group.create_dataset('words', data=np.array(words, dtype='S'))
                timestamp_group.create_dataset('embeddings', data=embeddings)
    
    def _load_h5(self, path: Path) -> None:
        """Load embeddings from HDF5 format."""
        with h5py.File(path, 'r') as f:
            # Metadata
            self.dimension = f.attrs['dimension']
            
            # Timestamps
            timestamps_group = f['timestamps']
            self.timestamps = []
            for i in range(f.attrs['num_timestamps']):
                timestamp = timestamps_group.attrs[f'timestamp_{i}']
                self.timestamps.append(timestamp)
            
            # Embeddings
            embeddings_group = f['embeddings']
            self.embeddings = {}
            self.vocabulary = {}
            
            for timestamp in self.timestamps:
                timestamp_group = embeddings_group[timestamp]
                
                words = [w.decode('utf-8') for w in timestamp_group['words'][:]]
                embeddings = timestamp_group['embeddings'][:]
                
                self.embeddings[timestamp] = {
                    word: embedding for word, embedding in zip(words, embeddings)
                }
                self.vocabulary[timestamp] = set(words)
        
        # Update statistics
        self.stats['total_words'] = len(set().union(*self.vocabulary.values()))
        self.stats['total_embeddings'] = sum(len(emb) for emb in self.embeddings.values())
        self.stats['memory_usage'] = self._calculate_memory_usage()
    
    def _save_pickle(self, path: Path) -> None:
        """Save embeddings to pickle format."""
        data = {
            'embeddings': self.embeddings,
            'vocabulary': self.vocabulary,
            'timestamps': self.timestamps,
            'dimension': self.dimension,
            'stats': self.stats
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def _load_pickle(self, path: Path) -> None:
        """Load embeddings from pickle format."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.vocabulary = data['vocabulary']
        self.timestamps = data['timestamps']
        self.dimension = data['dimension']
        self.stats = data['stats']
    
    def _cosine_similarity(self, emb1: Embedding, emb2: Embedding) -> float:
        """Compute cosine similarity between two embeddings."""
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_memory_usage(self) -> int:
        """Calculate approximate memory usage in bytes."""
        total_size = 0
        
        for timestamp, emb_dict in self.embeddings.items():
            for word, embedding in emb_dict.items():
                total_size += len(word.encode('utf-8'))  # Word string
                total_size += embedding.nbytes  # Embedding array
        
        return total_size
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return self.stats.copy()
    
    def optimize_memory(self) -> None:
        """Optimize memory usage by converting to appropriate dtypes."""
        for timestamp, emb_dict in self.embeddings.items():
            for word, embedding in emb_dict.items():
                if embedding.dtype != np.float32:
                    self.embeddings[timestamp][word] = embedding.astype(np.float32)
        
        # Update memory usage
        self.stats['memory_usage'] = self._calculate_memory_usage()
        logger.info("Memory optimization completed")
    
    def clear_cache(self) -> None:
        """Clear cache."""
        if self.cache:
            self.cache.clear()
        self.stats['cache_hits'] = 0
        self.stats['cache_misses'] = 0


class TemporalEmbeddingStore(EmbeddingStore):
    """
    Extended embedding store with enhanced temporal analysis capabilities.
    
    This class builds on EmbeddingStore to provide specialized functionality
    for semantic flow analysis including temporal interpolation, drift detection,
    and advanced temporal queries.
    """
    
    def __init__(self, embedding_dim: int, cache: Optional[BaseCache] = None):
        super().__init__(cache)
        self.embedding_dim = embedding_dim
        self.dimension = embedding_dim
        
        # Temporal analysis components
        self.temporal_interpolator = None
        self.drift_detector = None
        
        # Temporal statistics
        self.temporal_stats = {
            'vocabulary_stability': {},
            'embedding_drift_rates': {},
            'temporal_coverage': {},
            'interpolation_accuracy': {}
        }
        
        # Advanced indexing for temporal queries
        self.word_first_appearance = {}  # word -> timestamp
        self.word_last_appearance = {}   # word -> timestamp
        self.temporal_word_index = defaultdict(set)  # timestamp -> words
        
    def store_embedding(self, word: Word, timestamp: Timestamp, embedding: Embedding) -> None:
        """
        Store a single embedding with temporal tracking.
        
        Args:
            word: The word
            timestamp: The timestamp
            embedding: The embedding vector
        """
        # Validate embedding dimension
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}")
        
        # Store in the parent structure
        if timestamp not in self.embeddings:
            self.embeddings[timestamp] = {}
            self.vocabulary[timestamp] = set()
        
        self.embeddings[timestamp][word] = np.array(embedding, dtype=np.float32)
        self.vocabulary[timestamp].add(word)
        
        # Update timestamps
        if timestamp not in self.timestamps:
            self.timestamps.append(timestamp)
            self.timestamps.sort()
        
        # Update temporal tracking
        self._update_temporal_tracking(word, timestamp)
        
        # Update statistics
        self._update_temporal_stats()
        
    def _update_temporal_tracking(self, word: Word, timestamp: Timestamp) -> None:
        """Update temporal tracking indices."""
        # Track first and last appearances
        if word not in self.word_first_appearance:
            self.word_first_appearance[word] = timestamp
        else:
            if timestamp < self.word_first_appearance[word]:
                self.word_first_appearance[word] = timestamp
        
        if word not in self.word_last_appearance:
            self.word_last_appearance[word] = timestamp
        else:
            if timestamp > self.word_last_appearance[word]:
                self.word_last_appearance[word] = timestamp
        
        # Update temporal index
        self.temporal_word_index[timestamp].add(word)
    
    def _update_temporal_stats(self) -> None:
        """Update temporal statistics."""
        if len(self.timestamps) < 2:
            return
        
        # Vocabulary stability between consecutive timestamps
        for i in range(len(self.timestamps) - 1):
            t1, t2 = self.timestamps[i], self.timestamps[i + 1]
            vocab1 = self.get_vocabulary(t1)
            vocab2 = self.get_vocabulary(t2)
            
            if vocab1 and vocab2:
                intersection = len(vocab1 & vocab2)
                union = len(vocab1 | vocab2)
                stability = intersection / union if union > 0 else 0
                
                self.temporal_stats['vocabulary_stability'][f"{t1}_{t2}"] = stability
        
        # Update parent statistics
        self.stats['total_words'] = len(set().union(*self.vocabulary.values()))
        self.stats['total_embeddings'] = sum(len(emb) for emb in self.embeddings.values())
        self.stats['memory_usage'] = self._calculate_memory_usage()
    
    def get_word_temporal_span(self, word: Word) -> Tuple[Optional[Timestamp], Optional[Timestamp]]:
        """Get the temporal span (first, last) for a word."""
        first = self.word_first_appearance.get(word)
        last = self.word_last_appearance.get(word)
        return first, last
    
    def get_words_in_timespan(self, start_timestamp: Timestamp, end_timestamp: Timestamp) -> Set[Word]:
        """Get all words that appear in the given timespan."""
        words = set()
        for timestamp in self.timestamps:
            if start_timestamp <= timestamp <= end_timestamp:
                words.update(self.get_vocabulary(timestamp))
        return words
    
    def get_persistent_words(self, min_appearances: int = 2) -> Set[Word]:
        """Get words that appear in at least min_appearances timestamps."""
        word_counts = defaultdict(int)
        
        for vocab in self.vocabulary.values():
            for word in vocab:
                word_counts[word] += 1
        
        return {word for word, count in word_counts.items() if count >= min_appearances}
    
    def get_temporal_neighbors(self, word: Word, timestamp: Timestamp, k: int = 10) -> List[Tuple[Word, float]]:
        """
        Get k nearest neighbors at a specific timestamp with temporal consistency weighting.
        
        This method finds similar words while considering their temporal stability.
        """
        target_emb = self.get_embedding(word, timestamp)
        if target_emb is None:
            return []
        
        vocab = self.get_vocabulary(timestamp)
        similarities = []
        
        for other_word in vocab:
            if other_word == word:
                continue
            
            other_emb = self.get_embedding(other_word, timestamp)
            if other_emb is None:
                continue
            
            # Base similarity
            base_sim = self._cosine_similarity(target_emb, other_emb)
            
            # Temporal consistency bonus
            temporal_bonus = self._compute_temporal_consistency_bonus(word, other_word, timestamp)
            
            # Combined score
            final_score = base_sim * (1 + temporal_bonus)
            similarities.append((other_word, final_score))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def _compute_temporal_consistency_bonus(self, word1: Word, word2: Word, timestamp: Timestamp) -> float:
        """Compute temporal consistency bonus for word pairs."""
        # Find other timestamps where both words appear
        timeline1 = self.get_word_timeline(word1)
        timeline2 = self.get_word_timeline(word2)
        
        if len(timeline1) < 2 or len(timeline2) < 2:
            return 0.0
        
        # Compute average similarity across time
        common_times = []
        for t1, emb1 in timeline1:
            for t2, emb2 in timeline2:
                if t1 == t2 and t1 != timestamp:
                    sim = self._cosine_similarity(emb1, emb2)
                    common_times.append(sim)
        
        if not common_times:
            return 0.0
        
        # Bonus is proportional to consistency
        avg_similarity = np.mean(common_times)
        consistency_bonus = min(0.5, avg_similarity * 0.5)  # Max 50% bonus
        
        return consistency_bonus
    
    def compute_embedding_drift(self, word: Word, start_timestamp: Timestamp, end_timestamp: Timestamp) -> Optional[float]:
        """
        Compute semantic drift for a word between two timestamps.
        
        Returns the cosine distance between embeddings at start and end timestamps.
        """
        start_emb = self.get_embedding(word, start_timestamp)
        end_emb = self.get_embedding(word, end_timestamp)
        
        if start_emb is None or end_emb is None:
            return None
        
        # Cosine distance (1 - cosine similarity)
        similarity = self._cosine_similarity(start_emb, end_emb)
        drift = 1.0 - similarity
        
        return float(drift)
    
    def compute_trajectory_smoothness(self, word: Word) -> Optional[float]:
        """
        Compute smoothness of a word's trajectory over time.
        
        Returns average cosine similarity between consecutive time points.
        """
        timeline = self.get_word_timeline(word)
        
        if len(timeline) < 2:
            return None
        
        consecutive_similarities = []
        
        for i in range(len(timeline) - 1):
            _, emb1 = timeline[i]
            _, emb2 = timeline[i + 1]
            
            similarity = self._cosine_similarity(emb1, emb2)
            consecutive_similarities.append(similarity)
        
        return float(np.mean(consecutive_similarities))
    
    def detect_vocabulary_events(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect vocabulary events (word births, deaths, re-appearances).
        
        Returns:
            Dictionary with 'births', 'deaths', 'reappearances' lists
        """
        events = {
            'births': [],
            'deaths': [],
            'reappearances': []
        }
        
        if len(self.timestamps) < 2:
            return events
        
        # Track word presence across timestamps
        word_presence = defaultdict(list)  # word -> [(timestamp, present)]
        
        all_words = set()
        for vocab in self.vocabulary.values():
            all_words.update(vocab)
        
        for word in all_words:
            for timestamp in self.timestamps:
                present = word in self.get_vocabulary(timestamp)
                word_presence[word].append((timestamp, present))
        
        # Analyze each word's timeline
        for word, timeline in word_presence.items():
            prev_present = False
            word_born = False
            
            for timestamp, present in timeline:
                if present and not prev_present:
                    if not word_born:
                        # First appearance - birth
                        events['births'].append({
                            'word': word,
                            'timestamp': timestamp,
                            'event_type': 'birth'
                        })
                        word_born = True
                    else:
                        # Reappearance
                        events['reappearances'].append({
                            'word': word,
                            'timestamp': timestamp,
                            'event_type': 'reappearance'
                        })
                
                elif not present and prev_present:
                    # Disappearance - potential death
                    events['deaths'].append({
                        'word': word,
                        'timestamp': timestamp,
                        'event_type': 'death'
                    })
                
                prev_present = present
        
        return events
    
    def interpolate_embedding(self, word: Word, target_timestamp: Timestamp, method: str = 'linear') -> Optional[Embedding]:
        """
        Interpolate embedding for a word at a timestamp where it doesn't exist.
        
        Args:
            word: The word to interpolate
            target_timestamp: The target timestamp
            method: Interpolation method ('linear', 'spline', 'nearest')
            
        Returns:
            Interpolated embedding or None if not possible
        """
        timeline = self.get_word_timeline(word)
        
        if len(timeline) < 2:
            return None
        
        # Find bounding timestamps
        before_times = [(t, emb) for t, emb in timeline if t < target_timestamp]
        after_times = [(t, emb) for t, emb in timeline if t > target_timestamp]
        
        if not before_times and not after_times:
            return None
        
        if method == 'nearest':
            # Find nearest timestamp
            all_times = [(abs(target_timestamp - t), t, emb) for t, emb in timeline]
            all_times.sort()
            return all_times[0][2]
        
        elif method == 'linear' and before_times and after_times:
            # Linear interpolation between closest before and after
            before_time, before_emb = max(before_times, key=lambda x: x[0])
            after_time, after_emb = min(after_times, key=lambda x: x[0])
            
            # Interpolation weight
            total_span = after_time - before_time
            target_offset = target_timestamp - before_time
            weight = target_offset / total_span if total_span > 0 else 0.5
            
            # Linear interpolation
            interpolated = (1 - weight) * before_emb + weight * after_emb
            return interpolated
        
        elif before_times:
            # Only before times available - use latest
            return max(before_times, key=lambda x: x[0])[1]
        
        elif after_times:
            # Only after times available - use earliest
            return min(after_times, key=lambda x: x[0])[1]
        
        return None
    
    def compute_temporal_centrality(self, word: Word) -> Dict[str, float]:
        """
        Compute temporal centrality metrics for a word.
        
        Returns:
            Dictionary with centrality metrics across time
        """
        timeline = self.get_word_timeline(word)
        
        if not timeline:
            return {}
        
        centrality_metrics = {
            'temporal_span': 0.0,
            'persistence_ratio': 0.0,
            'average_similarity_to_center': 0.0,
            'temporal_stability': 0.0
        }
        
        # Temporal span
        if len(timeline) > 1:
            first_time = timeline[0][0]
            last_time = timeline[-1][0]
            total_possible_span = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 1
            actual_span = last_time - first_time
            centrality_metrics['temporal_span'] = actual_span / total_possible_span
        
        # Persistence ratio
        total_timestamps = len(self.timestamps)
        appearances = len(timeline)
        centrality_metrics['persistence_ratio'] = appearances / total_timestamps if total_timestamps > 0 else 0
        
        # Average similarity to vocabulary center at each timestamp
        center_similarities = []
        for timestamp, embedding in timeline:
            vocab = self.get_vocabulary(timestamp)
            vocab_embeddings = []
            
            for other_word in vocab:
                if other_word != word:
                    other_emb = self.get_embedding(other_word, timestamp)
                    if other_emb is not None:
                        vocab_embeddings.append(other_emb)
            
            if vocab_embeddings:
                # Compute center of vocabulary
                vocab_center = np.mean(vocab_embeddings, axis=0)
                center_sim = self._cosine_similarity(embedding, vocab_center)
                center_similarities.append(center_sim)
        
        if center_similarities:
            centrality_metrics['average_similarity_to_center'] = float(np.mean(center_similarities))
        
        # Temporal stability (smoothness of trajectory)
        stability = self.compute_trajectory_smoothness(word)
        if stability is not None:
            centrality_metrics['temporal_stability'] = stability
        
        return centrality_metrics
    
    def get_temporal_statistics(self) -> Dict[str, Any]:
        """Get comprehensive temporal statistics."""
        stats = self.get_statistics().copy()
        
        # Add temporal-specific statistics
        stats.update({
            'temporal_span': len(self.timestamps),
            'average_vocabulary_size': np.mean([len(vocab) for vocab in self.vocabulary.values()]) if self.vocabulary else 0,
            'vocabulary_stability': self.temporal_stats['vocabulary_stability'],
            'persistent_words_count': len(self.get_persistent_words()),
            'temporal_coverage': self._compute_temporal_coverage(),
            'embedding_dimension': self.embedding_dim
        })
        
        return stats
    
    def _compute_temporal_coverage(self) -> Dict[str, float]:
        """Compute temporal coverage statistics."""
        if not self.word_first_appearance or not self.word_last_appearance:
            return {}
        
        word_spans = []
        total_timeline = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 1
        
        for word in self.word_first_appearance:
            first = self.word_first_appearance[word]
            last = self.word_last_appearance[word]
            span = last - first
            coverage = span / total_timeline if total_timeline > 0 else 0
            word_spans.append(coverage)
        
        return {
            'average_word_coverage': float(np.mean(word_spans)) if word_spans else 0,
            'max_word_coverage': float(np.max(word_spans)) if word_spans else 0,
            'min_word_coverage': float(np.min(word_spans)) if word_spans else 0,
            'coverage_std': float(np.std(word_spans)) if word_spans else 0
        }
    
    def export_temporal_data(self) -> Dict[str, Any]:
        """Export temporal data for external analysis."""
        return {
            'embeddings': self.embeddings,
            'vocabulary': {k: list(v) for k, v in self.vocabulary.items()},
            'timestamps': self.timestamps,
            'word_first_appearance': self.word_first_appearance,
            'word_last_appearance': self.word_last_appearance,
            'temporal_stats': self.temporal_stats,
            'dimension': self.embedding_dim
        }