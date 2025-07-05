"""HDF5-based storage system for efficient semantic embedding management."""

import h5py
import numpy as np
import logging
from typing import Dict, List, Optional, Set, Any, Union
from pathlib import Path
from collections import defaultdict
import threading
from functools import lru_cache

from .types import Word, Timestamp, Embedding
from .base import BaseAnalyzer

logger = logging.getLogger(__name__)

class HDF5EmbeddingStore(BaseAnalyzer):
    """
    High-performance HDF5-based storage for semantic embeddings with lazy loading.
    
    Optimized for the gravitational field approach where metrics compute on full
    high-dimensional embeddings while visualization uses UMAP reduction separately.
    """
    
    def __init__(self, h5_path: str, mode: str = 'r', 
                 cache_size: int = 10000, embedding_dim: int = 768):
        super().__init__("HDF5EmbeddingStore")
        self.h5_path = Path(h5_path)
        self.mode = mode
        self.embedding_dim = embedding_dim
        self.cache_size = cache_size
        
        # Thread-safe caching
        self._lock = threading.RLock()
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Metadata tracking
        self._timestamps: Set[str] = set()
        self._vocabulary: Set[str] = set()
        self._metadata_loaded = False
        
        # Initialize storage
        self._init_storage()
    
    def _init_storage(self) -> None:
        """Initialize HDF5 file structure."""
        if self.mode in ['w', 'a'] and not self.h5_path.exists():
            self.h5_path.parent.mkdir(parents=True, exist_ok=True)
            
            with h5py.File(self.h5_path, 'w') as f:
                # Create groups
                f.create_group('embeddings')
                f.create_group('metadata')
                
                # Store configuration
                f.attrs['embedding_dim'] = self.embedding_dim
                f.attrs['version'] = '2.0'
                f.attrs['description'] = 'Semantic Flow Analyzer - Gravitational Field Storage'
        
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load vocabulary and timestamp metadata."""
        if self._metadata_loaded:
            return
            
        try:
            with h5py.File(self.h5_path, 'r') as f:
                # Load timestamps
                if 'embeddings' in f:
                    self._timestamps = set(f['embeddings'].keys())
                    
                    # Load vocabulary from first timestamp
                    if self._timestamps:
                        first_timestamp = next(iter(self._timestamps))
                        if first_timestamp in f['embeddings']:
                            self._vocabulary = set(f['embeddings'][first_timestamp].keys())
                
                logger.info(f"Loaded metadata: {len(self._timestamps)} timestamps, "
                           f"{len(self._vocabulary)} words")
                
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            
        self._metadata_loaded = True
    
    @lru_cache(maxsize=1000)
    def get_embedding(self, word: str, timestamp: str) -> Optional[np.ndarray]:
        """
        Get embedding for a specific word at a timestamp.
        
        Uses LRU cache for frequently accessed embeddings.
        """
        cache_key = f"{word}:{timestamp}"
        
        with self._lock:
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key].copy()
            
            self._cache_misses += 1
        
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if ('embeddings' in f and 
                    timestamp in f['embeddings'] and 
                    word in f['embeddings'][timestamp]):
                    
                    embedding = f['embeddings'][timestamp][word][:]
                    
                    # Cache the result
                    with self._lock:
                        if len(self._cache) >= self.cache_size:
                            # Remove oldest entries (simple FIFO)
                            oldest_key = next(iter(self._cache))
                            del self._cache[oldest_key]
                        
                        self._cache[cache_key] = embedding.copy()
                    
                    return embedding
                    
        except Exception as e:
            logger.warning(f"Failed to load embedding for {word}@{timestamp}: {e}")
        
        return None
    
    def get_embeddings_batch(self, words: List[str], timestamp: str, 
                           return_missing: bool = False) -> Dict[str, np.ndarray]:
        """
        Efficiently load multiple embeddings for the same timestamp.
        
        Optimized for gravitational field computation where we need
        k-nearest neighbors for a focal word.
        """
        results = {}
        missing_words = []
        
        # Check cache first
        with self._lock:
            for word in words:
                cache_key = f"{word}:{timestamp}"
                if cache_key in self._cache:
                    results[word] = self._cache[cache_key].copy()
                    self._cache_hits += 1
                else:
                    missing_words.append(word)
                    self._cache_misses += 1
        
        # Load missing words from disk
        if missing_words:
            try:
                with h5py.File(self.h5_path, 'r') as f:
                    if ('embeddings' in f and timestamp in f['embeddings']):
                        timestamp_group = f['embeddings'][timestamp]
                        
                        for word in missing_words:
                            if word in timestamp_group:
                                embedding = timestamp_group[word][:]
                                results[word] = embedding
                                
                                # Cache the result
                                with self._lock:
                                    cache_key = f"{word}:{timestamp}"
                                    if len(self._cache) < self.cache_size:
                                        self._cache[cache_key] = embedding.copy()
                            elif return_missing:
                                results[word] = None
                                
            except Exception as e:
                logger.warning(f"Failed to batch load embeddings: {e}")
        
        return results
    
    def store_embedding(self, word: str, timestamp: str, 
                       embedding: np.ndarray) -> None:
        """Store embedding in HDF5 file."""
        if self.mode == 'r':
            raise ValueError("Cannot store in read-only mode")
        
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Expected embedding dim {self.embedding_dim}, "
                           f"got {embedding.shape[0]}")
        
        try:
            with h5py.File(self.h5_path, 'a') as f:
                # Create timestamp group if needed
                if timestamp not in f['embeddings']:
                    f['embeddings'].create_group(timestamp)
                
                # Store embedding
                timestamp_group = f['embeddings'][timestamp]
                if word in timestamp_group:
                    del timestamp_group[word]  # Overwrite
                
                timestamp_group.create_dataset(
                    word, data=embedding, 
                    dtype=np.float32,
                    compression='gzip', 
                    compression_opts=9
                )
            
            # Update metadata
            self._timestamps.add(timestamp)
            self._vocabulary.add(word)
            
            # Update cache
            cache_key = f"{word}:{timestamp}"
            with self._lock:
                self._cache[cache_key] = embedding.copy()
            
        except Exception as e:
            logger.error(f"Failed to store embedding for {word}@{timestamp}: {e}")
            raise
    
    def get_timestamps(self) -> List[str]:
        """Get all available timestamps, sorted."""
        self._load_metadata()
        return sorted(list(self._timestamps))
    
    def get_vocabulary(self, timestamp: Optional[str] = None) -> List[str]:
        """Get vocabulary for a specific timestamp or all words."""
        if timestamp is None:
            self._load_metadata()
            return sorted(list(self._vocabulary))
        
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if ('embeddings' in f and timestamp in f['embeddings']):
                    return sorted(list(f['embeddings'][timestamp].keys()))
        except Exception as e:
            logger.warning(f"Failed to get vocabulary for {timestamp}: {e}")
        
        return []
    
    def get_temporal_embedding_matrix(self, words: List[str], 
                                    timestamps: List[str]) -> np.ndarray:
        """
        Get embedding matrix for gravitational field computation.
        
        Returns: shape (len(words), len(timestamps), embedding_dim)
        Missing embeddings are filled with zeros.
        """
        matrix = np.zeros((len(words), len(timestamps), self.embedding_dim))
        
        for t_idx, timestamp in enumerate(timestamps):
            embeddings = self.get_embeddings_batch(words, timestamp)
            
            for w_idx, word in enumerate(words):
                if word in embeddings:
                    matrix[w_idx, t_idx, :] = embeddings[word]
        
        return matrix
    
    def compute_gravitational_field_data(self, focal_word: str, 
                                       k_neighbors: int = 50,
                                       timestamps: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compute data needed for gravitational field visualization.
        
        This method efficiently prepares data for the gravitational field engine
        by computing high-dimensional similarities and flows.
        """
        if timestamps is None:
            timestamps = self.get_timestamps()
        
        # Get focal word trajectory
        focal_embeddings = []
        for timestamp in timestamps:
            emb = self.get_embedding(focal_word, timestamp)
            if emb is not None:
                focal_embeddings.append(emb)
            else:
                # Use zero vector for missing timestamps
                focal_embeddings.append(np.zeros(self.embedding_dim))
        
        focal_matrix = np.array(focal_embeddings)  # (T, D)
        
        # Find k-nearest neighbors across all timestamps
        all_similarities = defaultdict(list)
        neighbor_words = set()
        
        for t_idx, timestamp in enumerate(timestamps):
            if focal_embeddings[t_idx] is not None:
                vocab = self.get_vocabulary(timestamp)
                embeddings = self.get_embeddings_batch(vocab[:500], timestamp)  # Limit for efficiency
                
                # Compute similarities
                similarities = []
                words = []
                
                for word, embedding in embeddings.items():
                    if word != focal_word and embedding is not None:
                        sim = np.dot(focal_embeddings[t_idx], embedding) / (
                            np.linalg.norm(focal_embeddings[t_idx]) * np.linalg.norm(embedding)
                        )
                        similarities.append(sim)
                        words.append(word)
                
                # Get top-k neighbors
                if similarities:
                    top_indices = np.argsort(similarities)[-k_neighbors:]
                    for idx in top_indices:
                        word = words[idx]
                        neighbor_words.add(word)
                        all_similarities[word].append((t_idx, similarities[idx]))
        
        # Get embedding trajectories for neighbor words
        neighbor_list = list(neighbor_words)
        neighbor_matrix = self.get_temporal_embedding_matrix(neighbor_list, timestamps)
        
        return {
            'focal_word': focal_word,
            'focal_embeddings': focal_matrix,
            'neighbor_words': neighbor_list,
            'neighbor_embeddings': neighbor_matrix,
            'similarities': dict(all_similarities),
            'timestamps': timestamps,
            'embedding_dim': self.embedding_dim
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics."""
        with self._lock:
            total_requests = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
            
            return {
                'cache_size': len(self._cache),
                'max_cache_size': self.cache_size,
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        with self._lock:
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            with h5py.File(self.h5_path, 'r') as f:
                file_size = self.h5_path.stat().st_size
                
                return {
                    'file_path': str(self.h5_path),
                    'file_size_mb': file_size / (1024 * 1024),
                    'num_timestamps': len(self._timestamps),
                    'num_words': len(self._vocabulary),
                    'embedding_dim': self.embedding_dim,
                    'total_embeddings': len(self._timestamps) * len(self._vocabulary),
                    'cache_stats': self.get_cache_stats()
                }
        except Exception as e:
            logger.warning(f"Failed to get storage stats: {e}")
            return {'error': str(e)}

    def close(self) -> None:
        """Close storage and clean up resources."""
        self.clear_cache()
        logger.info(f"Closed HDF5 storage: {self.h5_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()