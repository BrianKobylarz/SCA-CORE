"""Reddit data loader for Semantic Flow Analyzer."""

import praw
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Union, Any, Tuple
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import pickle
import time

from ..core.types import Word, Timestamp, Embedding, EmbeddingMatrix
from ..core.base import BaseDataLoader
from ..core.embeddings import TemporalEmbeddingStore

logger = logging.getLogger(__name__)

class RedditDataLoader(BaseDataLoader):
    """
    Loads and processes Reddit data for semantic flow analysis.
    
    Supports loading from:
    - Reddit API (via PRAW)
    - Preprocessed embedding files
    - Cached data files
    """
    
    def __init__(self, 
                 client_id: Optional[str] = None,
                 client_secret: Optional[str] = None,
                 user_agent: Optional[str] = None,
                 cache_dir: Optional[str] = "./reddit_cache"):
        super().__init__("RedditDataLoader")
        
        # Reddit API credentials
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent or "SemanticFlowAnalyzer/1.0"
        
        # Cache configuration
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Reddit API client
        self._reddit_client = None
        
        # Data processing settings
        self.min_word_frequency = 5
        self.max_vocabulary_size = 10000
        self.embedding_dimension = 384  # Default for sentence-transformers
        
    def configure(self, **config) -> None:
        """Configure the data loader."""
        super().configure(**config)
        
        if 'min_word_frequency' in config:
            self.min_word_frequency = config['min_word_frequency']
        if 'max_vocabulary_size' in config:
            self.max_vocabulary_size = config['max_vocabulary_size']
        if 'embedding_dimension' in config:
            self.embedding_dimension = config['embedding_dimension']
    
    def _get_reddit_client(self) -> praw.Reddit:
        """Get authenticated Reddit client."""
        if self._reddit_client is None:
            if not all([self.client_id, self.client_secret]):
                raise ValueError("Reddit API credentials required")
            
            self._reddit_client = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )
        
        return self._reddit_client
    
    def load_from_subreddit(self,
                           subreddit_name: str,
                           start_date: str,
                           end_date: str,
                           time_granularity: str = 'monthly',
                           max_posts: int = 1000) -> TemporalEmbeddingStore:
        """
        Load Reddit data from a specific subreddit.
        
        Args:
            subreddit_name: Name of the subreddit
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            time_granularity: 'daily', 'weekly', or 'monthly'
            max_posts: Maximum posts to fetch per time period
            
        Returns:
            TemporalEmbeddingStore with loaded data
        """
        logger.info(f"Loading data from r/{subreddit_name}")
        
        # Check cache first
        cache_key = f"{subreddit_name}_{start_date}_{end_date}_{time_granularity}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            logger.info("Loaded data from cache")
            return cached_data
        
        # Generate time periods
        time_periods = self._generate_time_periods(start_date, end_date, time_granularity)
        
        # Initialize embedding store
        embedding_store = TemporalEmbeddingStore(self.embedding_dimension)
        
        # Load data for each time period
        reddit = self._get_reddit_client()
        subreddit = reddit.subreddit(subreddit_name)
        
        for timestamp, (period_start, period_end) in time_periods.items():
            logger.info(f"Processing period {timestamp}")
            
            try:
                # Fetch posts for this period
                posts_data = self._fetch_posts_for_period(
                    subreddit, period_start, period_end, max_posts
                )
                
                if not posts_data:
                    logger.warning(f"No posts found for period {timestamp}")
                    continue
                
                # Process text and generate embeddings
                embeddings = self._process_posts_to_embeddings(posts_data, timestamp)
                
                # Store embeddings
                for word, embedding in embeddings.items():
                    embedding_store.store_embedding(word, timestamp, embedding)
                
                logger.info(f"Stored {len(embeddings)} embeddings for {timestamp}")
                
            except Exception as e:
                logger.error(f"Error processing period {timestamp}: {e}")
                continue
        
        # Cache the results
        self._save_to_cache(cache_key, embedding_store)
        
        return embedding_store
    
    def load_from_file(self, file_path: str, format: str = 'auto') -> TemporalEmbeddingStore:
        """
        Load preprocessed embedding data from file.
        
        Args:
            file_path: Path to the embedding file
            format: File format ('json', 'pickle', 'h5', 'auto')
            
        Returns:
            TemporalEmbeddingStore with loaded data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Auto-detect format
        if format == 'auto':
            format = file_path.suffix.lower().lstrip('.')
        
        logger.info(f"Loading embeddings from {file_path} (format: {format})")
        
        if format == 'json':
            return self._load_from_json(file_path)
        elif format == 'pickle':
            return self._load_from_pickle(file_path)
        elif format in ['h5', 'hdf5']:
            return self._load_from_h5(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load(self, source: str, **kwargs) -> TemporalEmbeddingStore:
        """
        Load data from various sources.
        
        Args:
            source: Data source ('subreddit', 'file', or file path)
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            TemporalEmbeddingStore with loaded data
        """
        if source == 'subreddit':
            return self.load_from_subreddit(**kwargs)
        elif source == 'file' or Path(source).exists():
            file_path = kwargs.get('file_path', source)
            return self.load_from_file(file_path, kwargs.get('format', 'auto'))
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def validate(self, data: TemporalEmbeddingStore) -> List[str]:
        """Validate loaded data."""
        issues = []
        
        timestamps = data.get_timestamps()
        if len(timestamps) < 2:
            issues.append("Need at least 2 timestamps for flow analysis")
        
        # Check embedding dimensions
        for timestamp in timestamps:
            vocab = data.get_vocabulary(timestamp)
            if not vocab:
                issues.append(f"Empty vocabulary for timestamp {timestamp}")
                continue
            
            sample_word = next(iter(vocab))
            embedding = data.get_embedding(sample_word, timestamp)
            if embedding is None:
                issues.append(f"Missing embedding for word {sample_word} at {timestamp}")
            elif len(embedding) != self.embedding_dimension:
                issues.append(f"Dimension mismatch at {timestamp}: expected {self.embedding_dimension}, got {len(embedding)}")
        
        # Check vocabulary overlap
        if len(timestamps) >= 2:
            common_vocab = data.get_common_vocabulary(timestamps)
            if len(common_vocab) < 10:
                issues.append(f"Low vocabulary overlap: only {len(common_vocab)} common words")
        
        return issues
    
    def _generate_time_periods(self, start_date: str, end_date: str, granularity: str) -> Dict[str, Tuple[datetime, datetime]]:
        """Generate time periods for data collection."""
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        periods = {}
        current = start_dt
        
        if granularity == 'daily':
            delta = timedelta(days=1)
            date_format = '%Y-%m-%d'
        elif granularity == 'weekly':
            delta = timedelta(weeks=1)
            date_format = '%Y-W%U'
        elif granularity == 'monthly':
            delta = timedelta(days=30)  # Approximate
            date_format = '%Y-%m'
        else:
            raise ValueError(f"Unknown granularity: {granularity}")
        
        while current < end_dt:
            period_end = min(current + delta, end_dt)
            timestamp = current.strftime(date_format)
            periods[timestamp] = (current, period_end)
            current = period_end
        
        return periods
    
    def _fetch_posts_for_period(self, subreddit, start_time: datetime, end_time: datetime, max_posts: int) -> List[Dict[str, Any]]:
        """Fetch Reddit posts for a specific time period."""
        posts_data = []
        
        try:
            # Convert to Unix timestamps
            start_timestamp = int(start_time.timestamp())
            end_timestamp = int(end_time.timestamp())
            
            # Use pushshift API simulation (simplified)
            # In practice, you'd use the actual pushshift API or Reddit's search
            submissions = subreddit.new(limit=max_posts)
            
            for submission in submissions:
                if start_timestamp <= submission.created_utc <= end_timestamp:
                    posts_data.append({
                        'id': submission.id,
                        'title': submission.title,
                        'selftext': submission.selftext if hasattr(submission, 'selftext') else '',
                        'score': submission.score,
                        'created_utc': submission.created_utc,
                        'num_comments': submission.num_comments
                    })
                    
                # Rate limiting
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error fetching posts: {e}")
        
        return posts_data
    
    def _process_posts_to_embeddings(self, posts_data: List[Dict[str, Any]], timestamp: str) -> Dict[Word, Embedding]:
        """Process posts into word embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error("sentence-transformers not installed. Using dummy embeddings.")
            return self._create_dummy_embeddings(posts_data, timestamp)
        
        # Initialize sentence transformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Extract and process text
        all_text = []
        for post in posts_data:
            text = f"{post['title']} {post['selftext']}"
            all_text.append(text.lower())
        
        if not all_text:
            return {}
        
        # Extract vocabulary
        vocabulary = self._extract_vocabulary(all_text)
        
        # Generate embeddings for vocabulary
        embeddings = {}
        for word in vocabulary:
            try:
                # Generate embedding for word in context
                word_embedding = model.encode([word])[0]
                embeddings[word] = np.array(word_embedding, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Error generating embedding for word '{word}': {e}")
        
        return embeddings
    
    def _extract_vocabulary(self, texts: List[str]) -> Set[Word]:
        """Extract vocabulary from texts."""
        import re
        from collections import Counter
        
        # Simple tokenization and vocabulary extraction
        all_words = []
        for text in texts:
            # Basic preprocessing
            text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            words = text.split()
            all_words.extend([w for w in words if len(w) >= 3])  # Filter short words
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Filter by frequency and limit vocabulary size
        vocabulary = {
            word for word, count in word_counts.most_common(self.max_vocabulary_size)
            if count >= self.min_word_frequency
        }
        
        return vocabulary
    
    def _create_dummy_embeddings(self, posts_data: List[Dict[str, Any]], timestamp: str) -> Dict[Word, Embedding]:
        """Create dummy embeddings when sentence-transformers is not available."""
        logger.warning("Creating dummy embeddings - install sentence-transformers for real embeddings")
        
        # Extract simple vocabulary
        all_text = ' '.join([f"{post['title']} {post['selftext']}" for post in posts_data])
        words = set(all_text.lower().split())
        
        # Create random embeddings
        embeddings = {}
        for word in list(words)[:100]:  # Limit to 100 words
            embedding = np.random.randn(self.embedding_dimension).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            embeddings[word] = embedding
        
        return embeddings
    
    def _load_from_cache(self, cache_key: str) -> Optional[TemporalEmbeddingStore]:
        """Load data from cache."""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pickle"
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: TemporalEmbeddingStore) -> None:
        """Save data to cache."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.pickle"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved data to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def _load_from_json(self, file_path: Path) -> TemporalEmbeddingStore:
        """Load embeddings from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Determine embedding dimension
        first_timestamp = next(iter(data.keys()))
        first_word = next(iter(data[first_timestamp].keys()))
        embedding_dim = len(data[first_timestamp][first_word])
        
        # Create embedding store
        embedding_store = TemporalEmbeddingStore(embedding_dim)
        
        # Load embeddings
        for timestamp, word_embeddings in data.items():
            for word, embedding in word_embeddings.items():
                embedding_array = np.array(embedding, dtype=np.float32)
                embedding_store.store_embedding(word, timestamp, embedding_array)
        
        return embedding_store
    
    def _load_from_pickle(self, file_path: Path) -> TemporalEmbeddingStore:
        """Load embeddings from pickle file."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, TemporalEmbeddingStore):
            return data
        elif isinstance(data, dict):
            # Convert dict format to TemporalEmbeddingStore
            return self._convert_dict_to_store(data)
        else:
            raise ValueError(f"Unsupported pickle data type: {type(data)}")
    
    def _load_from_h5(self, file_path: Path) -> TemporalEmbeddingStore:
        """Load embeddings from HDF5 file."""
        import h5py
        
        with h5py.File(file_path, 'r') as f:
            # Get metadata
            embedding_dim = f.attrs.get('embedding_dimension', 384)
            
            # Create embedding store
            embedding_store = TemporalEmbeddingStore(embedding_dim)
            
            # Load embeddings
            embeddings_group = f['embeddings']
            for timestamp in embeddings_group.keys():
                timestamp_group = embeddings_group[timestamp]
                
                words = [w.decode('utf-8') for w in timestamp_group['words'][:]]
                embeddings = timestamp_group['embeddings'][:]
                
                for word, embedding in zip(words, embeddings):
                    embedding_store.store_embedding(word, timestamp, embedding)
        
        return embedding_store
    
    def _convert_dict_to_store(self, data: dict) -> TemporalEmbeddingStore:
        """Convert dictionary format to TemporalEmbeddingStore."""
        if 'embeddings' in data:
            # Handle nested format
            embeddings_data = data['embeddings']
        else:
            embeddings_data = data
        
        # Determine embedding dimension
        first_timestamp = next(iter(embeddings_data.keys()))
        first_word = next(iter(embeddings_data[first_timestamp].keys()))
        embedding_dim = len(embeddings_data[first_timestamp][first_word])
        
        # Create embedding store
        embedding_store = TemporalEmbeddingStore(embedding_dim)
        
        # Load embeddings
        for timestamp, word_embeddings in embeddings_data.items():
            for word, embedding in word_embeddings.items():
                if isinstance(embedding, list):
                    embedding = np.array(embedding, dtype=np.float32)
                embedding_store.store_embedding(word, timestamp, embedding)
        
        return embedding_store

class RedditDataProcessor:
    """Additional utilities for processing Reddit data."""
    
    @staticmethod
    def filter_by_subreddit_activity(embedding_store: TemporalEmbeddingStore, 
                                   min_activity_threshold: float = 0.1) -> TemporalEmbeddingStore:
        """Filter embeddings to keep only active words."""
        timestamps = embedding_store.get_timestamps()
        if len(timestamps) < 2:
            return embedding_store
        
        # Calculate activity scores
        word_activity = {}
        all_words = set()
        
        for timestamp in timestamps:
            vocab = embedding_store.get_vocabulary(timestamp)
            all_words.update(vocab)
        
        for word in all_words:
            appearances = sum(1 for ts in timestamps if word in embedding_store.get_vocabulary(ts))
            activity_score = appearances / len(timestamps)
            word_activity[word] = activity_score
        
        # Create filtered store
        filtered_store = TemporalEmbeddingStore(embedding_store.embedding_dim)
        
        for timestamp in timestamps:
            vocab = embedding_store.get_vocabulary(timestamp)
            for word in vocab:
                if word_activity.get(word, 0) >= min_activity_threshold:
                    embedding = embedding_store.get_embedding(word, timestamp)
                    if embedding is not None:
                        filtered_store.store_embedding(word, timestamp, embedding)
        
        return filtered_store
    
    @staticmethod
    def aggregate_subreddits(stores: List[TemporalEmbeddingStore]) -> TemporalEmbeddingStore:
        """Aggregate multiple subreddit embedding stores."""
        if not stores:
            raise ValueError("No stores provided")
        
        # Use dimension from first store
        embedding_dim = stores[0].embedding_dim
        aggregated_store = TemporalEmbeddingStore(embedding_dim)
        
        # Get all timestamps
        all_timestamps = set()
        for store in stores:
            all_timestamps.update(store.get_timestamps())
        
        # Aggregate embeddings by averaging
        for timestamp in sorted(all_timestamps):
            word_embeddings = {}
            
            for store in stores:
                vocab = store.get_vocabulary(timestamp)
                for word in vocab:
                    embedding = store.get_embedding(word, timestamp)
                    if embedding is not None:
                        if word not in word_embeddings:
                            word_embeddings[word] = []
                        word_embeddings[word].append(embedding)
            
            # Average embeddings for each word
            for word, embeddings_list in word_embeddings.items():
                if embeddings_list:
                    avg_embedding = np.mean(embeddings_list, axis=0)
                    aggregated_store.store_embedding(word, timestamp, avg_embedding)
        
        return aggregated_store