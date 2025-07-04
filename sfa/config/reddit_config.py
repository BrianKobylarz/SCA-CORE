"""Reddit-specific configuration for data handling."""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta

@dataclass
class RedditAPIConfig:
    """Reddit API configuration."""
    client_id: str = os.getenv('REDDIT_CLIENT_ID', '')
    client_secret: str = os.getenv('REDDIT_CLIENT_SECRET', '')
    user_agent: str = os.getenv('REDDIT_USER_AGENT', 'semantic_flow_analyzer_v0.1.0')
    
    # Rate limiting
    requests_per_minute: int = 60
    requests_per_hour: int = 3600
    retry_attempts: int = 3
    retry_delay: int = 5
    
    # Timeout settings
    request_timeout: int = 30
    read_timeout: int = 60

@dataclass
class RedditDataConfig:
    """Reddit data collection configuration."""
    
    # Subreddit settings
    target_subreddits: List[str] = field(default_factory=lambda: [
        'politics', 'worldnews', 'news', 'science', 'technology'
    ])
    
    # Time range
    start_date: str = '2020-01-01'
    end_date: str = '2024-01-01'
    temporal_resolution: str = 'daily'  # 'daily', 'weekly', 'monthly'
    
    # Content filtering
    min_score: int = 5
    min_comments: int = 2
    max_post_age_days: int = 365
    
    # Language filtering
    target_languages: List[str] = field(default_factory=lambda: ['en'])
    language_detection_threshold: float = 0.8
    
    # Content types
    include_posts: bool = True
    include_comments: bool = True
    include_comment_replies: bool = False
    max_comment_depth: int = 3
    
    # Text preprocessing
    min_text_length: int = 10
    max_text_length: int = 10000
    remove_deleted: bool = True
    remove_removed: bool = True
    
    # Sampling
    sample_size: Optional[int] = None
    sampling_strategy: str = 'random'  # 'random', 'top', 'controversial'

@dataclass
class RedditProcessingConfig:
    """Reddit data processing configuration."""
    
    # Text cleaning
    remove_urls: bool = True
    remove_usernames: bool = True
    remove_subreddit_mentions: bool = True
    normalize_whitespace: bool = True
    
    # Aggregation
    aggregation_level: str = 'daily'  # 'hourly', 'daily', 'weekly'
    vocabulary_size: int = 50000
    min_word_frequency: int = 5
    
    # Temporal alignment
    window_size: int = 7
    overlap_size: int = 3
    alignment_method: str = 'sliding'  # 'sliding', 'fixed'
    
    # Embedding generation
    embedding_batch_size: int = 1000
    embedding_workers: int = 4
    cache_embeddings: bool = True
    
    # Quality control
    detect_spam: bool = True
    detect_bots: bool = True
    spam_threshold: float = 0.8
    bot_threshold: float = 0.9

@dataclass
class RedditEventConfig:
    """Reddit event detection configuration."""
    
    # Event types to track
    political_events: bool = True
    cultural_events: bool = True
    breaking_news: bool = True
    viral_content: bool = True
    
    # Detection parameters
    burst_detection_window: int = 3
    burst_magnitude_threshold: float = 2.0
    viral_threshold: int = 1000
    
    # Event correlation
    cross_subreddit_correlation: bool = True
    temporal_correlation_window: int = 24  # hours
    
    # Known events (for validation)
    known_events: Dict[str, str] = field(default_factory=lambda: {
        '2020-03-11': 'COVID-19 WHO pandemic declaration',
        '2020-11-03': 'US Election Day',
        '2021-01-06': 'US Capitol riots',
        '2021-01-20': 'Biden inauguration',
        '2022-02-24': 'Russia invades Ukraine',
        '2022-11-08': 'US Midterm elections'
    })

@dataclass
class RedditSemanticConfig:
    """Reddit semantic analysis configuration."""
    
    # Subreddit-specific analysis
    subreddit_embeddings: bool = True
    cross_subreddit_flows: bool = True
    subreddit_clustering: bool = True
    
    # Community detection
    community_detection_method: str = 'louvain'
    community_resolution: float = 1.0
    min_community_size: int = 10
    
    # Semantic bridges
    detect_semantic_bridges: bool = True
    bridge_threshold: float = 0.5
    bridge_persistence: int = 3
    
    # Influence tracking
    track_user_influence: bool = True
    influence_metric: str = 'pagerank'
    influence_decay: float = 0.9
    
    # Lexical innovation
    detect_new_words: bool = True
    new_word_threshold: float = 0.8
    track_word_adoption: bool = True

@dataclass
class RedditConfig:
    """Main Reddit configuration combining all settings."""
    
    api: RedditAPIConfig = field(default_factory=RedditAPIConfig)
    data: RedditDataConfig = field(default_factory=RedditDataConfig)
    processing: RedditProcessingConfig = field(default_factory=RedditProcessingConfig)
    events: RedditEventConfig = field(default_factory=RedditEventConfig)
    semantic: RedditSemanticConfig = field(default_factory=RedditSemanticConfig)
    
    def validate(self) -> List[str]:
        """Validate Reddit configuration."""
        errors = []
        
        # Validate API credentials
        if not self.api.client_id:
            errors.append("Reddit client ID is required")
        if not self.api.client_secret:
            errors.append("Reddit client secret is required")
        
        # Validate date range
        try:
            start = datetime.strptime(self.data.start_date, '%Y-%m-%d')
            end = datetime.strptime(self.data.end_date, '%Y-%m-%d')
            if start >= end:
                errors.append("Start date must be before end date")
        except ValueError:
            errors.append("Invalid date format (use YYYY-MM-DD)")
        
        # Validate subreddits
        if not self.data.target_subreddits:
            errors.append("At least one target subreddit is required")
        
        # Validate thresholds
        if self.data.min_score < 0:
            errors.append("Minimum score must be non-negative")
        if self.data.min_comments < 0:
            errors.append("Minimum comments must be non-negative")
        
        return errors
    
    def get_date_range(self) -> List[str]:
        """Get list of dates in the specified range."""
        start = datetime.strptime(self.data.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.data.end_date, '%Y-%m-%d')
        
        dates = []
        current = start
        
        if self.data.temporal_resolution == 'daily':
            delta = timedelta(days=1)
        elif self.data.temporal_resolution == 'weekly':
            delta = timedelta(weeks=1)
        elif self.data.temporal_resolution == 'monthly':
            delta = timedelta(days=30)
        else:
            delta = timedelta(days=1)
        
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += delta
        
        return dates

# Default Reddit configuration
default_reddit_config = RedditConfig()