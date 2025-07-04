"""I/O modules for data loading, export, and embedding alignment."""

from .reddit_loader import RedditDataLoader, RedditDataProcessor
from .cache_manager import (
    CacheManager, MemoryCache, DiskCache, RedisCache, create_cache_manager
)

# Import other modules if they exist
try:
    from .export_manager import ExportManager
except ImportError:
    ExportManager = None

try:
    from .embedding_alignment import EmbeddingAlignment
except ImportError:
    EmbeddingAlignment = None

__all__ = [
    'RedditDataLoader', 'RedditDataProcessor',
    'CacheManager', 'MemoryCache', 'DiskCache', 'RedisCache', 'create_cache_manager'
]

# Add optional imports to __all__ if they exist
if ExportManager is not None:
    __all__.append('ExportManager')
if EmbeddingAlignment is not None:
    __all__.append('EmbeddingAlignment')