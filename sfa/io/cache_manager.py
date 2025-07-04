"""Cache management for Semantic Flow Analyzer."""

import json
import pickle
import hashlib
import time
from typing import Any, Dict, Optional, Set, Union, List
from pathlib import Path
from collections import OrderedDict
import threading
import logging

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..core.base import BaseCache

logger = logging.getLogger(__name__)

class MemoryCache(BaseCache):
    """
    In-memory cache with LRU eviction policy.
    """
    
    def __init__(self, max_size: int = 1000, ttl: Optional[int] = None):
        self.max_size = max_size
        self.default_ttl = ttl
        self._cache = OrderedDict()
        self._timestamps = {}
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                return None
            
            # Check TTL
            if self._is_expired(key):
                self.delete(key)
                return None
            
            # Move to end (most recently used)
            value = self._cache[key]
            self._cache.move_to_end(key)
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        with self._lock:
            # Remove if exists
            if key in self._cache:
                del self._cache[key]
            
            # Add new value
            self._cache[key] = value
            
            # Set timestamp
            if ttl is not None or self.default_ttl is not None:
                expiry_time = time.time() + (ttl or self.default_ttl)
                self._timestamps[key] = expiry_time
            
            # Evict if necessary
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                self.delete(oldest_key)
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._timestamps.pop(key, None)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            if key not in self._cache:
                return False
            
            if self._is_expired(key):
                self.delete(key)
                return False
            
            return True
    
    def _is_expired(self, key: str) -> bool:
        """Check if key is expired."""
        if key not in self._timestamps:
            return False
        
        return time.time() > self._timestamps[key]
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        with self._lock:
            expired_keys = [
                key for key in self._cache.keys()
                if self._is_expired(key)
            ]
            
            for key in expired_keys:
                self.delete(key)
            
            return len(expired_keys)

class DiskCache(BaseCache):
    """
    Disk-based cache with optional compression.
    """
    
    def __init__(self, 
                 cache_dir: str = "./cache",
                 max_size_mb: int = 1000,
                 use_compression: bool = True,
                 ttl: Optional[int] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_mb = max_size_mb
        self.use_compression = use_compression
        self.default_ttl = ttl
        
        # Metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._metadata = self._load_metadata()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            key_hash = self._hash_key(key)
            
            if key_hash not in self._metadata:
                return None
            
            metadata = self._metadata[key_hash]
            
            # Check TTL
            if self._is_expired(metadata):
                self.delete(key)
                return None
            
            # Load from disk
            try:
                file_path = self.cache_dir / f"{key_hash}.cache"
                if not file_path.exists():
                    # Metadata is stale
                    del self._metadata[key_hash]
                    self._save_metadata()
                    return None
                
                with open(file_path, 'rb') as f:
                    if self.use_compression:
                        import gzip
                        data = gzip.decompress(f.read())
                    else:
                        data = f.read()
                
                value = pickle.loads(data)
                
                # Update access time
                metadata['last_accessed'] = time.time()
                self._save_metadata()
                
                return value
                
            except Exception as e:
                logger.warning(f"Failed to load from disk cache: {e}")
                self.delete(key)
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        with self._lock:
            key_hash = self._hash_key(key)
            
            try:
                # Serialize value
                data = pickle.dumps(value)
                
                if self.use_compression:
                    import gzip
                    data = gzip.compress(data)
                
                # Save to disk
                file_path = self.cache_dir / f"{key_hash}.cache"
                with open(file_path, 'wb') as f:
                    f.write(data)
                
                # Update metadata
                metadata = {
                    'key': key,
                    'created': time.time(),
                    'last_accessed': time.time(),
                    'size_bytes': len(data),
                    'ttl': ttl or self.default_ttl
                }
                
                self._metadata[key_hash] = metadata
                self._save_metadata()
                
                # Cleanup if necessary
                self._cleanup_if_needed()
                
            except Exception as e:
                logger.error(f"Failed to save to disk cache: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self._lock:
            key_hash = self._hash_key(key)
            
            if key_hash not in self._metadata:
                return False
            
            # Remove file
            file_path = self.cache_dir / f"{key_hash}.cache"
            if file_path.exists():
                file_path.unlink()
            
            # Remove metadata
            del self._metadata[key_hash]
            self._save_metadata()
            
            return True
    
    def clear(self) -> None:
        """Clear all cache."""
        with self._lock:
            # Remove all cache files
            for file_path in self.cache_dir.glob("*.cache"):
                file_path.unlink()
            
            # Clear metadata
            self._metadata.clear()
            self._save_metadata()
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            key_hash = self._hash_key(key)
            
            if key_hash not in self._metadata:
                return False
            
            metadata = self._metadata[key_hash]
            
            if self._is_expired(metadata):
                self.delete(key)
                return False
            
            # Check if file exists
            file_path = self.cache_dir / f"{key_hash}.cache"
            if not file_path.exists():
                # Metadata is stale
                del self._metadata[key_hash]
                self._save_metadata()
                return False
            
            return True
    
    def _hash_key(self, key: str) -> str:
        """Generate hash for key."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _is_expired(self, metadata: Dict[str, Any]) -> bool:
        """Check if entry is expired."""
        if metadata.get('ttl') is None:
            return False
        
        created = metadata.get('created', 0)
        ttl = metadata.get('ttl', 0)
        
        return time.time() > (created + ttl)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        
        return {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _cleanup_if_needed(self) -> None:
        """Cleanup cache if size limit exceeded."""
        total_size_mb = sum(
            metadata.get('size_bytes', 0) 
            for metadata in self._metadata.values()
        ) / (1024 * 1024)
        
        if total_size_mb <= self.max_size_mb:
            return
        
        # Sort by last accessed time
        items = list(self._metadata.items())
        items.sort(key=lambda x: x[1].get('last_accessed', 0))
        
        # Remove oldest items
        removed_size = 0
        target_size = self.max_size_mb * 0.8  # Remove until 80% of limit
        
        for key_hash, metadata in items:
            if total_size_mb - (removed_size / (1024 * 1024)) <= target_size:
                break
            
            # Remove file
            file_path = self.cache_dir / f"{key_hash}.cache"
            if file_path.exists():
                file_path.unlink()
            
            removed_size += metadata.get('size_bytes', 0)
            del self._metadata[key_hash]
        
        if removed_size > 0:
            logger.info(f"Cleaned up {removed_size / (1024 * 1024):.2f} MB from disk cache")
            self._save_metadata()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size = sum(
                metadata.get('size_bytes', 0)
                for metadata in self._metadata.values()
            )
            
            return {
                'total_entries': len(self._metadata),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'cache_dir': str(self.cache_dir)
            }

class RedisCache(BaseCache):
    """
    Redis-based cache for distributed caching.
    """
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 key_prefix: str = 'sfa:',
                 ttl: Optional[int] = None):
        
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")
        
        self.key_prefix = key_prefix
        self.default_ttl = ttl
        
        # Create Redis connection
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False  # We handle binary data
        )
        
        # Test connection
        try:
            self.redis_client.ping()
        except redis.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            full_key = self.key_prefix + key
            data = self.redis_client.get(full_key)
            
            if data is None:
                return None
            
            return pickle.loads(data)
            
        except Exception as e:
            logger.warning(f"Failed to get from Redis cache: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        try:
            full_key = self.key_prefix + key
            data = pickle.dumps(value)
            
            if ttl is not None or self.default_ttl is not None:
                self.redis_client.setex(
                    full_key, 
                    ttl or self.default_ttl,
                    data
                )
            else:
                self.redis_client.set(full_key, data)
                
        except Exception as e:
            logger.error(f"Failed to set in Redis cache: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            full_key = self.key_prefix + key
            result = self.redis_client.delete(full_key)
            return result > 0
            
        except Exception as e:
            logger.warning(f"Failed to delete from Redis cache: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all cache."""
        try:
            # Get all keys with our prefix
            pattern = self.key_prefix + "*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                self.redis_client.delete(*keys)
                
        except Exception as e:
            logger.error(f"Failed to clear Redis cache: {e}")
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            full_key = self.key_prefix + key
            return self.redis_client.exists(full_key) > 0
            
        except Exception as e:
            logger.warning(f"Failed to check Redis cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = self.redis_client.info()
            
            # Count keys with our prefix
            pattern = self.key_prefix + "*"
            key_count = len(self.redis_client.keys(pattern))
            
            return {
                'total_entries': key_count,
                'redis_used_memory': info.get('used_memory', 0),
                'redis_used_memory_human': info.get('used_memory_human', '0B'),
                'connected_clients': info.get('connected_clients', 0)
            }
            
        except Exception as e:
            logger.warning(f"Failed to get Redis stats: {e}")
            return {}

class CacheManager:
    """
    Multi-tier cache manager with automatic fallback.
    
    Supports memory -> disk -> Redis cache hierarchy.
    """
    
    def __init__(self, 
                 use_memory: bool = True,
                 use_disk: bool = True,
                 use_redis: bool = False,
                 memory_config: Optional[Dict[str, Any]] = None,
                 disk_config: Optional[Dict[str, Any]] = None,
                 redis_config: Optional[Dict[str, Any]] = None):
        
        self.caches: List[BaseCache] = []
        
        # Initialize memory cache
        if use_memory:
            memory_config = memory_config or {}
            self.memory_cache = MemoryCache(**memory_config)
            self.caches.append(self.memory_cache)
        else:
            self.memory_cache = None
        
        # Initialize disk cache
        if use_disk:
            disk_config = disk_config or {}
            self.disk_cache = DiskCache(**disk_config)
            self.caches.append(self.disk_cache)
        else:
            self.disk_cache = None
        
        # Initialize Redis cache
        if use_redis and REDIS_AVAILABLE:
            redis_config = redis_config or {}
            try:
                self.redis_cache = RedisCache(**redis_config)
                self.caches.append(self.redis_cache)
            except (ImportError, ConnectionError) as e:
                logger.warning(f"Redis cache not available: {e}")
                self.redis_cache = None
        else:
            self.redis_cache = None
        
        if not self.caches:
            raise ValueError("At least one cache type must be enabled")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache hierarchy."""
        for i, cache in enumerate(self.caches):
            value = cache.get(key)
            if value is not None:
                # Populate higher-level caches
                for j in range(i):
                    self.caches[j].set(key, value)
                return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in all caches."""
        for cache in self.caches:
            cache.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete value from all caches."""
        deleted = False
        for cache in self.caches:
            if cache.delete(key):
                deleted = True
        return deleted
    
    def clear(self) -> None:
        """Clear all caches."""
        for cache in self.caches:
            cache.clear()
    
    def exists(self, key: str) -> bool:
        """Check if key exists in any cache."""
        for cache in self.caches:
            if cache.exists(key):
                return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all caches."""
        stats = {}
        
        if self.memory_cache:
            stats['memory'] = {
                'size': self.memory_cache.size(),
                'max_size': self.memory_cache.max_size
            }
        
        if self.disk_cache:
            stats['disk'] = self.disk_cache.get_stats()
        
        if self.redis_cache:
            stats['redis'] = self.redis_cache.get_stats()
        
        return stats
    
    def cleanup(self) -> Dict[str, int]:
        """Cleanup expired entries from all caches."""
        cleanup_stats = {}
        
        if self.memory_cache:
            cleanup_stats['memory'] = self.memory_cache.cleanup_expired()
        
        # Disk cache cleanup is automatic
        # Redis cleanup is automatic via TTL
        
        return cleanup_stats

# Convenience function to create cache manager
def create_cache_manager(cache_type: str = 'auto', **kwargs) -> CacheManager:
    """
    Create cache manager with sensible defaults.
    
    Args:
        cache_type: 'memory', 'disk', 'redis', 'auto'
        **kwargs: Additional configuration
    """
    if cache_type == 'memory':
        return CacheManager(use_memory=True, use_disk=False, use_redis=False, **kwargs)
    elif cache_type == 'disk':
        return CacheManager(use_memory=False, use_disk=True, use_redis=False, **kwargs)
    elif cache_type == 'redis':
        return CacheManager(use_memory=False, use_disk=False, use_redis=True, **kwargs)
    elif cache_type == 'auto':
        # Try Redis first, fallback to disk + memory
        try:
            return CacheManager(use_memory=True, use_disk=True, use_redis=True, **kwargs)
        except (ImportError, ConnectionError):
            return CacheManager(use_memory=True, use_disk=True, use_redis=False, **kwargs)
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")