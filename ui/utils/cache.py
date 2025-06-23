"""
Advanced Cache Manager for Bloomberg Terminal
High-performance caching with TTL, compression, and intelligent invalidation.
"""

import time
import pickle
import gzip
import hashlib
from typing import Any, Dict, Optional, Union, List
from datetime import datetime, timedelta
import logging
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)

class CacheManager:
    """High-performance cache manager with TTL and LRU eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with TTL check."""
        with self._lock:
            self._stats['total_requests'] += 1
            
            if key in self._cache:
                entry = self._cache[key]
                
                # Check TTL
                if self._is_expired(entry):
                    del self._cache[key]
                    self._stats['misses'] += 1
                    return default
                
                # Move to end (LRU)
                self._cache.move_to_end(key)
                self._stats['hits'] += 1
                
                # Decompress if needed
                return self._decompress_value(entry['value'])
            
            self._stats['misses'] += 1
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            compress: bool = False) -> None:
        """Set value in cache with optional TTL and compression."""
        with self._lock:
            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl
            
            # Compress large values
            if compress or self._should_compress(value):
                compressed_value = self._compress_value(value)
            else:
                compressed_value = value
            
            # Create cache entry
            entry = {
                'value': compressed_value,
                'timestamp': time.time(),
                'ttl': ttl,
                'compressed': compress or self._should_compress(value),
                'size': self._estimate_size(value)
            }
            
            # Add to cache
            self._cache[key] = entry
            self._cache.move_to_end(key)
            
            # Evict if necessary
            self._evict_if_needed()
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._reset_stats()
    
    def clear_expired(self) -> int:
        """Clear all expired entries and return count."""
        with self._lock:
            expired_keys = []
            
            for key, entry in self._cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = (self._stats['hits'] / self._stats['total_requests'] 
                       if self._stats['total_requests'] > 0 else 0)
            
            return {
                **self._stats,
                'hit_rate': hit_rate,
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage': self._estimate_memory_usage()
            }
    
    def get_keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get all cache keys, optionally filtered by pattern."""
        with self._lock:
            keys = list(self._cache.keys())
            
            if pattern:
                import re
                regex = re.compile(pattern)
                keys = [k for k in keys if regex.search(k)]
            
            return keys
    
    def extend_ttl(self, key: str, additional_seconds: int) -> bool:
        """Extend TTL for a specific key."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry['ttl'] += additional_seconds
                return True
            return False
    
    def get_remaining_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for a key."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                elapsed = time.time() - entry['timestamp']
                remaining = max(0, entry['ttl'] - elapsed)
                return int(remaining)
            return None
    
    def _is_expired(self, entry: Dict) -> bool:
        """Check if cache entry is expired."""
        if entry['ttl'] == 0:  # 0 means no expiration
            return False
        
        elapsed = time.time() - entry['timestamp']
        return elapsed > entry['ttl']
    
    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is full."""
        while len(self._cache) > self.max_size:
            # Remove oldest (first) item
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._stats['evictions'] += 1
    
    def _should_compress(self, value: Any) -> bool:
        """Determine if value should be compressed."""
        # Compress large objects (>1KB estimated)
        return self._estimate_size(value) > 1024
    
    def _compress_value(self, value: Any) -> bytes:
        """Compress value using gzip."""
        try:
            serialized = pickle.dumps(value)
            return gzip.compress(serialized)
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return value
    
    def _decompress_value(self, value: Any) -> Any:
        """Decompress value if it's compressed."""
        if isinstance(value, bytes):
            try:
                decompressed = gzip.decompress(value)
                return pickle.loads(decompressed)
            except Exception:
                # If decompression fails, return as-is
                return value
        return value
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return 64  # Default estimate
    
    def _estimate_memory_usage(self) -> int:
        """Estimate total memory usage of cache."""
        total_size = 0
        
        for entry in self._cache.values():
            total_size += entry.get('size', 64)
        
        return total_size
    
    def _reset_stats(self) -> None:
        """Reset cache statistics."""
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }

class MultiLevelCache:
    """Multi-level cache with different TTL for different data types."""
    
    def __init__(self):
        self.caches = {
            # Fast cache for frequently accessed data
            'fast': CacheManager(max_size=500, default_ttl=60),
            
            # Market data cache
            'market_data': CacheManager(max_size=200, default_ttl=30),
            
            # Model predictions cache
            'predictions': CacheManager(max_size=100, default_ttl=300),
            
            # Portfolio optimization cache
            'portfolio': CacheManager(max_size=50, default_ttl=600),
            
            # DCF valuation cache
            'valuation': CacheManager(max_size=100, default_ttl=3600),
            
            # News and sentiment cache
            'news': CacheManager(max_size=200, default_ttl=1800),
            
            # Generic long-term storage
            'storage': CacheManager(max_size=1000, default_ttl=86400)
        }
    
    def get(self, cache_type: str, key: str, default: Any = None) -> Any:
        """Get value from specific cache level."""
        if cache_type in self.caches:
            return self.caches[cache_type].get(key, default)
        return default
    
    def set(self, cache_type: str, key: str, value: Any, 
            ttl: Optional[int] = None, compress: bool = False) -> None:
        """Set value in specific cache level."""
        if cache_type in self.caches:
            self.caches[cache_type].set(key, value, ttl, compress)
    
    def delete(self, cache_type: str, key: str) -> bool:
        """Delete key from specific cache level."""
        if cache_type in self.caches:
            return self.caches[cache_type].delete(key)
        return False
    
    def clear(self, cache_type: Optional[str] = None) -> None:
        """Clear specific cache or all caches."""
        if cache_type and cache_type in self.caches:
            self.caches[cache_type].clear()
        else:
            for cache in self.caches.values():
                cache.clear()
    
    def clear_expired(self) -> Dict[str, int]:
        """Clear expired entries from all caches."""
        results = {}
        for cache_type, cache in self.caches.items():
            results[cache_type] = cache.clear_expired()
        return results
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all cache levels."""
        stats = {}
        for cache_type, cache in self.caches.items():
            stats[cache_type] = cache.get_stats()
        return stats
    
    def get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Create deterministic key from arguments
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()

# Global cache instance
global_cache = MultiLevelCache()

# Convenience functions
def cache_get(cache_type: str, key: str, default: Any = None) -> Any:
    """Get value from cache."""
    return global_cache.get(cache_type, key, default)

def cache_set(cache_type: str, key: str, value: Any, 
              ttl: Optional[int] = None, compress: bool = False) -> None:
    """Set value in cache."""
    global_cache.set(cache_type, key, value, ttl, compress)

def cache_delete(cache_type: str, key: str) -> bool:
    """Delete key from cache."""
    return global_cache.delete(cache_type, key)

def cache_clear(cache_type: Optional[str] = None) -> None:
    """Clear cache."""
    global_cache.clear(cache_type)

def cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return global_cache.get_all_stats()

def generate_cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments."""
    return global_cache.get_cache_key(*args, **kwargs)