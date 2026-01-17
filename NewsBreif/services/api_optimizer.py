"""
API Usage Optimizer for LiveBrief.
Minimal implementation for caching and request throttling.
"""
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from loguru import logger


class APICache:
    """Simple in-memory cache with TTL for API responses"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self._cache: Dict[str, Dict] = {}
        self._ttl = ttl_seconds
    
    def _make_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {k: v for k, v in sorted(kwargs.items())}
        key_str = json.dumps(key_data, sort_keys=True)
        return f"{prefix}:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry['timestamp'] < self._ttl:
                logger.debug(f"Cache HIT: {key[:50]}...")
                return entry['value']
            else:
                del self._cache[key]
                logger.debug(f"Cache EXPIRED: {key[:50]}...")
        return None
    
    def set(self, key: str, value: Any):
        """Store value in cache"""
        self._cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
        logger.debug(f"Cache SET: {key[:50]}...")
    
    def clear(self):
        """Clear all cached data"""
        self._cache.clear()
        logger.info("API cache cleared")
    
    def stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'total_entries': len(self._cache),
            'ttl_seconds': self._ttl
        }


# Global API cache instance
api_cache = APICache(ttl_seconds=3600)  # 1 hour TTL


class APIUsageTracker:
    """Track API usage for free tier management"""
    
    def __init__(self, daily_limit: int = 100):
        self._daily_limit = daily_limit
        self._requests_today = 0
        self._last_reset = datetime.now(timezone.utc).date()
        self._type_counts: Dict[str, int] = {}
    
    def reset_if_needed(self):
        """Reset counters if it's a new day"""
        today = datetime.now(timezone.utc).date()
        if today != self._last_reset:
            self._requests_today = 0
            self._type_counts.clear()
            self._last_reset = today
            logger.info("API usage counter reset for new day")
    
    def record_request(self, api_type: str):
        """Record an API request"""
        self.reset_if_needed()
        self._requests_today += 1
        self._type_counts[api_type] = self._type_counts.get(api_type, 0) + 1
    
    def can_make_request(self, api_type: str) -> bool:
        """Check if we can make more requests today"""
        self.reset_if_needed()
        # Check overall limit
        if self._requests_today >= self._daily_limit:
            logger.warning(f"Daily API limit reached ({self._daily_limit})")
            return False
        # Check per-type limit (10% of daily for safety)
        type_limit = max(1, self._daily_limit // 10)
        if self._type_counts.get(api_type, 0) >= type_limit:
            logger.warning(f"Per-type limit reached for {api_type}")
            return False
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current API usage status"""
        self.reset_if_needed()
        return {
            'requests_today': self._requests_today,
            'daily_limit': self._daily_limit,
            'remaining': self._daily_limit - self._requests_today,
            'by_type': self._type_counts.copy(),
            'percentage_used': round((self._requests_today / self._daily_limit) * 100, 1)
        }


# Global usage tracker
api_usage = APIUsageTracker(daily_limit=100)

