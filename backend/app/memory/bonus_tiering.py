import time
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

class MemoryItem(BaseModel):
    key: str
    content: str
    memory_type: str
    user_id: str
    created_at: float

class TieredMemoryGateway:
    """
    Simulates a production-grade tiered memory routing system (Hot Cache + Cold DB).
    Used in Case Studies (Chapter 14 / Bonus Chapter) to showcase latency control.
    """
    def __init__(self, ttl_seconds: int = 60):
        self.hot_cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds

    async def get_memory(self, user_id: str, key: str, db_fallback_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Attempts to read from hot cache. On miss, falls back to Database,
        populates hot cache, and returns with trace metadata.
        """
        now = time.time()
        cache_key = f"{user_id}:{key}"
        
        # Check cache hit & expiration
        if cache_key in self.hot_cache:
            entry = self.hot_cache[cache_key]
            if now - entry["cached_at"] < self.ttl:
                return {
                    "item": entry["item"],
                    "source": "hot_cache",
                    "latency_ms": 0.15,
                }

        # Cache miss -> Search DB fallback (Relational/pgvector simulation)
        db_item = None
        for item in db_fallback_items:
            if item.get("key") == key and item.get("user_id") == user_id:
                db_item = item
                break

        if not db_item:
            return {
                "item": None,
                "source": "db_miss",
                "latency_ms": 45.2,
            }

        # Populate cache
        self.hot_cache[cache_key] = {
            "item": db_item,
            "cached_at": now,
        }

        return {
            "item": db_item,
            "source": "database_fallback",
            "latency_ms": 45.2,
        }

    async def invalidate_user_cache(self, user_id: str) -> int:
        """Purges cache entries for a user (GDPR erase compliance)."""
        keys_to_remove = [k for k in self.hot_cache.keys() if k.startswith(f"{user_id}:")]
        for k in keys_to_remove:
            del self.hot_cache[k]
        return len(keys_to_remove)
