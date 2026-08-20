from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

class BlackboardEntry(BaseModel):
    key: str
    value: Any
    updated_by_agent: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class MultiAgentBlackboard:
    """
    A thread-safe, memory-based Shared Blackboard for multi-agent systems.
    Supports basic locks to prevent concurrent write collisions.
    """
    def __init__(self):
        self._store: Dict[str, BlackboardEntry] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def write(self, key: str, value: Any, agent_name: str) -> None:
        """
        Acquires a key-level lock and updates the blackboard entry.
        """
        async with self._global_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
        
        async with self._locks[key]:
            entry = BlackboardEntry(
                key=key,
                value=value,
                updated_by_agent=agent_name
            )
            self._store[key] = entry

    async def read(self, key: str) -> Optional[BlackboardEntry]:
        """
        Reads a value from the blackboard.
        """
        async with self._global_lock:
            return self._store.get(key)

    async def delete(self, key: str) -> bool:
        """
        Deletes a value from the blackboard if present.
        """
        async with self._global_lock:
            if key in self._store:
                del self._store[key]
                return True
            return False
