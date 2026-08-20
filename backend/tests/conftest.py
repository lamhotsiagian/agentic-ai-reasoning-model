import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Deterministic defaults so the fast suite needs no services running.
os.environ.setdefault("TOKEN_BEARER_URL", "/api/v1/auth/login")
os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("ACCESS_TOKEN_EXPIRY_MINS", "60")
os.environ.setdefault("REFRESH_TOKEN_EXPIRY_DAYS", "1")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_USER", "postgres")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_DATABASE", "langgraph_db")
os.environ.setdefault("PGVECTOR_COLLECTION_NAME", "test_collection")
