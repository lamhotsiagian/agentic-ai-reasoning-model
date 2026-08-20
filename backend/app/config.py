import logging
from pathlib import Path

from loguru import logger
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.reasoning.settings_fragment import ReasoningSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent

LOGS_DIR = BASE_DIR / "logs"
if not LOGS_DIR.exists():
    LOGS_DIR.mkdir()

logging.basicConfig(
    level=logging.INFO,  # For displaying the default model calling logs
)

logger.add(
    sink=LOGS_DIR / "api_{time:YYYYMMDD}.log",
    level="INFO",
    rotation="00:00",
    retention="7 days",
    compression="zip",
)


class Settings(ReasoningSettings, BaseSettings):
    api_key: str = Field(default="", alias="OPENAI_API_KEY")
    tavily_api_key: str = Field(default="")
    model_provider: str = Field(default="ollama")
    model_names: list[str] = Field(default=["llama3.2"])
    model_base_url: str | None = Field(default="http://localhost:11434")
    embeddings_model_name: str = Field(default="nomic-embed-text")
    embeddings_base_url: str | None = Field(default="http://localhost:11434")
    token_bearer_url: str
    jwt_secret: str
    jwt_algorithm: str
    access_token_expiry_mins: int
    refresh_token_expiry_days: int
    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_password: str
    postgres_database: str
    pgvector_collection_name: str

    # Memory Settings (Chapters 4, 6, 7, 10)
    memory_decay_rate: float = Field(default=0.05)       # Forgetting curve decay rate
    memory_max_age_days: int = Field(default=30)        # Privacy retention policy
    memory_consolidation_interval: int = Field(default=10) # Consolidate after this many turns
    memory_importance_threshold: float = Field(default=0.3) # Save only if >= threshold

    # Demo data: seed users, memories, and the reasoning corpus on startup
    # when the database is empty.
    seed_demo_data: bool = Field(default=True)

    # Read-only analytics replica used by the SQL tool. Access control lives
    # HERE, in the database role and row-level security, never in a WHERE
    # clause the model was asked to include.
    replica_host: str = Field(default="")
    replica_user: str = Field(default="reasoning_reader")
    replica_password: str = Field(default="reasoning_reader")

    @property
    def replica_uri(self) -> str:
        host = self.replica_host or self.postgres_host
        return (f"postgresql://{self.replica_user}:{self.replica_password}"
                f"@{host}:{self.postgres_port}/{self.postgres_database}")

    @property
    def database_uri(self) -> str:
        """Generate PostgreSQL connection string for sqlalchemy."""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"

    @property
    def checkpointer_uri(self) -> str:
        """Generate PostgreSQL connection string for checkpointer."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}?sslmode=disable"

    @property
    def pgvector_connection(self) -> str:
        """Generate PostgreSQL connection string for PGVector."""
        return f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="allow")


settings = Settings()  # type: ignore
