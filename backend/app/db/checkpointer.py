from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from loguru import logger
from app.config import settings

checkpointer: AsyncPostgresSaver | None = None

async def get_checkpointer() -> AsyncPostgresSaver:
    global checkpointer

    if checkpointer is not None:
        return checkpointer

    logger.info("Initializing fallback PostgresCheckpointer via connection pool...")
    checkpointer = AsyncPostgresSaver.from_conn_string(settings.checkpointer_uri)
    await checkpointer.setup()
    logger.info("✅ Fallback PostgresCheckpointer initialized successfully")
    return checkpointer

async def create_connection():
    """No-op kept for backwards compatibility in external script imports."""
    pass

async def close_connection() -> None:
    """No-op kept for backwards compatibility in external script imports."""
    pass
