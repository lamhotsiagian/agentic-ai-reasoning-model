from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from app.config import settings
from app.db.main import init_db
from app.db import reasoning_models  # noqa: F401  (registers tables)
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Running lifespan before the application startup!")
    await init_db()

    if settings.graph_enabled:
        try:
            from app.graph.client import ensure_schema
            await ensure_schema()
        except Exception as exc:      # the graph channel degrades gracefully
            logger.warning(f"Neo4j schema setup skipped: {exc}")
    
    if settings.seed_demo_data:
        try:
            from app.db.seed import seed_demo_data
            await seed_demo_data()

            from app.db.seed_reasoning import seed_graph, seed_reasoning_corpus
            await seed_reasoning_corpus()
            if settings.graph_enabled:
                await seed_graph()
        except Exception as e:  # never block startup on demo seeding
            logger.warning(f"Demo data seeding skipped: {e}")
            
    logger.info("Initializing checkpointer connection pool...")
    async with AsyncPostgresSaver.from_conn_string(settings.checkpointer_uri) as cp:
        await cp.setup()
        app.state.checkpointer = cp
        
        # Share it globally with the app.db.checkpointer module
        from app.db import checkpointer as cp_module
        cp_module.checkpointer = cp
        
        logger.info("✅ PostgresCheckpointer connection pool initialized successfully")
        yield
        
    if settings.graph_enabled:
        try:
            from app.graph.client import close_driver
            await close_driver()
        except Exception:
            pass

    logger.info("Running lifespan after the application shutdown!")
