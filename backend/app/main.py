from fastapi import FastAPI

from app.auth.routes import auth_router
from app.chat.routes import chat_router
from app.documents.routes import document_router
from app.lifespan import lifespan
from app.middleware import register_middleware
from app.threads.routes import thread_router
from app.users.routes import user_router
from app.memory.routes import memory_router
from app.reasoning.routes import reasoning_router

version = "v1"
version_prefix = f"/api/{version}"

app = FastAPI(
    title="Agentic AI Reasoning Model API",
    description=(
        "Reference implementation for *Agentic AI Reasoning Model System "
        "Design*: decomposition, planning, search, tools, GraphRAG, "
        "verification, and adaptive test-time compute."
    ),
    version=version,
    openapi_url=f"{version_prefix}/openapi.json",
    docs_url=f"{version_prefix}/docs",
    redoc_url=f"{version_prefix}/redoc",
    lifespan=lifespan,
    license_info={"name": "MIT License", "url": "https://opensource.org/license/mit"},
)

register_middleware(app)

app.include_router(auth_router, prefix=f"{version_prefix}/auth", tags=["AUTH"])
app.include_router(user_router, prefix=f"{version_prefix}/users", tags=["USERS"])
app.include_router(thread_router, prefix=f"{version_prefix}/threads", tags=["THREADS"])
app.include_router(chat_router, prefix=f"{version_prefix}/chat", tags=["CHAT"])
app.include_router(document_router, prefix=f"{version_prefix}/documents", tags=["DOCUMENTS"])
app.include_router(memory_router, prefix=f"{version_prefix}/memory", tags=["MEMORY"])
app.include_router(reasoning_router, prefix=f"{version_prefix}/reasoning", tags=["REASONING"])


@app.get(f"{version_prefix}/health")
async def health() -> dict[str, str]:
    """Liveness only. Readiness is gated on the checkpointer in lifespan."""
    return {"status": "ok"}
