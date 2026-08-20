from uuid import UUID
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, delete, update
from app.auth.dependencies import CurrentUserDep
from app.db.main import SessionDep
from app.db.models import Memory, MemoryConflict, MemoryConsolidation, Entity
from app.memory.schemas import (
    MemoryCreate, MemoryUpdate, MemoryResponse,
    MemorySearchResponse, MemoryStatsResponse, MemoryConflictResponse,
    ConflictResolveRequest, MemoryEvaluationResponse
)
from app.memory.reader import search_memories
from app.memory.service import add_memory_item
from app.memory.consolidator import consolidate_memories, detect_conflicts
from app.memory.privacy import forget_user, export_user_data

memory_router = APIRouter()

@memory_router.post("/", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_memory(
    data: MemoryCreate,
    current_user: CurrentUserDep,
    session: SessionDep
):
    mem = await add_memory_item(
        session=session,
        user_id=current_user.id,
        thread_id=data.thread_id,
        memory_type=data.memory_type,
        content=data.content,
        importance_score=data.importance_score,
        is_shared=data.is_shared,
        metadata_json=data.metadata_json
    )
    return mem

@memory_router.get("/", response_model=List[MemoryResponse])
async def list_memories(
    current_user: CurrentUserDep,
    session: SessionDep,
    memory_type: Optional[str] = None
):
    stmt = select(Memory).where(Memory.user_id == current_user.id, Memory.is_active == True)
    if memory_type:
        stmt = stmt.where(Memory.memory_type == memory_type)
    res = await session.execute(stmt)
    return res.scalars().all()

@memory_router.get("/search", response_model=List[MemorySearchResponse])
async def search_user_memories(
    query: str,
    current_user: CurrentUserDep,
    session: SessionDep,
    strategy: str = "combined",
    limit: int = 5
):
    results = await search_memories(
        session=session,
        user_id=str(current_user.id),
        query=query,
        strategy=strategy,
        limit=limit
    )
    # Map to schema response
    out = []
    for item in results:
        out.append(MemorySearchResponse(
            memory=MemoryResponse.model_validate(item["memory"]),
            relevance_score=item["relevance_score"],
            final_score=item["final_score"]
        ))
    return out

@memory_router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_single_memory(
    memory_id: UUID,
    current_user: CurrentUserDep,
    session: SessionDep
):
    stmt = select(Memory).where(Memory.id == memory_id, Memory.user_id == current_user.id)
    res = await session.execute(stmt)
    mem = res.scalar_one_or_none()
    if not mem:
        raise HTTPException(status_code=404, detail="Memory not found.")
        
    mem.is_active = False
    await session.commit()
    # Remove vectors
    try:
        from app.db.pgvector_utils import delete_document_from_pgvector
        await delete_document_from_pgvector([str(memory_id)])
    except Exception:
        pass

@memory_router.get("/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(
    current_user: CurrentUserDep,
    session: SessionDep
):
    stmt = select(Memory).where(Memory.user_id == current_user.id)
    res = await session.execute(stmt)
    memories = res.scalars().all()
    
    total = len(memories)
    active = sum(1 for m in memories if m.is_active)
    inactive = total - active
    
    counts = {}
    total_imp = 0.0
    for m in memories:
        if m.is_active:
            counts[m.memory_type] = counts.get(m.memory_type, 0) + 1
            total_imp += m.importance_score
            
    avg_imp = (total_imp / active) if active > 0 else 0.0
    
    return MemoryStatsResponse(
        total_count=total,
        counts_by_type=counts,
        avg_importance=round(avg_imp, 2),
        active_count=active,
        inactive_count=inactive
    )

@memory_router.post("/consolidate")
async def trigger_consolidation_endpoint(
    current_user: CurrentUserDep,
    session: SessionDep,
    thread_id: Optional[UUID] = None
):
    from app.config import settings
    res = await consolidate_memories(
        session=session,
        user_id=str(current_user.id),
        thread_id=thread_id,
        model_name=settings.model_names[0]
    )
    return res

@memory_router.get("/conflicts", response_model=List[MemoryConflictResponse])
async def list_conflicts(
    current_user: CurrentUserDep,
    session: SessionDep
):
    # Auto run detection first
    from app.config import settings
    await detect_conflicts(session, str(current_user.id), settings.model_names[0])
    
    stmt = select(MemoryConflict).where(MemoryConflict.user_id == current_user.id)
    res = await session.execute(stmt)
    conflicts = res.scalars().all()
    
    out = []
    for c in conflicts:
        m1 = await session.get(Memory, c.memory_id_old)
        m2 = await session.get(Memory, c.memory_id_new)
        if m1 and m2:
            out.append(MemoryConflictResponse(
                id=c.id,
                user_id=c.user_id,
                memory_id_old=c.memory_id_old,
                memory_id_new=c.memory_id_new,
                old_content=m1.content,
                new_content=m2.content,
                conflict_type=c.conflict_type,
                resolution=c.resolution,
                is_resolved=c.is_resolved,
                created_at=c.created_at
            ))
    return out

@memory_router.post("/resolve-conflict/{conflict_id}")
async def resolve_conflict_endpoint(
    conflict_id: UUID,
    req: ConflictResolveRequest,
    current_user: CurrentUserDep,
    session: SessionDep
):
    c = await session.get(MemoryConflict, conflict_id)
    if not c or c.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conflict record not found.")
        
    c.is_resolved = True
    c.resolution = req.resolution
    c.resolved_at = datetime.now()
    
    # Apply change to database memory nodes
    if req.keep_old:
        # Deactivate new one
        stmt = update(Memory).where(Memory.id == c.memory_id_new).values(is_active=False)
        await session.execute(stmt)
    else:
        # Deactivate old one
        stmt = update(Memory).where(Memory.id == c.memory_id_old).values(is_active=False)
        await session.execute(stmt)
        
    await session.commit()
    return {"message": "Conflict resolved successfully."}

@memory_router.post("/evaluate", response_model=MemoryEvaluationResponse)
async def evaluate_memory_metrics(
    query: str,
    expected_memory: str,
    current_user: CurrentUserDep,
    session: SessionDep
):
    """
    Chapter 9: Memory Evaluation
    Generates faithfulness, recall, and retrieval metrics.
    """
    import time
    start = time.time()
    
    # 1. Run combined retrieval
    results = await search_memories(
        session=session,
        user_id=str(current_user.id),
        query=query,
        strategy="combined",
        limit=5
    )
    
    latency = (time.time() - start) * 1000
    
    # Simple check for recall
    retrieved_texts = [r["memory"].content.lower() for r in results]
    recall = 1.0 if any(expected_memory.lower() in t or t in expected_memory.lower() for t in retrieved_texts) else 0.0
    
    # Mock precision check & faithfulness calculation
    precision = 1.0 if len(results) > 0 else 0.0
    faithfulness = 0.95 if recall > 0 else 0.50
    
    return MemoryEvaluationResponse(
        recall=recall,
        precision=precision,
        faithfulness=faithfulness,
        latency_ms=round(latency, 2)
    )

@memory_router.post("/forget-me", status_code=status.HTTP_200_OK)
async def forget_me_privacy_purge(
    current_user: CurrentUserDep,
    session: SessionDep
):
    await forget_user(session, str(current_user.id))
    return {"message": "All your personal memory assets have been permanently erased."}

@memory_router.get("/export")
async def export_user_data_endpoint(
    current_user: CurrentUserDep,
    session: SessionDep
):
    return await export_user_data(session, str(current_user.id))


# --- Chapter Labs UI Verification Endpoints ---
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from app.chat.langgraph_agent import create_model

class TripleRequest(BaseModel):
    text: str

@memory_router.post("/extract-triples")
async def extract_triples_endpoint(req: TripleRequest):
    from app.memory.graph_extractor import extract_graph_triples
    from app.config import settings
    
    model = create_model(model_name=settings.model_names[0])
    conversation = [HumanMessage(content=req.text)]
    res = await extract_graph_triples(conversation, model)
    return {"triples": [r.model_dump() for r in res]}

class RuleSchema(BaseModel):
    trigger: str
    forbidden_sub: str

class VerifyRequest(BaseModel):
    query: str
    rules: List[RuleSchema]

@memory_router.post("/verify-constraints")
async def verify_constraints_endpoint(
    req: VerifyRequest,
    current_user: CurrentUserDep,
    session: SessionDep
):
    from app.memory.neuro_symbolic import SymbolicConstraintEngine
    results = await search_memories(
        session=session,
        user_id=str(current_user.id),
        query=req.query,
        strategy="combined",
        limit=5
    )
    
    rules_dict = [{"trigger": r.trigger, "forbidden_sub": r.forbidden_sub} for r in req.rules]
    engine = SymbolicConstraintEngine(rules_dict)
    filtered = engine.verify_and_filter(results)
    
    return {
        "original_count": len(results),
        "filtered_count": len(filtered),
        "memories": [{"content": m["memory"].content, "score": m["final_score"]} for m in filtered]
    }

class ReflectRequest(BaseModel):
    query: str
    response: str
    memory_ids: List[UUID]

@memory_router.post("/reflect")
async def reflect_endpoint(
    req: ReflectRequest,
    current_user: CurrentUserDep,
    session: SessionDep
):
    from app.memory.reflector import reflect_on_retrieval_utility
    from app.config import settings
    
    # Fetch target memories to reflect on
    stmt = select(Memory).where(Memory.id.in_(req.memory_ids))
    res = await session.execute(stmt)
    memories = res.scalars().all()
    
    retrieved_list = [{"memory": m, "final_score": m.importance_score} for m in memories]
    model = create_model(model_name=settings.model_names[0])
    
    updated_count = await reflect_on_retrieval_utility(
        session=session,
        user_query=req.query,
        response=req.response,
        retrieved_memories=retrieved_list,
        model=model
    )
    return {"updated_count": updated_count}

class BlackboardWriteRequest(BaseModel):
    key: str
    value: str
    agent_name: str

from app.memory.blackboard import MultiAgentBlackboard
global_blackboard = MultiAgentBlackboard()

@memory_router.post("/blackboard")
async def blackboard_write_endpoint(req: BlackboardWriteRequest):
    await global_blackboard.write(req.key, req.value, req.agent_name)
    return {"message": f"Successfully wrote key '{req.key}' to blackboard."}

@memory_router.get("/blackboard/{key}")
async def blackboard_read_endpoint(key: str):
    entry = await global_blackboard.read(key)
    if not entry:
        raise HTTPException(status_code=404, detail="Key not found on blackboard.")
    return {
        "key": entry.key,
        "value": entry.value,
        "updated_by_agent": entry.updated_by_agent,
        "timestamp": entry.timestamp
    }

@memory_router.post("/run-stress-test")
async def run_stress_test_endpoint(
    current_user: CurrentUserDep,
    session: SessionDep
):
    from app.memory.stress_test import SyntheticUserSimulator, run_memory_stress_test
    from app.chat.langgraph_agent import build_retrival_graph, create_model
    from app.db.checkpointer import get_checkpointer
    from app.config import settings
    
    checkpointer = await get_checkpointer()
    graph = build_retrival_graph(checkpointer, settings.model_names[0])
    model = create_model(model_name=settings.model_names[0])
    
    simulator = SyntheticUserSimulator(
        email=current_user.email,
        persona_rules=["Be cooperative"],
        target_facts=["My favorite color is neon-blue-light-laser"]
    )
    
    success = await run_memory_stress_test(graph, simulator, model)
    return {"success": success}

class TieredRouteRequest(BaseModel):
    key: str
    db_fallback_items: List[Dict[str, Any]]

@memory_router.post("/tiered-gateway")
async def tiered_gateway_endpoint(
    req: TieredRouteRequest,
    current_user: CurrentUserDep
):
    from app.memory.bonus_tiering import TieredMemoryGateway
    if not hasattr(tiered_gateway_endpoint, "gateway"):
        tiered_gateway_endpoint.gateway = TieredMemoryGateway()
    
    # Add user_id to fallback items
    items = []
    for item in req.db_fallback_items:
        copy_item = item.copy()
        copy_item["user_id"] = str(current_user.id)
        items.append(copy_item)

    result = await tiered_gateway_endpoint.gateway.get_memory(
        user_id=str(current_user.id),
        key=req.key,
        db_fallback_items=items
    )
    return result

