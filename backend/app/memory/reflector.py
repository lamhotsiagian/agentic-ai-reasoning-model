from pydantic import BaseModel, Field
from typing import List
from uuid import UUID
from sqlalchemy import update

class UtilityScore(BaseModel):
    memory_id: str = Field(description="UUID of the evaluated memory.")
    was_helpful: bool = Field(description="True if the memory directly helped solve or personalize the response.")
    score_delta: float = Field(description="Adjustment value to apply (-0.2 to +0.2).")

class ReflectionEvaluation(BaseModel):
    evaluations: List[UtilityScore] = Field(description="Evaluations for each injected memory.")

async def reflect_on_retrieval_utility(
    session, user_query: str, response: str, retrieved_memories: List[dict], model
) -> int:
    """
    Evaluates utility of retrieved memories and adjusts database importance scores.
    """
    if not retrieved_memories:
        return 0

    mem_list = [{"id": str(m["memory"].id), "content": m["memory"].content} for m in retrieved_memories]
    prompt = (
        f"User Query: '{user_query}'\n"
        f"Agent Response: '{response}'\n"
        f"Retrieved Memories Injected into Context:\n{mem_list}\n\n"
        f"Evaluate which memories were helpful for personalizing or answering the query correctly. "
        f"For unhelpful or noisy memories, suggest negative score_delta. Output structured JSON."
    )
    
    structured_model = model.with_structured_output(ReflectionEvaluation)
    res = await structured_model.ainvoke(prompt)
    
    updated = 0
    for eval_item in res.evaluations:
        mem_uuid = UUID(eval_item.memory_id)
        # Find matching memory in database
        for m_dict in retrieved_memories:
            mem = m_dict["memory"]
            if mem.id == mem_uuid:
                # Apply delta clamp between [0.1, 1.0]
                new_score = max(0.1, min(1.0, mem.importance_score + eval_item.score_delta))
                mem.importance_score = new_score
                updated += 1
    
    if updated > 0:
        await session.commit()
    return updated
