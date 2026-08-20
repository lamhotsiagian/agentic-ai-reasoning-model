from pydantic import BaseModel, Field
from typing import List
from langchain_core.messages import BaseMessage

class MemoryTriple(BaseModel):
    subject: str = Field(description="The primary subject entity (e.g., 'Alice')")
    predicate: str = Field(description="The relationship verb or attribute link (e.g., 'works_at')")
    object: str = Field(description="The target object entity or value (e.g., 'Google DeepMind')")

class GraphMemoryExtraction(BaseModel):
    triples: List[MemoryTriple] = Field(description="List of extracted fact triples.")

async def extract_graph_triples(conversation: List[BaseMessage], model) -> List[MemoryTriple]:
    """
    Extracts Subject-Predicate-Object (SPO) triples from dialogue turns.
    These triples are indexed into a graph database to enable multi-hop reasoning.
    """
    dialogue_str = "\n".join([f"{m.type}: {m.content}" for m in conversation])
    prompt = (
        f"Analyze the following conversation and extract all stable relationship triples:\n"
        f"{dialogue_str}\n\n"
        f"Focus on entity connections, facts, and persistent relations. Output structured JSON."
    )
    # Configure the LLM to output schema-conforming JSON
    structured_model = model.with_structured_output(GraphMemoryExtraction)
    res = await structured_model.ainvoke(prompt)
    return res.triples
