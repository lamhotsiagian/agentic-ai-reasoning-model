"""The sufficiency gate (Chapter 6).

'Topically relevant but useless' is the retrieval failure that survives every
dashboard. Similarity looks healthy and the answer is wrong because no
passage contains the specific fact. The only detector that works is an
explicit predicate that also states WHAT is missing, and that statement is
what generates a good next query, so the check pays for itself twice.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.reasoning.llm import critic, structured_call
from app.reasoning.retrieval.loop import EvidenceSet


class Sufficiency(BaseModel):
    sufficient: bool
    missing: str = Field(
        default="",
        description="If not sufficient: the SPECIFIC fact that is absent.")
    next_query: str = Field(
        default="",
        description="A retrieval query that would find the missing fact.")
    conflicts: list[str] = Field(
        default_factory=list,
        description="Pairs of citations that state incompatible facts.")


SUFFICIENCY_PROMPT = """Question: {question}

Evidence gathered so far:
{evidence}

Decide whether this evidence is sufficient to answer the question completely
and correctly.

- If it is, set sufficient=true.
- If it is not, name the SPECIFIC missing fact in `missing` (not "more
  information"), and write a retrieval query in `next_query` that would find
  it.
- List any pair of citations that state incompatible facts in `conflicts`.

Topical relevance is not sufficiency: passages about the right subject that
do not contain the required fact are NOT enough."""


async def assess_sufficiency(question: str,
                             evidence: EvidenceSet) -> Sufficiency:
    if not evidence.passages:
        return Sufficiency(sufficient=False,
                           missing="no evidence retrieved at all",
                           next_query=question)

    result, _ = await structured_call(
        SUFFICIENCY_PROMPT.format(question=question,
                                  evidence=evidence.render()),
        schema=Sufficiency,
        model=critic(),
    )
    if not result.ok:
        # Fail towards continuing rather than answering on thin evidence.
        return Sufficiency(sufficient=False,
                           missing="sufficiency check failed to parse",
                           next_query=question)
    return result.value
