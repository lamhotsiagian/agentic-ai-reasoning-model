# agentic-ai-reasoning-model

Companion project for ***Agentic AI Reasoning Model System Design***
(AI Engineering Insider). A runnable reasoning system covering
decomposition, planning, search, tools, GraphRAG, verification, adaptive
test-time compute, and a multi-agent supervisor, served by FastAPI + LangGraph
behind a Next.js UI, sized to run on a laptop with Ollama.

Every measurement in the book can be reproduced here. That is the point of the
project: the labs exist so you can watch the behaviour rather than trust the
prose.

**Repository:** <https://github.com/lamhotsiagian/agentic-ai-reasoning-model>

---

## Chapter → module → lab

| Ch. | Topic | Module | Lab |
|-----|-------|--------|-----|
| 1 | Reasoning fundamentals | `app/reasoning/state.py`, `router.py`, `redaction.py` | `/labs/1` |
| 2 | Decomposition | `app/reasoning/decomposer.py`, `hybrid.py` | `/labs/2` |
| 3 | Planning | `app/reasoning/planner.py`, `progress.py`, `clarify.py` | `/labs/3` |
| 4 | Search | `app/reasoning/search.py` | `/labs/4` |
| 5 | Tools and actions | `app/reasoning/tools/` | `/labs/5` |
| 6 | Retrieval + GraphRAG | `app/reasoning/retrieval/`, `app/graph/` | `/labs/6` |
| 7 | Verification | `app/reasoning/verify.py` | `/labs/7` |
| 8 | Test-time compute | `app/reasoning/compute.py` | `/labs/8` |
| 9 | Multi-agent | `app/reasoning/agents/` | `/labs/9` |
| 10 | Production | `app/reasoning/routes.py`, `guardrails.py`, `tracing.py`, `eval/` | `/labs/10` |

---

## Quick start

```bash
cp env.example .env                 # review it; LABS_ENABLED must be false in prod

# 1. Models (two, deliberately different; see Chapter 7)
ollama pull qwen2.5:3b              # reasoner
ollama pull qwen3:1.7b              # critic / verifier / reranker
ollama pull nomic-embed-text        # embeddings

# 2. Storage
docker compose up -d postgres neo4j

# 3. Schema: pgvector, generated tsvector, RLS, and the read-only SQL role
psql "postgresql://postgres:test@localhost:5432/langgraph_db" \
     -f backend/migrations/001_reasoning.sql

# 4. Backend
cd backend && pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# 5. Frontend
cd ../frontend && npm install && npm run dev     # http://localhost:3000/labs
```

The whole stack also comes up with `docker compose up --build`, but running the
backend directly is faster for reading, since the labs restart on save.

---

## Architecture

```
Next.js UI ──► FastAPI ──► guardrails ──► budget controller
                                              │
                                              ▼
                                   LangGraph supervisor
                                   planner · researcher
                                   reasoner · critic · verifier
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
             hybrid retrieval           tool sandbox            verification
          pgvector · tsvector · Neo4j   python · sql · api          cascade
                    │                         │                         │
                    └─────────────────────────┼─────────────────────────┘
                                              ▼
                        Postgres: state · evidence · traces (separate retention)
                                              │
                                Langfuse · OpenTelemetry · eval harness
```

Three design decisions carry most of the weight:

**The budget is external.** A model asked to police its own spending keeps
deliberating. That is the behaviour outcome-reward training selects for. The
controller lives in `compute.py` and every exit path is bounded.

**The trace never crosses the network.** `redaction.py` projects internal steps
onto a whitelisted progress contract. Raw trajectories stay in an
access-controlled store with their own retention window, because that retention
is a legal decision rather than an engineering one.

**Authority lives outside the prompt.** Tool scoping (`tools/router.py`) and
database row-level security (`migrations/001_reasoning.sql`) are what hold when
prompt injection succeeds. The input scanner in `guardrails.py` is defence in
depth, not the boundary.

---

## Tests

```bash
cd backend
pytest                              # 53 unit tests, no services required
pytest -m integration               # requires Postgres, Neo4j, and Ollama
python -m app.reasoning.eval        # decomposed evaluation report
```

The unit suite runs without any service because the mechanisms worth testing
(budget enforcement, DAG validation, reversible-first ordering, SQL AST
validation, RRF, arithmetic re-execution, adaptive-N stopping) are all
deterministic. Tool fixtures (`app/reasoning/eval/fixtures.py`) belong in CI:
tool-call accuracy regresses silently when somebody rewords a description
"for clarity".

---

## Safety notes

- `LABS_ENABLED=true` exposes raw trajectories and internal step payloads.
  **Set it to `false` in any deployment that is not a local reading
  environment.**
- The in-process Python sandbox in `tools/sandbox.py` runs out-of-process with
  an import allowlist and resource caps. It is a **development** fallback. In
  production, run each execution in a gVisor or Firecracker container with the
  network denied by policy.
- SQL is validated by parsing the AST, never by a keyword blocklist. Blocklists
  are evaded by comments, casing, and concatenation, and the evasions look
  accidental in review.
- The `reasoning_reader` role is `SELECT`-only at the database level, with
  row-level security bound to the requesting tenant. Never rely on a `WHERE`
  clause the model was asked to include.
# agentic-ai-reasoning-model
