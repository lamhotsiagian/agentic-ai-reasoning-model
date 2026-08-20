.PHONY: up down migrate models backend frontend test eval fmt

up:            ## start postgres + neo4j
	docker compose up -d postgres neo4j

down:
	docker compose down

models:        ## pull the two reasoning models + embeddings
	ollama pull qwen2.5:3b
	ollama pull qwen3:1.7b
	ollama pull nomic-embed-text

migrate:       ## pgvector, tsvector, RLS, and the read-only SQL role
	psql "postgresql://$${POSTGRES_USER:-postgres}:$${POSTGRES_PASSWORD:-test}@$${POSTGRES_HOST:-localhost}:$${POSTGRES_PORT:-5432}/$${POSTGRES_DATABASE:-langgraph_db}" \
		-f backend/migrations/001_reasoning.sql

backend:
	cd backend && uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

test:          ## fast unit suite; no services required
	cd backend && pytest -q

eval:          ## decomposed evaluation report
	cd backend && python -m app.reasoning.eval

fmt:
	cd backend && python -m ruff check --fix app tests || true
