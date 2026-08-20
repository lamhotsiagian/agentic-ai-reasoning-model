-- ===========================================================================
--  Reasoning layer bootstrap.
--  Run once against the application database:
--      psql "$POSTGRES_URI" -f migrations/001_reasoning.sql
-- ===========================================================================
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- --- vector column + indexes ------------------------------------------------
ALTER TABLE reasoning_chunks
    ADD COLUMN IF NOT EXISTS embedding vector(768);          -- nomic-embed-text

CREATE INDEX IF NOT EXISTS ix_chunks_embedding
    ON reasoning_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- --- sparse channel: generated tsvector, so it can never drift from content -
ALTER TABLE reasoning_chunks
    ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED;

CREATE INDEX IF NOT EXISTS ix_chunks_tsv
    ON reasoning_chunks USING gin (tsv);

-- ===========================================================================
--  Row-level security. This is the control that holds when prompt injection
--  succeeds: even a perfectly injected query returns only permitted rows.
--  Never rely on a WHERE clause the model was asked to include.
-- ===========================================================================
ALTER TABLE reasoning_chunks    ENABLE ROW LEVEL SECURITY;
ALTER TABLE reasoning_documents ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation_chunks ON reasoning_chunks;
CREATE POLICY tenant_isolation_chunks ON reasoning_chunks
    USING (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS tenant_isolation_docs ON reasoning_documents;
CREATE POLICY tenant_isolation_docs ON reasoning_documents
    USING (tenant_id = current_setting('app.tenant_id', true));

-- ===========================================================================
--  The SQL tool's role. SELECT only, at the DATABASE level -- grants, not
--  prompts. No DDL, no DML, no function execution.
-- ===========================================================================
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'reasoning_reader') THEN
        CREATE ROLE reasoning_reader LOGIN PASSWORD 'reasoning_reader';
    END IF;
END $$;

REVOKE ALL ON ALL TABLES IN SCHEMA public FROM reasoning_reader;
GRANT USAGE ON SCHEMA public TO reasoning_reader;
GRANT SELECT ON reasoning_chunks, reasoning_documents TO reasoning_reader;
ALTER ROLE reasoning_reader SET statement_timeout = '8s';
ALTER ROLE reasoning_reader SET default_transaction_read_only = on;

-- reasoning_reader is NOT a superuser, so RLS applies to it.
ALTER TABLE reasoning_chunks    FORCE ROW LEVEL SECURITY;
ALTER TABLE reasoning_documents FORCE ROW LEVEL SECURITY;
