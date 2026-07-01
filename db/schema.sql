CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS filing_chunks (
    id               SERIAL PRIMARY KEY,
    ticker           TEXT NOT NULL,
    filing_date      DATE NOT NULL,
    form             TEXT NOT NULL,
    chunk_index      INTEGER NOT NULL,
    total_chunks     INTEGER NOT NULL,
    chunk_text       TEXT NOT NULL,
    sentiment_label  TEXT,
    sentiment_score  DOUBLE PRECISION,
    embedding        vector(384)
);

CREATE INDEX IF NOT EXISTS filing_chunks_embedding_idx
    ON filing_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS filing_chunks_ticker_idx ON filing_chunks (ticker);

CREATE TABLE IF NOT EXISTS filing_sentiment (
    id                   SERIAL PRIMARY KEY,
    ticker               TEXT NOT NULL,
    filing_date          DATE NOT NULL,
    form                 TEXT NOT NULL,
    avg_sentiment_score  DOUBLE PRECISION NOT NULL,
    sentiment_shift      DOUBLE PRECISION NOT NULL,
    material_shift       BOOLEAN NOT NULL,
    UNIQUE (ticker, filing_date, form)
);

CREATE INDEX IF NOT EXISTS filing_sentiment_ticker_idx ON filing_sentiment (ticker);
