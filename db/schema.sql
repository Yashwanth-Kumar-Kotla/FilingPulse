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
    embedding        vector(1536)
);

-- No ivfflat index: at this corpus size (tens of thousands of rows), pgvector's
-- own guidance is that exact brute-force search stays fast and is more accurate
-- than an approximate index — ivfflat only pays off at ~1M+ rows. It was also
-- measured at 4.5x the size of the actual embedding data, which matters on a
-- disk-constrained hosted Postgres plan.

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

CREATE TABLE IF NOT EXISTS subscriptions (
    id                          SERIAL PRIMARY KEY,
    email                       TEXT NOT NULL,
    ticker                      TEXT NOT NULL,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_notified_filing_date   DATE,
    UNIQUE (email, ticker)
);

CREATE INDEX IF NOT EXISTS subscriptions_ticker_idx ON subscriptions (ticker);
