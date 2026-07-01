import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://filingpulse:filingpulse@localhost:5432/filingpulse"
)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 100

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = _get_client()
    embeddings = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embeddings.extend(item.embedding for item in response.data)

    return embeddings


def _load_sentiment_lookup(sentiment_path: str = "data/sentiment.json") -> dict:
    records = json.loads(Path(sentiment_path).read_text())
    lookup = {}
    for record in records:
        key = (record["ticker"], record["filing_date"], record["form"])
        for chunk_score in record["chunk_scores"]:
            lookup[(key, chunk_score["chunk_index"])] = chunk_score
    return lookup


def store_chunks(
    chunks_path: str = "data/chunks.json",
    sentiment_path: str = "data/sentiment.json",
    tone_shifts_path: str = "data/tone_shifts.json",
):
    chunks = json.loads(Path(chunks_path).read_text())
    sentiment_lookup = _load_sentiment_lookup(sentiment_path)
    tone_shifts = json.loads(Path(tone_shifts_path).read_text())

    engine = create_engine(DATABASE_URL)

    print(f"  Embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")
    embeddings = embed_texts([c["text"] for c in chunks])

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE filing_chunks RESTART IDENTITY"))

        for chunk, embedding in zip(chunks, embeddings):
            key = (chunk["ticker"], chunk["filing_date"], chunk["form"])
            chunk_score = sentiment_lookup.get((key, chunk["chunk_index"]), {})

            conn.execute(
                text("""
                    INSERT INTO filing_chunks
                        (ticker, filing_date, form, chunk_index, total_chunks,
                         chunk_text, sentiment_label, sentiment_score, embedding)
                    VALUES
                        (:ticker, :filing_date, :form, :chunk_index, :total_chunks,
                         :chunk_text, :sentiment_label, :sentiment_score, :embedding)
                """),
                {
                    "ticker": chunk["ticker"],
                    "filing_date": chunk["filing_date"],
                    "form": chunk["form"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks"],
                    "chunk_text": chunk["text"],
                    "sentiment_label": chunk_score.get("label"),
                    "sentiment_score": chunk_score.get("numeric"),
                    "embedding": str(embedding),
                },
            )

        print(f"  Inserted {len(chunks)} chunks into filing_chunks")

        for shift in tone_shifts:
            conn.execute(
                text("""
                    INSERT INTO filing_sentiment
                        (ticker, filing_date, form, avg_sentiment_score,
                         sentiment_shift, material_shift)
                    VALUES
                        (:ticker, :filing_date, :form, :avg_sentiment_score,
                         :sentiment_shift, :material_shift)
                    ON CONFLICT (ticker, filing_date, form) DO UPDATE SET
                        avg_sentiment_score = EXCLUDED.avg_sentiment_score,
                        sentiment_shift = EXCLUDED.sentiment_shift,
                        material_shift = EXCLUDED.material_shift
                """),
                shift,
            )

        print(f"  Inserted/updated {len(tone_shifts)} rows in filing_sentiment")


if __name__ == "__main__":
    print("Embedding chunks and storing in pgvector...")
    store_chunks()
    print("Done.")
