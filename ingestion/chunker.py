# ingestion/chunker.py

from transformers import AutoTokenizer
import json
from pathlib import Path

TOKENIZER = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
CHUNK_SIZE = 512
OVERLAP    = 50

def chunk_text(text: str, metadata: dict) -> list[dict]:
    """
    Split cleaned filing text into 512-token chunks with 50-token overlap.
    Each chunk carries its metadata so nothing gets orphaned downstream.
    
    Returns list of dicts:
    {
        "text": str,
        "ticker": str,
        "filing_date": str,
        "form": str,           # 10-K or 10-Q
        "chunk_index": int,
        "total_chunks": int
    }
    """
    tokens = TOKENIZER.encode(text, add_special_tokens=False)
    
    chunks = []
    start  = 0
    index  = 0

    while start < len(tokens):
        end         = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text   = TOKENIZER.decode(chunk_tokens, skip_special_tokens=True)

        chunks.append({
            "text":        chunk_text,
            "ticker":      metadata["ticker"],
            "filing_date": metadata["filing_date"],
            "form":        metadata["form"],
            "chunk_index": index,
        })

        if end == len(tokens):
            break

        start += CHUNK_SIZE - OVERLAP
        index += 1

    # Inject total_chunks now that we know the count
    for chunk in chunks:
        chunk["total_chunks"] = len(chunks)

    return chunks


def chunk_all_filings(cleaned_dir: str = "data/cleaned") -> list[dict]:
    """
    Reads every .txt file in data/cleaned/, chunks it, returns
    a flat list of all chunks across all filings.
    Filename format expected: {TICKER}_{FORM}_{DATE}.txt
    e.g. AAPL_10-Q_2024-01-15.txt
    """
    all_chunks = []
    cleaned_path = Path(cleaned_dir)

    for filepath in sorted(cleaned_path.glob("*.txt")):
        # Parse metadata from filename
        stem  = filepath.stem            # AAPL_10-Q_2024-01-15
        parts = stem.split("_")

        if len(parts) < 3:
            print(f"Skipping malformed filename: {filepath.name}")
            continue

        metadata = {
            "ticker":      parts[0],
            "form":        parts[1],     # 10-K or 10-Q
            "filing_date": parts[2],
        }

        text = filepath.read_text(encoding="utf-8")

        if not text.strip():
            print(f"Skipping empty file: {filepath.name}")
            continue

        chunks = chunk_text(text, metadata)
        all_chunks.extend(chunks)
        print(f"{filepath.name} → {len(chunks)} chunks")

    return all_chunks


if __name__ == "__main__":
    all_chunks = chunk_all_filings()
    print(f"\nTotal chunks across all filings: {len(all_chunks)}")

    # Save for inspection before moving to FinBERT
    out_path = Path("data/chunks_preview.json")
    with open(out_path, "w") as f:
        json.dump(all_chunks[:20], f, indent=2)   # first 20 only for sanity check
    print(f"Preview saved to {out_path}")