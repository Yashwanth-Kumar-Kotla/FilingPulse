import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "yiyanghkust/finbert-tone"
LABEL_TO_NUMERIC = {"positive": 1, "neutral": 0, "negative": -1}

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def score_chunk(text: str) -> dict:
    tokenizer, model = _load_model()
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    label_id = int(torch.argmax(probs).item())
    label = model.config.id2label[label_id].lower()
    confidence = float(probs[label_id].item())

    return {
        "label": label,
        "score": confidence,
        "numeric": LABEL_TO_NUMERIC.get(label, 0),
    }


def score_all_chunks(chunks_path: str = "data/chunks.json") -> list[dict]:
    chunks = json.loads(Path(chunks_path).read_text())

    grouped = defaultdict(list)
    for chunk in chunks:
        key = (chunk["ticker"], chunk["filing_date"], chunk["form"])
        grouped[key].append(chunk)

    results = []
    for (ticker, filing_date, form), filing_chunks in grouped.items():
        chunk_scores = []
        weighted_sum = 0.0
        weight_total = 0

        for chunk in sorted(filing_chunks, key=lambda c: c["chunk_index"]):
            scored = score_chunk(chunk["text"])
            token_count = len(chunk["text"].split())

            chunk_scores.append({
                "chunk_index": chunk["chunk_index"],
                "label": scored["label"],
                "score": scored["score"],
                "numeric": scored["numeric"],
            })

            weighted_sum += scored["numeric"] * token_count
            weight_total += token_count

        avg_sentiment_score = weighted_sum / weight_total if weight_total else 0.0

        results.append({
            "ticker": ticker,
            "filing_date": filing_date,
            "form": form,
            "avg_sentiment_score": avg_sentiment_score,
            "chunk_scores": chunk_scores,
        })

        print(f"  {ticker} {form} {filing_date} → avg sentiment {avg_sentiment_score:.3f}")

    return results


if __name__ == "__main__":
    print("Scoring chunks with FinBERT...")
    results = score_all_chunks()

    out = Path("data/sentiment.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} filing sentiment records to {out}")
