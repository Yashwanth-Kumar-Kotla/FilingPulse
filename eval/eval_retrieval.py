import json
from pathlib import Path

from rag.query import retrieve_chunks

K_VALUES = [1, 3, 5]


def _is_match(chunk: dict, item: dict) -> bool:
    return (
        chunk["ticker"] == item["expected_ticker"]
        and str(chunk["filing_date"]) == item["expected_date"]
        and chunk["form"] == item["expected_form"]
    )


def evaluate(golden_set_path: str = "eval/golden_set.json") -> dict:
    golden_set = json.loads(Path(golden_set_path).read_text())

    hits_at_k = {k: 0 for k in K_VALUES}

    for item in golden_set:
        chunks = retrieve_chunks(item["question"])
        for k in K_VALUES:
            if any(_is_match(c, item) for c in chunks[:k]):
                hits_at_k[k] += 1

    total = len(golden_set)
    precision = {k: hits / total for k, hits in hits_at_k.items()}
    return {"total": total, "hits": hits_at_k, "precision": precision}


if __name__ == "__main__":
    print("Evaluating retrieval against golden set...")
    results = evaluate()

    print(f"\n{'k':<5}{'hits':<8}{'precision@k':<15}")
    for k in K_VALUES:
        print(f"{k:<5}{results['hits'][k]:<8}{results['precision'][k]:<15.3f}")
