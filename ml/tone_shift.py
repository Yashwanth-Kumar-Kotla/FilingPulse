import json
from pathlib import Path

MATERIAL_SHIFT_THRESHOLD = 0.15


def compute_tone_shifts(sentiment_path: str = "data/sentiment.json") -> list[dict]:
    records = json.loads(Path(sentiment_path).read_text())
    records.sort(key=lambda r: (r["ticker"], r["filing_date"]))

    results = []
    last_score_by_ticker = {}

    for record in records:
        ticker = record["ticker"]
        current_avg = record["avg_sentiment_score"]
        previous_avg = last_score_by_ticker.get(ticker)

        sentiment_shift = (
            current_avg - previous_avg if previous_avg is not None else 0.0
        )
        material_shift = abs(sentiment_shift) > MATERIAL_SHIFT_THRESHOLD

        results.append({
            "ticker": ticker,
            "filing_date": record["filing_date"],
            "form": record["form"],
            "avg_sentiment_score": current_avg,
            "sentiment_shift": sentiment_shift,
            "material_shift": material_shift,
        })

        last_score_by_ticker[ticker] = current_avg

    return results


if __name__ == "__main__":
    print("Computing quarter-over-quarter tone shifts...")
    shifts = compute_tone_shifts()

    material_count = sum(1 for s in shifts if s["material_shift"])
    print(f"  {len(shifts)} filings processed, {material_count} material shifts flagged")

    out = Path("data/tone_shifts.json")
    out.write_text(json.dumps(shifts, indent=2))
    print(f"Saved to {out}")
