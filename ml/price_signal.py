import json
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
from scipy import stats


def _get_price_on_or_after(history, target_date: datetime):
    for date, row in history.iterrows():
        if date.to_pydatetime().replace(tzinfo=None) >= target_date:
            return float(row["Close"])
    return None


def _five_day_return(ticker_history, filing_date: str) -> float | None:
    day0 = datetime.strptime(filing_date, "%Y-%m-%d")
    price0 = _get_price_on_or_after(ticker_history, day0)
    if price0 is None:
        return None

    naive_index = ticker_history.index.tz_localize(None)
    window = ticker_history[naive_index >= day0]
    if len(window) < 6:
        return None
    price5 = float(window.iloc[5]["Close"])

    return (price5 / price0 - 1) * 100


def compute_price_signal(tone_shifts_path: str = "data/tone_shifts.json") -> dict:
    records = json.loads(Path(tone_shifts_path).read_text())

    tickers = sorted({r["ticker"] for r in records})
    histories = {}
    for ticker in tickers:
        print(f"  Fetching price history for {ticker}...")
        histories[ticker] = yf.Ticker(ticker).history(period="5y")

    enriched = []
    shift_returns = []
    baseline_returns = []

    for record in records:
        ticker_history = histories.get(record["ticker"])
        if ticker_history is None or ticker_history.empty:
            continue

        price_change_pct = _five_day_return(ticker_history, record["filing_date"])
        if price_change_pct is None:
            continue

        enriched.append({**record, "price_change_pct": price_change_pct})

        if record["material_shift"]:
            shift_returns.append(price_change_pct)
        else:
            baseline_returns.append(price_change_pct)

    summary = {"shift_n": len(shift_returns), "baseline_n": len(baseline_returns)}

    if len(shift_returns) >= 2 and len(baseline_returns) >= 2:
        t_stat, p_value = stats.ttest_ind(shift_returns, baseline_returns, equal_var=False)
        summary["shift_mean_return_pct"] = sum(shift_returns) / len(shift_returns)
        summary["baseline_mean_return_pct"] = sum(baseline_returns) / len(baseline_returns)
        summary["t_stat"] = float(t_stat)
        summary["p_value"] = float(p_value)
    else:
        summary["note"] = "Not enough samples in one or both groups for a t-test"

    return {"records": enriched, "summary": summary}


if __name__ == "__main__":
    print("Computing price signal around material tone shifts...")
    result = compute_price_signal()

    out = Path("data/price_signal.json")
    out.write_text(json.dumps(result, indent=2))

    print(f"\nSaved {len(result['records'])} enriched records to {out}")
    print("\nSummary:")
    for key, value in result["summary"].items():
        print(f"  {key}: {value}")
