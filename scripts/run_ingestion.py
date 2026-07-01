import json
import time
from pathlib import Path

from ingestion.edgar_client import get_ticker_cik_map, get_filing_list, download_filing
from ingestion.cleaner import clean_html, is_meaningful
from ingestion.chunker import chunk_all_filings

RAW_DIR     = Path("data/raw")
CLEANED_DIR = Path("data/cleaned")
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

TEST_MODE    = True   # flip to False for full 50-company run
TEST_TICKERS = ["AAPL", "JPM", "CRWD"]

ALL_TICKERS = [
    "JPM", "GS", "MS", "BAC", "WFC", "C", "SCHW", "V", "MA",
    "AXP", "BLK", "SPGI", "ICE", "CME", "PNC", "USB", "TFC", "COF",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "CRM", "ORCL",
    "ADBE", "INTC", "CSCO", "IBM", "CRWD", "PANW", "FTNT", "NOW",
    "INTU", "AMD", "QCOM", "TXN",
    "XOM", "CVX", "OXY", "F", "GM", "DAL", "AAL", "CCL",
    "FCX", "X", "TSLA", "UBER",
]


def run():
    tickers = TEST_TICKERS if TEST_MODE else ALL_TICKERS

    # Step 1: ticker → CIK
    print("=" * 50)
    print("Step 1: Building ticker → CIK map...")
    print("=" * 50)
    ticker_cik = get_ticker_cik_map(tickers)
    Path("data/tickers.json").write_text(json.dumps(ticker_cik, indent=2))
    print(f"Resolved {len(ticker_cik)}/{len(tickers)} tickers\n")

    total_downloaded = 0
    total_skipped    = 0
    total_failed     = 0

    for ticker, cik in ticker_cik.items():
        print(f"Processing {ticker} (CIK: {cik})")
        print("-" * 40)

        # Step 2: get filing list
        filings = get_filing_list(cik, years_back=5)
        print(f"  Found {len(filings)} filings")

        if not filings:
            print(f"  No filings found, skipping\n")
            continue

        for filing in filings:
            date      = filing["filingDate"]
            form      = filing["form"].replace("/", "-")
            accession = filing["accessionNumber"]
            doc       = filing["primaryDocument"]

            filename   = f"{ticker}_{form}_{date}"
            raw_path   = RAW_DIR     / f"{filename}.html"
            clean_path = CLEANED_DIR / f"{filename}.txt"

            if clean_path.exists():
                print(f"    Skipping {filename} (already exists)")
                total_skipped += 1
                continue

            # Step 3a: download
            print(f"    Downloading {filename}...", end=" ")
            raw_html = download_filing(cik, accession, doc)

            if not raw_html:
                print("FAILED")
                total_failed += 1
                continue

            raw_path.write_text(raw_html, encoding="utf-8")

            # Step 3b: clean
            clean_text = clean_html(raw_html)

            if not is_meaningful(clean_text):
                print("EMPTY after cleaning")
                total_failed += 1
                continue

            clean_path.write_text(clean_text, encoding="utf-8")
            print(f"OK ({len(clean_text):,} chars)")
            total_downloaded += 1

            time.sleep(0.15)

        print()

    print("=" * 50)
    print(f"Downloaded : {total_downloaded}")
    print(f"Skipped    : {total_skipped}")
    print(f"Failed     : {total_failed}")
    print("=" * 50)

    # Step 4: chunk
    print("\nStep 4: Chunking all cleaned filings...")
    all_chunks = chunk_all_filings(str(CLEANED_DIR))
    print(f"  Total chunks: {len(all_chunks):,}")

    chunks_path = Path("data/chunks.json")
    chunks_path.write_text(json.dumps(all_chunks, indent=2))
    print(f"  Saved to {chunks_path}")

    if all_chunks:
        s = all_chunks[0]
        print(f"\nSample chunk:")
        print(f"  Ticker      : {s['ticker']}")
        print(f"  Form        : {s['form']}")
        print(f"  Filing date : {s['filing_date']}")
        print(f"  Chunks      : {s['chunk_index']} of {s['total_chunks']}")
        print(f"  Preview     : {s['text'][:200]}...")

    print("\nIngestion complete. Ready for ml/sentiment.py")


if __name__ == "__main__":
    run()
