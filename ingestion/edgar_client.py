# ingestion/edgar_client.py

import requests
import time

HEADERS = {"User-Agent": "Yashwanth Kotla 20891a1228yashwanth@gmail.com"}

MY_TICKERS = [
    # Financials
    "JPM", "GS", "MS", "BAC", "WFC", "C", "SCHW", "V", "MA",
    "AXP", "BLK", "SPGI", "ICE", "CME", "PNC", "USB", "TFC", "COF",
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "CRM", "ORCL",
    "ADBE", "INTC", "CSCO", "IBM", "CRWD", "PANW", "FTNT", "NOW",
    "INTU", "AMD", "QCOM", "TXN",
    # Volatile
    "XOM", "CVX", "OXY", "F", "GM", "DAL", "AAL", "CCL",
    "FCX", "X", "TSLA", "UBER",
]


def get_ticker_cik_map(tickers: list[str] = MY_TICKERS) -> dict:
    """
    Step 1: Downloads SEC master ticker→CIK file.
    Returns dict like {'AAPL': '0000320193', 'JPM': '0000019617'}
    """
    url  = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()

    full_map = {
        v["ticker"]: str(v["cik_str"]).zfill(10)
        for v in data.values()
    }

    result = {}
    for ticker in tickers:
        if ticker in full_map:
            result[ticker] = full_map[ticker]
        else:
            print(f"  Warning: {ticker} not found in SEC map")

    return result


def get_filing_list(cik: str, years_back: int = 5,
                    headers: dict = HEADERS) -> list[dict]:
    """
    Step 2: Given a CIK, returns list of 10-K and 10-Q filings
    from the last `years_back` years.
    Each item: {form, filingDate, accessionNumber, primaryDocument}
    """
    from datetime import datetime, timedelta
    cutoff = (datetime.today() - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")

    url  = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    recent   = data["filings"]["recent"]
    filings  = []

    for i in range(len(recent["form"])):
        form = recent["form"][i]
        date = recent["filingDate"][i]

        if form not in ("10-K", "10-Q"):
            continue
        if date < cutoff:
            continue

        filings.append({
            "form":            form,
            "filingDate":      date,
            "accessionNumber": recent["accessionNumber"][i],
            "primaryDocument": recent["primaryDocument"][i],
        })

    return filings


def get_filing_url(cik: str, accession_number: str,
                   primary_document: str) -> str:
    """
    Builds the direct URL to a filing document on SEC EDGAR.
    """
    accession_clean = accession_number.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession_clean}/{primary_document}"
    )


def download_filing(cik: str, accession_number: str,
                    primary_document: str,
                    headers: dict = HEADERS) -> str | None:
    """
    Step 3a: Downloads raw HTML of a single filing.
    Returns HTML string or None if download fails.
    """
    url = get_filing_url(cik, accession_number, primary_document)
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"    Download failed: {url} — {e}")
        return None


if __name__ == "__main__":
    # Only runs when you execute edgar_client.py directly
    # Safe to test here without breaking other imports
    print("Testing edgar_client.py...")

    cik_map = get_ticker_cik_map(["AAPL", "JPM", "CRWD"])
    print(f"CIK map: {cik_map}")

    filings = get_filing_list(cik_map["AAPL"])
    print(f"AAPL filings found: {len(filings)}")
    print(f"Most recent: {filings[0]}")