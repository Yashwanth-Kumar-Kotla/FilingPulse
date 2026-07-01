import os
import sqlite3
from contextlib import contextmanager

import resend

DB_PATH = os.getenv("SUBSCRIPTIONS_DB_PATH", "subscriptions.db")


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                ticker TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_notified_filing_date TEXT,
                UNIQUE (email, ticker)
            )
        """)


def subscribe(email: str, ticker: str):
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO subscriptions (email, ticker) VALUES (?, ?)",
            (email, ticker.upper()),
        )


def get_subscribers(ticker: str) -> list[str]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT email FROM subscriptions WHERE ticker = ?", (ticker.upper(),)
        ).fetchall()
    return [row[0] for row in rows]


def update_last_notified(email: str, ticker: str, filing_date: str):
    with _connect() as conn:
        conn.execute(
            """
            UPDATE subscriptions
            SET last_notified_filing_date = ?
            WHERE email = ? AND ticker = ?
            """,
            (filing_date, email, ticker.upper()),
        )


def notify_shift(ticker: str, filing_date: str, shift_value: float, top_chunk_text: str):
    resend.api_key = os.getenv("RESEND_API_KEY")

    subscribers = get_subscribers(ticker)
    if not subscribers:
        return

    direction = "improved" if shift_value > 0 else "deteriorated"
    subject = f"FilingPulse Alert: {ticker} management tone {direction}"
    body = (
        f"<p>{ticker}'s management tone {direction} by {abs(shift_value):.2f} "
        f"in the {filing_date} filing.</p>"
        f"<p><strong>Excerpt:</strong> {top_chunk_text[:500]}...</p>"
    )

    for email in subscribers:
        with _connect() as conn:
            already_notified = conn.execute(
                """
                SELECT last_notified_filing_date FROM subscriptions
                WHERE email = ? AND ticker = ?
                """,
                (email, ticker.upper()),
            ).fetchone()

        if already_notified and already_notified[0] == filing_date:
            continue

        resend.Emails.send({
            "from": "alerts@filingpulse.dev",
            "to": email,
            "subject": subject,
            "html": body,
        })
        update_last_notified(email, ticker, filing_date)


if __name__ == "__main__":
    init_db()
    print(f"Initialized subscriptions DB at {DB_PATH}")
