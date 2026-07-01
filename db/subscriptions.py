import os

import resend
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://filingpulse:filingpulse@localhost:5432/filingpulse"
)

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL)
    return _engine


def init_db():
    with _get_engine().begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                id                          SERIAL PRIMARY KEY,
                email                       TEXT NOT NULL,
                ticker                      TEXT NOT NULL,
                created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_notified_filing_date   DATE,
                UNIQUE (email, ticker)
            )
        """))


def subscribe(email: str, ticker: str):
    with _get_engine().begin() as conn:
        conn.execute(
            text("""
                INSERT INTO subscriptions (email, ticker)
                VALUES (:email, :ticker)
                ON CONFLICT (email, ticker) DO NOTHING
            """),
            {"email": email, "ticker": ticker.upper()},
        )


def get_subscribers(ticker: str) -> list[str]:
    with _get_engine().connect() as conn:
        rows = conn.execute(
            text("SELECT email FROM subscriptions WHERE ticker = :ticker"),
            {"ticker": ticker.upper()},
        ).fetchall()
    return [row[0] for row in rows]


def update_last_notified(email: str, ticker: str, filing_date: str):
    with _get_engine().begin() as conn:
        conn.execute(
            text("""
                UPDATE subscriptions
                SET last_notified_filing_date = :filing_date
                WHERE email = :email AND ticker = :ticker
            """),
            {"filing_date": filing_date, "email": email, "ticker": ticker.upper()},
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
        with _get_engine().connect() as conn:
            already_notified = conn.execute(
                text("""
                    SELECT last_notified_filing_date FROM subscriptions
                    WHERE email = :email AND ticker = :ticker
                """),
                {"email": email, "ticker": ticker.upper()},
            ).fetchone()

        if already_notified and str(already_notified[0]) == filing_date:
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
    print("Ensured subscriptions table exists.")
