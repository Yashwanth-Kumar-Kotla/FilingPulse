import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from db.subscriptions import init_db, subscribe
from rag.query import answer_question

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://filingpulse:filingpulse@localhost:5432/filingpulse"
)

app = FastAPI(title="FilingPulse API")
engine = create_engine(DATABASE_URL)


@app.on_event("startup")
def on_startup():
    init_db()


class QueryRequest(BaseModel):
    question: str
    ticker: str | None = None
    date: str | None = None


class SubscribeRequest(BaseModel):
    email: str
    ticker: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sentiment")
def sentiment(ticker: str):
    query = text("""
        SELECT ticker, filing_date, form, avg_sentiment_score,
               sentiment_shift, material_shift
        FROM filing_sentiment
        WHERE ticker = :ticker
        ORDER BY filing_date
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"ticker": ticker.upper()}).mappings().all()

    return {"ticker": ticker.upper(), "history": [dict(row) for row in rows]}


@app.post("/query")
def query(req: QueryRequest):
    return answer_question(req.question, req.ticker, req.date)


@app.post("/subscribe")
def subscribe_endpoint(req: SubscribeRequest):
    subscribe(req.email, req.ticker)
    return {"status": "subscribed", "email": req.email, "ticker": req.ticker.upper()}
