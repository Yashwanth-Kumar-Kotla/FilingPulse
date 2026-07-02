import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import create_engine, text

from db.subscriptions import init_db, subscribe
from rag.query import answer_question

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://filingpulse:filingpulse@localhost:5432/filingpulse"
)

app = FastAPI(title="FilingPulse API")
engine = create_engine(DATABASE_URL)

# In-memory, per-IP rate limiting. Resets on redeploy/restart and only works
# correctly for a single instance — acceptable here since /query costs real
# OpenAI spend per call and this is a single-instance deployment.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.on_event("startup")
def on_startup():
    init_db()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
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
@limiter.limit("10/minute")
def query(request: Request, req: QueryRequest):
    return answer_question(req.question, req.ticker, req.date)


@app.post("/subscribe")
@limiter.limit("5/hour")
def subscribe_endpoint(request: Request, req: SubscribeRequest):
    subscribe(req.email, req.ticker)
    return {"status": "subscribed", "email": req.email, "ticker": req.ticker.upper()}
