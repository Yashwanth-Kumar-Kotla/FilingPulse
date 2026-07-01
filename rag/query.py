import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, text

from db.vector_store import embed_texts

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://filingpulse:filingpulse@localhost:5432/filingpulse"
)
TOP_K = 5
SIMILARITY_THRESHOLD = 0.3

SYSTEM_PROMPT = (
    "Answer only from the provided context. Cite which filing "
    "(ticker, date, form) each piece of information comes from."
)

REWRITE_SYSTEM_PROMPT = (
    "Rewrite the user's question into the formal, disclosure-style language "
    "used in SEC 10-K/10-Q filings (Risk Factors, MD&A), preserving its intent. "
    "Output only the rewritten question, nothing else."
)

# Words that signal the user is asking about our own computed tone/sentiment
# metric (which lives in filing_sentiment) rather than filing content directly.
SENTIMENT_METRIC_WORDS = {"sentiment", "tone"}
POSITIVE_DIRECTION_WORDS = {
    "peak", "peaked", "high", "highest", "improve", "improved", "improvement",
    "better", "increase", "increased", "up", "rise", "rose",
}
NEGATIVE_DIRECTION_WORDS = {
    "decline", "declined", "drop", "dropped", "worse", "worsen", "worsened",
    "decrease", "decreased", "low", "lowest", "down", "fall", "fell",
}

_engine = None
_llm = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL)
    return _engine


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _llm


def rewrite_query(question: str) -> str:
    """Rephrase a casually-worded question into filing-style language so its
    embedding lands closer to how SEC filings actually talk, improving recall
    against MiniLM/OpenAI cosine similarity search."""
    llm = _get_llm()
    response = llm.invoke([
        ("system", REWRITE_SYSTEM_PROMPT),
        ("user", question),
    ])
    return response.content.strip() or question


def _is_sentiment_metric_question(question: str) -> bool:
    words = set(question.lower().replace("?", "").split())
    return bool(words & SENTIMENT_METRIC_WORDS)


def find_sentiment_extremum(ticker: str, question: str) -> dict | None:
    """For questions about our own computed sentiment/tone metric (e.g. 'why
    did tone peak/decline for X'), look up the actual quarter that matches —
    this is structured data, not something any filing states in words."""
    words = set(question.lower().replace("?", "").split())

    if words & NEGATIVE_DIRECTION_WORDS:
        order_by = "avg_sentiment_score ASC"
    elif words & POSITIVE_DIRECTION_WORDS:
        order_by = "avg_sentiment_score DESC"
    else:
        order_by = "ABS(sentiment_shift) DESC"

    query = text(f"""
        SELECT ticker, filing_date, form, avg_sentiment_score, sentiment_shift
        FROM filing_sentiment
        WHERE ticker = :ticker
        ORDER BY {order_by}
        LIMIT 1
    """)

    with _get_engine().connect() as conn:
        row = conn.execute(query, {"ticker": ticker.upper()}).mappings().first()

    return dict(row) if row else None


def retrieve_chunks(question: str, ticker: str | None = None, date: str | None = None) -> list[dict]:
    question_embedding = embed_texts([question])[0]

    filters = []
    params = {"embedding": str(question_embedding), "top_k": TOP_K}

    if ticker:
        filters.append("ticker = :ticker")
        params["ticker"] = ticker.upper()
    if date:
        filters.append("filing_date = :date")
        params["date"] = date

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    query = text(f"""
        SELECT ticker, filing_date, form, chunk_text,
               1 - (embedding <=> :embedding) AS similarity
        FROM filing_chunks
        {where_clause}
        ORDER BY embedding <=> :embedding
        LIMIT :top_k
    """)

    with _get_engine().connect() as conn:
        rows = conn.execute(query, params).mappings().all()

    return [dict(row) for row in rows]


def answer_question(question: str, ticker: str | None = None, date: str | None = None) -> dict:
    extremum = None
    if ticker and not date and _is_sentiment_metric_question(question):
        extremum = find_sentiment_extremum(ticker, question)
        if extremum:
            date = str(extremum["filing_date"])

    rewritten_question = rewrite_query(question)
    chunks = retrieve_chunks(rewritten_question, ticker, date)

    # A structured sentiment-extremum match is already grounded in real
    # numbers, so it bypasses the semantic-similarity gate; otherwise a
    # narrowly-scoped single-filing search can score lower than open search.
    if not extremum and (not chunks or chunks[0]["similarity"] < SIMILARITY_THRESHOLD):
        return {
            "answer": "No relevant information found.",
            "citations": [],
        }

    context = "\n\n".join(
        f"[{c['ticker']} {c['form']} {c['filing_date']}]: {c['chunk_text']}"
        for c in chunks
    )

    if extremum:
        context = (
            f"[Computed FilingPulse metric] {extremum['ticker']} {extremum['form']} "
            f"filed {extremum['filing_date']}: average sentiment score = "
            f"{extremum['avg_sentiment_score']:.3f} (quarter-over-quarter shift = "
            f"{extremum['sentiment_shift']:+.3f}).\n\n"
        ) + context

    llm = _get_llm()
    response = llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("user", f"Context:\n{context}\n\nQuestion: {question}"),
    ])

    citations = [
        {
            "ticker": c["ticker"],
            "filing_date": str(c["filing_date"]),
            "form": c["form"],
            "excerpt": c["chunk_text"][:300],
        }
        for c in chunks
    ]

    return {"answer": response.content, "citations": citations}


if __name__ == "__main__":
    result = answer_question("Why did management tone shift this quarter?")
    print(result["answer"])
    for citation in result["citations"]:
        print(f"  - {citation['ticker']} {citation['form']} {citation['filing_date']}")
