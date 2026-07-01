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
    chunks = retrieve_chunks(question, ticker, date)

    if not chunks or chunks[0]["similarity"] < SIMILARITY_THRESHOLD:
        return {
            "answer": "No relevant information found.",
            "citations": [],
        }

    context = "\n\n".join(
        f"[{c['ticker']} {c['form']} {c['filing_date']}]: {c['chunk_text']}"
        for c in chunks
    )

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
