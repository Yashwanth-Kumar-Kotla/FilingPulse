# FilingPulse

**FilingPulse** detects quarter-over-quarter shifts in management tone across SEC 10-K/10-Q filings, correlates those shifts against forward stock price movement, and lets you ask natural-language questions about *why* tone shifted — with answers grounded in cited source passages, not hallucinated.

It is not a "chat with your PDFs" RAG demo. The differentiator is the analytical layer underneath the chat: FinBERT sentiment scoring → statistical tone-shift detection → price correlation → retrieval-augmented Q&A that can reason over *both* the computed metrics and the underlying filing text.

Current corpus: **49 companies** (financials, tech, cyclicals), **967 filings**, **73,290 chunks**, spanning ~5 years / ~20 quarters each.

---

## Table of contents

1. [System architecture](#system-architecture)
2. [The data pipeline, step by step](#the-data-pipeline-step-by-step)
3. [The RAG query flow](#the-rag-query-flow)
4. [Repository layout](#repository-layout)
5. [Tech stack](#tech-stack)
6. [Running it](#running-it)
7. [API reference](#api-reference)
8. [Key design decisions](#key-design-decisions)
9. [Known limitations](#known-limitations)

---

## System architecture

```mermaid
flowchart TB
    subgraph external["External data sources"]
        EDGAR["SEC EDGAR API\n(10-K / 10-Q filings)"]
        YF["yfinance\n(stock prices)"]
    end

    subgraph pipeline["Offline pipeline (scripts/run_ingestion.py + ml/*)"]
        ING["ingestion/\nedgar_client → cleaner → chunker"]
        SENT["ml/sentiment.py\nFinBERT scoring"]
        SHIFT["ml/tone_shift.py\nQoQ delta detection"]
        PRICE["ml/price_signal.py\nprice correlation + t-test"]
    end

    subgraph storage["Storage"]
        PG[("Postgres + pgvector\nfiling_chunks, filing_sentiment")]
        SQLITE[("SQLite\nsubscriptions.db")]
        JSON["data/*.json\nintermediate artifacts"]
    end

    subgraph serving["Serving layer"]
        RAG["rag/query.py\nretrieval + grounded LLM answer"]
        API["FastAPI\n/sentiment /query /subscribe"]
        UI["Streamlit dashboard\ntrend chart + Q&A + alerts"]
    end

    EDGAR --> ING --> JSON
    ING --> SENT --> JSON
    SENT --> SHIFT --> JSON
    SHIFT --> PRICE --> JSON
    YF --> PRICE
    JSON -->|"db/vector_store.py\n(OpenAI embeddings)"| PG
    PG --> RAG --> API
    SQLITE --> API
    API --> UI
```

Everything above the "Serving layer" runs offline/batch (it's the analytical backbone); everything in the serving layer is what a user actually touches.

---

## The data pipeline, step by step

```mermaid
flowchart LR
    S1["1. Ticker → CIK\nmap (SEC master file)"] --> S2["2. Filing list\nper CIK, 10-K/10-Q,\nlast 5 years"]
    S2 --> S3a["3a. Download\nraw HTML"]
    S3a --> S3b["3b. Clean HTML →\nMD&A + Risk Factors text"]
    S3b --> S4["4. Chunk\n512 tokens / 50 overlap"]
    S4 --> S5["5. FinBERT\nsentiment per chunk"]
    S5 --> S6["6. QoQ tone-shift\ndetection (|Δ| > 0.15)"]
    S6 --> S7["7. Price correlation\n(yfinance + t-test)"]
    S6 --> S8["8. Embed + store\nin pgvector"]
    S8 --> S9["9. RAG query\n(retrieve + LLM answer)"]
    S9 --> S10["10. Retrieval eval\nprecision@k"]
```

| Step | Module | What it does | Why it matters |
|---|---|---|---|
| 1 | `ingestion/edgar_client.py` | Resolves each ticker to its 10-digit CIK via SEC's `company_tickers.json`. | EDGAR indexes everything by CIK, not ticker. |
| 2 | `ingestion/edgar_client.py` | Pulls the filing list for a CIK, filters to `10-K`/`10-Q`, last 5 years. **Pages through SEC's historical submission files** for high-volume filers (banks file thousands of 8-Ks/notes, which push older 10-Ks/10-Qs out of the `recent` window). | Without pagination, JPM/GS/BAC-style filers silently lose most of their history — this was a real bug we found and fixed. |
| 3a/3b | `ingestion/edgar_client.py`, `ingestion/cleaner.py` | Downloads raw HTML, strips scripts/tables/nav, **strips inline-XBRL hidden fact blocks** (`ix:header`/`display:none`), then extracts just the MD&A and Risk Factors sections via regex section markers. | Modern SEC filings embed the *entire* XBRL data model as hidden text in the HTML — without stripping it, "cleaned" text is mostly machine-readable tag soup, not prose. Also fixed a `DOTALL` regex bug that could silently swallow 99% of a filing's text (see [Known limitations](#known-limitations)). |
| 4 | `ingestion/chunker.py` | Splits cleaned text into 512-token chunks (FinBERT's hard limit) with 50-token overlap. | Overlap prevents losing context at chunk boundaries — important for both sentiment accuracy and RAG retrieval coherence. |
| 5 | `ml/sentiment.py` | Runs FinBERT (`yiyanghkust/finbert-tone`) on every chunk → `{label, confidence, numeric}`, aggregated into a token-weighted average per filing. | Domain-tuned model beats general LLMs on financial jargon, costs nothing per call, and is deterministic (no sampling variance). |
| 6 | `ml/tone_shift.py` | Sorts by ticker + date, computes `sentiment_shift = current − previous`, flags `material_shift` when `|shift| > 0.15`. | Turns a raw score into an actionable signal: "did tone change enough to matter this quarter?" |
| 7 | `ml/price_signal.py` | For material-shift filings, pulls 5-trading-day forward returns via yfinance and compares against baseline (non-shift) filings with a Welch's t-test. | Tests whether the tone signal actually predicts anything about price — the whole point of the project. |
| 8 | `db/vector_store.py` | Embeds every chunk with OpenAI `text-embedding-3-small` (1536-dim, batched 100/call) and stores in pgvector alongside its sentiment label. | See [Key design decisions](#key-design-decisions) for why we moved off a local embedding model. |
| 9 | `rag/query.py` | Answers natural-language questions — see the [dedicated flow below](#the-rag-query-flow). | |
| 10 | `eval/eval_retrieval.py` | Runs a hand-written golden set of Q&A pairs through retrieval, reports precision@1/3/5. | |

**Current run stats** (49/50 tickers — `X`/U.S. Steel dropped out of SEC's ticker map after its 2025 delisting):

- 967 filings → 73,290 chunks
- 328 filings (34%) flagged with a material tone shift
- Shift filings: **+0.29%** mean 5-day return vs **+1.03%** for baseline filings (t = −2.08, **p = 0.038** — statistically significant, and in the *opposite* direction of the naive "positive tone → positive return" hypothesis)

---

## The RAG query flow

This is the part users actually interact with, and it's more than "embed → search → ask an LLM" — two extra steps exist specifically to stop it from failing on realistic questions.

```mermaid
sequenceDiagram
    participant U as User
    participant Q as rag/query.py
    participant PG as pgvector
    participant FS as filing_sentiment
    participant LLM as GPT-4o-mini

    U->>Q: "why did sentiment peak for JPM?"
    Q->>Q: is this about our own sentiment/tone<br/>metric + a direction word?
    alt sentiment-metric question (e.g. peak/decline/shift)
        Q->>FS: SELECT the actual peak/trough/<br/>biggest-shift filing for this ticker
        FS-->>Q: filing_date, avg_sentiment_score, sentiment_shift
        Q->>Q: pin retrieval to that specific filing
    end
    Q->>LLM: rewrite question into filing-style language
    LLM-->>Q: "...material factors contributing to<br/>an improved risk outlook..."
    Q->>Q: embed rewritten question (OpenAI text-embedding-3-small)
    Q->>PG: cosine similarity search (top 5, optionally date-scoped)
    PG-->>Q: candidate chunks + similarity scores
    Q->>Q: groundedness gate:<br/>similarity < 0.3 AND no structured match → refuse
    Q->>LLM: [computed metric, if any] + retrieved chunks + original question
    LLM-->>Q: cited answer
    Q-->>U: answer + citations (ticker, form, date, excerpt)
```

**Why each piece exists:**

- **Query rewriting** — casual phrasing ("why did the sentiment went down") embeds poorly against formal filing prose. Rewriting into disclosure-style language before embedding measurably improves match quality (in testing, one failing query went from 0.03 cosine similarity to 0.40 after rewriting).
- **Sentiment-aware retrieval** — "sentiment peaked" describes a value in the `filing_sentiment` table, not something any filing says in words. Pure semantic search over chunk text can never answer that reliably — it needs a structured lookup first, then retrieval scoped to the *correct* filing.
- **Groundedness gate** (similarity < 0.3 → refuse) — the system prompt tells the LLM to answer only from context; the gate stops it from ever being handed context that's actually unrelated, which is what makes the "no relevant information found" response trustworthy rather than a coin flip.

---

## Repository layout

```
FilingPulse/
├── .env                      # API keys — never commit
├── docker-compose.yml        # postgres+pgvector / api / streamlit services
├── Dockerfile
├── requirements.txt
│
├── data/
│   ├── raw/                  # downloaded HTML filings (gitignored)
│   ├── cleaned/               # stripped plain text per filing (gitignored)
│   ├── chunks.json            # all chunks after chunking (gitignored)
│   ├── sentiment.json         # per-filing + per-chunk FinBERT scores
│   ├── tone_shifts.json       # QoQ deltas + material_shift flags
│   ├── price_signal.json      # price correlation + t-test summary
│   └── tickers.json           # ticker → CIK map
│
├── ingestion/
│   ├── edgar_client.py        # Steps 1-3: CIK map, filing list (+ pagination), download
│   ├── cleaner.py             # Step 3b: HTML → clean narrative text
│   └── chunker.py             # Step 4: 512-token chunks, 50-token overlap
│
├── ml/
│   ├── sentiment.py           # Step 5: FinBERT scoring per chunk
│   ├── tone_shift.py          # Step 6: QoQ delta + material-shift flag
│   └── price_signal.py        # Step 7: yfinance pull + t-test
│
├── db/
│   ├── schema.sql             # pgvector table definitions
│   ├── vector_store.py        # Step 8: OpenAI embeddings → pgvector
│   └── subscriptions.py       # SQLite subscription CRUD + Resend alerts
│
├── rag/
│   └── query.py               # Step 9: rewrite → retrieve → ground → answer
│
├── api/
│   └── main.py                # FastAPI: /sentiment /query /subscribe /health
│
├── app/
│   └── streamlit_app.py       # trend chart + Q&A + alert signup UI
│
├── eval/
│   ├── golden_set.json        # hand-written Q&A pairs for retrieval eval
│   └── eval_retrieval.py      # Step 10: precision@k scorer
│
└── scripts/
    └── run_ingestion.py       # orchestrates Steps 1-4
```

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Sentiment | FinBERT (`yiyanghkust/finbert-tone`) | Finance-domain fine-tuning beats general LLMs on filing jargon; free, deterministic, runs on CPU. |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) | Better retrieval quality on casually-phrased questions than a general-purpose local model (see [Key design decisions](#key-design-decisions)); embedding the entire 73K-chunk corpus costs well under $1. |
| Vector DB | pgvector (Postgres extension) | SQL + vector search in one engine; `ivfflat` cosine index. |
| RAG / LLM | LangChain (`langchain-openai`) + GPT-4o-mini | Cheap, fast, good enough for grounded short-answer synthesis. |
| Backend | FastAPI + Uvicorn | Async, typed, minimal boilerplate for 4 endpoints. |
| Frontend | Streamlit + Plotly | Fast to build an internal analytical dashboard; not meant to be a production consumer UI. |
| Subscriptions | SQLite + Resend | A single email-alert list doesn't need its own Postgres schema. |
| Data sources | SEC EDGAR (free, no auth) + yfinance | No paid data licensing required for either filings or prices. |
| Infra | Docker Compose | One command brings up Postgres; API/Streamlit can run in containers or directly in a venv. |

---

## Running it

```bash
# 1. Start pgvector
docker-compose up db -d

# 2. Install dependencies (Python 3.11 — 3.13+ lacks prebuilt torch/scipy wheels)
python3.11 -m venv venv
./venv/bin/pip install -r requirements.txt

# 3. Ingest filings (edit scripts/run_ingestion.py: TEST_MODE = True/False)
./venv/bin/python -m scripts.run_ingestion

# 4. Score sentiment (CPU-bound — ~0.1-0.25s/chunk; full 50-company run takes hours)
./venv/bin/python -m ml.sentiment

# 5. Detect tone shifts
./venv/bin/python -m ml.tone_shift

# 6. Correlate with price
./venv/bin/python -m ml.price_signal

# 7. Embed + store in pgvector
./venv/bin/python -m db.vector_store

# 8. (optional) Evaluate retrieval quality
./venv/bin/python -m eval.eval_retrieval

# 9. Start the API
./venv/bin/uvicorn api.main:app --reload

# 10. Start the dashboard
./venv/bin/streamlit run app/streamlit_app.py
```

`.env` requires:

```
OPENAI_API_KEY=...       # embeddings + RAG answer generation
RESEND_API_KEY=...       # only needed for email alerts
DATABASE_URL=postgresql://filingpulse:filingpulse@localhost:5432/filingpulse
```

---

## API reference

| Endpoint | Method | Body / Params | Returns |
|---|---|---|---|
| `/health` | GET | — | `{"status": "ok"}` |
| `/sentiment` | GET | `?ticker=AAPL` | Full tone-shift history for that ticker |
| `/query` | POST | `{"question": str, "ticker": str\|null, "date": str\|null}` | `{"answer": str, "citations": [...]}` |
| `/subscribe` | POST | `{"email": str, "ticker": str}` | Subscribes an email to tone-shift alerts for that ticker |

---

## Key design decisions

- **FinBERT over GPT for sentiment.** Domain fine-tuning wins on financial jargon, costs nothing per call at 73K-chunk scale, and is deterministic — no sampling variance to explain away.
- **Strip tables, keep only MD&A + Risk Factors.** Financial tables (revenue, EPS line items) inflate chunk count without carrying tone signal — only narrative prose reflects management's actual language choices.
- **512-token chunks / 50-token overlap.** 512 is FinBERT's hard input ceiling; the overlap avoids losing context exactly at a chunk boundary, which matters for both sentiment accuracy and RAG coherence.
- **49 targeted companies over S&P 500 breadth.** A deep 20-quarter time series per company is more analytically meaningful than one snapshot each across 500 companies — and cyclical sectors (airlines, energy, autos) were included deliberately because they show genuine tone swings that heavily lawyered mega-cap language often doesn't.
- **OpenAI embeddings over a local model.** We started with `all-MiniLM-L6-v2` (free, self-hosted) but found it was too sensitive to phrasing — a real user question ("why did the sentiment went down for CRWD?") scored 0.03 cosine similarity against clearly-relevant filing text, while a more formally-phrased version of the same question scored 0.40. Switching to `text-embedding-3-small` (kept at full 1536 dimensions, not truncated) fixed this directly, and at ~$0.60 to embed the entire 73K-chunk corpus, cost was never the constraint.
- **Groundedness guard (similarity < 0.3 → refuse).** Prevents the LLM from answering when nothing relevant was actually retrieved — this is what makes "no relevant information found" a meaningful signal instead of the model just guessing anyway.
- **Sentiment-aware retrieval for metric questions.** "Why did tone peak/decline" describes a value in `filing_sentiment`, not text any filing contains. Rather than ask an LLM to explain a number it's never shown, we look the number up directly and scope retrieval to the filing it came from.
- **SQLite for subscriptions, not another Postgres schema.** A single email/ticker list doesn't need vector search or SQL joins — keeping it separate keeps the pgvector schema focused on retrieval.
- **`langchain-community` avoided.** It's sunset/archived; all pgvector integration goes through raw SQLAlchemy instead of a LangChain vectorstore wrapper, and `langchain-openai` is used directly for the LLM call.

---

## Known limitations

- **`eval/golden_set.json` expected dates/hints are placeholders**, not verified against real filing dates for all 49 tickers — treat `eval_retrieval.py` output as a smoke test, not a real precision benchmark, until the golden set is rebuilt from actual filings.
- **Two LLM calls per RAG query** (rewrite + answer) roughly double per-query latency and cost versus a single-call design — still cheap on GPT-4o-mini, but worth knowing if usage scales up.
- **`X` (U.S. Steel) is missing** from the corpus — it fell out of SEC's `company_tickers.json` after its 2025 acquisition/delisting, not a pipeline bug.
- **FinBERT sentiment scoring is CPU-bound and slow at scale** — roughly 0.1-0.25s/chunk, meaning a full 50-company/5-year run takes multiple hours on a laptop CPU. There's no GPU acceleration path configured.
- **Historical bugs fixed during development, worth knowing if extending the cleaner/ingestion code:**
  - `ingestion/cleaner.py`'s original boilerplate-stripping regex used `.*?\n` with `DOTALL` — since `get_text()` output has very few literal newlines, this could silently consume almost an entire filing's text before finding the next `\n`. Now bounded to a fixed character span instead.
  - Inline-XBRL filings hide their entire tagged-fact data model as text inside `<ix:header>`/`display:none` blocks — without stripping these first, "cleaned" narrative text was mostly machine tags, not prose.
  - `ingestion/edgar_client.py`'s filing-list fetch only read SEC's `recent` filings block, which silently drops older 10-Ks/10-Qs for high-volume filers (banks issuing thousands of 8-Ks/structured notes) — now pages through historical submission files when needed.
