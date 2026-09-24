# 🧠 Enterprise-Grade RAG

**A RAG system that decides where to look, instead of always looking in the same place.**

Most RAG demos wire one vector store to one LLM and call it done — every question gets searched the same way, whether or not that's the right place to search. This project gives Claude **tools** instead of a fixed pipeline: an internal HR policy index, a financial-filings index, and live web search. For each incoming question, the agent reasons about which of those it actually needs — zero, one, or several — calls them, reads what comes back, and decides whether it has enough to answer or needs to call another tool. Only then does it write one grounded, cited answer.

That's the difference between a toy chatbot and something you could put in front of real users: a compound question spanning HR policy *and* company financials gets both sources pulled in the same turn, and a question about neither falls back to the web instead of hitting a dead end.

## 🏗️ How it works

![Enterprise RAG agent architecture](assets/agent_architecture.svg)

1. A user question enters the agent loop along with three tool definitions.
2. The agent decides what it needs and calls:
   - **`search_hr_docs`** — internal HR policy (PTO, leave, benefits, payroll, onboarding, performance reviews), retrieved from a Qdrant vector index built from the employee handbook.
   - **`search_10k_docs`** — Uber & Lyft 2023 10-K filings (revenue, costs, risk factors), retrieved from a separate Qdrant index so financial and HR content never mix.
   - **`search_web`** — live web search (Firecrawl) for anything outside the indexed documents, current events, or version/date-sensitive facts. Before the agent ever sees them, results pass through an LLM grading step that drops irrelevant or stale pages.
3. Each tool result goes back to the agent, which reasons again — enough to answer, or another call needed?
4. Once satisfied, the agent exits the loop and writes one answer grounded in whatever it retrieved.

## 🎬 See it run

This is a real transcript from `python demo.py` — it ingests the actual HR handbook and 10-K filings from `data/`, then fires three questions at the agent loop.

**HR question → routed to `search_hr_docs`:**
```
=== How many PTO days do I get? ===
Based on the HR policy documents, as a full-time employee, you get 15 days of
paid time off (PTO) per calendar year.

- Accrual rate: 1.25 days per month (totaling 15 days annually)
- Carryover: up to 5 days into the next calendar year; the rest is forfeited
  on December 31st
- Request process: submit through the HR portal at least 5 business days in
  advance; managers approve/deny within 2 business days
```

**Financial question → routed to `search_10k_docs`:**
```
=== What was Uber's 2023 revenue? ===
Based on Uber's 2023 10-K filing, Uber's 2023 revenue was $37.281 billion,
representing a 17% increase compared to 2022 revenue of $31.877 billion.

Driven by:
- Mobility: +$5.8 billion (31% increase in Gross Bookings)
- Delivery: +$1.3 billion (14% increase in Gross Bookings)
- Freight: -$1.7 billion (Gross Bookings declined 25%)
```

**Question outside both indexes → falls back to `search_web`, with grading:**
```
=== What's the latest version of Python released? ===
[Note: time-sensitive query — results restricted to the past month]
Based on the search results, Python 3.14.6 is the latest stable release.
```

Every answer is grounded in the actual retrieved text, not the model's own memory — the agent never had this data in its training, it read it out of `data/` and the live web at query time.

## 🚀 Run it yourself

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .example.env .env   # fill in ANTHROPIC_API_KEY and FIRECRAWL_API_KEY
python demo.py
```

No Qdrant account needed — with `QDRANT_URL` unset, it falls back to an in-process, in-memory Qdrant instance (non-persistent between runs, which is fine for the demo).

To serve it over HTTP instead of running the script:

```bash
uvicorn enterprise_rag.api:app --reload
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many PTO days do I get?"}'
```

`/query` runs a raw Anthropic SDK tool-use loop; `/query_langchain` runs an equivalent LangChain loop against the same tools — both are wired up so the orchestration layer is swappable without touching the tools themselves.

## 🧰 Built with

Claude (`anthropic` / `langchain-anthropic`), Qdrant for vector search, local `sentence-transformers` embeddings, Firecrawl for web search, and FastAPI. A few extra libraries (`rank_bm25`, `chromadb`, `faiss-cpu`) are installed for retrieval work that's planned but not wired in yet — see Roadmap.

## 📌 Status

**Implemented**

- Both agent loops (Anthropic SDK + LangChain), sharing one tool set, tested end to end including max-turn exhaustion and API-error handling
- `search_web`, `search_hr_docs`, `search_10k_docs` tools, tested
- LLM-based relevance grading + time-sensitivity handling for `search_web`
- FastAPI endpoints (`/query`, `/query_langchain`), tested, returning 500 on agent failure
- Real ingestion: loads and token-aware chunks the actual documents under `data/` (HR handbook, Uber/Lyft 10-Ks) into Qdrant — `demo.py` runs this end to end, not against hardcoded example chunks
- GitHub Actions CI running the test suite on every push

**Roadmap — not yet done**

- Test coverage for `ingestion.py` / `pipeline.py`
- Relevance grading + corrective retry (rewrite query / retry / fall back) for `search_hr_docs` and `search_10k_docs` — today that loop only exists for `search_web`
- Query rewriting — normalizing casual/compound user queries before retrieval
- Hybrid retrieval (dense + BM25 fusion)
- Persistent Qdrant deployment for production use
- An evaluation suite (retrieval precision, answer groundedness) to measure the impact of the corrective loop once it exists
- A demoable UI (Streamlit/Gradio) on top of the FastAPI endpoints
