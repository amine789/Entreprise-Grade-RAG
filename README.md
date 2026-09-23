# 🧠 Enterprise-Grade RAG

**A production-style Retrieval-Augmented Generation system that doesn't just retrieve — it *decides where to look*.**

Most RAG demos bolt a single vector store onto an LLM and call it a day. This project goes further: Claude is given **tools**, not a fixed path — it reasons about each incoming query, decides which knowledge source(s) it actually needs (an internal HR policy index, a financial-filings index, live web search — zero, one, or several), calls them, reads the results, and repeats until it has enough to answer. It's the difference between a toy chatbot and something you could actually put in front of enterprise users.

## 🏗️ Architecture

![Enterprise RAG agent architecture](assets/agent_architecture.svg)

The flow is an **agentic tool-use loop**, not a one-shot classifier:

1. **User Query** enters the **Claude agent loop** along with three tool definitions.
2. The agent reasons about what it needs and calls **`search_hr_docs`** (Qdrant: internal HR policy documents — PTO/leave policy, benefits, payroll, onboarding, the employee handbook), **`search_10k_docs`** (Qdrant: Uber & Lyft 10-K annual filings — financial performance, revenue, operating metrics), **`search_web`** (SerpAPI/Google — current events, comparisons, anything outside the indexed corpora), or any combination of them — a compound question can trigger more than one tool in the same turn.
3. Each tool's result goes back to the agent, which reasons again: enough context to answer, or another call needed?
4. Once satisfied, the agent **exits the loop** and generates one coherent, cited answer from everything it gathered.

This is **agentic retrieval** — instead of committing to a single path up front, the model controls its own retrieval strategy at runtime, which is what lets it actually handle compound questions spanning more than one knowledge source.

## ✨ Why this is more than a basic RAG pipeline

| Capability | What it buys you |
|---|---|
| 🧭 **Agentic tool selection** | No manual keyword rules or fixed branches — the agent reasons about which sources it actually needs, including more than one per query |
| 🗂️ **Domain-partitioned vector stores** | Cleaner embeddings, less cross-domain noise, faster and more relevant retrieval |
| 🌐 **Live web fallback** | Never a dead end — queries outside the knowledge base still get answered |
| 🧩 **Composable retrieval** | Hybrid search building blocks (dense + BM25), multiple vector DB backends, and swappable embedding models |

## 🧰 Tech Stack

- **Orchestration:** LangChain (`langchain`, `langchain-anthropic`, `langchain-community`, `langchain-text-splitters`)
- **LLM:** Anthropic Claude via `anthropic` / `langchain-anthropic`
- **Vector Stores:** Qdrant (`qdrant-client`), Chroma (`chromadb`, `langchain-chroma`), FAISS (`faiss-cpu`)
- **Embeddings:** `langchain-huggingface`, `sentence-transformers`
- **Hybrid / Sparse Retrieval:** `rank_bm25`, `scikit-learn`
- **Web Search Fallback:** `duckduckgo-search`, `ddgs`, `requests`, `bs4`, `html2text`
- **Document Loaders:** `pypdf`, `docx2txt`, `wikipedia`
- **Data & Utilities:** `pandas`, `numpy`, `torch`, `datasets`, `kagglehub`, `tiktoken`, `einops`, `scipy`, `ipywidgets`, `matplotlib`

## 📁 Project Structure

```
.
├── notebooks/              # exploratory notebooks (routing experiments, etc.)
├── src/enterprise_rag/     # installable package (router, ingestion, retrieval code)
├── assets/                 # images used in docs
├── requirements.txt
└── pyproject.toml
```

## 🚀 Getting Started

### 1. Create a virtual environment

> ⚠️ Note: `numpy<2`, `torch`, `faiss-cpu`, and `chromadb` don't yet ship prebuilt wheels for the newest Python releases (e.g. 3.14). Use **Python 3.11** for a smooth install.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Configure your environment

Copy `.example.env` to `.env` and fill in your API keys:

```env
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
SERPAPI_API_KEY=your_serpapi_key
```

> No Qdrant account yet? Leave `QDRANT_URL` unset and the notebooks fall back to an in-process, in-memory Qdrant instance (`AsyncQdrantClient(location=":memory:")`) — no server or key required, just non-persistent between runs.

### 4. Run the agent

Point the pipeline at your document sets (internal HR policy docs, 10-K filings, etc.), build the Qdrant indices, and let the agent start answering queries with its tools.

## 📌 Status

- ✅ Agent loops — both a raw Anthropic SDK tool-use loop and a LangChain loop, sharing the same tool set, tested
- ✅ Tools — `search_web` (Firecrawl + LLM-based relevance grading), `search_hr_docs`, `search_10k_docs` (Qdrant vector search)
- ✅ FastAPI endpoints (`/query`, `/query_langchain`) wrapping both loops, tested, with CI running the suite on push
- ✅ Qdrant ingestion primitive (`ingestion.py`, `pipeline.py`)
- 🚧 In progress: loading/chunking real source documents into the indices (currently only demo/hardcoded chunks are ingested), relevance grading + corrective retry for the internal Qdrant tools (only `search_web` has it today), hybrid dense+BM25 retrieval, query rewriting (prototyped in `notebooks/3. query_rewriter.ipynb`, not yet wired into the agent), an evaluation suite, and a demoable UI

This README reflects the target design shown in the diagram above; the checklist tracks how much of it is implemented.
