import asyncio
from pathlib import Path

from enterprise_rag.tools import qdrant
from enterprise_rag.ingestion import ingest_documents
from enterprise_rag.loaders import load_directory_chunks
from enterprise_rag.agent import run_agent_langchain

VECTOR_SIZE = 768
DATA_DIR = Path(__file__).parent / "data"


async def main():
    hr_chunks = load_directory_chunks(DATA_DIR / "hr")
    tenk_chunks = load_directory_chunks(DATA_DIR / "10k")

    await ingest_documents(qdrant, "hr_data", hr_chunks, VECTOR_SIZE)
    await ingest_documents(qdrant, "10k_data", tenk_chunks, VECTOR_SIZE)

    for query in [
        "How many PTO days do I get?",
        "What was Uber's 2023 revenue?",
        "What's the latest version of Python released?",
    ]:
        print(f"\n=== {query} ===")
        answer = await run_agent_langchain(query)
        print(answer)


if __name__ == "__main__":
    asyncio.run(main())
