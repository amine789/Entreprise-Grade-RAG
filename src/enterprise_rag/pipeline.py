from qdrant_client import AsyncQdrantClient

from enterprise_rag.retrieval import COLLECTIONS
from enterprise_rag.ingestion import ingest_documents


async def ingestion_pipeline(
    qdrant: AsyncQdrantClient, action: str, data: list[dict], vector_size: int
) -> None:
    if action in COLLECTIONS:
        await ingest_documents(qdrant, COLLECTIONS[action], data, vector_size)
