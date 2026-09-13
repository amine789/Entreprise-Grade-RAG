from uuid import uuid4

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from enterprise_rag.utils import get_text_embeddings


async def ingest_documents(
    qdrant: AsyncQdrantClient,
    collection_name: str,
    chunks: list[dict],
    tokenizer,
    model,
    vector_size: int,
) -> None:
    """Embed chunks and upsert them into Qdrant with source metadata.

    Each chunk must have a "content" key, and should include "source"
    (e.g. a filename) and optional "page" so retrieved hits can later
    be cited back to a real document instead of just a list position.
    """
    if not await qdrant.collection_exists(collection_name):
        await qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    points = [
        PointStruct(
            id=str(uuid4()),
            vector=get_text_embeddings(chunk["content"], tokenizer, model).tolist(),
            payload={
                "content": chunk["content"],
                "source": chunk.get("source", "unknown"),
                "page": chunk.get("page"),
            },
        )
        for chunk in chunks
    ]
    await qdrant.upsert(collection_name=collection_name, points=points)
