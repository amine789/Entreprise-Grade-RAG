from qdrant_client import AsyncQdrantClient

from enterprise_rag.generation import rag_formatted_response
from enterprise_rag.utils import get_text_embeddings

COLLECTIONS = {
    "ANTHROPIC_QUERY": "anthropic_data",
    "10K_DOCUMENT_QUERY": "10k_data",
}


async def retrieve_and_response(
    qdrant: AsyncQdrantClient, user_query: str, action: str
) -> str:
    query_embedding = get_text_embeddings(user_query)
    text_hits = await qdrant.query_points(
        collection_name=COLLECTIONS[action],
        query=query_embedding,
        limit=3,
    )
    contents = [
        {
            "content": point.payload["content"],
            "source": point.payload.get("source", "unknown"),
            "page": point.payload.get("page"),
        }
        for point in text_hits.points
    ]
    return rag_formatted_response(user_query, contents)
