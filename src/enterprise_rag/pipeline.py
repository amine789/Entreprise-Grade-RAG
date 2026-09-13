from qdrant_client import AsyncQdrantClient

from enterprise_rag.generation import rag_formatted_response
from enterprise_rag.retrieval import retrieve_and_response
from enterprise_rag.router import route_query


async def handle_query(qdrant: AsyncQdrantClient, user_query: str) -> str:
    route_result = route_query(user_query)
    action = route_result["action"]
    reason = route_result["reason"]
    short_answer = route_result.get("answer", "")

    print(f"Route: {action}")
    print(f"Reason: {reason}")

    if short_answer:
        return short_answer

    if action == "WEB_SEARCH":
        # search_web is not implemented yet
        search_results = search_web(user_query)
        return rag_formatted_response(user_query, search_results)

    return await retrieve_and_response(qdrant, user_query, action)
