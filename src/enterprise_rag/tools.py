import os
import requests
from qdrant_client import AsyncQdrantClient
from enterprise_rag.utils import get_text_embeddings, is_time_sensitive
from dotenv import load_dotenv

load_dotenv()

qdrant_url = os.environ.get("QDRANT_URL")
if qdrant_url:
    qdrant = AsyncQdrantClient(url=qdrant_url, api_key=os.environ.get("QDRANT_API_KEY"))
else:
    qdrant = AsyncQdrantClient(location=":memory:")


FIRECRAWL_URL = "https://api.firecrawl.dev/v2/search"
REQUEST_TIMEOUT = 10  # seconds
MAX_RESULTS = 5
MAX_DESCRIPTION_CHARS = 500


def search_web(query: str) -> str:
    """Search the web for a query. Returns the title, url, and description
    of the top results. Use this for current information, documentation,
    or research not covered by existing project files. Results have no
    publish date attached -- a page may be outdated, so do not assume a
    result's "latest" or "current" claim still holds; hedge accordingly."""
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        return "[error] FIRECRAWL_API_KEY is not set."

    payload = {"query": query, "limit": MAX_RESULTS}
    if is_time_sensitive(query):
        payload["tbs"] = "qdr:m"  # restrict to results from the past month

    try:
        response = requests.post(
            FIRECRAWL_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        return f"[error] web search request failed: {type(e).__name__}: {e}"

    if response.status_code != 200:
        return f"[error] web search failed: HTTP {response.status_code} — {response.text}"

    body = response.json()
    if not body.get("success"):
        return f"[error] web search failed: {body}"

    results = body.get("data", {}).get("web", [])
    if not results:
        return f"no results found for: {query}"

    lines = []
    for r in results[:MAX_RESULTS]:
        description = r.get("description", "")[:MAX_DESCRIPTION_CHARS]
        lines.append(f"{r.get('title', '(no title)')}\n{r.get('url', '')}\n{description}")

    output = "\n\n".join(lines)
    if is_time_sensitive(query):
        output = (
            "[Note: this is a time-sensitive query, so results were "
            "restricted to the past month. Individual pages still carry no "
            "exact publish date, so verify specific figures/versions rather "
            "than presenting them as certain.]\n\n"
        ) + output
    return output




async def search_hr_docs(user_query: str) -> str:
    """Search internal HR policy documents. Use this for questions about
    PTO/vacation and leave policy, benefits enrollment, payroll and expense
    reimbursement, onboarding, the employee handbook, or the performance
    review process."""
    query_embedding = get_text_embeddings(user_query)
    text_hits = await qdrant.query_points(
        collection_name="hr_data",
        query=query_embedding,
        limit=3,
    )
    if not text_hits.points:
        return f"no HR policy documents matched: {user_query}"

    lines = []
    for point in text_hits.points:
        source = point.payload.get("source", "unknown")
        page = point.payload.get("page")
        heading = f"{source}, page {page}" if page is not None else source
        lines.append(f"{heading}\n{point.payload['content']}")
    return "\n\n".join(lines)


async def search_10k_docs(user_query: str) -> str:
    """Search Uber and Lyft 10-K annual filings. Use this for questions
    about company financials, revenue, operating costs, or filing
    disclosures."""
    query_embedding = get_text_embeddings(user_query)
    text_hits = await qdrant.query_points(
        collection_name="10k_data",
        query=query_embedding,
        limit=3,
    )
    if not text_hits.points:
        return f"no 10-K filing documents matched: {user_query}"

    lines = []
    for point in text_hits.points:
        source = point.payload.get("source", "unknown")
        page = point.payload.get("page")
        heading = f"{source}, page {page}" if page is not None else source
        lines.append(f"{heading}\n{point.payload['content']}")
    return "\n\n".join(lines)


