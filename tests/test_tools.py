from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests

from enterprise_rag.tools import search_hr_docs, search_10k_docs, search_web


@pytest.mark.asyncio
async def test_hr_docs():
    fake_point = MagicMock()
    fake_point.payload = {
        "content": "Full-time employees accrue 15 days of PTO per year.",
        "source": "employee_handbook.pdf",
        "page": 12,
    }
    fake_result = MagicMock()
    fake_result.points = [fake_point]

    with patch("enterprise_rag.tools.qdrant") as mock_qdrant, \
         patch("enterprise_rag.tools.get_text_embeddings") as mock_embed:
        mock_qdrant.query_points = AsyncMock(return_value=fake_result)
        mock_embed.return_value = [0.1, 0.2, 0.3]

        result = await search_hr_docs("How many PTO days do I get?")

    assert result == (
        "employee_handbook.pdf, page 12\n"
        "Full-time employees accrue 15 days of PTO per year."
    )
    mock_qdrant.query_points.assert_awaited_once_with(
        collection_name="hr_data",
        query=[0.1, 0.2, 0.3],
        limit=3,
    )


@pytest.mark.asyncio
async def test_10k_docs():
    fake_point = MagicMock()
    fake_point.payload = {
        "content": "Uber Technologies reported total revenue of $37.3 billion for fiscal year 2023.",
        "source": "uber_10k_2023.pdf",
        "page": 52,
    }
    fake_result = MagicMock()
    fake_result.points = [fake_point]

    with patch("enterprise_rag.tools.qdrant") as mock_qdrant, \
         patch("enterprise_rag.tools.get_text_embeddings") as mock_embed:
        mock_qdrant.query_points = AsyncMock(return_value=fake_result)
        mock_embed.return_value = [0.4, 0.5, 0.6]

        result = await search_10k_docs("What was Uber's 2023 revenue?")

    assert result == (
        "uber_10k_2023.pdf, page 52\n"
        "Uber Technologies reported total revenue of $37.3 billion for fiscal year 2023."
    )
    mock_qdrant.query_points.assert_awaited_once_with(
        collection_name="10k_data",
        query=[0.4, 0.5, 0.6],
        limit=3,
    )


def test_web_missing_api_key(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)

    result = search_web("explain TCP")

    assert result == "[error] FIRECRAWL_API_KEY is not set."


def test_web_request_exception(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")
    with patch("enterprise_rag.tools.requests.post",
               side_effect=requests.RequestException("boom")):
        result = search_web('Explain TCP')
    assert result.startswith("[error] web search request failed:")
    assert "boom" in result


def test_web_non_200_status(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

    fake_resp = MagicMock()
    fake_resp.status_code = 500
    fake_resp.text = "server error"

    with patch("enterprise_rag.tools.requests.post", return_value=fake_resp):
        result = search_web("explain TCP")

    assert "[error] web search failed: HTTP 500" in result
    assert "server error" in result


def test_web_success_false(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"success": False, "error": "bad query"}

    with patch("enterprise_rag.tools.requests.post", return_value=fake_resp):
        result = search_web("explain TCP")

    assert result.startswith("[error] web search failed:")


def test_web_no_results(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"success": True, "data": {"web": []}}

    with patch("enterprise_rag.tools.requests.post", return_value=fake_resp):
        result = search_web("explain TCP")

    assert result == "no results found for: explain TCP"


def test_web_formats_results(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "success": True,
        "data": {"web": [
            {"title": "TCP Explained", "url": "https://example.com/tcp", "description": "TCP is a protocol."},
        ]},
    }

    with patch("enterprise_rag.tools.requests.post", return_value=fake_resp):
        result = search_web("explain TCP")

    assert result == "TCP Explained\nhttps://example.com/tcp\nTCP is a protocol."


def test_web_time_sensitive_adds_tbs_and_note(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "success": True,
        "data": {"web": [
            {"title": "Python Releases", "url": "https://python.org", "description": "3.14.7 is latest."},
        ]},
    }

    with patch("enterprise_rag.tools.requests.post", return_value=fake_resp) as mock_post:
        result = search_web("what is the latest python version")

    assert result.startswith("[Note: this is a time-sensitive query")
    sent_payload = mock_post.call_args.kwargs["json"]
    assert sent_payload["tbs"] == "qdr:m"


def test_web_non_time_sensitive_has_no_tbs_or_note(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "success": True,
        "data": {"web": [
            {"title": "TCP Explained", "url": "https://example.com/tcp", "description": "TCP is a protocol."},
        ]},
    }

    with patch("enterprise_rag.tools.requests.post", return_value=fake_resp) as mock_post:
        result = search_web("explain TCP")

    assert not result.startswith("[Note:")
    sent_payload = mock_post.call_args.kwargs["json"]
    assert "tbs" not in sent_payload


def test_web_description_truncated(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

    long_description = "x" * 600
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "success": True,
        "data": {"web": [
            {"title": "Long Page", "url": "https://example.com/long", "description": long_description},
        ]},
    }

    with patch("enterprise_rag.tools.requests.post", return_value=fake_resp):
        result = search_web("explain TCP")

    description_line = result.splitlines()[2]
    assert len(description_line) == 500