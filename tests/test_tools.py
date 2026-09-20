from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest
import requests
from enterprise_rag.agent import run_agent_sdk, MAX_TURNS, run_agent_langchain
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

    fake_grading_response = MagicMock()
    fake_grading_response.content = "[1]"

    with patch("enterprise_rag.tools.requests.post", return_value=fake_resp), \
         patch("enterprise_rag.tools.llm") as mock_llm:
        mock_llm.invoke.return_value = fake_grading_response
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

    fake_grading_response = MagicMock()
    fake_grading_response.content = "[1]"

    with patch("enterprise_rag.tools.requests.post", return_value=fake_resp) as mock_post, \
         patch("enterprise_rag.tools.llm") as mock_llm:
        mock_llm.invoke.return_value = fake_grading_response
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

    fake_grading_response = MagicMock()
    fake_grading_response.content = "[1]"

    with patch("enterprise_rag.tools.requests.post", return_value=fake_resp) as mock_post, \
         patch("enterprise_rag.tools.llm") as mock_llm:
        mock_llm.invoke.return_value = fake_grading_response
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

    fake_grading_response = MagicMock()
    fake_grading_response.content = "[1]"

    with patch("enterprise_rag.tools.requests.post", return_value=fake_resp), \
         patch("enterprise_rag.tools.llm") as mock_llm:
        mock_llm.invoke.return_value = fake_grading_response
        result = search_web("explain TCP")

    description_line = result.splitlines()[2]
    assert len(description_line) == 500


def test_web_grading_filters_irrelevant_results(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "success": True,
        "data": {"web": [
            {"title": "TCP Explained", "url": "https://example.com/tcp", "description": "TCP is a protocol."},
            {"title": "Unrelated Page", "url": "https://example.com/other", "description": "Nothing to do with it."},
        ]},
    }

    fake_grading_response = MagicMock()
    fake_grading_response.content = "[1]"

    with patch("enterprise_rag.tools.requests.post", return_value=fake_resp), \
         patch("enterprise_rag.tools.llm") as mock_llm:
        mock_llm.invoke.return_value = fake_grading_response
        result = search_web("explain TCP")

    assert "TCP Explained" in result
    assert "Unrelated Page" not in result


def test_web_grading_all_filtered_out(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "success": True,
        "data": {"web": [
            {"title": "Unrelated Page", "url": "https://example.com/other", "description": "Nothing to do with it."},
        ]},
    }

    fake_grading_response = MagicMock()
    fake_grading_response.content = "[]"

    with patch("enterprise_rag.tools.requests.post", return_value=fake_resp), \
         patch("enterprise_rag.tools.llm") as mock_llm:
        mock_llm.invoke.return_value = fake_grading_response
        result = search_web("explain TCP")

    assert result == "no relevant results found for: explain TCP"


def test_web_grading_fails_open_on_bad_response(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-key")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "success": True,
        "data": {"web": [
            {"title": "TCP Explained", "url": "https://example.com/tcp", "description": "TCP is a protocol."},
        ]},
    }

    fake_grading_response = MagicMock()
    fake_grading_response.content = "not valid json"

    with patch("enterprise_rag.tools.requests.post", return_value=fake_resp), \
         patch("enterprise_rag.tools.llm") as mock_llm:
        mock_llm.invoke.return_value = fake_grading_response
        result = search_web("explain TCP")

    assert result == "TCP Explained\nhttps://example.com/tcp\nTCP is a protocol."


@pytest.mark.asyncio
async def test_agent_sdk_no_tool_call():
    fake_block = MagicMock()
    fake_block.type = "text"
    fake_block.text = "Recursion is when a function calls itself."

    fake_res = MagicMock()
    fake_res.content = [fake_block]
    fake_res.stop_reason = "end_turn"

    with patch("enterprise_rag.agent.client") as mock_client:
        mock_client.messages.create.return_value = fake_res

        result = await run_agent_sdk("Explain recursion")

    assert result == "Recursion is when a function calls itself."


@pytest.mark.asyncio
async def test_agent_sdk_tool_call_then_answer():
    fake_tool_use_block = MagicMock()
    fake_tool_use_block.type = "tool_use"
    fake_tool_use_block.name = "search_web"
    fake_tool_use_block.input = {"query": "latest python version"}
    fake_tool_use_block.id = "call_1"

    first_res = MagicMock()
    first_res.content = [fake_tool_use_block]
    first_res.stop_reason = "tool_use"

    fake_text_block = MagicMock()
    fake_text_block.type = "text"
    fake_text_block.text = "Python 3.14 is the latest version."

    second_res = MagicMock()
    second_res.content = [fake_text_block]
    second_res.stop_reason = "end_turn"

    mock_search_web = MagicMock(return_value="fake search results")

    with patch("enterprise_rag.agent.client") as mock_client, \
         patch("enterprise_rag.agent.TOOLS", {"search_web": mock_search_web}):
        mock_client.messages.create.side_effect = [first_res, second_res]

        result = await run_agent_sdk("What is the latest Python version?")

    assert result == "Python 3.14 is the latest version."
    assert mock_client.messages.create.call_count == 2
    mock_search_web.assert_called_once_with(query="latest python version")


@pytest.mark.asyncio
async def test_agent_sdk_max_turns_exhausted():
    fake_tool_use_block = MagicMock()
    fake_tool_use_block.type = "tool_use"
    fake_tool_use_block.name = "search_web"
    fake_tool_use_block.input = {"query": "anything"}
    fake_tool_use_block.id = "call_x"

    fake_tool_use_response = MagicMock()
    fake_tool_use_response.content = [fake_tool_use_block]
    fake_tool_use_response.stop_reason = "tool_use"

    mock_search_web = MagicMock(return_value="fake search results")

    with patch("enterprise_rag.agent.client") as mock_client, \
         patch("enterprise_rag.agent.TOOLS", {"search_web": mock_search_web}):
        mock_client.messages.create.return_value = fake_tool_use_response

        result = await run_agent_sdk("What is the latest Python version?")

    assert mock_client.messages.create.call_count == MAX_TURNS
    assert result == ""


@pytest.mark.asyncio
async def test_agent_sdk_api_error():
    fake_error = anthropic.APIError("rate limited", MagicMock(), body=None)

    with patch("enterprise_rag.agent.client") as mock_client:
        mock_client.messages.create.side_effect = fake_error

        result = await run_agent_sdk("Explain recursion")

    assert result == "[error] agent request failed: APIError: rate limited"
    assert mock_client.messages.create.call_count == 1


@pytest.mark.asyncio
async def test_agent_langchain_no_tool_call():
    fake_res = MagicMock()
    fake_res.tool_calls = []
    fake_res.content = "Recursion is when a function calls itself."

    with patch("enterprise_rag.agent.llm_with_tools") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=fake_res)

        result = await run_agent_langchain("Explain recursion")

    assert result == "Recursion is when a function calls itself."


@pytest.mark.asyncio
async def test_agent_langchain_tool_call_then_answer():
    first_res = MagicMock()
    first_res.tool_calls = [
        {"name": "search_web", "args": {"query": "latest python version"}, "id": "call_1"}
    ]

    second_res = MagicMock()
    second_res.tool_calls = []
    second_res.content = "Python 3.14 is the latest version."

    mock_tool = MagicMock()
    mock_tool.ainvoke = AsyncMock(return_value="fake search results")

    with patch("enterprise_rag.agent.llm_with_tools") as mock_llm, \
         patch("enterprise_rag.agent.LC_TOOLS_BY_NAME", {"search_web": mock_tool}):
        mock_llm.ainvoke = AsyncMock(side_effect=[first_res, second_res])

        result = await run_agent_langchain("What is the latest Python version?")

    assert result == "Python 3.14 is the latest version."
    assert mock_llm.ainvoke.await_count == 2
    mock_tool.ainvoke.assert_awaited_once_with({"query": "latest python version"})


@pytest.mark.asyncio
async def test_agent_langchain_max_turns_exhausted():
    fake_tool_use_response = MagicMock()
    fake_tool_use_response.tool_calls = [
        {"name": "search_web", "args": {"query": "anything"}, "id": "call_x"}
    ]
    fake_tool_use_response.content = ""

    mock_tool = MagicMock()
    mock_tool.ainvoke = AsyncMock(return_value="fake search results")

    with patch("enterprise_rag.agent.llm_with_tools") as mock_llm, \
         patch("enterprise_rag.agent.LC_TOOLS_BY_NAME", {"search_web": mock_tool}):
        mock_llm.ainvoke = AsyncMock(return_value=fake_tool_use_response)

        result = await run_agent_langchain("What is the latest Python version?")

    assert mock_llm.ainvoke.await_count == MAX_TURNS
    assert result == ""


@pytest.mark.asyncio
async def test_agent_langchain_api_error():
    fake_error = anthropic.APIError("rate limited", MagicMock(), body=None)

    with patch("enterprise_rag.agent.llm_with_tools") as mock_llm:
        mock_llm.ainvoke = AsyncMock(side_effect=fake_error)

        result = await run_agent_langchain("Explain recursion")

    assert result == "[error] agent request failed: APIError: rate limited"
    assert mock_llm.ainvoke.await_count == 1


