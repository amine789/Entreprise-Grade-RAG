from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from enterprise_rag.api import app

client = TestClient(app)


def test_query_returns_agent_answer():
    with patch("enterprise_rag.api.run_agent_sdk",
               new=AsyncMock(return_value="Recursion is when a function calls itself.")):
        response = client.post("/query", json={"question": "Explain recursion"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Recursion is when a function calls itself."}


def test_query_langchain_returns_agent_answer():
    with patch("enterprise_rag.api.run_agent_langchain",
               new=AsyncMock(return_value="Recursion is when a function calls itself.")):
        response = client.post("/query_langchain", json={"question": "Explain recursion"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Recursion is when a function calls itself."}


def test_query_missing_question_returns_422():
    response = client.post("/query", json={})

    assert response.status_code == 422


def test_query_langchain_missing_question_returns_422():
    response = client.post("/query_langchain", json={})

    assert response.status_code == 422


def test_query_agent_error_returns_500():
    with patch("enterprise_rag.api.run_agent_sdk",
               new=AsyncMock(return_value="[error] agent request failed: APIError: rate limited")):
        response = client.post("/query", json={"question": "Explain recursion"})

    assert response.status_code == 500
    assert response.json() == {"detail": "[error] agent request failed: APIError: rate limited"}


def test_query_langchain_agent_error_returns_500():
    with patch("enterprise_rag.api.run_agent_langchain",
               new=AsyncMock(return_value="[error] agent request failed: APIError: rate limited")):
        response = client.post("/query_langchain", json={"question": "Explain recursion"})

    assert response.status_code == 500
    assert response.json() == {"detail": "[error] agent request failed: APIError: rate limited"}


