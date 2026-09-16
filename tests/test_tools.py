from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from enterprise_rag.tools import search_hr_docs, search_10k_docs


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
