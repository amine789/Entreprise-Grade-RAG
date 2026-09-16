import asyncio

from enterprise_rag.tools import qdrant
from enterprise_rag.ingestion import ingest_documents
from enterprise_rag.agent import run_agent_langchain

VECTOR_SIZE = 768

hr_chunks = [
    {
        "content": "Full-time employees accrue 15 days of paid time off (PTO) per year during their first three years of employment, increasing to 20 days after three years of continuous service.",
        "source": "employee_handbook.pdf",
        "page": 12,
    },
    {
        "content": "Employees are eligible for up to 12 weeks of paid parental leave following the birth, adoption, or foster placement of a child, provided they have completed at least 90 days of employment.",
        "source": "employee_handbook.pdf",
        "page": 18,
    },
    {
        "content": "Open enrollment for medical, dental, and vision benefits occurs once a year in November. Employees may also enroll within 30 days of a qualifying life event such as marriage or the birth of a child.",
        "source": "benefits_guide.pdf",
        "page": 4,
    },
]

tenk_chunks = [
    {
        "content": "Uber Technologies reported total revenue of $37.3 billion for fiscal year 2023, representing a 17% increase compared to fiscal year 2022.",
        "source": "uber_10k_2023.pdf",
        "page": 52,
    },
    {
        "content": "Lyft's operating expenses for fiscal year 2023 totaled $4.6 billion, driven primarily by insurance costs and sales and marketing spend.",
        "source": "lyft_10k_2023.pdf",
        "page": 47,
    },
    {
        "content": "Uber's 10-K filing identifies driver classification litigation as a material risk factor, noting that reclassification of drivers as employees in certain jurisdictions could materially increase operating costs.",
        "source": "uber_10k_2023.pdf",
        "page": 21,
    },
]


async def main():
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
