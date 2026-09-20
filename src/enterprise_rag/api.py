from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from enterprise_rag.agent import run_agent_sdk, run_agent_langchain


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


app = FastAPI()


def _raise_if_agent_error(answer: str) -> None:
    if answer.startswith("[error]"):
        raise HTTPException(status_code=500, detail=answer)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    answer = await run_agent_sdk(request.question)
    _raise_if_agent_error(answer)
    return QueryResponse(answer=answer)


@app.post("/query_langchain", response_model=QueryResponse)
async def query_langchain(request: QueryRequest) -> QueryResponse:
    answer = await run_agent_langchain(request.question)
    _raise_if_agent_error(answer)
    return QueryResponse(answer=answer)
