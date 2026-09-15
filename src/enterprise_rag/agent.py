import inspect
from dotenv import load_dotenv
import anthropic
from anthropic import Anthropic
import os
from enterprise_rag.tools import search_web, search_hr_docs, search_10k_docs
from enterprise_rag.prompt import AGENT_SYSTEM_PROMPT
load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

from typing import Callable

TOOLS: dict[str, Callable] = {
    "search_web": search_web,
    "search_hr_docs": search_hr_docs,
    "search_10k_docs": search_10k_docs,
}

TOOL_SCHEMAS = [
    {
        "name": "search_web",
        "description": search_web.__doc__,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "search_hr_docs",
        "description": search_hr_docs.__doc__,
        "input_schema": {
            "type": "object",
            "properties": {"user_query": {"type": "string"}},
            "required": ["user_query"],
        },
    },
    {
        "name": "search_10k_docs",
        "description": search_10k_docs.__doc__,
        "input_schema": {
            "type": "object",
            "properties": {"user_query": {"type": "string"}},
            "required": ["user_query"],
        },
    },
]
MAX_TURNS = 5


async def run_agent_sdk(prompt: str, system=AGENT_SYSTEM_PROMPT,
          model="claude-haiku-4-5") -> str:
    messages = [{"role": "user", "content": prompt}]

    for _ in range(MAX_TURNS):
        try:
            res = client.messages.create(
                model=model,
                max_tokens=1024,
                system=system,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
        except anthropic.APIError as e:
            return f"[error] agent request failed: {type(e).__name__}: {e}"
        has_text = False
        text = "".join(b.text for b in res.content if b.type == "text")
        messages.append({"role": "assistant", "content": res.content})

        if res.stop_reason != "tool_use":
            return text
        for block in res.content:
                if block.type == "text":
                    print(f"Agent: {block.text}")
                    has_text = True
        tool_use_blocks = [b for b in res.content if b.type == "tool_use"]
        if not tool_use_blocks:
            if not has_text:
                print(f"[debug] stop_reason={res.stop_reason!r}")
                print(f"[debug] content={res.content!r}")
                raise RuntimeError(
                                "Loop terminated with no tool calls and no text content. "
                                "Check the API response and the termination logic."
        )


        
        tool_results = []
        for call in tool_use_blocks:
            tool_fn = TOOLS[call.name]
            if inspect.iscoroutinefunction(tool_fn):
                result = await tool_fn(**call.input)
            else:
                result = tool_fn(**call.input)
            print(f"  [observation] {result}")  # Observe
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": result,
            })
        messages.append({"role": "user", "content": tool_results})

    return text  # MAX_TURNS exhausted; return the last text seen


