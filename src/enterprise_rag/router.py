import json
import re

from enterprise_rag.llm_model import llm


def route_query(user_query: str) -> dict:
    router_system_prompt = f'''
    As a professional query router, classify user input into one of
    three categories:

    1. "HUMAN_RESOURCES_QUERY": Questions about internal policy, PTO/vacation and leave policy,
        benefits enrollment, payroll and expense reibursement, onboarding, the employee handbook or 
        performance review process

    2. "10K_DOCUMENT_QUERY": Questions about company financials, 10-K
       annual reports, Uber or Lyft revenue, operating costs, or filing
       disclosures.

    3. "WEB_SEARCH": Everything else -- general knowledge, technology
       trends, comparisons, or anything not in the internal document
       collections.

    Always respond in this exact JSON format:
    {{
        "action": "HUMAN_RESOURCES_QUERY" or "10K_DOCUMENT_QUERY" or
                  "WEB_SEARCH",
        "reason": "one sentence justification for the routing decision",
        "answer": "AT MOST 5 words if trivially obvious, else leave empty"
    }}

    User: {user_query}
    '''
    try:
        response = llm.invoke(router_system_prompt)
        json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
        return json.loads(json_match.group())
    except (json.JSONDecodeError, AttributeError) as err:
        return {
            "action": "WEB_SEARCH",
            "reason": f"Routing error: {err}",
            "answer": ""
        }
