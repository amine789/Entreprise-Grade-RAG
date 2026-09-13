from enterprise_rag.llm_model import llm


def rag_formatted_response(user_query: str, context: list) -> str:
    context_block = "\n\n".join(
        f"[{i + 1}] (Source: {chunk['source']}, page {chunk.get('page', 'N/A')})\n"
        f"{chunk['content']}"
        for i, chunk in enumerate(context)
    )
    rag_prompt = f"""
    Based on the given context, answer the user query: {user_query}
    Context:
    {context_block}
    Use numbered citations [1][2][3] referencing the context chunks above.
    Begin directly with the answer.
    """
    response = llm.invoke(rag_prompt)
    return response.content
