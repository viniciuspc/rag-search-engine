from .gemini_client import generate_content

def run_rag(query: str, formatted_results: list[str]) -> str:
    docs = {chr(10).join(formatted_results)}
    
    prompt = prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""

    result = generate_content(prompt)
    
    return result if result is not None else ""