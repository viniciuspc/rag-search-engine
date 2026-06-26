import json
from .gemini_client import generate_content

def calculate_batch_rank(query: str, doc_list_str: str) -> list[int]:
    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

    result = generate_content(prompt)
    
    return json.loads(result) if result is not None else []
    
    
def safe_float(val, default=None):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        # ValueError handles bad strings ('abc')
        # TypeError handles bad types (None, lists, etc.)
        return default