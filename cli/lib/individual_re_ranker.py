from .gemini_client import generate_content

def calculate_rank(query: str, doc: dict) -> float:
    prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""

    result = generate_content(prompt)
    
    return safe_float(result, 0.0)
    
    
def safe_float(val, default=None):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        # ValueError handles bad strings ('abc')
        # TypeError handles bad types (None, lists, etc.)
        return default