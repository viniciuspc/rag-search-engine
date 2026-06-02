from .gemini_client import generate_content
def correct_spell(query: str) -> str:
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
        Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
        Preserve punctuation and capitalization unless a change is required for a typo fix.
        If there are no spelling errors, or if you're unsure, output the original query unchanged.
        Output only the final query text, nothing else.
        User query: "{query}"
        """
    
    response = generate_content(prompt)
    
    if response:
        return response
    else:
        return query