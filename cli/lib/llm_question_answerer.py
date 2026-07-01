from .gemini_client import generate_content

def answer_question(question: str, context: list[str]) -> str:
    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

    answer = generate_content(prompt)
    
    return answer if answer is not None else ""