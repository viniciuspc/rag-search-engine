import mimetypes
import base64
from lib.openai_client import generate_content, OpenAIResponse

def describe_image_command(image_path: str, query: str) -> str | None:
    mime = get_mime_from_image_path(image_path)
    
    print(f"Query: {query}")
    print(f"Image path: {image_path}")
    print(f"Mime: {mime}")
    
    f = open(image_path, mode="rb") 
    img_bytes = f.read()
    openai_response = get_response_from_llm(img_bytes, mime, query)
    
    content: str = openai_response.content if openai_response.content else ""
    
    print(f"Rewritten query: {content.strip()}")
    if openai_response.total_tokens is not None:
        print(f"Total tokens:    {openai_response.total_tokens}")
    
    
def get_mime_from_image_path(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    return mime or "image/jpeg"

def get_response_from_llm(img_bytes: bytes, mime: str, query: str) -> OpenAIResponse:
    system_prompt = """"
Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""
    data_url = f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": query.strip()},
            ],
        }
    ]
    
    response = generate_content(messages)
    
    return response
    
    
    