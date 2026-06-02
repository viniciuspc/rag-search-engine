import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

def generate_content(prompt: str) -> str | None:
    
    response = client.models.generate_content(
              model="gemma-4-31b-it",
              contents=prompt,
          )
    
    
    return response.text