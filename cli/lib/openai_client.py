import os
from dotenv import load_dotenv
from openai import OpenAI

class OpenAIResponse():
    content: str | None
    total_tokens: int | None
    
    

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
    )

def generate_content(messages) -> OpenAIResponse:
    
    response = client.chat.completions.create(
              model="openrouter/free",
              messages= messages
          )
    
    openai_response = OpenAIResponse()
    
    if response.usage is not None:
        openai_response.total_tokens = response.usage.total_tokens
        
    openai_response.content = response.choices[0].message.content
    
    return openai_response