import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

def main():
    
    query = "briish bear"
    content = f"""Fix any spelling errors in the user-provided movie search query below.
        Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
        Preserve punctuation and capitalization unless a change is required for a typo fix.
        If there are no spelling errors, or if you're unsure, output the original query unchanged.
        Output only the final query text, nothing else.
        User query: "{query}"
        """
    
    response = client.models.generate_content(
              model="gemma-4-31b-it",
              contents=content,
          )
    
    print(response.text)
    
    if response.usage_metadata is not None:
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Response tokens: {response.usage_metadata.candidates_token_count}")

if __name__ == "__main__":
    main()