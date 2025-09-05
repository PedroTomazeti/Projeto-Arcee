import os
from dotenv import load_dotenv
from google import genai

def load_genai():
    load_dotenv(dotenv_path=r"C:\Projeto-Arcee\.env.local")
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if not API_KEY:
        raise ValueError("Defina sua GOOGLE_API_KEY em .env.local")
    
    client = genai.Client(api_key=API_KEY)

    return client