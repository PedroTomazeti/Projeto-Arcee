from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# Carrega .env.local
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env.local'))

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Defina sua GOOGLE_API_KEY como variável de ambiente.")

client = genai.Client(api_key=api_key)

def chat():
    print("🤖 Arcee iniciado! (digite 'sair' para encerrar)\n")
    history = []

    while True:
        user_input = input("Você: ")
        if user_input.lower() in ["sair", "exit", "quit"]:
            print("Arcee: Até mais, chefe!")
            break

        history.append({"role": "user", "content": user_input})

        # Limitar histórico para acelerar
        max_turns = 5
        history_to_send = history[-max_turns:]

        # Gerar resposta
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=["\n".join(
                [f"Você: {h['content']}" if h['role']=='user' else f"Arcee: {h['content']}" 
                 for h in history_to_send]
            )],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)  # desativa "Pensamento"
            )
        )

        answer = response.text
        print(f"Arcee: {answer}\n")
        history.append({"role": "model", "content": answer})

if __name__ == "__main__":
    chat()
