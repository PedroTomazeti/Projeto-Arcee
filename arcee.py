import sqlite3
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# -----------------------
# Configuração do SQLite
# -----------------------
DB_PATH = "arcee_context.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT,
        content TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memory_summary (
        id INTEGER PRIMARY KEY,
        summary TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_message(role, content):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()

def get_recent_messages(limit=5):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

def get_memory_summary():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT summary FROM memory_summary WHERE id=1")
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else ""

def update_memory_summary(summary):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO memory_summary (id, summary) VALUES (1, ?)", (summary,))
    conn.commit()
    conn.close()

# Inicializa o banco
init_db()

# -----------------------
# Configuração do Gemini
# -----------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env.local'))
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Defina sua GOOGLE_API_KEY como variável de ambiente.")

client = genai.Client(api_key=api_key)

# -----------------------
# System Instruction
# -----------------------
system_instruction_text = """
Você é Arcee, um assistente pessoal estilo Jarvis.
Seja educado, útil e objetivo.
Use respostas curtas quando possível.
Sempre mantenha o contexto da conversa.
"""

# -----------------------
# Função de resumo do histórico
# -----------------------
def summarize_history(old_turns):
    prompt = "Resuma brevemente em 1-2 linhas o seguinte histórico de conversa, mantendo informações importantes:\n\n"
    prompt += "\n".join([f"Você: {h['content']}" if h['role']=='user' else f"Arcee: {h['content']}" for h in old_turns])
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            system_instruction=system_instruction_text
        )
    )
    return response.output_text.strip()

# -----------------------
# Função principal de chat
# -----------------------
def chat():
    print("🤖 Arcee iniciado! (digite 'sair' para encerrar)")
    print("💡 Dica: digite '/pensar' no início da mensagem para ativar Pensamento profundo.\n")

    memory_summary = ""

    while True:
        user_input = input("Você: ")
        if user_input.lower() in ["sair", "exit", "quit"]:
            print("Arcee: Até mais, chefe!")
            break

        # Detecta Pensamento profundo
        if user_input.startswith("/pensar"):
            thinking_budget = 100
            user_input = user_input.replace("/pensar", "", 1).strip()
        else:
            thinking_budget = 0

        # Salva a mensagem do usuário
        save_message("user", user_input)

        # Recupera histórico recente
        history = get_recent_messages(limit=5)

        # Atualiza memória resumida se houver mensagens antigas
        all_messages = get_recent_messages(limit=1000)  # pega tudo para resumir histórico antigo
        if len(all_messages) > 5:
            old_turns = all_messages[:-5]
            memory_summary = summarize_history(old_turns)
            update_memory_summary(memory_summary)

        # Monta prompt final
        prompt_parts = []
        if memory_summary:
            prompt_parts.append(f"Memória resumida: {memory_summary}")
        prompt_parts.append("\n".join([f"Você: {h['content']}" if h['role']=='user' else f"Arcee: {h['content']}" for h in history]))
        prompt_parts.append(f"Você: {user_input}")
        prompt = "\n".join(prompt_parts)

        # Gera resposta com System Instruction
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
                system_instruction=system_instruction_text
            )
        )

        answer = response.text
        print(f"Arcee: {answer}\n")

        # Salva resposta no banco
        save_message("model", answer)

# -----------------------
# Executa o chat
# -----------------------
if __name__ == "__main__":
    chat()
