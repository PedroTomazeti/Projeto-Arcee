import os
import sqlite3
import json
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from modules.profile_manager import extract_profile_from_text
from .genai_config import load_genai

# ==============================
# Configurações gerais
# ==============================
DB_PATH = "arcee_context.db"
CHAT_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "text-embedding-004"  # ajuste se necessário
ASSETS_SYSTEM_INSTR_PATH = os.path.join("assets", "system_instruction.txt")

# Quantidades
RECENT_TURNS = 5                 # últimos turnos do diálogo enviados ao modelo
SUMMARY_CHUNK_SIZE = 20          # a cada N mensagens, cria um resumo incremental
MAX_SUMMARIES_TO_USE = 5         # quantos resumos incrementais incluir no prompt
TOP_K_SEMANTIC = 4               # quantos trechos relevantes por busca semântica incluir

# ==============================
# Inicialização do cliente Gemini
# ==============================
load_dotenv(dotenv_path=r"C:\Projeto-Arcee\.env.local")
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Defina sua GOOGLE_API_KEY em .env.local")

client = genai.Client(api_key=API_KEY)

# ==============================
# Banco de dados (SQLite)
# ==============================
SCHEMA_SQL = r"""
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  nome TEXT,
  preferencias TEXT,   -- JSON (ex: {"tom":"formal","resposta_curta":true})
  dados_pessoais TEXT  -- JSON (ex: {"aniversario":"2025-09-05"})
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT,
  role TEXT,       -- 'user' | 'model'
  content TEXT,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS summaries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT,
  summary TEXT,
  start_msg_id INTEGER,
  end_msg_id INTEGER,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS embeddings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  message_id INTEGER,
  vector BLOB
);

CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, id);
CREATE INDEX IF NOT EXISTS idx_embeddings_msg ON embeddings(message_id);
CREATE INDEX IF NOT EXISTS idx_summaries_user ON summaries(user_id, id);
"""

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA_SQL)

# ==============================
# Utilidades de usuário / perfis
# ==============================
def get_or_create_user(user_id: str = "default") -> Dict[str, Any]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, nome, preferencias, dados_pessoais FROM users WHERE id=?", (user_id,))
        row = cur.fetchone()
        if row:
            return {
                "id": row[0],
                "nome": row[1] or "",
                "preferencias": json.loads(row[2]) if row[2] else {},
                "dados_pessoais": json.loads(row[3]) if row[3] else {},
            }
        # cria padrão
        cur.execute(
            "INSERT INTO users(id, nome, preferencias, dados_pessoais) VALUES(?,?,?,?)",
            (user_id, "", json.dumps({"tom": "formal", "resposta_curta": True}), json.dumps({}))
        )
        conn.commit()
        return {"id": user_id, "nome": "", "preferencias": {"tom": "formal", "resposta_curta": True}, "dados_pessoais": {}}

def update_user_prefs(user_id: str, prefs: Dict[str, Any]):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT preferencias FROM users WHERE id=?", (user_id,))
        row = cur.fetchone()
        current = json.loads(row[0]) if row and row[0] else {}
        current.update(prefs)
        cur.execute("UPDATE users SET preferencias=? WHERE id=?", (json.dumps(current), user_id))
        conn.commit()

# ==============================
# Mensagens e resumos
# ==============================
def save_message(user_id: str, role: str, content: str) -> int:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO messages(user_id, role, content) VALUES(?,?,?)", (user_id, role, content))
        msg_id = cur.lastrowid
        conn.commit()
        return msg_id

def get_recent_messages(user_id: str, limit: int = RECENT_TURNS) -> List[Dict[str, str]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT role, content FROM messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )
        rows = cur.fetchall()
    # retorna na ordem cronológica
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

def get_all_messages_with_ids(user_id: str) -> List[Tuple[int, str, str]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, role, content FROM messages WHERE user_id=? ORDER BY id ASC", (user_id,))
        return cur.fetchall()

def get_last_summarized_msg_id(user_id: str) -> int:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT MAX(end_msg_id) FROM summaries WHERE user_id=?", (user_id,))
        row = cur.fetchone()
        return row[0] or 0

def add_summary(user_id: str, summary: str, start_id: int, end_id: int):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO summaries(user_id, summary, start_msg_id, end_msg_id) VALUES(?,?,?,?)",
            (user_id, summary, start_id, end_id),
        )
        conn.commit()

def get_recent_summaries(user_id: str, limit: int = MAX_SUMMARIES_TO_USE) -> List[str]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT summary FROM summaries WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )
        rows = cur.fetchall()
    return [r[0] for r in reversed(rows)]

# ==============================
# Embeddings (semântica)
# ==============================
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def embed_text(text: str) -> np.ndarray:
    """Gera embedding usando o modelo de embeddings da Gemini API.
    OBS: dependendo da versão da lib, o campo de retorno pode variar.
    Ajuste abaixo caso necessário (resp.embeddings[0].values, resp.data[0].embedding, etc.).
    """
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[text]
    )
    # Tentativas de extração comuns (mantenha a que funcionar no seu ambiente):
    vec = None
    try:
        vec = resp.embeddings[0].values
    except Exception:
        try:
            vec = resp.data[0].embedding.values
        except Exception:
            raise RuntimeError("Não foi possível extrair o vetor de embedding. Ajuste a função embed_text conforme a versão da lib.")
    return np.array(vec, dtype=np.float32)

def save_embedding(message_id: int, vector: np.ndarray):
    blob = pickle.dumps(vector)
    with get_conn() as conn:
        conn.execute("INSERT INTO embeddings(message_id, vector) VALUES(?, ?)", (message_id, blob))
        conn.commit()

def get_all_embeddings(user_id: str) -> List[Tuple[int, np.ndarray, str]]:
    """Retorna (message_id, vetor, content) para todas as mensagens do user."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT e.message_id, e.vector, m.content
            FROM embeddings e
            JOIN messages m ON m.id = e.message_id
            WHERE m.user_id=?
            ORDER BY e.message_id ASC
            """,
            (user_id,),
        )
        rows = cur.fetchall()
    out = []
    for mid, blob, content in rows:
        vec = pickle.loads(blob)
        out.append((mid, vec, content))
    return out

def semantic_search(user_id: str, query: str, top_k: int = TOP_K_SEMANTIC) -> List[str]:
    qvec = embed_text(query)
    corpus = get_all_embeddings(user_id)  # (id, vec, content)
    if not corpus:
        return []
    scored = []
    for mid, vec, content in corpus:
        scored.append((content, _cosine_sim(qvec, vec)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:top_k]]
# ==============================
# System Instruction + perfil
# ==============================
def load_system_instruction() -> str:
    if not os.path.exists(ASSETS_SYSTEM_INSTR_PATH):
        return (
            "Você é ARCEE, uma assistente de IA. Mantenha respostas claras, objetivas, sem se reapresentar em cada turno."
        )
    with open(ASSETS_SYSTEM_INSTR_PATH, "r", encoding="utf-8") as f:
        base = f.read()
    # reforço para evitar repetição
    base += "\n\nDiretriz: Evite cumprimentos longos ou se reapresentar. Continue a conversa de forma natural."
    return base

def build_profile_snippet(user: Dict[str, Any]) -> str:
    prefs = user.get("preferencias", {})
    dados = user.get("dados_pessoais", {})
    parts = []
    if user.get("nome"):
        parts.append(f"Nome do usuário: {user['nome']}")
    if prefs:
        parts.append("Preferências: " + json.dumps(prefs, ensure_ascii=False))
    if dados:
        parts.append("Dados pessoais relevantes: " + json.dumps(dados, ensure_ascii=False))
    return "\n".join(parts)

# ==============================
# Resumo incremental
# ==============================
def summarize_block(msgs: List[Tuple[int, str, str]]) -> str:
    """Resumo de um bloco de mensagens [(id, role, content), ...]."""
    if not msgs:
        return ""
    formatted = "\n".join([f"Você: {c}" if r == "user" else f"Arcee: {c}" for (_id, r, c) in msgs])
    prompt = (
        "Resuma em 2-3 frases o histórico abaixo, preservando decisões, preferências e fatos importantes.\n\n" + formatted
    )
    resp = client.models.generate_content(
        model=CHAT_MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            system_instruction=load_system_instruction(),
        ),
    )
    try:
        return resp.output_text.strip()
    except Exception:
        return resp.text.strip() if hasattr(resp, "text") else ""

def maybe_create_incremental_summary(user_id: str):
    """Se houver mensagens novas suficientes desde o último resumo, cria um novo resumo incremental."""
    last_end = get_last_summarized_msg_id(user_id)
    all_msgs = get_all_messages_with_ids(user_id)
    # Filtra mensagens ainda não resumidas
    pending = [m for m in all_msgs if m[0] > last_end]
    if len(pending) >= SUMMARY_CHUNK_SIZE:
        # pega o próximo bloco
        block = pending[:SUMMARY_CHUNK_SIZE]
        summary = summarize_block(block)
        start_id, end_id = block[0][0], block[-1][0]
        add_summary(user_id, summary, start_id, end_id)
# ==============================
# Chat loop
# ==============================
def build_prompt(user_id: str, user_input: str) -> str:
    # Perfil
    user = get_or_create_user(user_id)
    profile = build_profile_snippet(user)

    # Resumos incrementais recentes
    summaries = get_recent_summaries(user_id, limit=MAX_SUMMARIES_TO_USE)

    # Busca semântica (memórias relevantes)
    semantic_clues = semantic_search(user_id, user_input, top_k=TOP_K_SEMANTIC)

    # Últimos turnos
    recent = get_recent_messages(user_id, limit=RECENT_TURNS)

    parts = []
    if profile:
        parts.append("Perfil do usuário:\n" + profile)
    if summaries:
        parts.append("Memória incremental recente:\n- " + "\n- ".join(summaries))
    if semantic_clues:
        parts.append("Memórias relevantes encontradas:\n- " + "\n- ".join(semantic_clues))
    if recent:
        parts.append("Últimos turnos:\n" + "\n".join([f"Você: {h['content']}" if h['role']=="user" else f"Arcee: {h['content']}" for h in recent]))
    parts.append(f"Você: {user_input}")
    return "\n\n".join(parts)

def chat(user_id: str = "default"):
    print("🤖 Arcee iniciada! (digite 'sair' para encerrar)")
    print("💡 Dica: use '/pensar ' no início para raciocínio profundo. Use '/perfil {json}' para ajustar preferências.\n")

    system_instruction_text = load_system_instruction()

    while True:
        user_input = input("Você: ")
        if user_input.lower() in ["sair", "exit", "quit"]:
            print("Arcee: Até mais!")
            break

        # Comando de ajuste de perfil (JSON)
        if user_input.startswith("/perfil"):
            payload = user_input.replace("/perfil", "", 1).strip()
            try:
                prefs = json.loads(payload)
                update_user_prefs(user_id, prefs)
                print("Arcee: Preferências atualizadas.")
            except Exception as e:
                print(f"Arcee: Não consegui atualizar o perfil. Envie um JSON válido. Erro: {e}")
            continue

        # Detecta pensamento profundo
        if user_input.startswith("/pensar"):
            thinking_budget = 120
            user_input = user_input.replace("/pensar", "", 1).strip()
        else:
            thinking_budget = 0

        # Salva mensagem e embedding
        msg_id = save_message(user_id, "user", user_input)
        extract_profile_from_text(user_id, user_input)
        try:
            vec = embed_text(user_input)
            save_embedding(msg_id, vec)
        except Exception as e:
            # Embedding é opcional; não quebra o chat
            print(f"(Aviso) Falha ao gerar embedding: {e}")

        # Resumo incremental se necessário
        maybe_create_incremental_summary(user_id)

        # Monta prompt final
        prompt = build_prompt(user_id, user_input)

        # Responde
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
                system_instruction=system_instruction_text,
            ),
        )

        try:
            answer = response.output_text
        except Exception:
            answer = response.text if hasattr(response, "text") else "(sem resposta)"

        print(f"\nArcee: {answer}\n")

        # Salva resposta do modelo (e, opcionalmente, embedding dela para buscas futuras)
        mid_model = save_message(user_id, "model", answer)
        # Você pode embutir respostas do modelo também, se desejar melhorar a busca semântica:
        try:
            vec_m = embed_text(answer)
            save_embedding(mid_model, vec_m)
        except Exception:
            pass

if __name__ == "__main__":
    init_db()
    get_or_create_user("default")
    chat("default")
