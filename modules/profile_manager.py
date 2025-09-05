import json
from google import genai
from google.genai import types
from arcee import get_conn

client = genai.Client()

# -----------------------
# Função para extrair perfil implícito
# -----------------------
def extract_profile_from_text(user_id: str, text: str):
    """
    Recebe o texto do usuário e tenta extrair informações que podem atualizar seu perfil.
    Retorna um dicionário JSON com campos atualizados.
    """
    prompt = (
        "Extraia informações de perfil do usuário do texto a seguir e retorne como JSON. "
        "Inclua preferências, interesses, hábitos e características pessoais. "
        "Não invente informações. Se não houver nada, retorne um objeto vazio.\n\n"
        f"Texto do usuário: {text}"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )

        output_text = response.output_text if hasattr(response, "output_text") else response.text
        # Tenta converter em JSON
        profile_update = json.loads(output_text)
        if isinstance(profile_update, dict):
            update_user_profile_in_db(user_id, profile_update)
    except Exception as e:
        print(f"(Aviso) Não foi possível extrair perfil: {e}")

# -----------------------
# Função de atualização incremental
# -----------------------
def update_user_profile_in_db(user_id: str, new_data: dict):
    if not new_data:
        return

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT preferencias, dados_pessoais FROM users WHERE id=?", (user_id,))
        row = cur.fetchone()

        if row:
            prefs = json.loads(row[0]) if row[0] else {}
            dados = json.loads(row[1]) if row[1] else {}
            # Merge incremental
            prefs.update(new_data.get("preferencias", {}))
            dados.update(new_data.get("dados_pessoais", {}))
            cur.execute(
                "UPDATE users SET preferencias=?, dados_pessoais=? WHERE id=?",
                (json.dumps(prefs, ensure_ascii=False), json.dumps(dados, ensure_ascii=False), user_id)
            )
            conn.commit()
