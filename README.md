# Projeto-Arcee

Arcee é uma assistente virtual baseada no modelo de IA **Gemini 2.5 Flash** do Google, inspirada no estilo Jarvis da Marvel.  
Fornece suporte estratégico, informações, decisões e ajuda com tarefas, mantendo eficiência, segurança e clareza.

---

## 🔹 Arcee - Fase 1

Protótipo inicial de assistente pessoal em **chat de texto no terminal**, com histórico e memória resumida.

---

## 🔹 Funcionalidades da Fase 1

- Chat de texto no terminal com respostas rápidas.
- Histórico de conversa **persistente** usando SQLite.
- **Memória resumida** de conversas antigas para manter contexto sem sobrecarregar o modelo.
- **Pensamento ativável**: use `/pensar` no início da mensagem para raciocínio mais profundo.
- **Personalidade e System Instruction** definidas em `assets/system_instruction.txt`.
- Utiliza a **API Gemini 2.5 Flash** do Google.

---

## 🔹 Pré-requisitos

- Python 3.10+
- Conta Google Cloud com API Key para Gemini 2.5 Flash
- Instalar dependências:
```bash
pip install -r requirements.txt
