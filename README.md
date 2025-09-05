# Projeto-Arcee

Arcee é uma assistente virtual baseada no modelo de IA **Gemini 2.5 Flash** do Google, inspirada no estilo Jarvis da Marvel.  
Fornece suporte estratégico, informações, decisões e ajuda com tarefas, mantendo eficiência, segurança e clareza.

---
## 🔹 Sobre Arcee (A.R.C.E.E)

A.R.C.E.E (Autonomous Reasoning & Control Expert Entity) é uma assistente virtual avançada inspirada no estilo Jarvis da Marvel.
Ela oferece suporte estratégico, ajuda com tarefas, decisões e informações, mantendo eficiência, segurança, clareza e profissionalismo em todas as interações.

Cada letra do acrônimo representa:

- A – Autonomous → Atua de forma autônoma, antecipando necessidades e propondo soluções.

- R – Reasoning → Capacidade de raciocínio lógico e analítico para decisões inteligentes.

- C – Control → Gerencia tarefas, informações e priorizações de forma organizada.

- E – Expert → Fornece respostas precisas e confiáveis com base em conhecimento e contexto.

- E – Entity → Uma entidade virtual consistente, com personalidade definida, simulação de empatia e comportamento adaptativo.

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
