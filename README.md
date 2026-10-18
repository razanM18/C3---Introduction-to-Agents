---
title: CedarSun Energy Assistant
emoji: ☀️
sdk: gradio
app_file: app.py
pinned: false
---

# CedarSun Energy: Agent-Powered Business Assistant (C3)

EECE 503P/798S Agentic Systems. **Zeina**, a solar advisor of a fictitious Lebanese residential solar company,
answers questions from the company documents, collects leads, logs feedback and unanswered questions, and estimates a rough
system size with a Python tool.

## Project structure

```
business_bot/
├── me/
│   ├── about_business.pdf        company profile and technical knowledge base
│   └── business_summary.txt      short summary used in the prompt
├── business_agent.ipynb          main notebook: persona, instructions, router, tools, agent loop, reflection, Gradio
├── app.py                        same code as a script (python app.py)
├── tests/test_offline.py         offline tests (no API key needed)
├── .env                          your API key (NOT uploaded)
├── .env.example                  template for .env
└── requirements.txt
```

## Setup

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and put your key in `GEMINI_API_KEY` (default) or set `LLM_PROVIDER=openai` and fill `OPENAI_API_KEY`.
   Get a Gemini key at https://aistudio.google.com/apikey.
   Free alternatives for development: `LLM_PROVIDER=groq` (key from console.groq.com), `cerebras`, `openrouter` (free `:free` models), or `ollama` (runs locally, no key).
   `LLM_FALLBACKS=gemini,openai` makes the app switch to another provider if the main one is rate-limited or down.
3. Run `business_agent.ipynb` top to bottom (Jupyter, VS Code or Colab), or run `python app.py` for the Gradio chat.

**Colab:** upload the folder (at least `me/`), add the key under *Secrets* as `GEMINI_API_KEY`, and run all cells.

## How it maps to the assignment

| Requirement | Where |
|---|---|
| Business identity (PDF and TXT) | `me/` |
| Persona and instructions, labelled and separate | notebook section 3 (`PERSONA`, `INSTRUCTIONS`) |
| Router: general / lead / feedback | section 6 (`route_message`): keyword rules, then a small LLM classifier |
| Tools with MCP-style schemas | sections 4 and 5: `record_customer_interest`, `record_feedback`, plus `estimate_system_size` |
| System prompt, chat history, tool calls | section 7 (`run_agent`) |
| Gradio ChatInterface | section 10 (`gr.ChatInterface` inside the landing and call flow) |
| Bonus: reflection with logged draft and revision | section 8, logs in `logs/reflection.jsonl` |

Leads, feedback and reflection logs are written to `logs/` (git-ignored).

## Features beyond the assignment

* Landing page with common questions, a short "calling an advisor" screen, then the Gradio ChatInterface with Zeina.
* Replies in the customer's language: English, French, Lebanese Arabic and Lebanese Arabizi (`detect_language()` in `app.py`).
* Provider fallbacks and `PDF_MAX_WORDS` (only the PDF sections that match the question are sent) for free-tier token limits.

## The website (`website/`)

A static site (home, solutions, how it works, about, FAQ, contact) with the chat built in: a round chat button opens the
assistant in a window, and the "Request your free survey" form saves leads to `logs/leads.jsonl` through the assistant.

```
python app.py                              # terminal 1: the assistant on http://127.0.0.1:7860
cd website && python -m http.server 8000   # terminal 2: the website on http://127.0.0.1:8000
```

`website/config.js` holds `chatUrl`, the address of the assistant. Point it at your Hugging Face Space URL when you deploy.
GitHub Pages can host the website folder, but it cannot run `app.py`, so the assistant itself must run locally or on a Space.

## Deploying to Hugging Face Spaces (bonus)

Create a Gradio Space, upload `app.py`, `requirements.txt`, `me/` and `logo.svg`, and add your key under *Settings > Secrets*
(for example `GEMINI_API_KEY`). The header at the top of this README is the Space configuration.

## Offline tests

`python tests/test_offline.py` checks the tools, the router rules, and the agent loop with a scripted fake model. No API key needed.

## Notes

* Both providers are called through the `openai` library. Gemini is reached through its OpenAI-compatible endpoint.
* The agent never quotes prices: every price is a written quotation after a free site survey (see the PDF, Section 12).
* CedarSun Energy is fictional. All names, contacts and figures are illustrative.
