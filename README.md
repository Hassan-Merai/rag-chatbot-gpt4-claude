```markdown
# 🧠 Modular RAG Chatbot with GPT-4, Claude, and Open-Source LLMs

This project is a flexible, production-ready **Retrieval-Augmented Generation (RAG)** chatbot that supports both **open-source language models** (e.g., LLaMA2, Mistral) and **commercial LLM APIs** such as **OpenAI's GPT-4** and **Anthropic's Claude**.

The chatbot retrieves relevant chunks from custom documents using **semantic search**, constructs prompts with contextual grounding, and generates high-quality answers using the selected LLM backend.

---

## 🔍 Features

- 🔁 **Switchable LLM Backend**: Use either:
  - 🧠 Open-source models (e.g., LLaMA2, Mistral)
  - 🤖 GPT-4 (OpenAI API)
  - 🤖 Claude (Anthropic API)
- 📁 PDF/text document ingestion
- 🔎 Chunking and semantic retrieval using **FAISS** and **Sentence Transformers**
- 🚀 Backend API using **FastAPI**
- 💬 Web chat interface built with **Streamlit**
- 🐳 Docker-ready architecture (coming soon)
- 🔐 `.env` support for API key management

---

## 📁 Current Project Structure

```

rag-chatbot-gpt4-claude/
├── backend/
│   ├── main.py
│   └── services/
│       ├── open\_source.py
│       ├── openai.py
│       └── anthropic.py
├── retrieval/
│   ├── vector\_store.py
│   └── embed\_utils.py
├── frontend/
│   └── app.py
├── data/                     # Source documents (PDFs, text files)
├── .env                      # API keys (not tracked by Git)
├── requirements.txt
├── README.md

````

---

## 🛠️ Setup (So Far)

### 1. Clone the repository and create environment

```bash
git clone https://github.com/yourusername/rag-chatbot-gpt4-claude.git
cd rag-chatbot-gpt4-claude
conda create -n ragbot python=3.10 -y
conda activate ragbot
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API keys to `.env`

```
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

---

## ✅ Status: Day 1 Complete

* [x] Project folder structure set up
* [x] Conda environment created
* [x] Core Python dependencies installed
* [x] Initial files and folders created
* [x] Git initialized

---

## 🔜 Coming Next

* Document ingestion
* Chunking + embedding with SentenceTransformers
* FAISS vector store creation

---
