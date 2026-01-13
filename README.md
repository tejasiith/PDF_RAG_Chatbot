# PDF-RAG-Chatbot

A **Retrieval-Augmented Generation (RAG) chatbot** built using LangChain and OpenAI, allowing you to **ask questions from PDF documents** like research papers, theses, or notes.

---

## Features

- Chat with your PDF documents using AI.
- Uses **FAISS vector embeddings** for fast retrieval.
- Handles large PDFs by splitting them into chunks.
- Returns answers only based on your documents ("I don't know" if not in context).
- Built with **Streamlit** for an interactive UI.

---

## Tech Stack

- Python 3.10+
- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- OpenAI GPT-3.5 / GPT-4
- Streamlit for UI
- PyPDFLoader for PDF parsing

---
