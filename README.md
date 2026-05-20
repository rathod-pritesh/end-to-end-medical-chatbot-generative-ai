# 🏥 End-to-End Medical Chatbot (Generative AI)

A Retrieval-Augmented Generation (RAG) based medical chatbot that answers questions from medical documents using LLMs and vector search.

---

## 🔍 Overview

This project builds an intelligent medical Q&A chatbot that:
- Loads and processes medical PDF documents
- Chunks and embeds them using HuggingFace sentence transformers
- Stores embeddings in a Pinecone vector database
- Retrieves relevant context and answers questions using the Groq LLM (LLaMA 3.3 70B)

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq — `llama-3.3-70b-versatile` |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | Pinecone |
| RAG Framework | LangChain + langchain-classic |
| PDF Loader | LangChain `PyPDFLoader` |
| Environment | Python 3.11, virtualenv |

---

---

## 🔑 API Keys Required

- **Pinecone** — [https://www.pinecone.io](https://www.pinecone.io)
- **Groq** — [https://console.groq.com](https://console.groq.com)

---
---

## 📝 License

This project is for educational purposes.
