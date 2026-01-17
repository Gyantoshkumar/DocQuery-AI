# 🦙 DocQuery-AI (Local Version)

> A 100% local RAG system using Ollama + Llama 3.1 and Hugging Face Embeddings.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge)

## ⚠️ Prerequisites

**You MUST install Ollama first:**
1. Download from [ollama.com/download](https://ollama.com/download)
2. Install the application
3. Open a **new terminal** and run:
   ```bash
   ollama pull llama3.1
   ```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```
*If that doesn't work, try:*
```bash
python -m streamlit run app.py
```

### 3. Features
- **Upload PDFs**: Ingest multiple documents.
- **Chat**: Ask questions about your content.
- **Reset**: Use "Reset All" to clear documents and history.


## 🛠️ Tech Stack

| Component | Technology | Local? |
|-----------|------------|--------|
| **LLM** | Ollama (Llama 3.1:8b) | ✅ Yes |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | ✅ Yes |
| **Vector DB** | FAISS | ✅ Yes |
| **UI** | Streamlit | ✅ Yes |

## 📁 Structure

```
DocQuery-AI/
├── src/
│   ├── rag_engine.py    # Local RAG logic
│   └── config.py        # Model settings
├── app.py               # UI
└── requirements.txt     # Dependencies
```
