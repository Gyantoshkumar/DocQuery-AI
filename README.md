# 🔮 DocQuery-AI

> Intelligent Document Q&A System powered by Google Gemini & LangChain

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge)

## ✨ Features

- 📄 **Multi-PDF Support** - Upload and query multiple documents simultaneously
- 🤖 **AI-Powered Answers** - Leverages Google Gemini 1.5 Flash for intelligent responses
- 📑 **Source Citations** - Every answer shows exactly where the information comes from
- 💬 **Conversational Interface** - Natural chat-style Q&A with memory
- 🎨 **Modern UI** - Beautiful dark theme with glassmorphism effects
- ⚡ **Fast Processing** - Efficient vector search using ChromaDB
- 🔒 **Secure** - API keys stored securely, never exposed in code

## 🚀 Live Demo

[**Try DocQuery-AI →**](https://docquery-ai.streamlit.app)

## 📸 Screenshots

![DocQuery-AI Interface](https://via.placeholder.com/800x450?text=DocQuery-AI+Interface)

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| LLM | Google Gemini 1.5 Flash |
| Embeddings | Google text-embedding-004 |

## 📋 Prerequisites

- Python 3.9+
- Google AI API Key ([Get one here](https://aistudio.google.com/apikey))

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Gyantoshkumar/DocQuery-AI.git
cd DocQuery-AI
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

### 5. Open in Browser

Navigate to `http://localhost:8501`

## 🌐 Deploy to Streamlit Cloud (Free Hosting)

1. **Fork this repository** to your GitHub account

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Click "New app"** and select your forked repository

4. **Set the main file path** to `app.py`

5. **Add your API key** in the "Advanced settings" → "Secrets":
   ```toml
   GOOGLE_API_KEY = "your-api-key-here"
   ```

6. **Click Deploy!** 🎉

## 🔧 Configuration

### Environment Variables

Create a `.env` file (copy from `.env.example`):

```env
GOOGLE_API_KEY=your_api_key_here
```

### Streamlit Secrets (for deployment)

Add to `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your-api-key-here"
```

## 📖 How to Use

1. **Enter your API Key** in the sidebar
2. **Upload PDF files** using the file uploader
3. **Click "Process Documents"** to create the knowledge base
4. **Ask questions** in the chat input
5. **View answers** with source citations

## 🤔 Example Questions

- "What is the main topic of this document?"
- "Summarize the key findings"
- "What are the recommendations mentioned?"
- "Explain the methodology used"
- "Find information about [specific topic]"

## 📁 Project Structure

```
DocQuery-AI/
├── app.py                 # Main Streamlit application
├── rag_engine.py          # RAG pipeline with LangChain
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit theme configuration
├── .env.example           # Environment variables template
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) - For the amazing web framework
- [LangChain](https://langchain.com/) - For the powerful RAG framework
- [Google Gemini](https://ai.google.dev/) - For the incredible AI capabilities
- [ChromaDB](https://www.trychroma.com/) - For the efficient vector storage

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/Gyantoshkumar">Gyantosh Kumar</a>
</p>
