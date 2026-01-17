"""
DocQuery-AI: Intelligent Document Q&A System
=============================================
A beautiful, production-ready RAG application built with Streamlit and Google Gemini.
Upload PDFs and get AI-powered answers with source citations.
"""

import streamlit as st
from rag_engine import RAGEngine, validate_api_key
import time

# Page configuration
st.set_page_config(
    page_title="DocQuery-AI | Intelligent Document Q&A",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: shimmer 15s infinite linear;
    }
    
    @keyframes shimmer {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        animation: fadeIn 0.3s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 15%;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.25);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        color: #e0e0e0;
        margin-right: 15%;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
    }
    
    .message-role {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
        opacity: 0.8;
    }
    
    .message-content {
        line-height: 1.6;
    }
    
    /* Source citation card */
    .source-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 0.75rem;
        backdrop-filter: blur(10px);
    }
    
    .source-card-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .source-content {
        font-size: 0.85rem;
        color: #a0a0a0;
        font-style: italic;
        line-height: 1.5;
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: rgba(102, 126, 234, 0.1) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        color: #e0e0e0;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
    }
    
    /* Status cards */
    .status-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    
    .status-card.processing {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .status-card.error {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%);
        border-color: rgba(239, 68, 68, 0.3);
    }
    
    /* Feature cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .feature-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        background: rgba(255,255,255,0.06);
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
    }
    
    .feature-title {
        font-weight: 600;
        color: #e0e0e0;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-size: 0.85rem;
        color: #888;
    }
    
    /* Processed files list */
    .file-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        font-size: 0.85rem;
        color: #a0b4f4;
    }
    
    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        margin: 2rem 0;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.05);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "api_key_validated" not in st.session_state:
        st.session_state.api_key_validated = False


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🔮 DocQuery-AI</h1>
        <p>Upload your documents and unlock insights with AI-powered intelligence</p>
    </div>
    """, unsafe_allow_html=True)


def render_features():
    """Render feature cards for first-time users."""
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">📄</div>
            <div class="feature-title">Multi-PDF Support</div>
            <div class="feature-desc">Upload multiple documents at once</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AI-Powered</div>
            <div class="feature-desc">Google Gemini for smart answers</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📑</div>
            <div class="feature-title">Source Citations</div>
            <div class="feature-desc">See exactly where answers come from</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">💬</div>
            <div class="feature-title">Conversational</div>
            <div class="feature-desc">Natural chat-style interface</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_message(role: str, content: str, sources: list = None):
    """Render a chat message with optional sources."""
    role_class = "user-message" if role == "user" else "assistant-message"
    role_label = "You" if role == "user" else "DocQuery-AI"
    role_emoji = "👤" if role == "user" else "🔮"
    
    sources_html = ""
    if sources and role == "assistant":
        for i, source in enumerate(sources[:3], 1):  # Show max 3 sources
            sources_html += f"""
            <div class="source-card">
                <div class="source-card-header">
                    📎 Source {i}: {source['source']} (Page {source['page']})
                </div>
                <div class="source-content">"{source['content']}"</div>
            </div>
            """
    
    st.markdown(f"""
    <div class="chat-message {role_class}">
        <div class="message-role">{role_emoji} {role_label}</div>
        <div class="message-content">{content}</div>
        {sources_html}
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with configuration and file upload."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google AI API Key",
            type="password",
            placeholder="Enter your API key...",
            help="Get your API key from aistudio.google.com"
        )
        
        if api_key and not st.session_state.api_key_validated:
            with st.spinner("Validating API key..."):
                is_valid, message = validate_api_key(api_key)
                if is_valid:
                    st.session_state.api_key_validated = True
                    st.session_state.rag_engine = RAGEngine(api_key)
                    st.success("✅ API key validated!")
                else:
                    st.error(message)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # File upload section
        st.markdown("## 📄 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF documents to query"
        )
        
        if uploaded_files and st.session_state.api_key_validated:
            if st.button("🚀 Process Documents", use_container_width=True):
                with st.spinner("Processing documents..."):
                    # Prepare files for processing
                    pdf_files = []
                    for file in uploaded_files:
                        pdf_files.append((file.name, file.read()))
                        file.seek(0)  # Reset file pointer
                    
                    # Process PDFs
                    success, message = st.session_state.rag_engine.process_pdfs(pdf_files)
                    
                    if success:
                        st.session_state.documents_loaded = True
                        st.success(message)
                        
                        # Show processed files
                        st.markdown("### 📚 Loaded Documents")
                        for fname in st.session_state.rag_engine.processed_files:
                            st.markdown(f'<span class="file-chip">📄 {fname}</span>', unsafe_allow_html=True)
                    else:
                        st.error(message)
        
        elif uploaded_files and not st.session_state.api_key_validated:
            st.warning("⚠️ Please enter a valid API key first")
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Actions section
        st.markdown("## 🔧 Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.rag_engine:
                    st.session_state.rag_engine.clear_memory()
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset All", use_container_width=True):
                st.session_state.messages = []
                st.session_state.documents_loaded = False
                if st.session_state.rag_engine:
                    st.session_state.rag_engine.reset()
                st.rerun()
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # About section
        st.markdown("## ℹ️ About")
        st.markdown("""
        **DocQuery-AI** is a RAG-powered document assistant that helps you extract insights from your PDFs using Google's Gemini AI.
        
        Made with ❤️ using Streamlit & LangChain
        """)


def main():
    """Main application entry point."""
    init_session_state()
    render_header()
    render_sidebar()
    
    # Main content area
    if not st.session_state.documents_loaded:
        st.markdown("### 👋 Welcome to DocQuery-AI!")
        st.markdown("Upload your PDF documents from the sidebar to get started.")
        render_features()
        
        # Sample questions
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 💡 Example Questions You Can Ask")
        st.markdown("""
        - *"What is the main topic of this document?"*
        - *"Summarize the key findings."*
        - *"What are the recommendations mentioned?"*
        - *"Explain the methodology used."*
        - *"Find information about [specific topic]."*
        """)
    
    else:
        # Chat interface
        st.markdown("### 💬 Chat with Your Documents")
        
        # Display chat messages
        for message in st.session_state.messages:
            render_message(
                message["role"],
                message["content"],
                message.get("sources", None)
            )
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })
            render_message("user", prompt)
            
            # Get AI response
            with st.spinner("Thinking..."):
                answer, sources = st.session_state.rag_engine.ask_question(prompt)
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
            render_message("assistant", answer, sources)
            
            # Force refresh to show new messages properly
            time.sleep(0.1)
            st.rerun()


if __name__ == "__main__":
    main()
