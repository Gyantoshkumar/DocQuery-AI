"""
DocQuery-AI: Local App
======================
Streamlit interface for Local RAG using Ollama.
"""

import streamlit as st
from src.rag_engine import RAGEngine
from src.config import LLM_MODEL
import time

st.set_page_config(page_title="DocQuery-AI (Local)", page_icon="🦙", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()
if "messages" not in st.session_state:
    st.session_state.messages = []

def main():
    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1>🦙 DocQuery-AI (Local)</h1>
        <p>Running locally with <b>Ollama ({LLM_MODEL})</b> + HuggingFace Embeddings</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📄 Documents")
        files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        
        if files and st.button("Process Documents", use_container_width=True):
            with st.spinner("Processing locally..."):
                pdf_data = [(f.name, f.read()) for f in files]
                for f in files: f.seek(0)
                
                success, msg = st.session_state.rag.process_pdfs(pdf_data)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
        
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("Reset All", type="primary", use_container_width=True):
                st.session_state.rag.reset()
                st.session_state.messages = []
                st.rerun()

    # Chat Area
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['source']}** (p.{s['page']}): {s['content']}...")

    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking (Local LLM)..."):
                response, sources = st.session_state.rag.ask_question(prompt)
                st.markdown(response)
                with st.expander("Sources"):
                     for s in sources:
                        st.markdown(f"**{s['source']}** (p.{s['page']}): {s['content']}...")
        
        st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})

if __name__ == "__main__":
    main()
