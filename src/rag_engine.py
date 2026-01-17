"""
DocQuery-AI: Local RAG Engine
=============================
Core RAG pipeline using Ollama (Llama 3.1) and HuggingFace Embeddings.
"""

from typing import List, Tuple, Optional
import os
import tempfile
import hashlib

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import LLM_MODEL, EMBEDDING_MODEL

class RAGEngine:
    """
    Local RAG Engine using Ollama and HuggingFace.
    """
    
    def __init__(self):
        """Initialize the Local RAG Engine."""
        
        # Initialize local embeddings (runs on CPU/GPU)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        
        # Initialize local LLM via Ollama
        self.llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0.3
        )
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.vector_store: Optional[FAISS] = None
        self.processed_files: List[str] = []
        self.chat_history: List[Tuple[str, str]] = []
        
    def process_pdfs(self, pdf_files: List[Tuple[str, bytes]]) -> Tuple[bool, str]:
        """Process uploaded PDF files and create vector store."""
        try:
            all_documents = []
            
            for filename, file_content in pdf_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file_content)
                    tmp_path = tmp_file.name
                
                try:
                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()
                    for doc in documents:
                        doc.metadata["source"] = filename
                    all_documents.extend(documents)
                    self.processed_files.append(filename)
                finally:
                    os.unlink(tmp_path)
            
            if not all_documents:
                return False, "No content could be extracted."
            
            chunks = self.text_splitter.split_documents(all_documents)
            
            # Create FAISS vector store with local embeddings
            self.vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            
            self.chat_history = []
            return True, f"✅ Successfully processed {len(pdf_files)} files ({len(chunks)} chunks)."
            
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
    
    def ask_question(self, question: str) -> Tuple[str, List[dict]]:
        """Ask a question using the local model."""
        if not self.vector_store:
            return "Please upload documents first.", []
        
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(question)
            
            # Simple context formatting
            context = "\n\n".join(doc.page_content for doc in docs)
            
            prompt = ChatPromptTemplate.from_template("""You are an intelligent document assistant.
            
CONTEXT:
{context}

QUESTION: {question}

ANSWER:""")
            
            chain = prompt | self.llm | StrOutputParser()
            
            answer = chain.invoke({
                "context": context,
                "question": question
            })
            
            self.chat_history.append((question, answer))
            
            # Source metadata
            sources = []
            seen = set()
            for doc in docs:
                content = doc.page_content[:200]
                if content not in seen:
                    seen.add(content)
                    sources.append({
                        "source": doc.metadata.get("source", "Unknown"),
                        "content": content,
                        "page": doc.metadata.get("page", "N/A")
                    })
            
            return answer, sources
            
        except Exception as e:
            return f"Error: {str(e)}", []
    
    def reset(self):
        self.vector_store = None
        self.processed_files = []
        self.chat_history = []
