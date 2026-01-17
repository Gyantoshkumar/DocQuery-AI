"""
DocQuery-AI: RAG Engine
========================
Core RAG (Retrieval Augmented Generation) pipeline using LangChain and Google Gemini.
Refactored to use simpler approach for Python 3.14 compatibility.
"""

import os
import tempfile
from typing import List, Tuple, Optional
import hashlib

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class RAGEngine:
    """
    RAG Engine for processing PDFs and answering questions using Google Gemini.
    Uses simple LCEL chains for Python 3.14 compatibility.
    """
    
    def __init__(self, google_api_key: str):
        """
        Initialize the RAG Engine with Google API credentials.
        
        Args:
            google_api_key: Google AI API key for Gemini access
        """
        self.google_api_key = google_api_key
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
        # Initialize embeddings model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=google_api_key
        )
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_api_key,
            temperature=0.3
        )
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Vector store (initialized when documents are loaded)
        self.vector_store: Optional[FAISS] = None
        self.processed_files: List[str] = []
        self.chat_history: List[Tuple[str, str]] = []
        
    def _get_file_hash(self, file_content: bytes) -> str:
        """Generate a hash for file content to track processed files."""
        return hashlib.md5(file_content).hexdigest()
    
    def _format_docs(self, docs) -> str:
        """Format retrieved documents into a string."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _format_chat_history(self) -> str:
        """Format chat history into a string."""
        if not self.chat_history:
            return "No previous conversation."
        
        formatted = []
        for human, ai in self.chat_history[-5:]:  # Keep last 5 exchanges
            formatted.append(f"Human: {human}")
            formatted.append(f"Assistant: {ai}")
        return "\n".join(formatted)
    
    def process_pdfs(self, pdf_files: List[Tuple[str, bytes]]) -> Tuple[bool, str]:
        """
        Process uploaded PDF files and create vector store.
        
        Args:
            pdf_files: List of tuples (filename, file_content)
            
        Returns:
            Tuple of (success, message)
        """
        try:
            all_documents = []
            
            for filename, file_content in pdf_files:
                # Create temporary file to process PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file_content)
                    tmp_path = tmp_file.name
                
                try:
                    # Load PDF
                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()
                    
                    # Add source metadata
                    for doc in documents:
                        doc.metadata["source"] = filename
                    
                    all_documents.extend(documents)
                    self.processed_files.append(filename)
                    
                finally:
                    # Clean up temp file
                    os.unlink(tmp_path)
            
            if not all_documents:
                return False, "No content could be extracted from the PDFs."
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(all_documents)
            
            # Create vector store
            self.vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            
            # Clear chat history for new documents
            self.chat_history = []
            
            total_chunks = len(chunks)
            return True, f"✅ Successfully processed {len(pdf_files)} PDF(s) with {total_chunks} text chunks."
            
        except Exception as e:
            return False, f"❌ Error processing PDFs: {str(e)}"
    
    def ask_question(self, question: str) -> Tuple[str, List[dict]]:
        """
        Ask a question about the loaded documents.
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (answer, source_documents)
        """
        if not self.vector_store:
            return "Please upload PDF documents first before asking questions.", []
        
        try:
            # Get retriever
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            
            # Retrieve relevant documents
            docs = retriever.invoke(question)
            context = self._format_docs(docs)
            chat_history = self._format_chat_history()
            
            # Create prompt
            prompt = ChatPromptTemplate.from_template("""You are DocQuery-AI, an intelligent document assistant. Your role is to help users understand and extract information from their uploaded documents.

INSTRUCTIONS:
- Answer questions based ONLY on the provided context from the documents
- If the answer is not in the documents, clearly state that
- Be concise but comprehensive
- Use bullet points for lists
- Quote relevant passages when appropriate
- Always be helpful and professional

CONTEXT FROM DOCUMENTS:
{context}

PREVIOUS CONVERSATION:
{chat_history}

USER QUESTION: {question}

DOCQUERY-AI RESPONSE:""")
            
            # Create simple chain using LCEL
            chain = prompt | self.llm | StrOutputParser()
            
            # Get response
            answer = chain.invoke({
                "context": context,
                "chat_history": chat_history,
                "question": question
            })
            
            # Update chat history
            self.chat_history.append((question, answer))
            
            # Format source documents
            sources = []
            seen_contents = set()
            
            for doc in docs:
                content = doc.page_content[:300]  # Truncate for display
                content_hash = hash(content)
                
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    sources.append({
                        "content": content + "..." if len(doc.page_content) > 300 else content,
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", "N/A")
                    })
            
            return answer, sources
            
        except Exception as e:
            return f"Error generating response: {str(e)}", []
    
    def clear_memory(self):
        """Clear conversation history."""
        self.chat_history = []
    
    def reset(self):
        """Reset the engine, clearing all loaded documents and memory."""
        self.vector_store = None
        self.processed_files = []
        self.chat_history = []


def validate_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Validate the Google API key by making a simple embedding request.
    
    Args:
        api_key: Google AI API key
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )
        # Test with a simple embedding
        embeddings.embed_query("test")
        return True, "API key is valid!"
    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower():
            return False, "Invalid API key. Please check your Google AI API key."
        return False, f"Error validating API key: {error_msg}"
