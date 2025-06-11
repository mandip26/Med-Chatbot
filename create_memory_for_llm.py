"""
Vector Store Creation Module

This module loads PDF documents, creates text chunks, generates embeddings,
and stores them in a FAISS vector database for retrieval-based QA.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
def load_pdf_files(data_path: str) -> List[Document]:
    """
    Load PDF files from the specified directory.
    
    Args:
        data_path: Path to the directory containing PDF files
        
    Returns:
        List of Document objects loaded from PDFs
        
    Raises:
        FileNotFoundError: If the data path doesn't exist
        ValueError: If no PDF files are found
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    try:
        loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        
        if not documents:
            raise ValueError(f"No PDF files found in {data_path}")
            
        logger.info(f"Successfully loaded {len(documents)} PDF pages from {data_path}")
        return documents
        
    except Exception as e:
        logger.error(f"Error loading PDF files: {str(e)}")
        raise

def create_chunks(extracted_data: List[Document], 
                 chunk_size: int = CHUNK_SIZE, 
                 chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        extracted_data: List of documents to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        text_chunks = text_splitter.split_documents(extracted_data)
        
        logger.info(f"Created {len(text_chunks)} text chunks")
        return text_chunks
        
    except Exception as e:
        logger.error(f"Error creating chunks: {str(e)}")
        raise

def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """
    Initialize the embedding model.
    
    Args:
        model_name: Name of the embedding model to use
        
    Returns:
        HuggingFaceEmbeddings instance
    """
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        logger.info(f"Initialized embedding model: {model_name}")
        return embedding_model
        
    except Exception as e:
        logger.error(f"Error initializing embedding model: {str(e)}")
        raise

def create_vectorstore(text_chunks: List[Document], 
                      embedding_model: HuggingFaceEmbeddings,
                      db_path: str = DB_FAISS_PATH) -> None:
    """
    Create and save FAISS vector database.
    
    Args:
        text_chunks: List of document chunks
        embedding_model: Embedding model to use
        db_path: Path to save the vector database
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Create FAISS database
        db = FAISS.from_documents(text_chunks, embedding_model)
        
        # Save to local storage
        db.save_local(db_path)
        
        logger.info(f"Vector database saved successfully to {db_path}")
        
    except Exception as e:
        logger.error(f"Error creating vector database: {str(e)}")
        raise

def main() -> None:
    """Main function to orchestrate the vector store creation process."""
    try:
        logger.info("Starting vector store creation process...")
        
        # Step 1: Load PDF files
        documents = load_pdf_files(DATA_PATH)
        
        # Step 2: Create chunks
        text_chunks = create_chunks(documents)
        
        # Step 3: Get embedding model
        embedding_model = get_embedding_model()
        
        # Step 4: Create and save vector store
        create_vectorstore(text_chunks, embedding_model)
        
        logger.info("Vector store creation completed successfully!")
        
    except Exception as e:
        logger.error(f"Vector store creation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
