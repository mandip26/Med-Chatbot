"""
Configuration settings for the MediBot application.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data"
VECTORSTORE_PATH = BASE_DIR / "vectorstore" / "db_faiss"

# Model configurations
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Text processing settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# LLM settings
TEMPERATURE = 0.2
MAX_LENGTH = "512"

# Retrieval settings
RETRIEVAL_K = 3

# Environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")

# Streamlit settings
STREAMLIT_CONFIG = {
    "page_title": "MediBot - Medical Assistant",
    "page_icon": "🏥",
    "layout": "wide"
}

# Logging configuration
LOGGING_LEVEL = "INFO"
LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Prompt templates
DEFAULT_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's questions.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Provide accurate and helpful medical information while emphasizing that this is not a substitute for professional medical advice.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

# File patterns
SUPPORTED_FILE_TYPES = [".pdf"]
PDF_GLOB_PATTERN = "*.pdf"

# Error messages
ERROR_MESSAGES = {
    "no_hf_token": "Hugging Face token not found. Please set the HF_TOKEN environment variable.",
    "vectorstore_not_found": "Vector database not found. Please run create_memory_for_llm.py first.",
    "no_pdf_files": "No PDF files found in the data directory.",
    "invalid_query": "Please enter a valid question.",
    "llm_load_error": "Failed to load the language model.",
    "vectorstore_load_error": "Failed to load the knowledge base."
}
