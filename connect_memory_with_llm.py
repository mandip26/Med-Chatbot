"""
Memory-LLM Connection Module

This module connects the FAISS vector database with a language model
to create a retrieval-based question-answering system.
"""

import os
import logging
from typing import Optional, Dict, Any

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_LENGTH = "512"
DEFAULT_K = 3

def load_llm(huggingface_repo_id: str = DEFAULT_MODEL, 
             hf_token: Optional[str] = None,
             temperature: float = DEFAULT_TEMPERATURE,
             max_length: str = DEFAULT_MAX_LENGTH) -> HuggingFaceEndpoint:
    """
    Load and configure the Hugging Face language model.
    
    Args:
        huggingface_repo_id: Repository ID for the model
        hf_token: Hugging Face API token
        temperature: Sampling temperature for generation
        max_length: Maximum length for generated text
        
    Returns:
        Configured HuggingFaceEndpoint instance
        
    Raises:
        ValueError: If HF_TOKEN is not provided
    """
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
        
    if not hf_token:
        raise ValueError("HF_TOKEN not found. Please set the HF_TOKEN environment variable.")
    
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=temperature,
            model_kwargs={"token": hf_token, "max_length": max_length}
        )
        logger.info(f"Successfully loaded LLM: {huggingface_repo_id}")
        return llm
        
    except Exception as e:
        logger.error(f"Error loading LLM: {str(e)}")
        raise

def set_custom_prompt(custom_prompt_template: str) -> PromptTemplate:
    """
    Create a custom prompt template for the QA chain.
    
    Args:
        custom_prompt_template: The template string for the prompt
        
    Returns:
        PromptTemplate object
    """
    try:
        prompt = PromptTemplate(
            template=custom_prompt_template, 
            input_variables=["context", "question"]
        )
        return prompt
        
    except Exception as e:
        logger.error(f"Error creating prompt template: {str(e)}")
        raise

def load_vectorstore(db_path: str = DB_FAISS_PATH, 
                    embedding_model_name: str = EMBEDDING_MODEL) -> FAISS:
    """
    Load the FAISS vector database.
    
    Args:
        db_path: Path to the FAISS database
        embedding_model_name: Name of the embedding model
        
    Returns:
        FAISS vector database instance
        
    Raises:
        FileNotFoundError: If the database path doesn't exist
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Vector database not found at {db_path}. "
                               "Please run create_memory_for_llm.py first.")
    
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
        logger.info(f"Successfully loaded vector database from {db_path}")
        return db
        
    except Exception as e:
        logger.error(f"Error loading vector database: {str(e)}")
        raise

def create_qa_chain(llm: HuggingFaceEndpoint, 
                   vectorstore: FAISS,
                   prompt_template: str,
                   k: int = DEFAULT_K) -> RetrievalQA:
    """
    Create a retrieval-based QA chain.
    
    Args:
        llm: Language model instance
        vectorstore: FAISS vector database
        prompt_template: Custom prompt template
        k: Number of documents to retrieve
        
    Returns:
        RetrievalQA chain instance
    """
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': k}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt(prompt_template)}
        )
        logger.info("Successfully created QA chain")
        return qa_chain
        
    except Exception as e:
        logger.error(f"Error creating QA chain: {str(e)}")
        raise

def main() -> None:
    """Main function to demonstrate the QA system."""
    # Default prompt template
    custom_prompt_template = """
    Use the pieces of information provided in the context to answer user's questions.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Provide accurate and helpful information while being concise.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """
    
    try:
        # Step 1: Load LLM
        logger.info("Loading language model...")
        llm = load_llm()
        
        # Step 2: Load vector database
        logger.info("Loading vector database...")
        vectorstore = load_vectorstore()
        
        # Step 3: Create QA chain
        logger.info("Creating QA chain...")
        qa_chain = create_qa_chain(llm, vectorstore, custom_prompt_template)
        
        # Step 4: Interactive query loop
        print("\n🏥 MediBot Console Interface")
        print("=" * 40)
        print("Ask medical questions or type 'quit' to exit")
        print("=" * 40)
        
        while True:
            user_query = input("\n❓ Your question: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not user_query:
                print("Please enter a valid question.")
                continue
                
            try:
                print("\n🔍 Searching knowledge base...")
                response = qa_chain.invoke({'query': user_query})
                
                print(f"\n🤖 MediBot: {response['result']}")
                
                # Optionally show source documents
                show_sources = input("\n📚 Show source documents? (y/n): ").lower()
                if show_sources == 'y':
                    print("\n📄 Source Documents:")
                    for i, doc in enumerate(response['source_documents'], 1):
                        print(f"\n--- Source {i} ---")
                        content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        print(content)
                        
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                print(f"❌ Error: {str(e)}")
                
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"❌ Application error: {str(e)}")

if __name__ == "__main__":
    main()

print("RESULT:", response["result"])
print("SOURCE DOCUMENTS:", response["source_documents"])

