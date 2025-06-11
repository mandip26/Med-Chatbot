import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
import logging
from typing import Optional

DB_FAISS_PATH = "vectorstore/db_faiss"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def get_vectorstore() -> Optional[FAISS]:
    """
    Load the FAISS vectorstore with embeddings.
    
    Returns:
        FAISS vectorstore or None if loading fails
    """
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        
        if not os.path.exists(DB_FAISS_PATH):
            logger.error(f"Vectorstore path not found: {DB_FAISS_PATH}")
            return None
            
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        logger.info("Vectorstore loaded successfully")
        return db
    except Exception as e:
        logger.error(f"Error loading vectorstore: {str(e)}")
        return None

def set_custom_prompt(custom_prompt_template: str) -> PromptTemplate:
    """
    Create a custom prompt template for the QA chain.
    
    Args:
        custom_prompt_template: The template string for the prompt
        
    Returns:
        PromptTemplate object
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id: str, hf_token: str) -> Optional[HuggingFaceEndpoint]:
    """
    Load the Hugging Face language model.
    
    Args:
        huggingface_repo_id: Repository ID for the model
        hf_token: Hugging Face API token
        
    Returns:
        HuggingFaceEndpoint or None if loading fails
    """
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=0.2,
            model_kwargs={"token": hf_token, "max_length": "512"}
        )
        return llm
    except Exception as e:
        logger.error(f"Error loading LLM: {str(e)}")
        return None

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(
        page_title="MediBot - Medical Assistant",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 MediBot - Your Medical Assistant")
    st.markdown("*Ask questions about medical topics and get informed answers based on our knowledge base.*")

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Chat input
    prompt = st.chat_input("Ask me a medical question...")

    if prompt:
        # Validate input
        if not prompt.strip():
            st.error("Please enter a valid question.")
            return
            
        # Display user message
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Custom prompt template
        custom_prompt_template = """
        Use the pieces of information provided in the context to answer user's questions.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Provide accurate and helpful medical information while emphasizing that this is not a substitute for professional medical advice.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        # Configuration
        huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
        hf_token = os.environ.get("HF_TOKEN")

        if not hf_token:
            st.error("❌ Hugging Face token not found. Please set the HF_TOKEN environment variable.")
            return

        try:
            with st.spinner("🔍 Searching knowledge base..."):
                # Load vectorstore
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("❌ Failed to load the knowledge base. Please check if the vectorstore exists.")
                    return

                # Load LLM
                llm = load_llm(huggingface_repo_id, hf_token)
                if llm is None:
                    st.error("❌ Failed to load the language model.")
                    return

                # Create QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)}
                )
                
                # Get response
                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                source_documents = response["source_documents"]
                
                # Display response
                st.chat_message('assistant').markdown(result)
                st.session_state.messages.append({'role': 'assistant', 'content': result})
                
                # Optional: Show source information
                if st.checkbox("Show source documents", key=f"sources_{len(st.session_state.messages)}"):
                    with st.expander("📚 Source Documents"):
                        for i, doc in enumerate(source_documents):
                            st.write(f"**Source {i+1}:**")
                            st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                            st.write("---")

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            st.error(f"❌ An error occurred: {str(e)}")
            st.error("Please try again or contact support if the issue persists.")

if __name__ == '__main__':
    main()