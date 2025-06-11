# 🏥 MediBot - AI-Powered Medical Assistant

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.43.0-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.19-green.svg)](https://github.com/langchain-ai/langchain)

MediBot is an intelligent medical assistant powered by advanced AI technologies. It uses Retrieval-Augmented Generation (RAG) to provide accurate medical information based on a curated knowledge base of medical documents.

## 🌟 Features

- **📚 Knowledge Base**: Processes PDF medical documents to create a searchable knowledge base
- **🔍 Intelligent Retrieval**: Uses FAISS vector database for fast and accurate document retrieval
- **🤖 AI-Powered Responses**: Leverages Mistral-7B language model for generating contextual answers
- **💬 Interactive Chat Interface**: User-friendly Streamlit web interface with chat functionality
- **📖 Source Attribution**: Shows source documents for transparency and verification
- **🛡️ Safe Responses**: Designed to acknowledge limitations and avoid hallucinations

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Documents │───▶│  Text Chunking   │───▶│   Vector Store  │
│                 │    │   & Embedding    │    │    (FAISS)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Retrieval QA    │◀───│  Document       │
│                 │    │     Chain        │    │  Retrieval      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌──────────────────┐
                       │   LLM Response   │
                       │  (Mistral-7B)    │
                       └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Hugging Face API token ([Get one here](https://huggingface.co/settings/tokens))

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/medibot.git
   cd medibot
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows with bash
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   echo "HF_TOKEN=your_hugging_face_token_here" > .env
   ```

### Setup Knowledge Base

1. **Add PDF documents**

   ```bash
   # Place your medical PDF files in the data/ directory
   mkdir -p data
   # Copy your PDF files to data/
   ```

2. **Create vector database**
   ```bash
   python create_memory_for_llm.py
   ```

### Run the Application

**Web Interface (Recommended)**

```bash
streamlit run medibot.py
```

**Console Interface**

```bash
python connect_memory_with_llm.py
```

## 📁 Project Structure

```
medibot/
├── 📄 README.md                    # Project documentation
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore                   # Git ignore rules
├── 📄 .env                         # Environment variables
├── 📄 config.py                    # Configuration settings
├── 📄 Pipfile                      # Pipenv configuration
├── 📄 Pipfile.lock                 # Pipenv lock file
│
├── 🐍 create_memory_for_llm.py     # Vector database creation
├── 🐍 medibot.py                   # Streamlit web interface
├── 🐍 connect_memory_with_llm.py   # Console interface
│
├── 📁 data/                        # PDF documents directory
│   └── 📄 *.pdf                    # Medical PDF files
│
└── 📁 vectorstore/                 # Vector database storage
    └── 📁 db_faiss/                # FAISS database files
        ├── 📄 index.faiss
        └── 📄 index.pkl
```

## 🔧 Configuration

### Environment Variables

| Variable   | Description            | Required |
| ---------- | ---------------------- | -------- |
| `HF_TOKEN` | Hugging Face API token | ✅ Yes   |

### Model Configuration

Edit `config.py` to customize:

- **Embedding Model**: Default is `sentence-transformers/all-MiniLM-L6-v2`
- **Language Model**: Default is `mistralai/Mistral-7B-Instruct-v0.3`
- **Chunk Size**: Default is 500 characters
- **Retrieval Count**: Default is 3 documents

## 📊 Usage Examples

### Adding New Documents

1. Place PDF files in the `data/` directory
2. Run the vector database creation script:
   ```bash
   python create_memory_for_llm.py
   ```

### Example Queries

- "What are the symptoms of hypertension?"
- "How is diabetes diagnosed?"
- "What are the side effects of aspirin?"
- "Explain the treatment options for asthma"

## 🛠️ API Reference

### Core Functions

#### `create_memory_for_llm.py`

- `load_pdf_files(data_path)`: Load PDF documents from directory
- `create_chunks(documents)`: Split documents into chunks
- `get_embedding_model()`: Initialize embedding model
- `create_vectorstore()`: Create and save FAISS database

#### `medibot.py`

- `get_vectorstore()`: Load cached vector database
- `load_llm()`: Initialize language model
- `set_custom_prompt()`: Create prompt template


## 📈 Performance Optimization

### Memory Usage

- Use `sentence-transformers/all-MiniLM-L6-v2` for balanced performance
- Adjust chunk size based on document complexity
- Consider using FAISS with GPU for large datasets

### Response Speed

- Reduce retrieval count for faster responses
- Use smaller embedding models for speed
- Implement caching for frequently asked questions

## 🔒 Security Considerations

- **API Tokens**: Never commit tokens to version control
- **Data Privacy**: Ensure medical documents comply with regulations
- **Input Validation**: All user inputs are validated and sanitized
- **Error Handling**: Comprehensive error handling prevents crashes

## 🚨 Limitations & Disclaimers

⚠️ **Important Medical Disclaimer**

MediBot is designed for educational and informational purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns.

### Technical Limitations

- Responses are limited to information in the knowledge base
- May not have access to the latest medical research
- Context window limitations of the underlying language model
- Potential for occasional inaccuracies in AI-generated responses


### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints
- Write unit tests for new features


## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [Streamlit](https://streamlit.io/) for the web interface
- [Hugging Face](https://huggingface.co/) for model hosting
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search


## 🔄 Changelog

### v1.0.0 (2025-06-11)

- Initial release
- RAG-based medical question answering
- Streamlit web interface
- FAISS vector database integration
- Comprehensive error handling and logging

---

**Made with ❤️ for the medical community**
