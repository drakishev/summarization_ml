# Local LLM with RAG

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system for offline document summarization and question-answering. It leverages Llama 3.2 1B, LangChain, Ollama, Chroma, and Streamlit to process multi-format documents locally, ensuring data privacy. The system supports document ingestion, text segmentation, embedding storage, and interactive querying via command-line and web interfaces.

## Dependencies
- chromadb==0.6.3
- langchain==0.3.18
- langchain-community==0.3.17
- langchain-ollama==0.2.3
- numpy
- ollama==0.4.7
- pypdf==5.3.0
- python-docx
- requests
- sentence-transformers
- streamlit==1.42.0
- tiktoken
- tqdm==4.67.1
- unstructured
- watchdog==6.0.0

## Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd local-llm-with-rag
   ```
2. Create a Python virtual environment (Python >=3.12):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install and start Ollama:
   ```bash
   # Follow instructions at https://ollama.com/download
   ollama pull llama3.2:1b
   ```
5. Ensure the `storage` directory exists for Chroma persistence:
   ```bash
   mkdir -p storage
   ```

## Usage
- **Command-Line Interface**:
  Run `app.py` to interact via terminal:
  ```bash
  python app.py -m llama3.2:1b -p <path-to-documents>
  ```
  Enter questions or type `exit` to quit.

- **Web Interface**:
  Run `ui.py` for a Streamlit-based interface:
  ```bash
  streamlit run ui.py
  ```
  Upload documents, adjust chunk settings, and ask questions via the browser.