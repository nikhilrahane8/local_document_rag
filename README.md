# Local PDF RAG Assistant

This project is a local Retrieval-Augmented Generation (RAG) pipeline that allows you to chat with a PDF document. It uses LangChain, ChromaDB, HuggingFace for local embeddings, and Ollama to run the LLM entirely offline.

## Setup Instructions

1. **Install Dependencies:**
   `pip install -r requirements.txt`

2. **Install Ollama:**
   Download and install [Ollama](https://ollama.com/).
   Pull the required model by running:
   `ollama pull qwen2.5:0.5b`

3. **Run the Application:**
   `python app.py`

## sentence transformers are required for hood of huggingFace to work locally