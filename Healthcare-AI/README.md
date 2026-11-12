# Healthcare AI Assistant

A beginner-friendly Healthcare AI application built using Python, Streamlit, Ollama, and Retrieval-Augmented Generation (RAG).

## Project Overview

This project explores how Large Language Models (LLMs) and RAG can be used to build a healthcare information assistant using locally hosted AI models.

## Features

- Interactive healthcare assistant using Streamlit.
- Local LLM integration through Ollama.
- PDF document ingestion.
- Text extraction and chunking.
- Semantic search using vector embeddings.
- ChromaDB for persistent vector storage.
- Context-grounded responses using retrieved document content.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| User interface | Streamlit |
| Local LLM runtime | Ollama |
| Chat model | llama3.2:3b |
| Embedding model | all-minilm |
| Vector database | ChromaDB |
| RAG | Retrieval-Augmented Generation |

## Architecture

```text
User
 |
 v
Streamlit Application
 |
 v
User Query
 |
 v
Embedding Model (all-minilm)
 |
 v
ChromaDB Vector Search
 |
 v
Relevant Document Chunks
 |
 v
Ollama LLM (llama3.2:3b)
 |
 v
Context-Grounded Response
```

## Project Structure

```text
healthcare_ai_v01/
|
|-- app.py
|-- rag_app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|
|-- docs/
|   |-- architecture.md
|
|-- sample_documents/
    |-- README.md
```

## Prerequisites

Install the following software:

1. Python 3.11 or a compatible version supported by your dependencies.
2. Git for Windows.
3. Ollama.
4. Visual Studio Code (optional).

Official installation links:

- Python: https://www.python.org/downloads/
- Git: https://git-scm.com/install/windows
- Ollama: https://ollama.com/download
- VS Code: https://code.visualstudio.com/

## Installation and Setup

### 1. Clone the repository

```powershell
git clone https://github.com/karunaVibeCoder/AI-Apps.git
```

Navigate to the project directory containing these application files.

### 2. Create a Python virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Ollama models

Start Ollama and download the required models:

```powershell
ollama pull llama3.2:3b
ollama pull all-minilm
```

Check installed models:

```powershell
ollama list
```

### 5. Run the Healthcare Assistant

```powershell
streamlit run app.py
```

### 6. Run the RAG application

```powershell
streamlit run rag_app.py
```

Streamlit normally opens the application at:

http://localhost:8501

## RAG Workflow

1. Upload a healthcare PDF document.
2. Extract text from the PDF.
3. Split the text into smaller chunks.
4. Generate embeddings using all-minilm.
5. Store embeddings in ChromaDB.
6. Convert the user query into an embedding.
7. Retrieve relevant document chunks.
8. Pass the retrieved context to the Ollama LLM.
9. Generate a response based on the retrieved information.

## Troubleshooting

### Ollama connection issue

Verify that Ollama is running:

```powershell
ollama list
```

### Missing Python packages

```powershell
pip install -r requirements.txt
```

### Streamlit command not found

```powershell
python -m streamlit run app.py
```

### ChromaDB data

The local vector database is stored in `healthcare_chroma_db/`. It is excluded from Git and can be recreated by processing the source documents.

## Future Enhancements

- LangChain and LangGraph workflows.
- Model Context Protocol (MCP) integration.
- Healthcare document summarization.
- Agentic AI for healthcare workflows.
- Automated document evaluation and testing.

## Disclaimer

This project is for educational and demonstration purposes only. It is not a medical diagnostic system and must not be used as a substitute for professional medical advice.

Do not upload real patient records or personally identifiable health information.

## Author

Healthcare AI Learning Project

Repository: https://github.com/karunaVibeCoder/AI-Apps
