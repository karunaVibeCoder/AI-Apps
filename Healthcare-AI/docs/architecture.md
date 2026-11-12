# Healthcare AI Architecture

## Overview

The application uses locally hosted AI models to generate healthcare information responses.

## Components

1. **Streamlit:** Provides the user interface.
2. **Ollama:** Runs the local LLM.
3. **Llama 3.2 (3B):** Generates responses.
4. **all-minilm:** Converts text into vector embeddings.
5. **ChromaDB:** Stores and retrieves document embeddings.

## RAG Data Flow

PDF Upload ? Text Extraction ? Chunking ? Embedding Generation ? ChromaDB Storage ? Semantic Retrieval ? LLM Response.

## Privacy

The project is designed for local model execution. Do not upload real patient records or sensitive health information.
