# Healthcare AI Assistant

A local healthcare knowledge assistant built with Python, Streamlit, Ollama, ChromaDB, RAG, MCP and UTCP.

The application searches indexed healthcare PDF documents first. If retrieved passages do not meet the configured relevance threshold, the UTCP workflow can call the local Ollama model as a clearly labelled general-knowledge fallback.

> **Safety notice:** This is an educational prototype, not a medical device. It does not diagnose conditions or prescribe treatment. Use public or synthetic documents only; do not upload confidential patient records or personally identifiable health information. Consult a qualified healthcare professional for medical decisions.

## Features

- Upload healthcare PDF documents through Streamlit.
- Extract and chunk PDF text, generate embeddings and store vectors in ChromaDB.
- Use Ollama `all-minilm` for embeddings and `llama3.2:3b` for local answer generation.
- Expose healthcare knowledge tools through MCP.
- Discover and invoke HTTP tools through a UTCP manual.
- Search documents before answering questions.
- Route to a background local-LLM job when no sufficiently relevant passage is found.
- Display answer provenance, retrieval distance, fallback reason and source metadata.

## Architecture

```text
User
 |
 v
Streamlit: mcp_rag_app.py
 |                         |
 v                         v
MCP client (stdio)      UTCP client (HTTP)
 |                         |
 v                         v
mcp_server.py           utcp_server.py
                           |
                      UTCP manual/tools
                           |
             +-------------+-------------+
             |                           |
             v                           v
       ChromaDB retrieval           Ollama Llama
             |                       fallback
             +-------------+-------------+
                           |
                           v
                 healthcare_service.py
```

### Question-answering flow

1. The Streamlit app passes the question to `utcp_client.py`.
2. The UTCP client calls `search_healthcare_knowledge`.
3. The UTCP server retrieves passages from ChromaDB and evaluates the best distance against the configured threshold.
4. If context is relevant, the client generates an answer using the retrieved document context.
5. If context is not relevant, the client starts a background LLM job and polls its status.
6. The UI labels the answer as either document-grounded or general local-LLM knowledge.

The current relevance threshold in `utcp_server.py` is `0.55`. For the current distance metric, the implementation treats a best distance at or below this threshold as relevant. Validate and tune this value against your own document collection; it is not a guarantee of factual or medical correctness.

## Technology stack

| Component | Purpose |
|---|---|
| Python | Runtime |
| Streamlit | User interface |
| Ollama | Local model runtime |
| `llama3.2:3b` | Answer-generation model |
| `all-minilm` | Embedding model |
| ChromaDB | Vector database |
| RAG | Retrieval-augmented generation |
| MCP | Standardized client-server tool integration |
| UTCP | Tool discovery and invocation using call templates |
| FastAPI / Uvicorn | UTCP HTTP server |

## Project structure

```text
Healthcare-AI/
├── app.py                         # Original Healthcare Assistant
├── rag_app.py                     # Original RAG application
├── mcp_rag_app.py                 # Streamlit MCP + UTCP interface
├── mcp_server.py                  # MCP server and healthcare tools
├── mcp_client_test.py             # MCP connectivity test
├── utcp_server.py                 # UTCP manual and HTTP endpoints
├── utcp_client.py                 # RAG-first and LLM-fallback workflow
├── services/
│   ├── __init__.py
│   └── healthcare_service.py      # Shared ingestion, retrieval and LLM logic
├── requirements.txt
└── README.md
```

The ChromaDB persistence directory is configured in `services/healthcare_service.py` and is outside the source folder in the current local setup. Do not commit the database directory.

## Prerequisites and setup

- Python and Git installed.
- Ollama installed and running.
- Dependencies listed in `requirements.txt`.

Pull the models:

```powershell
ollama pull llama3.2:3b
ollama pull all-minilm
```

Verify Ollama:

```powershell
ollama list
Invoke-RestMethod http://localhost:11434/api/tags
```

Install project dependencies:

```powershell
cd C:\Users\hp\healthcare_ai_v01\Healthcare-AI
python -m pip install -r requirements.txt
```

If a dependency is missing, add the tested package to `requirements.txt` rather than relying on an undocumented global installation.

## Run the application

Use separate PowerShell windows. Keep the UTCP server running while using the integrated Streamlit application.

### Terminal 1 — UTCP server

```powershell
cd C:\Users\hp\healthcare_ai_v01\Healthcare-AI
python utcp_server.py
```

Endpoints:

- Health check: `http://127.0.0.1:8000/health`
- UTCP manual: `http://127.0.0.1:8000/utcp`

Verify the advertised tools from another terminal:

```powershell
(Invoke-RestMethod http://127.0.0.1:8000/utcp).tools.name
```

Expected tools:

```text
search_healthcare_knowledge
start_local_llm_job
get_local_llm_job_status
```

If Windows reports `WinError 10048`, port 8000 may already be occupied by a running UTCP server. Check `/health` before starting another instance.

### Terminal 2 — Streamlit

```powershell
cd C:\Users\hp\healthcare_ai_v01\Healthcare-AI
streamlit run mcp_rag_app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

### Optional — MCP test

```powershell
cd C:\Users\hp\healthcare_ai_v01\Healthcare-AI
python mcp_client_test.py
```

### Optional — standalone UTCP test

```powershell
cd C:\Users\hp\healthcare_ai_v01\Healthcare-AI
python utcp_client.py
```

Enter a question and type `exit` to quit.

## MCP tools

| Tool | Purpose |
|---|---|
| `search_healthcare_knowledge` | Search indexed healthcare PDF content |
| `get_healthcare_knowledge_count` | Return the number of indexed knowledge chunks |

The Streamlit application currently uses MCP over local stdio for the knowledge-count operation. The standalone MCP test can verify MCP connectivity and tool discovery.

## UTCP tools

| Tool | Purpose |
|---|---|
| `search_healthcare_knowledge` | Retrieve document passages and return relevance information |
| `start_local_llm_job` | Start local Ollama generation in a background worker and return a job ID |
| `get_local_llm_job_status` | Retrieve background job status and result |

The background-job pattern avoids holding a single UTCP HTTP tool call open for the full duration of local model generation. The client polls the status tool until completion, failure or timeout.

## Answer provenance

The workflow returns one of these source labels:

- `healthcare_documents_rag` — generated using retrieved healthcare-document context.
- `local_llm_general_knowledge` — no sufficiently relevant document passage was found, so the local model answered using its general model knowledge.

The application can also report retrieval distance, threshold, fallback reason and source metadata. A retrieval match does not prove that a passage is correct, complete or medically appropriate.

## Basic test plan

| Test | Example/action | Expected behaviour |
|---|---|---|
| MCP connectivity | `python mcp_client_test.py` | MCP initializes and tools are discovered |
| UTCP discovery | Request `/utcp` | Three UTCP tools are listed |
| PDF ingestion | Upload a public/synthetic PDF | Chunks are indexed |
| Document-grounded question | Ask a question answered by an uploaded PDF | `healthcare_documents_rag` |
| General-knowledge fallback | Ask a question not covered by uploaded PDFs | `local_llm_general_knowledge` |
| Source display | Inspect retrieval details in Streamlit | Filename, page and distance shown when available |

Development tests previously exercised both paths: a question about precautions when sick returned a document-grounded answer, while “Explain diabetes in simple terms” triggered the local-LLM fallback. These observations are not a formal clinical or model-quality evaluation.

## Configuration and operational notes

- Ollama API: `http://localhost:11434`.
- UTCP server: `http://127.0.0.1:8000`.
- Chat model: `llama3.2:3b`.
- Embedding model: `all-minilm`.
- Relevance threshold: configured in `utcp_server.py`.
- The prototype stores background job state in process memory. Use one Uvicorn worker; multiple workers would not share the same in-memory job dictionary.
- Keep the UTCP server running while Streamlit uses UTCP.
- Do not expose development servers publicly without authentication, authorization, input validation, rate limiting and network protections.

## Git hygiene

Do not commit:

- ChromaDB persistence directories or generated vector files.
- Confidential, private or patient PDFs.
- Virtual environments, `__pycache__` or `.pyc` files.
- `.env` files, API keys, passwords or other secrets.
- Temporary logs or generated model files.

Review `.gitignore` before committing. Commit source code, safe sample configuration, dependency declarations and documentation.

## Current implementation status

Implemented and exercised locally:

- Ollama models and shared healthcare service.
- PDF ingestion and ChromaDB retrieval.
- MCP server and client test.
- UTCP manual and HTTP tools.
- RAG-first question answering.
- Relevance-based background LLM fallback.
- Streamlit integration file.

The standalone UTCP RAG and fallback paths and the integrated Streamlit UI have been tested locally for document-grounded answers and general-knowledge fallback. These tests do not certify every workflow or clinical answer.

## Future enhancements

- Add a LangGraph orchestrator and explicit workflow state.
- Introduce specialized agents for retrieval, summarization and safety review.
- Build an evaluation set for retrieval precision, fallback routing and answer quality.
- Add structured logs, job expiry/cleanup and improved error handling.
- Add automated tests and CI checks.
- Add security controls before shared or public deployment.

---

**Project:** Healthcare AI Learning Project  
**Repository:** https://github.com/karunaVibeCoder/AI-Apps  
**Scope:** Local educational prototype using public or synthetic healthcare knowledge documents.

