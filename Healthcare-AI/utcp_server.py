from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any
from uuid import uuid4

import requests
from fastapi import FastAPI, HTTPException, Query
from utcp.data.utcp_manual import UtcpManual
from utcp.python_specific_tooling.tool_decorator import utcp_tool
from utcp_http.http_call_template import HttpCallTemplate

from services.healthcare_service import (
    OLLAMA_URL,
    CHAT_MODEL,
    retrieve_context,
    get_knowledge_count,
)

# Lower cosine distance means a closer semantic match.
RELEVANCE_DISTANCE_THRESHOLD = 0.55

# LLM work runs outside the HTTP request so UTCP's 30-second HTTP
# timeout does not interrupt longer local model generations.
executor = ThreadPoolExecutor(max_workers=2)
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = Lock()

app = FastAPI(
    title="Healthcare UTCP Server",
    description="UTCP tools for healthcare document retrieval and local LLM fallback.",
    version="1.1.0",
)


def _run_local_llm(question: str) -> str:
    system_prompt = """
You are a general healthcare education assistant.

Provide general educational information in clear, simple language.
Do not claim that your answer comes from the user's indexed documents.
Do not diagnose, prescribe medicines, or provide individualized treatment.
State uncertainty and recommend a qualified healthcare professional for
personal medical decisions. For potentially life-threatening symptoms,
advise immediate emergency medical care.
If the question is unrelated to healthcare, answer it as general educational
information and do not present yourself as a medical professional.
"""
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=240,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def _finish_llm_job(job_id: str, question: str) -> None:
    try:
        answer = _run_local_llm(question)
        with jobs_lock:
            jobs[job_id].update(
                {
                    "status": "completed",
                    "answer": answer,
                    "answer_source": "local_llm_general_knowledge",
                    "document_grounded": False,
                }
            )
    except Exception as error:
        with jobs_lock:
            jobs[job_id].update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "Healthcare UTCP Server"}


@app.get("/utcp")
def utcp_discovery() -> Any:
    return UtcpManual.create_from_decorators(manual_version="1.1.0")


@utcp_tool(
    tool_call_template=HttpCallTemplate(
        name="search_healthcare_knowledge",
        url="http://127.0.0.1:8000/api/search_healthcare_knowledge",
        http_method="GET",
    ),
    description="Search indexed healthcare PDFs and return context, sources, distances and relevance.",
    tags=["healthcare", "rag", "documents", "search"],
)
@app.get("/api/search_healthcare_knowledge")
def search_healthcare_knowledge(
    question: str = Query(..., min_length=3),
    top_k: int = Query(4, ge=1, le=10),
) -> dict[str, Any]:
    try:
        if get_knowledge_count() == 0:
            return {
                "question": question,
                "context": "",
                "sources": [],
                "best_distance": None,
                "threshold": RELEVANCE_DISTANCE_THRESHOLD,
                "relevant": False,
                "reason": "No healthcare document chunks are indexed.",
            }

        context, sources = retrieve_context(question, top_k)
        distances = [item["distance"] for item in sources]
        best_distance = min(distances) if distances else None
        relevant = (
            best_distance is not None
            and best_distance <= RELEVANCE_DISTANCE_THRESHOLD
            and bool(context.strip())
        )

        return {
            "question": question,
            "context": context,
            "sources": sources,
            "best_distance": best_distance,
            "threshold": RELEVANCE_DISTANCE_THRESHOLD,
            "relevant": relevant,
            "reason": (
                "Relevant document context found."
                if relevant
                else "No retrieved passage met the relevance threshold."
            ),
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Healthcare knowledge search failed: {error}",
        ) from error


@utcp_tool(
    tool_call_template=HttpCallTemplate(
        name="start_local_llm_job",
        url="http://127.0.0.1:8000/api/start_local_llm_job",
        http_method="GET",
    ),
    description="Start a background Ollama LLM answer job and return its job ID immediately.",
    tags=["healthcare", "ollama", "local-llm", "fallback"],
)
@app.get("/api/start_local_llm_job")
def start_local_llm_job(
    question: str = Query(..., min_length=3),
) -> dict[str, str]:
    job_id = str(uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "question": question,
            "answer_source": "local_llm_general_knowledge",
            "document_grounded": False,
        }

    executor.submit(_finish_llm_job, job_id, question)
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Local LLM job started. Poll the job status endpoint.",
    }


@utcp_tool(
    tool_call_template=HttpCallTemplate(
        name="get_local_llm_job_status",
        url="http://127.0.0.1:8000/api/local_llm_job_status",
        http_method="GET",
    ),
    description="Check the status and retrieve the result of a background local LLM job.",
    tags=["healthcare", "ollama", "local-llm", "job-status"],
)
@app.get("/api/local_llm_job_status")
def get_local_llm_job_status(
    job_id: str = Query(..., min_length=10),
) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job ID.")
        return dict(job)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
