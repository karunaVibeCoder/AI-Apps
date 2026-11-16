import asyncio
import json
from typing import Any

from utcp.utcp_client import UtcpClient

from services.healthcare_service import generate_answer

UTCP_MANUAL_URL = "http://127.0.0.1:8000/utcp"
MANUAL_NAME = "healthcare"
POLL_INTERVAL_SECONDS = 2
MAX_LLM_WAIT_SECONDS = 270


def to_dict(result: Any) -> dict:
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    if hasattr(result, "model_dump"):
        value = result.model_dump()
        if isinstance(value, dict):
            return value
    raise TypeError(
        f"Unexpected UTCP tool result type: {type(result).__name__}: {result}"
    )


async def answer_question(question: str, top_k: int = 4) -> dict:
    client = await UtcpClient.create(
        config={
            "manual_call_templates": [
                {
                    "name": MANUAL_NAME,
                    "call_template_type": "http",
                    "url": UTCP_MANUAL_URL,
                    "http_method": "GET",
                }
            ]
        }
    )

    try:
        search_result = await client.call_tool(
            f"{MANUAL_NAME}.search_healthcare_knowledge",
            {"question": question, "top_k": top_k},
        )
        search_data = to_dict(search_result)

        if search_data.get("relevant") is True and search_data.get("context"):
            answer = generate_answer(question, search_data["context"])
            return {
                "question": question,
                "answer": answer,
                "answer_source": "healthcare_documents_rag",
                "document_grounded": True,
                "best_distance": search_data.get("best_distance"),
                "threshold": search_data.get("threshold"),
                "sources": search_data.get("sources", []),
            }

        # The start tool returns quickly; long model generation happens server-side.
        start_result = await client.call_tool(
            f"{MANUAL_NAME}.start_local_llm_job",
            {"question": question},
        )
        start_data = to_dict(start_result)
        job_id = start_data.get("job_id")
        if not job_id:
            raise RuntimeError(f"LLM job response did not contain job_id: {start_data}")

        elapsed = 0
        while elapsed < MAX_LLM_WAIT_SECONDS:
            status_result = await client.call_tool(
                f"{MANUAL_NAME}.get_local_llm_job_status",
                {"job_id": job_id},
            )
            status_data = to_dict(status_result)
            status = status_data.get("status")

            if status == "completed":
                return {
                    "question": question,
                    "answer": status_data.get("answer", ""),
                    "answer_source": "local_llm_general_knowledge",
                    "document_grounded": False,
                    "fallback_reason": search_data.get(
                        "reason", "No relevant document context was found."
                    ),
                    "best_distance": search_data.get("best_distance"),
                    "threshold": search_data.get("threshold"),
                    "sources": search_data.get("sources", []),
                    "model": "llama3.2:3b",
                }

            if status == "failed":
                raise RuntimeError(status_data.get("error", "Local LLM job failed."))

            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            elapsed += POLL_INTERVAL_SECONDS

        raise TimeoutError(
            f"Local LLM job did not finish within {MAX_LLM_WAIT_SECONDS} seconds."
        )
    finally:
        await client.close()


async def main() -> None:
    print("Healthcare UTCP client with background LLM fallback")
    print(f"Discovery manual: {UTCP_MANUAL_URL}")
    print("Enter a question, or type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if len(question) < 3:
            print("Please enter a question with at least 3 characters.\n")
            continue

        try:
            result = await answer_question(question)
            print("\nAnswer:")
            print(result["answer"])
            print("\nAnswer source:", result["answer_source"])
            print("Document grounded:", result["document_grounded"])

            if result.get("best_distance") is not None:
                print("Best retrieval distance:", result["best_distance"])
                print("Relevance threshold:", result.get("threshold"))

            if result.get("fallback_reason"):
                print("Fallback reason:", result["fallback_reason"])

            if result.get("sources"):
                print("Retrieved sources:")
                for source in result["sources"]:
                    print(
                        f"- {source.get('source')} | "
                        f"Page {source.get('page')} | "
                        f"Distance {source.get('distance')}"
                    )
            print("\n" + "-" * 70 + "\n")
        except Exception as error:
            print(f"\nUTCP request failed: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    asyncio.run(main())
