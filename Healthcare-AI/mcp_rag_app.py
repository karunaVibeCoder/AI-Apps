import asyncio
from pathlib import Path
import sys

import streamlit as st

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from services.healthcare_service import ingest_pdf
from utcp_client import answer_question


PROJECT_DIR = Path(__file__).resolve().parent
MCP_SERVER_FILE = PROJECT_DIR / "mcp_server.py"


async def call_mcp_tool(tool_name, arguments=None):
    """Start the local MCP server, call one tool, and close the session."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER_FILE)],
        cwd=str(PROJECT_DIR),
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(
                tool_name,
                arguments=arguments or {},
            )
            text_parts = [
                item.text for item in result.content if hasattr(item, "text")
            ]
            if not text_parts:
                raise RuntimeError(
                    f"MCP tool '{tool_name}' returned no text result."
                )
            if getattr(result, "isError", False):
                raise RuntimeError("\n".join(text_parts))
            return "\n".join(text_parts)


def run_mcp_tool(tool_name, arguments=None):
    """Run the async MCP client from Streamlit's synchronous script."""
    return asyncio.run(call_mcp_tool(tool_name, arguments))


def run_utcp_question(question, top_k=4):
    """Run the tested UTCP RAG-first, local-LLM-fallback workflow."""
    return asyncio.run(answer_question(question, top_k=top_k))


st.set_page_config(
    page_title="Healthcare AI Assistant",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Healthcare Knowledge Assistant")
st.caption("MCP + UTCP | Local Ollama + ChromaDB + RAG")

st.warning(
    "Educational prototype only. Use public or synthetic documents. "
    "Do not upload confidential patient records. This tool does not "
    "provide a diagnosis or replace qualified medical advice."
)

st.subheader("1. Upload healthcare knowledge documents")
uploaded_files = st.file_uploader(
    "Upload healthcare PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.button("Process documents", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        for uploaded_file in uploaded_files:
            try:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    chunk_count = ingest_pdf(uploaded_file)
                st.success(
                    f"{uploaded_file.name}: {chunk_count} chunks processed."
                )
            except Exception as error:
                st.error(f"Could not process {uploaded_file.name}: {error}")

st.divider()
st.subheader("2. Knowledge database status")

if st.button("Check indexed document count"):
    try:
        with st.spinner("Checking through the MCP server..."):
            count_message = run_mcp_tool("get_healthcare_knowledge_count")
        st.info(count_message)
    except Exception as error:
        st.error(f"MCP connection or count failed: {error}")

st.divider()
st.subheader("3. Ask a healthcare knowledge question")

question = st.text_input(
    "Enter your question",
    placeholder="Example: What precautions should I take when I am sick?",
)

top_k = st.slider(
    "Number of document passages to retrieve",
    min_value=1,
    max_value=8,
    value=4,
)

if st.button("Ask Healthcare Assistant", type="primary"):
    if len(question.strip()) < 3:
        st.warning("Please enter a question with at least 3 characters.")
    else:
        try:
            with st.spinner(
                "Searching healthcare documents. If no relevant passage is "
                "found, UTCP will call the local Llama model..."
            ):
                result = run_utcp_question(question.strip(), top_k=top_k)

            st.subheader("AI Answer")

            answer_source = result.get("answer_source", "unknown")
            if answer_source == "healthcare_documents_rag":
                st.success("Source: Your indexed healthcare documents")
                st.caption("This answer was generated using retrieved document context.")
            elif answer_source == "local_llm_general_knowledge":
                st.info("Source: Local Llama model — general knowledge")
                st.caption(
                    "No sufficiently relevant passage was found in your "
                    "indexed PDFs. This answer is not grounded in your documents."
                )
            else:
                st.warning(f"Answer source: {answer_source}")

            st.write(result.get("answer", "No answer was returned."))

            with st.expander("Retrieval details and sources"):
                st.write(
                    f"**Document grounded:** "
                    f"{result.get('document_grounded', False)}"
                )

                best_distance = result.get("best_distance")
                threshold = result.get("threshold")
                if best_distance is not None:
                    st.write(f"**Best retrieval distance:** {best_distance}")
                if threshold is not None:
                    st.write(f"**Relevance threshold:** {threshold}")

                fallback_reason = result.get("fallback_reason")
                if fallback_reason:
                    st.write(f"**Fallback reason:** {fallback_reason}")

                sources = result.get("sources") or []
                if sources:
                    st.write("**Retrieved passages:**")
                    for source in sources:
                        if isinstance(source, dict):
                            source_name = source.get("source", "Unknown source")
                            page = source.get("page", "Unknown")
                            distance = source.get("distance", "Unknown")
                            st.write(
                                f"- {source_name} | Page {page} | "
                                f"Distance {distance}"
                            )
                        else:
                            st.write(f"- {source}")
                else:
                    st.write("No document sources were returned.")

                model_name = result.get("model")
                if model_name:
                    st.write(f"**Fallback model:** {model_name}")

        except Exception as error:
            st.error(
                f"Healthcare assistant error: {type(error).__name__}: {error}"
            )
            st.info(
                "Check that the UTCP server is running at "
                "http://127.0.0.1:8000 and that Ollama is running."
            )

st.divider()
st.caption(
    "Healthcare AI Learning Project | MCP + UTCP + Local LLM + RAG + ChromaDB"
)
