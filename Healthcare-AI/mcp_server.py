from mcp.server import MCPServer

from services.healthcare_service import (
    retrieve_context,
    get_knowledge_count
)

mcp = MCPServer("Healthcare Knowledge Server")


@mcp.tool()
def search_healthcare_knowledge(question: str, top_k: int = 4) -> str:
    """Search existing healthcare knowledge documents."""
    try:
        context, sources = retrieve_context(question, top_k)

        return (
            f"Retrieved healthcare context:\n\n{context}\n\n"
            f"Sources:\n{sources}"
        )
    except Exception as error:
        return f"Knowledge search failed: {error}"


@mcp.tool()
def get_healthcare_knowledge_count() -> str:
    """Return the number of indexed healthcare document chunks."""
    try:
        return (
            f"Indexed healthcare document chunks: "
            f"{get_knowledge_count()}"
        )
    except Exception as error:
        return f"Knowledge count failed: {error}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
