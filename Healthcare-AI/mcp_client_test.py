
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    project_dir = Path(__file__).resolve().parent

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
        cwd=str(project_dir)
    )

    print("Connecting to Healthcare MCP server...", flush=True)

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:

            await session.initialize()
            print("MCP connection initialized successfully.", flush=True)

            tools = await session.list_tools()

            print("\nAvailable MCP tools:", flush=True)
            for tool in tools.tools:
                print("-", tool.name, flush=True)

            print("\nTesting knowledge count...", flush=True)

            count_result = await session.call_tool(
                "get_healthcare_knowledge_count",
                arguments={}
            )

            for item in count_result.content:
                if hasattr(item, "text"):
                    print(item.text, flush=True)

            print("\nTesting healthcare knowledge search...", flush=True)

            search_result = await session.call_tool(
                "search_healthcare_knowledge",
                arguments={
                    "question": "What precautions should I take when I am sick?",
                    "top_k": 4
                }
            )

            for item in search_result.content:
                if hasattr(item, "text"):
                    print(item.text, flush=True)

            print("\nMCP testing completed.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
