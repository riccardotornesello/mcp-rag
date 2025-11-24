import asyncio
from fastmcp import Client, FastMCP

# HTTP server
client = Client("http://localhost:8000/mcp")


async def main():
    async with client:
        # Basic server interaction
        await client.ping()

        # List available operations
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()

        print("Available tools:", tools)
        print("Available resources:", resources)
        print("Available prompts:", prompts)

        # Execute operations
        result = await client.call_tool(
            "ingest_tool",
            {
                "source": "https://raw.githubusercontent.com/datapizza-labs/datapizza-ai/refs/heads/main/README.md"
            },
        )
        print(result)

        result = await client.call_tool(
            "retrieve_tool", {"query": "What is Datapizza?"}
        )
        print(result)


asyncio.run(main())
