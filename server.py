import os
from typing_extensions import TypedDict, Optional

from fastmcp import FastMCP
from datapizza.core.clients import Client
from datapizza.core.embedder import BaseEmbedder
from datapizza.core.vectorstore import Vectorstore

from rag import init_embedder, init_rewriter, init_vectorstore, ingest, retrieve


mcp = FastMCP(name="RagServer")

rewriter: Client | None = None
embedder: BaseEmbedder | None = None
model_name: str | None = None
vectorstore: Vectorstore | None = None
collection_name: str | None = None


class RewrittenChunk(TypedDict):
    text: str
    source: Optional[str]


@mcp.tool
def ingest_tool(source: str) -> None:
    """Ingest documents from the specified source path.

    Args:
        source (str): Path or URL to the source documents.
    """

    global embedder, vectorstore, collection_name

    ingest(
        source=source,
        embedder_client=embedder,
        vectorstore=vectorstore,
        collection_name=collection_name,
    )


@mcp.tool
def retrieve_tool(query: str) -> list[RewrittenChunk]:
    """Retrieve relevant chunks based on the user query.

    Args:
        query (str): User query string.
    """

    global rewriter, embedder, vectorstore, collection_name, model_name

    res = retrieve(
        query=query,
        rewriter_client=rewriter,
        embedder_client=embedder,
        vectorstore=vectorstore,
        collection_name=collection_name,
        k=5,
        vector_name=model_name,
    )

    return [
        {"text": chunk.text, "source": chunk.metadata.get("source")} for chunk in res
    ]


if __name__ == "__main__":
    # Configuration
    rewriter_provider = os.getenv("REWRITER_PROVIDER") or os.getenv("PROVIDER") or ""
    embedder_provider = os.getenv("EMBEDDER_PROVIDER") or os.getenv("PROVIDER") or ""

    collection_name = os.getenv("COLLECTION_NAME", "my_documents")
    embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))

    # Initialize components
    rewriter = init_rewriter(rewriter_provider)
    embedder, model_name = init_embedder(embedder_provider)
    vectorstore = init_vectorstore(
        vectorstore_type="qdrant",
        location=":memory:",
        collection_name=collection_name,
        vector_name=model_name,
        embedding_dimensions=embedding_dimensions,
    )

    # Start the server
    mcp.run(transport="http", port=8000)
