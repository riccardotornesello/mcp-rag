import os
import sys

from datapizza.clients.google import GoogleClient
from datapizza.core.clients import Client
from datapizza.core.embedder import BaseEmbedder
from datapizza.core.vectorstore import VectorConfig, Vectorstore
from datapizza.embedders import ChunkEmbedder
from datapizza.embedders.google import GoogleEmbedder
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.parsers.text_parser import TextParser
from datapizza.modules.rewriters import ToolRewriter
from datapizza.modules.splitters import NodeSplitter
from datapizza.pipeline import DagPipeline, IngestionPipeline
from datapizza.type import Chunk
from datapizza.vectorstores.qdrant import QdrantVectorstore

from modules.loaders.text_loader import TextLoader
from modules.rewriters.dummy_rewriter import DummyRewriter


def init_rewriter(
    provider: str, api_key: str | None = None, model_name: str | None = None
) -> Client | None:
    api_key = api_key or os.getenv("REWRITER_API_KEY") or os.getenv("API_KEY")
    model_name = model_name or os.getenv("REWRITER_MODEL_NAME")

    match provider:
        case "":
            return None

        case "GOOGLE":
            if not api_key:
                raise ValueError("API key for rewriter is not set.")

            return GoogleClient(
                api_key=api_key,
                model=model_name or "gemini-2.0-flash",
            )

        case _:
            raise ValueError(f"Unsupported provider: {provider}")


def init_embedder(
    provider: str, api_key: str | None = None, model_name: str | None = None
) -> [BaseEmbedder, str]:
    api_key = api_key or os.getenv("EMBEDDER_API_KEY") or os.getenv("API_KEY")
    model_name = model_name or os.getenv("EMBEDDER_MODEL_NAME")

    match provider:
        case "GOOGLE":
            if not api_key:
                raise ValueError("API key for rewriter is not set.")

            model_name = model_name or "gemini-embedding-001"

            return (
                GoogleEmbedder(
                    api_key=api_key,
                    model_name=model_name,
                ),
                model_name,
            )

        case _:
            raise ValueError(f"Unsupported provider: {provider}")


def init_vectorstore(
    vectorstore_type: str,
    location: str,
    collection_name: str,
    vector_name: str,
    embedding_dimensions: int,
) -> Vectorstore:
    match vectorstore_type:
        case "qdrant":
            vectorstore = QdrantVectorstore(location=location)
            vectorstore.create_collection(
                collection_name,
                vector_config=[
                    VectorConfig(name=vector_name, dimensions=embedding_dimensions)
                ],
            )
            return vectorstore

        case _:
            raise ValueError(f"Unsupported vectorstore: {vectorstore}")


def ingest(
    source: str,
    source_type: str,
    embedder_client: BaseEmbedder,
    vectorstore: Vectorstore,
    collection_name: str,
    max_char: int = 1000,
    metadata: dict | None = None,
):
    modules = []

    match source_type:
        case "raw":
            modules.extend([TextParser()])
        case "text":
            modules.extend([TextLoader(), TextParser()])
        case "web":
            # TODO: web loader
            raise NotImplementedError("Web loader not implemented yet.")
        case _:
            modules.extend([DoclingParser()])

    modules.extend(
        [
            # TODO: media captioner
            NodeSplitter(max_char=max_char),
            ChunkEmbedder(client=embedder_client),
        ]
    )

    ingestion_pipeline = IngestionPipeline(
        modules=modules,
        vector_store=vectorstore,
        collection_name=collection_name,
    )

    ingestion_pipeline.run(source, metadata={**(metadata or {}), "source": source})


def retrieve(
    query: str,
    rewriter_client: Client | None,
    embedder_client: BaseEmbedder,
    vectorstore: Vectorstore,
    collection_name: str,
    k: int = 5,
    vector_name: str | None = None,
) -> list[Chunk]:
    dag_pipeline = DagPipeline()

    if rewriter_client:
        query_rewriter = ToolRewriter(
            client=rewriter_client,
            system_prompt="Rewrite user queries to improve retrieval accuracy.",
        )
    else:
        query_rewriter = DummyRewriter()

    dag_pipeline.add_module("rewriter", query_rewriter)
    dag_pipeline.add_module("embedder", embedder_client)
    dag_pipeline.add_module("retriever", vectorstore)

    dag_pipeline.connect("rewriter", "embedder", target_key="text")
    dag_pipeline.connect("embedder", "retriever", target_key="query_vector")

    result = dag_pipeline.run(
        {
            "rewriter": {"user_prompt": query},
            "prompt": {"user_prompt": query},
            "retriever": {
                "collection_name": collection_name,
                "k": k,
                "vector_name": vector_name,
            },
        }
    )

    return result["retriever"]


if __name__ == "__main__":
    # Configuration
    rewriter_provider = os.getenv("REWRITER_PROVIDER") or os.getenv("PROVIDER") or ""
    embedder_provider = os.getenv("EMBEDDER_PROVIDER") or os.getenv("PROVIDER") or ""

    collection_name = os.getenv("COLLECTION_NAME", "my_documents")
    embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))

    file_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "https://raw.githubusercontent.com/datapizza-labs/datapizza-ai/refs/heads/main/README.md"
    )
    file_type = sys.argv[2] if len(sys.argv) > 2 else ""
    query = sys.argv[3] if len(sys.argv) > 3 else "What is Datapizza?"

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

    # Ingest data
    ingest(
        source=file_path,
        source_type=file_type,
        embedder_client=embedder,
        vectorstore=vectorstore,
        collection_name=collection_name,
    )

    print("Ingestion completed...")

    # Retrieve data
    res = retrieve(
        query=query,
        rewriter_client=rewriter,
        embedder_client=embedder,
        vectorstore=vectorstore,
        collection_name=collection_name,
        k=5,
        vector_name=model_name,
    )
    print(
        [{"text": chunk.text, "source": chunk.metadata.get("source")} for chunk in res]
    )
