# MCP-RAG: Model Context Protocol (MCP) Server

This repository implements an **MCP server** for Retrieval-Augmented Generation (RAG) workflows. The server exposes a modular framework for loading, parsing, rewriting, and serving data, making it easy to build and extend RAG pipelines.

## What is an MCP Server?

The **Model Context Protocol (MCP) server** is a standard interface for serving context and tools to language models. It allows clients to query for information, retrieve documents, and use custom tools in a unified way. This repo provides a Python implementation of an MCP server, designed for extensibility and experimentation.

## How to Use the MCP Server

### 1. Install Requirements

Make sure you have Python 3.8+ installed. Then, install dependencies using [uv](https://github.com/astral-sh/uv):

```bash
uv pip install -r requirements.txt  # or uv pip install .
```

### 2. Start the MCP Server

Run the server with:

```bash
python server.py
```

This will start the MCP server, which listens for incoming requests and serves context and tool results.

### 3. Interact with the Server

You can interact with the server using HTTP requests, a client library, or by integrating it with an LLM agent that supports MCP. The server exposes endpoints for querying documents, running tools, and more.

## Tools Overview

The MCP server provides several modular tools, each in its own directory:

### Loaders (`modules/loaders/`)

- **text_loader.py**: Loads text data from files or other sources into the system.

### Parsers (`modules/parsers/`)

- **web_base.py**: Parses and extracts information from web-based sources.

### Rewriters (`modules/rewriters/`)

- **dummy_rewriter.py**: Example rewriter that demonstrates how to modify or augment loaded data.

## Extending the Server

To add new tools, simply create new Python modules in the appropriate directory (`loaders`, `parsers`, or `rewriters`). Each tool should follow the interface conventions used in the existing modules.

## Testing

Run the test suite with:

```bash
python test.py
```

## License

See [LICENSE](LICENSE) for details.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
