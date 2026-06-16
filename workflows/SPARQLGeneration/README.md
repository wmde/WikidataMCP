# SPARQL Generation Workflow

This workflow uses LangChain tool calling and structured output, LangGraph,
Ollama, and the hosted Wikidata MCP server to generate grounded SPARQL queries.

Each workflow step owns its prompt, structured output schema, allowed tools, and
runtime logic:

1. `discover` finds candidate QIDs and PIDs.
2. `inspect_structure` verifies graph relationships.
3. `generate_sparql` creates a query from verified identifiers.
4. `validate_sparql` executes and refines the query.

Shared code stays deliberately small:

- `main.py` initializes the plain dictionary state and connects the four
  steps with LangGraph.

## Run

Install the dependencies and ensure Ollama is running with the selected model:

```bash
cd workflows/SPARQLGeneration
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py "Who are the presidents of France?"
```

Use a different model or MCP endpoint:

```bash
python main.py "Who are the presidents of France?" \
  --model qwen2.5:14b \
  --mcp-url http://localhost:8000/mcp/
```

Use `python -m pip` rather than `pip` so dependencies are installed for the
same Python interpreter that runs the workflow. The isolated virtual environment
avoids conflicts with older LangChain packages installed elsewhere.
