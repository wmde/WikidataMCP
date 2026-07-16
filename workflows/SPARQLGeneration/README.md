# SPARQL Generation Workflow

This workflow uses LangChain tool calling, LangGraph, Ollama, and the hosted
Wikidata MCP server to generate grounded SPARQL queries.

The workflow is designed for small local models that tend to assume too much.
Each agent receives a narrow prompt, a narrow tool set, and prose context from
earlier stages. Tool-using stages do not use structured output and the code does
not parse their prose summaries into trusted state. Each tool-using stage writes
its own findings note; later stages can use earlier notes as context, but they
are not responsible for rewriting the whole discovery.

## Workflow

1. `discover` searches only with `search_items` and `search_properties`.
   It selects candidate QIDs/PIDs and names examples plus counterexamples that
   should help later stages understand Wikidata structure.
2. `inspect_structure` writes three independent findings notes:
   item statements with `get_statements`, statement details with
   `get_statement_values`, then class hierarchy with
   `get_instance_and_subclass_hierarchy`. The code bundles Step 1-4 notes under
   stage headers for SPARQL generation without interpreting their contents.
3. `generate_sparql` receives all discovery findings and critique notes, but no
   tools. It uses structured output to return one read-only SPARQL query. Python
   validates the query and executes it against Wikidata.
4. `validate_sparql` writes three independent critique notes:
   result item statements, result statement details, and result hierarchy. The
   code bundles Step 6-8 notes for the next generation attempt without treating
   the hierarchy note as the whole critique.
5. LangGraph loops from `validate_sparql` back to `generate_sparql`, stopping
   when the query or critique stabilizes, or when the configured cycle limit is
   reached.

Shared code:

- `main.py` initializes the plain dictionary state and connects the four
  steps with LangGraph.
- `steps/agent.py` centralizes LangChain model/tool setup and streaming logs.
- `steps/workflow_utils.py` handles mechanical helpers: tool output text
  normalization, prompt compaction, read-only checks, and result formatting.

## Run

Install the dependencies and ensure Ollama is running with the selected model:

```bash
cd workflows/SPARQLGeneration
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py "Who are the presidents of France?"
```

Use a different model, MCP endpoint, or refinement limit:

```bash
python main.py "Who are the presidents of France?" \
  --model qwen2.5:14b \
  --mcp-url http://localhost:8000/mcp/ \
  --max-refinement-cycles 3
```

Use `python -m pip` rather than `pip` so dependencies are installed for the
same Python interpreter that runs the workflow. The isolated virtual environment
avoids conflicts with older LangChain packages installed elsewhere.
