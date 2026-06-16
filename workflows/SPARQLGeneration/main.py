"""Run the SPARQL generation workflow."""

import argparse
import asyncio
from collections.abc import Sequence
from functools import partial

from langgraph.graph import END, START, StateGraph
from steps import discover, generate_sparql, inspect_structure, validate_sparql


async def run_workflow(question: str, model_name: str, mcp_url: str) -> dict:
    """Build and run the complete SPARQL generation workflow."""
    builder = StateGraph(dict)
    builder.add_node("discover", partial(discover.run, model_name=model_name, mcp_url=mcp_url))
    builder.add_node(
        "inspect_structure",
        partial(inspect_structure.run, model_name=model_name, mcp_url=mcp_url),
    )
    builder.add_node("generate_sparql", partial(generate_sparql.run, model_name=model_name))
    builder.add_node(
        "validate_sparql",
        partial(validate_sparql.run, model_name=model_name, mcp_url=mcp_url),
    )

    builder.add_edge(START, "discover")
    builder.add_edge("discover", "inspect_structure")
    builder.add_edge("inspect_structure", "generate_sparql")
    builder.add_edge("generate_sparql", "validate_sparql")
    builder.add_edge("validate_sparql", END)

    return await builder.compile().ainvoke(
        {
            "question": question,
            "relevant_qids": [],
            "relevant_pids": [],
            "evidence": [],
            "relationships": [],
            "sparql_hints": "",
            "sparql": "",
            "result": "",
            "validation_reason": "",
        }
    )


async def main(argv: Sequence[str] | None = None) -> None:
    """Run the workflow and print its final result."""
    parser = argparse.ArgumentParser(description="Generate grounded Wikidata SPARQL.")
    parser.add_argument("question", help="Natural-language Wikidata question.")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model name.")
    parser.add_argument(
        "--mcp-url",
        default="https://wd-mcp.wmcloud.org/mcp/",
        help="Wikidata MCP streamable HTTP endpoint.",
    )
    args = parser.parse_args(argv)

    state = await run_workflow(args.question, model_name=args.model, mcp_url=args.mcp_url)

    print("\n=== Final SPARQL ===")
    print(state["sparql"])
    print("\n=== Validation ===")
    print(state["validation_reason"])
    print("\n=== Result ===")
    print(state["result"])


if __name__ == "__main__":
    asyncio.run(main())
