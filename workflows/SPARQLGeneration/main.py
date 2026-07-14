"""Run the SPARQL generation workflow."""

import argparse
import asyncio
from collections.abc import Sequence

from langgraph.graph import END, START, StateGraph
from steps.discover import DiscoverStep
from steps.generate_sparql import GenerateSparqlStep
from steps.inspect_structure import InspectStructureStep
from steps.validate_sparql import ValidateSparqlStep


async def run_workflow(question: str, model_name: str, mcp_url: str) -> dict:
    """Build and run the complete SPARQL generation workflow."""
    discover = DiscoverStep(model_name=model_name, mcp_url=mcp_url)
    inspect_structure = InspectStructureStep(model_name=model_name, mcp_url=mcp_url)
    generate_sparql = GenerateSparqlStep(model_name=model_name)
    validate_sparql = ValidateSparqlStep(model_name=model_name, mcp_url=mcp_url)

    builder = StateGraph(dict)
    builder.add_node("discover", discover.run)
    builder.add_node("inspect_structure", inspect_structure.run)
    builder.add_node("generate_sparql", generate_sparql.run)
    builder.add_node("validate_sparql", validate_sparql.run)

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
            "relevant_items": [],
            "relevant_properties": [],
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
