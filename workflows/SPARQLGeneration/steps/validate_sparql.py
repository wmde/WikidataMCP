"""Step 4: execute, validate, and refine generated SPARQL."""

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """\
You are validating a Wikidata SPARQL query against its execution result.

Accept it only when the result plausibly answers the user's question.
When refinement is needed, return a complete corrected query using only allowed identifiers.
Preserve all valid values for multi-valued relationships.
Never add LIMIT merely to hide valid rows or make an answer appear simpler.
Do not infer the meaning of an unlabeled identifier unless verified relationships support it.
Never invent QIDs or PIDs.
"""

ERROR_PREFIXES = (
    "MCP error",
    "SPARQL query returned no data",
    "Unexpected server error",
    "Wikidata is currently unavailable",
)
UPDATE_KEYWORDS = ("INSERT", "DELETE", "LOAD", "CLEAR", "CREATE", "DROP", "MOVE", "COPY", "ADD")
WIKIDATA_ID_PATTERN = re.compile(r"\b([PQ]\d+)\b")


class ValidationOutput(BaseModel):
    """Semantic validation and optional query refinement."""

    accepted: bool = Field(description="Whether the query and result answer the question.")
    reason: str = Field(description="Brief validation explanation.")
    refined_sparql: str = Field(default="", description="Complete corrected query when refinement is needed.")


async def run(
    state: dict,
    model_name: str,
    mcp_url: str,
    max_attempts: int = 3,
) -> dict:
    """Execute and refine SPARQL while enforcing evidence-grounded identifiers."""
    print("\n=== Step 4: Validate SPARQL ===", flush=True)
    client = MultiServerMCPClient(
        {
            "wikidata": {
                "transport": "streamable_http",
                "url": mcp_url
            }
        }
    )
    tools = {tool.name: tool for tool in await client.get_tools()}
    if "execute_sparql" not in tools:
        raise ValueError("MCP server is missing tool: execute_sparql")
    model = ChatOllama(model=model_name, temperature=0).with_structured_output(ValidationOutput)
    query = state["sparql"]
    result_text = ""
    final_reason = state["validation_reason"]
    allowed_ids = set(WIKIDATA_ID_PATTERN.findall("\n".join(state["evidence"])))
    relationships = "\n- ".join(state["relationships"]) or "none"

    for attempt in range(1, max_attempts + 1):
        unsupported = set(WIKIDATA_ID_PATTERN.findall(query)) - allowed_ids
        issue = ""
        if unsupported:
            issue = f"Unsupported identifiers: {', '.join(sorted(unsupported))}"
        elif any(re.search(rf"\b{keyword}\b", query.upper()) for keyword in UPDATE_KEYWORDS):
            issue = "Only read-only SPARQL queries are allowed."
        if issue:
            result_text = issue
        else:
            try:
                result = await tools["execute_sparql"].ainvoke({"sparql": query, "K": 10})
                if isinstance(result, tuple) and result:
                    result = result[0]
                if isinstance(result, list):
                    blocks = [
                        block["text"]
                        for block in result
                        if isinstance(block, dict)
                        and block.get("type") == "text"
                        and isinstance(block.get("text"), str)
                    ]
                    result_text = "\n".join(blocks) if blocks else json.dumps(result, ensure_ascii=False)
                elif isinstance(result, str):
                    result_text = result
                else:
                    result_text = json.dumps(result, ensure_ascii=False)
            except Exception as exc:
                issue = f"Execution failed: {exc}"
                result_text = issue
            if result_text.startswith(ERROR_PREFIXES):
                issue = result_text

        prompt = (
            f"Question: {state['question']}\n\n"
            f"SPARQL:\n{query}\n\n"
            f"Execution result:\n{result_text}\n\n"
            f"Verified relationships:\n- {relationships}\n\n"
            f"Mechanical issue: {issue or 'none'}\n"
            f"Allowed identifiers: {', '.join(allowed_ids) or 'none'}"
        )
        output = await model.ainvoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
        if not isinstance(output, ValidationOutput):
            output = ValidationOutput.model_validate(output)
        final_reason = output.reason

        if output.accepted and not issue:
            print(f"Accepted on attempt {attempt}: {output.reason}")
            break

        refined = output.refined_sparql.strip()
        if not refined or refined == query:
            print(f"Stopped on attempt {attempt}: {output.reason}")
            break
        if attempt == max_attempts:
            final_reason = f"{output.reason} Refinement was not executed because the attempt limit was reached."
            break

        query = refined
        print(f"Refining after attempt {attempt}: {output.reason}")

    return {
        **state,
        "sparql": query,
        "result": result_text,
        "validation_reason": final_reason,
    }
