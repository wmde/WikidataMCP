"""Step 2: inspect and verify Wikidata graph structure."""

import asyncio
import json
import re

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

TOOL_NAMES = (
    "get_statements",
    "get_statement_values",
    "get_instance_and_subclass_hierarchy",
)
WIKIDATA_ID_PATTERN = re.compile(r"\b([PQ]\d+)\b")

SYSTEM_PROMPT = """\
You are verifying how Wikidata models the user's question.

Inspect candidate entities, relevant statement details, and class hierarchies.
Describe concrete graph relationships needed by SPARQL in concise plain text.
Select additional required QIDs or PIDs only when they visibly occur in tool results.
Stop inspecting once the required SPARQL graph pattern is verified.
Never invent identifiers.
"""


class StructureOutput(BaseModel):
    """Verified graph structure required by the query."""

    relationships: list[str] = Field(description="Verified entity-property-value relationships.")
    required_qids: list[str] = Field(description="Additional required QIDs visible in tool results.")
    required_pids: list[str] = Field(description="Additional required PIDs visible in tool results.")
    sparql_hints: str = Field(description="Grounded advice for constructing the query.")


async def run(state: dict, model_name: str, mcp_url: str) -> dict:
    """Inspect statements and hierarchy for the discovered identifiers."""
    print("\n=== Step 2: Inspect Structure ===", flush=True)
    prompt = (
        f"Question: {state['question']}\n"
        f"Candidate QIDs: {', '.join(state['relevant_qids']) or 'none'}\n"
        f"Candidate PIDs: {', '.join(state['relevant_pids']) or 'none'}"
    )
    client = MultiServerMCPClient(
        {
            "wikidata": {
                "transport": "streamable_http",
                "url": mcp_url
            }
        }
    )
    available_tools = {tool.name: tool for tool in await client.get_tools()}
    missing = set(TOOL_NAMES) - available_tools.keys()
    if missing:
        raise ValueError(f"MCP server is missing tools: {', '.join(sorted(missing))}")
    tools = {name: available_tools[name] for name in TOOL_NAMES}

    model = ChatOllama(model=model_name, temperature=0)
    model_with_tools = model.bind_tools(list(tools.values()))
    messages: list[BaseMessage] = [HumanMessage(content=prompt)]

    for _ in range(5):
        response = await model_with_tools.ainvoke([SystemMessage(content=SYSTEM_PROMPT), *messages])
        messages.append(response)
        if not response.tool_calls:
            break

        for call in response.tool_calls:
            name = call["name"]
            print(f"  [tool] {name}({call['args']})", flush=True)
            try:
                async with asyncio.timeout(35):
                    result = await tools[name].ainvoke(call["args"])
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
                    result = "\n".join(blocks) if blocks else json.dumps(result, ensure_ascii=False)
                elif not isinstance(result, str):
                    result = json.dumps(result, ensure_ascii=False)
                message = ToolMessage(content=result, name=name, tool_call_id=call["id"])
            except Exception as exc:
                message = ToolMessage(
                    content=f"Tool failed: {exc}",
                    name=name,
                    tool_call_id=call["id"],
                    status="error",
                )
            messages.append(message)

    structured_model = model.with_structured_output(StructureOutput)
    output = await structured_model.ainvoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *messages,
            HumanMessage(content="Return your findings in the required structured format."),
        ]
    )
    if not isinstance(output, StructureOutput):
        output = StructureOutput.model_validate(output)
    new_evidence = [
        f"{message.name}:\n{message.content}"
        for message in messages
        if isinstance(message, ToolMessage) and message.name in TOOL_NAMES and message.status != "error"
    ]
    all_evidence = [*state["evidence"], *new_evidence]
    allowed_ids = set(WIKIDATA_ID_PATTERN.findall("\n".join(all_evidence)))
    qids = list(
        dict.fromkeys(
            [
                *state["relevant_qids"],
                *(qid for qid in output.required_qids if qid in allowed_ids and qid.startswith("Q")),
            ]
        )
    )
    pids = list(
        dict.fromkeys(
            [
                *state["relevant_pids"],
                *(pid for pid in output.required_pids if pid in allowed_ids and pid.startswith("P")),
            ]
        )
    )

    print(f"Relationships: {output.relationships}")
    print(f"SPARQL hints: {output.sparql_hints}")
    return {
        **state,
        "relevant_qids": qids,
        "relevant_pids": pids,
        "evidence": all_evidence,
        "relationships": output.relationships,
        "sparql_hints": output.sparql_hints,
    }
