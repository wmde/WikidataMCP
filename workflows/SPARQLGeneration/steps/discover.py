"""Step 1: discover relevant Wikidata entities and properties."""

import asyncio
import json
import re

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

TOOL_NAMES = ("search_items", "search_properties")
WIKIDATA_ID_PATTERN = re.compile(r"\b([PQ]\d+)\b")

SYSTEM_PROMPT = """\
You are discovering Wikidata identifiers needed to answer a question.

Use search_items and search_properties with varied natural-language searches.
Select only QIDs and PIDs that visibly occur in the tool results.
Stop searching once you have enough identifiers to answer the question.
Do not use memorized or invented identifiers.
"""


class DiscoveryOutput(BaseModel):
    """Relevant identifiers selected from search results."""

    relevant_qids: list[str] = Field(description="Relevant QIDs visible in search results.")
    relevant_pids: list[str] = Field(description="Relevant PIDs visible in search results.")
    reasoning: str = Field(description="Brief explanation of why these identifiers matter.")


async def run(state: dict, model_name: str, mcp_url: str) -> dict:
    """Discover and verify candidate QIDs and PIDs."""
    print("\n=== Step 1: Discover ===", flush=True)
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
    messages: list[BaseMessage] = [HumanMessage(content=f"Question: {state['question']}")]

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

    structured_model = model.with_structured_output(DiscoveryOutput)
    output = await structured_model.ainvoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *messages,
            HumanMessage(content="Return your findings in the required structured format."),
        ]
    )
    if not isinstance(output, DiscoveryOutput):
        output = DiscoveryOutput.model_validate(output)
    evidence = [
        f"{message.name}:\n{message.content}"
        for message in messages
        if isinstance(message, ToolMessage) and message.name in TOOL_NAMES and message.status != "error"
    ]
    allowed_ids = set(WIKIDATA_ID_PATTERN.findall("\n".join(evidence)))
    qids = list(dict.fromkeys(qid for qid in output.relevant_qids if qid in allowed_ids and qid.startswith("Q")))
    pids = list(dict.fromkeys(pid for pid in output.relevant_pids if pid in allowed_ids and pid.startswith("P")))

    print(f"QIDs: {qids}")
    print(f"PIDs: {pids}")
    print(f"Reasoning: {output.reasoning}")
    return {
        **state,
        "relevant_qids": qids,
        "relevant_pids": pids,
        "evidence": evidence,
    }
