"""Step 3: generate SPARQL from verified structure."""

import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """\
You are generating a Wikidata SPARQL query from verified graph evidence.

Use only the QIDs and PIDs supplied in the prompt. Never invent identifiers.
Use wdt:P31/wdt:P279* only when supported by the verified relationships.
Use DISTINCT where multi-valued relationships could inflate results.
Include the Wikidata label service when labels help answer the question.
Do not add LIMIT merely to hide valid rows or simplify a multi-valued answer.
Return only one complete read-only SELECT, ASK, CONSTRUCT, or DESCRIBE query.
"""
WIKIDATA_ID_PATTERN = re.compile(r"\b([PQ]\d+)\b")


class GenerationOutput(BaseModel):
    """Generated Wikidata SPARQL query."""

    sparql: str = Field(description="One complete Wikidata SPARQL query.")


async def run(state: dict, model_name: str) -> dict:
    """Generate SPARQL and mechanically check its identifiers."""
    print("\n=== Step 3: Generate SPARQL ===", flush=True)
    relationships = "\n- ".join(state["relationships"]) or "none"
    prompt = (
        f"Question: {state['question']}\n\n"
        f"Verified relationships:\n- {relationships}\n\n"
        f"SPARQL hints: {state['sparql_hints'] or 'none'}\n"
        f"Allowed QIDs: {', '.join(state['relevant_qids']) or 'none'}\n"
        f"Allowed PIDs: {', '.join(state['relevant_pids']) or 'none'}"
    )
    model = ChatOllama(model=model_name, temperature=0).with_structured_output(GenerationOutput)
    output = await model.ainvoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
    if not isinstance(output, GenerationOutput):
        output = GenerationOutput.model_validate(output)
    unsupported = set(WIKIDATA_ID_PATTERN.findall(output.sparql)) - set(
        WIKIDATA_ID_PATTERN.findall("\n".join(state["evidence"]))
    )
    reason = ""
    if unsupported:
        reason = f"Generated query contains unsupported identifiers: {', '.join(sorted(unsupported))}"

    print(output.sparql)
    return {**state, "sparql": output.sparql, "validation_reason": reason}
