"""Step 4: execute, validate, and refine generated SPARQL."""

import json
import re

from steps.agent import AgentFactory, AgentRunner


class ValidateSparqlStep:
    """Execute, validate, and refine generated SPARQL."""

    TOOL_NAMES = ("execute_sparql",)
    TOOL_CALL_LIMIT = 5
    SYSTEM_PROMPT = """\
    You are validating a Wikidata SPARQL query against its execution result.

    Accept it only when the result plausibly answers the user's question.
    When refinement is needed, return a complete corrected query using only allowed identifiers.
    Preserve all valid values for multi-valued relationships.
    Never add LIMIT merely to hide valid rows or make an answer appear simpler.
    Do not infer the meaning of an unlabeled identifier unless verified relationships support it.
    Never invent QIDs or PIDs.
    """

    def __init__(self, model_name: str, mcp_url: str, max_attempts: int = 3) -> None:
        """Configure the SPARQL-validation step."""
        self.agent_factory = AgentFactory(model_name=model_name, mcp_url=mcp_url)
        self.max_attempts = max_attempts

    async def run(self, state: dict) -> dict:
        """Execute and refine SPARQL until accepted or attempts are exhausted."""
        print("\n=== Step 4: Validate SPARQL ===", flush=True)
        runner = await self.agent_factory.create(
            system_prompt=self.SYSTEM_PROMPT,
            tool_names=self.TOOL_NAMES,
            tool_call_limit=self.TOOL_CALL_LIMIT,
        )

        query = state["sparql"]
        result_text = ""
        final_reason = state["validation_reason"]

        for attempt in range(1, self.max_attempts + 1):
            finalized = await self.finalize(state, {"query": query}, runner)
            output = finalized["output"]
            result_text = finalized["result_text"]
            issue = finalized["issue"]
            final_reason = output.reason

            if output.accepted and not issue:
                print(f"Accepted on attempt {attempt}: {output.reason}")
                break

            refined = output.refined_sparql.strip()
            if not refined or refined == query:
                print(f"Stopped on attempt {attempt}: {output.reason}")
                break
            if attempt == self.max_attempts:
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

    async def finalize(self, state: dict, result: dict, runner: AgentRunner) -> dict:
        """Execute one query and semantically verify its result."""
        query = result["query"]
        allowed_ids = set(WIKIDATA_ID_PATTERN.findall("\n".join(state["evidence"])))
        unsupported = set(WIKIDATA_ID_PATTERN.findall(query)) - allowed_ids
        issue = ""
        if unsupported:
            issue = f"Unsupported identifiers: {', '.join(sorted(unsupported))}"
        elif any(re.search(rf"\b{keyword}\b", query.upper()) for keyword in UPDATE_KEYWORDS):
            issue = "Only read-only SPARQL queries are allowed."

        result_text = issue
        if not issue:
            try:
                execution_result = await runner.tools["execute_sparql"].ainvoke({"sparql": query, "K": 10})
                if isinstance(execution_result, tuple) and execution_result:
                    execution_result = execution_result[0]
                if isinstance(execution_result, list):
                    blocks = [
                        block["text"]
                        for block in execution_result
                        if isinstance(block, dict)
                        and block.get("type") == "text"
                        and isinstance(block.get("text"), str)
                    ]
                    result_text = "\n".join(blocks) if blocks else json.dumps(execution_result, ensure_ascii=False)
                elif isinstance(execution_result, str):
                    result_text = execution_result
                else:
                    result_text = json.dumps(execution_result, ensure_ascii=False)
            except Exception as exc:
                issue = f"Execution failed: {exc}"
                result_text = issue
            if result_text.startswith(ERROR_PREFIXES):
                issue = result_text

        relationships = "\n- ".join(state["relationships"]) or "none"
        prompt = (
            f"Question: {state['question']}\n\n"
            f"SPARQL:\n{query}\n\n"
            f"Execution result:\n{result_text}\n\n"
            f"Verified relationships:\n- {relationships}\n\n"
            f"Mechanical issue: {issue or 'none'}\n"
            f"Allowed identifiers: {', '.join(allowed_ids) or 'none'}"
        )
        agent_result = await runner.invoke_agent({"messages": [{"role": "user", "content": prompt}]}, "validate sparql")
        output = agent_result["structured_response"]
        if not isinstance(output, ValidationOutput):
            output = ValidationOutput.model_validate(output)
        return {"output": output, "result_text": result_text, "issue": issue}
