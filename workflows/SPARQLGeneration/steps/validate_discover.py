"""Step 4: execute, validate, and refine generated SPARQL."""

from pydantic import BaseModel, Field

from steps.agent import AgentFactory


class ValidationOutput(BaseModel):
    """Semantic validation and optional query refinement."""

    sufficient: bool = Field(
        description="Whether the discovery summary has sufficient information for writing the SPARQL query."
    )
    reason: str = Field(
        description="In case the discovery is insufficient, a brief exaplanation of what is missing."
    )


class ValidateDiscoveryStep:
    """Execute, validate, and refine generated SPARQL."""

    TOOL_NAMES = (
        "get_statements",
        "get_statement_values",
        "get_instance_and_subclass_hierarchy"
    )
    OUTPUT_MODEL = ValidationOutput
    SYSTEM_PROMPT = """\
    You verify whether the discovered Wikidata evidence is sufficient for generating a SPARQL query that answers the user question.
    You are not the query-writing or answer-producing step.

    The discovery summary is sufficient only when it gives a later SPARQL agent enough grounded information to write the query without Wikidata access.

    Check that the summary contains:
    - Useful entities: QIDs needed as fixed values, anchors, or domain concepts in the query.
    - Useful relationships: PIDs needed as graph edges or filters.
    - Useful classes: QIDs from instance-of/subclass-of evidence that help restrict or filter results.
    - Example items: concrete Wikidata items that show how this domain is modeled for this question.

    If the summary is insufficient, explain what specific information is missing and what should be searched or inspected next.
    Do not write SPARQL, answer the original question, or ask the user follow-up questions.
    """  # noqa: E501

    def __init__(self, model_name: str, mcp_url: str) -> None:
        """Configure the discovery-validation step."""
        self.agent_factory = AgentFactory(model_name=model_name, mcp_url=mcp_url)

    async def run(self, state: dict) -> dict:
        """Execute and refine SPARQL until accepted or attempts are exhausted."""
        print("\n=== Step 2: Validate Discovery ===", flush=True)
        runner = await self.agent_factory.create(
            system_prompt=self.SYSTEM_PROMPT,
            output_model=ValidationOutput,
            tool_names=self.TOOL_NAMES,
        )

        result = await runner.invoke_agent(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Question: {state['question']}\n\n"
                            f"Discovery Summary: {state['discovery_summary']}."
                        ),
                    }
                ]
            }
        )

        output = result.get("structured_response")
        if not isinstance(output, ValidationOutput):
            output = ValidationOutput.model_validate(output)

        return {
            **state,
            "sufficient": output.sufficient,
            "reason": output.reason
        }
