"""Step 3: generate SPARQL from verified structure."""

import os

import requests
from pydantic import BaseModel, Field

from steps.agent import AgentStep

WD_QUERY_URI = os.environ.get("WD_QUERY_URI", "https://query.wikidata.org/sparql")
USER_AGENT = os.environ.get("USER_AGENT", "Wikidata MCP SPARQL Generation (embedding@wikimedia.de)")

class SPARQLOutput(BaseModel):
    """Generated Wikidata SPARQL query."""

    sparql: str = Field(description="One complete Wikidata SPARQL query.")


class GenerateSparqlStep(AgentStep):
    """Generate and verify a grounded SPARQL query."""

    TOOL_NAMES = ("execute_sparql",)
    TOOL_CALL_LIMIT = 5
    OUTPUT_MODEL = SPARQLOutput
    SYSTEM_PROMPT = """\
    You generate a Wikidata SPARQL query that answers the user question using the provided discovery summary.

    Never invent QIDs or PIDs; use identifiers only when supported by the discovery summary.
    Do not use workarounds, approximations, or shortcuts; the query must fully translate the user's question into Wikidata graph patterns.
    Return only one complete SPARQL query.
    """  # noqa: E501

    async def run(self, state: dict) -> dict:
        """Generate SPARQL and return its mechanically verified state."""
        print("\n=== Step 3: Generate SPARQL ===", flush=True)
        messages = await self.run_with_tool(state)
        state = await self.run_sparql_generation(state, messages)
        return state

    async def run_with_tool(self, state: dict) -> dict:
        """Run the SPARQL generation step with tools."""
        print("\n=== Step 3.1: Test with MCP ===", flush=True)
        self.agent = None
        self.OUTPUT_MODEL = None
        self.TOOL_NAMES = ("execute_sparql",)
        self.TOOL_CALL_LIMIT = 5
        await self.setup()

        result = await self.invoke_agent(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Question: {state['question']}\n\n"
                            f"Discovery Summary: {state['discovery_summary']}"
                            "Generate and execute SPARQL queries to answer the question."
                        ),
                    }
                ]
            }
        )

        return result['messages']

    async def run_sparql_generation(self, state: dict, messages: list) -> dict:
        """Run the SPARQL generation step without tools."""
        print("\n=== Step 3.2: Generate SPARQL ===", flush=True)
        self.agent = None
        self.OUTPUT_MODEL = SPARQLOutput
        self.TOOL_NAMES = ()
        await self.setup()

        sparql = ""
        sparql_results = None
        content = (
            f"Question: {state['question']}\n\n"
            f"Discovery Summary: {state['discovery_summary']}"
        )

        for _ in range(5):
            result = await self.invoke_agent(
                {
                    "messages": [
                        *messages,
                        {
                            "role": "user",
                            "content": content,
                        }
                    ]
                }
            )
            messages = result['messages']

            output = result.get("structured_response")
            if not isinstance(output, SPARQLOutput):
                output = SPARQLOutput.model_validate(output)

            new_sparql = output.sparql.strip()
            if new_sparql == sparql:
                break
            sparql = new_sparql

            try:
                sparql_results = self._execute_sparql(sparql)
            except ValueError as e:
                content = (
                    f"Question: {state['question']}\n\n"
                    f"Discovery Summary: {state['discovery_summary']}\n\n"
                    f"Previous SPARQL query: {sparql}\n\n"
                    f"The SPARQL is invalid: {e}\n"
                    "Please refine the query to correct the error."
                )
                sparql_results = str(e)
                continue
            except Exception as e:
                sparql_results = str(e)
                continue

            if len(sparql_results) == 0:
                content = (
                    f"Question: {state['question']}\n\n"
                    f"Discovery Summary: {state['discovery_summary']}\n\n"
                    f"Previous SPARQL query: {sparql}\n\n"
                    "The SPARQL returned no results.\n"
                    "Reference the discovery summary. If the question should plausibly have results, refine the query."
                )
                continue

            break

        return {
            **state,
            "sparql": sparql,
            "sparql_results": sparql_results,
        }

    def _execute_sparql(self, sparql: str) -> dict:
        """Execute a SPARQL query against the Wikidata endpoint."""
        result = requests.get(
            WD_QUERY_URI,
            params={
                "query": sparql,
                "format": "json",
            },
            headers={"User-Agent": f"{USER_AGENT}"}
        )

        if result.status_code == 400:
            error_message = result.text.split("	at ")[0]
            raise ValueError(error_message)
        result.raise_for_status()

        result_bindings = result.json()["results"]["bindings"]
        return result_bindings