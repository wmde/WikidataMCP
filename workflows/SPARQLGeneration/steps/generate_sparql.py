"""Step 5: generate SPARQL from refined prose findings and execute it."""

from __future__ import annotations

import os

import requests
from pydantic import BaseModel, Field

from steps.agent import AgentFactory
from steps.workflow_utils import (
    compact_text,
    format_sparql_bindings,
    is_read_only_sparql,
)

WD_QUERY_URI = os.environ.get("WD_QUERY_URI", "https://query.wikidata.org/sparql")
USER_AGENT = os.environ.get("USER_AGENT", "Wikidata MCP SPARQL Generation (embedding@wikimedia.de)")


class SPARQLOutput(BaseModel):
    """Generated Wikidata SPARQL query."""

    sparql: str = Field(description="One complete read-only Wikidata SPARQL query.")


class GenerateSparqlStep:
    """Generate one SPARQL query without tools, then execute it mechanically."""

    SYSTEM_PROMPT = """\
    You are Step 5 of a Wikidata SPARQL generation workflow: write SPARQL only.

    Use only the QIDs and PIDs grounded in the discovery and critique summaries.
    Never invent QIDs or PIDs.
    Return exactly one complete read-only Wikidata SPARQL query.

    Discovery and critique notes are grouped by stage.
    If a stage note contains no grounded Wikidata findings for the question, treat that stage as unavailable.
    Use subclass expansion only when the discovery summary recommends it.
    If critique notes identify a missing or incorrect pattern, fix the query directly.
    """  # noqa: E501

    def __init__(self, model_name: str, mcp_url: str | None = None) -> None:
        """Configure the SPARQL-generation step."""
        self.agent_factory = AgentFactory(model_name=model_name, mcp_url=mcp_url)

    async def run(self, state: dict) -> dict:
        """Generate SPARQL, execute it, and return execution text for critique."""
        print("\n=== Step 5: Generate and Execute SPARQL ===", flush=True)
        runner = await self.agent_factory.create(
            system_prompt=self.SYSTEM_PROMPT,
            output_model=SPARQLOutput,
        )

        prompt = self._build_prompt(state)
        result = await runner.invoke_agent({"messages": [{"role": "user", "content": prompt}]}, "generate sparql")
        output = result.get("structured_response")
        if not isinstance(output, SPARQLOutput):
            output = SPARQLOutput.model_validate(output)
        sparql = output.sparql.strip()

        sparql_error = self._validate_query(sparql)
        bindings: list[dict] = []
        result_text = sparql_error
        if not sparql_error:
            try:
                bindings = self._execute_sparql(sparql)
                result_text = format_sparql_bindings(bindings)
            except Exception as exc:
                sparql_error = f"Execution failed: {exc}"
                result_text = sparql_error

        sparql_history = [*state.get("sparql_history", []), sparql]
        result_history = [*state.get("result_history", []), result_text]

        return {
            **state,
            "sparql": sparql,
            "sparql_results": result_text,
            "sparql_result_bindings": bindings,
            "sparql_result_empty": not bindings and not sparql_error,
            "sparql_error": sparql_error,
            "sparql_history": sparql_history,
            "result_history": result_history,
            "result": result_text,
        }

    def _build_prompt(self, state: dict) -> str:
        """Build the model prompt for initial generation or refinement."""
        critique = state.get("critique_summary", "")
        previous_sparql = state.get("sparql", "")
        parts = [
            f"Question:\n{state['question']}",
            f"Discovery Findings:\n{self._format_discovery_findings(state)}",
        ]
        if previous_sparql:
            parts.append(f"Previous SPARQL:\n{previous_sparql}")
        if critique:
            parts.append(f"Critique and refinement notes:\n{self._format_critique_findings(state)}")
        parts.append("Return only the complete SPARQL query.")
        return "\n\n".join(parts)

    def _format_discovery_findings(self, state: dict) -> str:
        """Format discovery stage notes without interpreting them."""
        sections = [
            ("Step 1 Search Findings", state.get("search_summary", "")),
            ("Step 2 Item Statement Findings", state.get("item_inspection_summary", "")),
            ("Step 3 Statement Detail Findings", state.get("statement_inspection_summary", "")),
            ("Step 4 Class Hierarchy Findings", state.get("class_inspection_summary", "")),
        ]
        if not any(summary for _, summary in sections):
            sections = [("Discovery Summary", state.get("discovery_summary", ""))]
        return self._format_stage_sections(sections, max_chars_per_section=3500)

    def _format_critique_findings(self, state: dict) -> str:
        """Format critique stage notes without interpreting them."""
        sections = [
            ("Step 6 Result Item Findings", state.get("result_inspection_summary", "")),
            ("Step 7 Result Statement Findings", state.get("statement_validation_summary", "")),
            ("Step 8 Result Class Findings", state.get("class_validation_summary", "")),
        ]
        if not any(summary for _, summary in sections):
            sections = [("Critique Summary", state.get("critique_summary", ""))]
        return self._format_stage_sections(sections, max_chars_per_section=3000)

    @staticmethod
    def _format_stage_sections(sections: list[tuple[str, str]], max_chars_per_section: int) -> str:
        """Apply prompt limits per stage so middle stages stay visible."""
        return "\n\n".join(
            f"## {heading}\n"
            f"{compact_text(summary or 'No findings reported by this stage.', max_chars=max_chars_per_section)}"
            for heading, summary in sections
        )

    def _validate_query(self, sparql: str) -> str:
        """Run cheap local checks before querying Wikidata."""
        if not sparql.strip():
            return "No SPARQL query was generated."
        if not is_read_only_sparql(sparql):
            return "Only read-only SELECT, ASK, CONSTRUCT, or DESCRIBE SPARQL queries are allowed."
        return ""

    def _execute_sparql(self, sparql: str) -> list[dict]:
        """Execute a SPARQL query against the Wikidata endpoint."""
        response = requests.get(
            WD_QUERY_URI,
            params={
                "query": sparql,
                "format": "json",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )

        if response.status_code == 400:
            error_message = response.text.split("\tat ")[0]
            raise ValueError(error_message)
        response.raise_for_status()

        return response.json()["results"]["bindings"]
