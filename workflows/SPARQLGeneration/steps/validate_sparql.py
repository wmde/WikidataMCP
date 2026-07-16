"""Steps 6-8: inspect SPARQL results and build refinement notes."""

from __future__ import annotations

from steps.agent import AgentFactory
from steps.workflow_utils import (
    compact_text,
    message_text,
)


class ValidateSparqlStep:
    """Critique generated SPARQL/results with narrow inspection tools."""

    RESULT_PROMPT = """\
    You are Step 6 of a Wikidata SPARQL generation workflow: inspect result items.

    Your only tool is get_statements. Use it to inspect items returned by the generated SPARQL.
    Compare the result rows against the question and discovery summary.
    Look for missing constraints, wrong result types, unexpected empty values, and false positives.
    Compare result rows with any example and counterexample items in the discovery summary.

    Write a result-item findings note.
    Use discovery findings as context, but summarize only what this step learned from get_statements.
    Include returned items inspected, false positives, missing constraints, unexpected empty values, and concrete
    improvements the next SPARQL attempt may need.
    Do not write a final critique or SPARQL plan.
    """  # noqa: E501

    STATEMENT_PROMPT = """\
    You are Step 7 of a Wikidata SPARQL generation workflow: inspect result statement values.

    Your only tool is get_statement_values. Use it to inspect relevant entity-property pairs from the generated SPARQL,
    discovery summary, Step 6 critique, and returned result items.
    Check whether statement values, qualifiers, ranks, or deprecated values make the current query too broad, too narrow, or wrong.
    Pay special attention to properties whose values include both answer values and non-answer values.
    Do not write SPARQL.

    Write a result-statement findings note.
    Use Step 6 as context, but summarize only what this step learned from get_statement_values.
    Include statement values, qualifiers, ranks, deprecated-value concerns, and whether the current query is too
    broad, too narrow, or using the wrong relationship.
    """  # noqa: E501

    HIERARCHY_PROMPT = """\
    You are Step 8 of a Wikidata SPARQL generation workflow: inspect class assumptions.

    Your only tool is get_instance_and_subclass_hierarchy. Use it to inspect class filters, returned result items,
    and class-like QIDs used or implied by the generated SPARQL.
    Check whether class filtering is correct, too broad, too narrow, missing subclass expansion, or using the wrong class.
    Compare broad and narrow classes when counterexamples show that a generic class would overcount.
    Do not write SPARQL.

    Write a result-class findings note.
    Use Step 6 as context, but summarize only what this step learned from get_instance_and_subclass_hierarchy.
    Include class filters that are correct, too broad, too narrow, missing subclass expansion, or wrong.
    """  # noqa: E501

    def __init__(self, model_name: str, mcp_url: str, max_refinement_cycles: int = 3) -> None:
        """Configure the SPARQL-validation stage."""
        self.agent_factory = AgentFactory(model_name=model_name, mcp_url=mcp_url)
        self.max_refinement_cycles = max_refinement_cycles

    async def run(self, state: dict) -> dict:
        """Run result, statement, and hierarchy critique, then decide whether to refine."""
        print("\n=== Step 6-8: Inspect Results and Critique ===", flush=True)
        previous_critique = state.get("critique_summary", "")

        result_summary = await self.inspect_result_items(state)
        statement_summary = await self.inspect_result_statements(state, result_summary)
        class_summary = await self.inspect_result_hierarchy(state, result_summary)
        critique_summary = self.build_critique_summary(result_summary, statement_summary, class_summary)

        cycle = state.get("refinement_cycle", 0) + 1
        should_refine, reason = self._should_refine(state, previous_critique, critique_summary, cycle)

        critique_history = [
            *state.get("critique_history", []),
            {
                "cycle": cycle,
                "result_summary": result_summary,
                "statement_summary": statement_summary,
                "class_summary": class_summary,
                "critique_summary": critique_summary,
                "should_refine": should_refine,
                "reason": reason,
            },
        ]

        return {
            **state,
            "result_inspection_summary": result_summary,
            "statement_validation_summary": statement_summary,
            "class_validation_summary": class_summary,
            "critique_summary": critique_summary,
            "critique_history": critique_history,
            "refinement_cycle": cycle,
            "should_refine": should_refine,
            "validation_reason": reason,
        }

    @staticmethod
    def build_critique_summary(result_summary: str, statement_summary: str, class_summary: str) -> str:
        """Bundle each critique stage without interpreting model prose."""
        sections = [
            ("Step 6 Result Item Findings", result_summary),
            ("Step 7 Result Statement Findings", statement_summary),
            ("Step 8 Result Class Findings", class_summary),
        ]
        return "\n\n".join(
            f"## {heading}\n{(summary or '').strip() or 'No findings reported by this stage.'}"
            for heading, summary in sections
        )

    async def inspect_result_items(self, state: dict) -> str:
        """Inspect statements for returned result items."""
        print("\n=== Step 6: Inspect Result Items ===", flush=True)
        runner = await self.agent_factory.create(
            system_prompt=self.RESULT_PROMPT,
            tool_names=("get_statements",),
            tool_call_limit=6,
        )

        result = await runner.invoke_agent(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Question:\n{state['question']}\n\n"
                            "Discovery Summary:\n"
                            f"{compact_text(state.get('discovery_summary', ''), max_chars=10000)}\n\n"
                            f"Generated SPARQL:\n{state.get('sparql', '')}\n\n"
                            f"Execution result:\n{compact_text(state.get('sparql_results', ''), max_chars=8000)}\n\n"
                            f"Execution issue:\n{state.get('sparql_error') or 'none'}\n\n"
                            "Use get_statements to inspect relevant returned items before writing only the "
                            "Step 6 result-item findings note."
                        ),
                    }
                ]
            },
            "inspect result items",
        )
        return message_text(result)

    async def inspect_result_statements(self, state: dict, result_summary: str) -> str:
        """Inspect statement values for result items and query properties."""
        print("\n=== Step 7: Inspect Result Statements ===", flush=True)
        runner = await self.agent_factory.create(
            system_prompt=self.STATEMENT_PROMPT,
            tool_names=("get_statement_values",),
            tool_call_limit=8,
        )

        result = await runner.invoke_agent(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Question:\n{state['question']}\n\n"
                            "Discovery Summary:\n"
                            f"{compact_text(state.get('discovery_summary', ''), max_chars=9000)}\n\n"
                            f"Generated SPARQL:\n{state.get('sparql', '')}\n\n"
                            f"Execution result:\n{compact_text(state.get('sparql_results', ''), max_chars=7000)}\n\n"
                            f"Step 6 Result Item Findings:\n{result_summary}\n\n"
                            "Use get_statement_values to inspect relevant result statement values before "
                            "writing only the Step 7 result-statement findings note."
                        ),
                    }
                ]
            },
            "inspect result statements",
        )
        return message_text(result)

    async def inspect_result_hierarchy(self, state: dict, result_summary: str) -> str:
        """Inspect hierarchy for classes and result items."""
        print("\n=== Step 8: Inspect Result Classes ===", flush=True)
        runner = await self.agent_factory.create(
            system_prompt=self.HIERARCHY_PROMPT,
            tool_names=("get_instance_and_subclass_hierarchy",),
            tool_call_limit=6,
        )

        result = await runner.invoke_agent(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Question:\n{state['question']}\n\n"
                            "Discovery Summary:\n"
                            f"{compact_text(state.get('discovery_summary', ''), max_chars=9000)}\n\n"
                            f"Generated SPARQL:\n{state.get('sparql', '')}\n\n"
                            f"Execution result:\n{compact_text(state.get('sparql_results', ''), max_chars=7000)}\n\n"
                            f"Step 6 Result Item Findings:\n{result_summary}\n\n"
                            "Use get_instance_and_subclass_hierarchy to inspect relevant classes or result "
                            "items before writing only the Step 8 result-class findings note."
                        ),
                    }
                ]
            },
            "inspect result hierarchy",
        )
        return message_text(result)

    def _should_refine(
        self,
        state: dict,
        previous_critique: str,
        critique_summary: str,
        cycle: int,
    ) -> tuple[bool, str]:
        """Decide whether LangGraph should loop back to SPARQL generation."""
        max_cycles = state.get("max_refinement_cycles", self.max_refinement_cycles)
        if cycle >= max_cycles:
            return False, f"Stopped after refinement cycle {cycle}; max refinement cycles reached."

        history = state.get("sparql_history", [])
        stable_query = len(history) >= 2 and history[-1].strip() == history[-2].strip()
        if stable_query:
            return False, "Stopped because the generator produced the same SPARQL as the previous cycle."

        stable_critique = bool(previous_critique) and previous_critique.strip() == critique_summary.strip()
        if stable_critique:
            return False, "Stopped because the critique summary did not change."

        return True, f"Refining after cycle {cycle}; running another SPARQL attempt with the latest critique."
