"""Steps 6-8: inspect SPARQL results and build refinement notes."""

from __future__ import annotations

from steps.agent import AgentFactory
from steps.workflow_utils import (
    compact_text,
    message_text,
)


class ValidateSparqlStep:
    """Validate generated SPARQL/results with narrow inspection tools."""

    RESULT_PROMPT = """\
    You are Step 6 of a Wikidata SPARQL generation workflow: inspect result items.

    Your only tool is get_statements. Use it to inspect items returned by the generated SPARQL.
    Use discovery context, the generated SPARQL, and execution results as staged evidence.
    Focus on whether returned rows match the question and the discovered examples or counterexamples.

    Write only a Step 6 result-item findings note containing:
    - Returned items inspected, with QIDs and why each matters.
    - Result items that look supported, unsupported, or still uncertain.
    - Missing constraints, wrong result types, unexpected empty values, or false positives found from item statements.
    - Result item patterns that Step 7 should inspect next.
    - "No grounded findings for this step" when there are no returned items or no useful item-statement evidence.
    """  # noqa: E501

    STATEMENT_PROMPT = """\
    You are Step 7 of a Wikidata SPARQL generation workflow: inspect result statement values.

    Your only tool is get_statement_values. Use it to inspect relevant entity-property pairs.
    Use the corresponding Step 3 statement-detail findings, the generated SPARQL, and execution results as staged evidence.
    Focus on whether statement values, qualifiers, ranks, or deprecated values explain result correctness.

    Write only a Step 7 result-statement findings note containing:
    - Entity-property pairs inspected, with QIDs/PIDs and why each matters.
    - Relevant values, with value QIDs or literals, ranks, and qualifier PIDs when they affect the question.
    - Query property assumptions marked as supported, unsupported, or still uncertain.
    - Statement patterns that the next SPARQL attempt should preserve or refine.
    - "No grounded findings for this step" when there is no useful statement-value evidence.
    """  # noqa: E501

    HIERARCHY_PROMPT = """\
    You are Step 8 of a Wikidata SPARQL generation workflow: inspect class assumptions.

    Your only tool is get_instance_and_subclass_hierarchy. Use it to inspect relevant result items, class filters,
    and class-like QIDs used or implied by the generated SPARQL.
    Use the corresponding Step 4 class-hierarchy findings, the generated SPARQL, and execution results as staged evidence.
    Focus on class filters that could explain correct results, false positives, or missing results.

    Write only a Step 8 result-class findings note containing:
    - Entities or classes inspected, with QIDs.
    - Relevant instance-of/subclass-of paths found by the tool.
    - Class filters marked as supported, too broad, too narrow, missing subclass expansion, or still uncertain.
    - Example and counterexample items that clarify class filtering.
    - "No grounded findings for this step" when there is no useful class-hierarchy evidence.
    """  # noqa: E501

    EMPTY_RESULT_SUMMARY = (
        "No rows were returned by the generated SPARQL. Step 6 has no result items to inspect. "
        "Steps 7 and 8 should inspect query assumptions using the focused discovery context."
    )

    def __init__(self, model_name: str, mcp_url: str, max_refinement_cycles: int = 3) -> None:
        """Configure the SPARQL-validation stage."""
        self.agent_factory = AgentFactory(model_name=model_name, mcp_url=mcp_url)
        self.max_refinement_cycles = max_refinement_cycles

    async def run(self, state: dict) -> dict:
        """Run result, statement, and hierarchy validation, then decide whether to refine."""
        print("\n=== Step 6-8: Inspect Results and Validation Findings ===", flush=True)
        previous_critique = state.get("critique_summary", "")

        if state.get("sparql_result_empty"):
            print("\n=== Step 6: Inspect Result Items ===", flush=True)
            result_summary = self.EMPTY_RESULT_SUMMARY
            print(result_summary, flush=True)
        else:
            result_summary = await self.inspect_result_items(state)
        statement_summary = await self.inspect_result_statements(state)
        class_summary = await self.inspect_result_hierarchy(state)
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
        """Bundle each validation stage without interpreting model prose."""
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
            tool_call_limit=5,
        )

        result = await runner.invoke_agent(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Question:\n{state['question']}\n\n"
                            "Corresponding discovery context from Step 2 item-statement findings:\n"
                            f"{self._format_result_item_context(state)}\n\n"
                            f"Generated SPARQL:\n{state.get('sparql', '')}\n\n"
                            f"Execution result:\n{compact_text(state.get('sparql_results', ''), max_chars=8000)}\n\n"
                            f"Execution issue:\n{state.get('sparql_error') or 'none'}\n\n"
                            "Use get_statements to inspect actual returned item IDs before writing only the "
                            "Step 6 result-item findings note."
                        ),
                    }
                ]
            },
            "inspect result items",
        )
        return message_text(result)

    async def inspect_result_statements(self, state: dict) -> str:
        """Inspect statement values for result items and query properties."""
        print("\n=== Step 7: Inspect Result Statements ===", flush=True)
        runner = await self.agent_factory.create(
            system_prompt=self.STATEMENT_PROMPT,
            tool_names=("get_statement_values",),
            tool_call_limit=5,
        )

        result = await runner.invoke_agent(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Question:\n{state['question']}\n\n"
                            "Corresponding discovery context from Step 3 statement-detail findings:\n"
                            f"{self._format_statement_validation_context(state)}\n\n"
                            f"Generated SPARQL:\n{state.get('sparql', '')}\n\n"
                            f"Execution result:\n{compact_text(state.get('sparql_results', ''), max_chars=7000)}\n\n"
                            f"Execution issue:\n{state.get('sparql_error') or 'none'}\n\n"
                            "Use get_statement_values to inspect relevant statement values before "
                            "writing only the Step 7 result-statement findings note."
                        ),
                    }
                ]
            },
            "inspect result statements",
        )
        return message_text(result)

    async def inspect_result_hierarchy(self, state: dict) -> str:
        """Inspect hierarchy for classes and result items."""
        print("\n=== Step 8: Inspect Result Classes ===", flush=True)
        runner = await self.agent_factory.create(
            system_prompt=self.HIERARCHY_PROMPT,
            tool_names=("get_instance_and_subclass_hierarchy",),
            tool_call_limit=3,
        )

        result = await runner.invoke_agent(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Question:\n{state['question']}\n\n"
                            "Corresponding discovery context from Step 4 class-hierarchy findings:\n"
                            f"{self._format_class_validation_context(state)}\n\n"
                            f"Generated SPARQL:\n{state.get('sparql', '')}\n\n"
                            f"Execution result:\n{compact_text(state.get('sparql_results', ''), max_chars=7000)}\n\n"
                            f"Execution issue:\n{state.get('sparql_error') or 'none'}\n\n"
                            "Use get_instance_and_subclass_hierarchy to inspect relevant classes or result "
                            "items before writing only the Step 8 result-class findings note."
                        ),
                    }
                ]
            },
            "inspect result hierarchy",
        )
        return message_text(result)

    def _format_result_item_context(self, state: dict) -> str:
        """Keep Step 6 context focused on the corresponding item-statement findings."""
        return self._format_context_sections(
            (
                ("Step 2 Item Statement Findings", state.get("item_inspection_summary", ""), 4000),
            )
        )

    def _format_statement_validation_context(self, state: dict) -> str:
        """Keep Step 7 context focused on the corresponding statement-detail findings."""
        return self._format_context_sections(
            (
                ("Step 3 Statement Detail Findings", state.get("statement_inspection_summary", ""), 4000),
            )
        )

    def _format_class_validation_context(self, state: dict) -> str:
        """Keep Step 8 context focused on the corresponding class-hierarchy findings."""
        return self._format_context_sections(
            (
                ("Step 4 Class Hierarchy Findings", state.get("class_inspection_summary", ""), 4000),
            )
        )

    @staticmethod
    def _format_context_sections(sections: tuple[tuple[str, str, int], ...]) -> str:
        """Format selected stage notes without passing the full combined summary."""
        return "\n\n".join(
            f"## {heading}\n{compact_text(summary or 'No findings reported by this stage.', max_chars=max_chars)}"
            for heading, summary, max_chars in sections
        )

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
            return False, "Stopped because the validation findings did not change."

        return (
            True,
            f"Refining after cycle {cycle}; running another SPARQL attempt with the latest validation findings.",
        )
