"""Steps 2-4: inspect statements, statement values, and class hierarchy."""

from __future__ import annotations

from steps.agent import AgentFactory
from steps.workflow_utils import message_text


class InspectStructureStep:
    """Refine the search summary with grounded Wikidata structure evidence."""

    ITEM_PROMPT = """\
    You are Step 2 of a Wikidata SPARQL generation workflow: inspect items.

    Your only tool is get_statements. Use it to inspect candidate items from Step 1.
    Select statements that are relevant to answering the question.
    Compare examples and counterexamples to find which statements separate correct answers from false positives.
    Do not assume that a searched property is correct until item statements support it.

    Write an item-statement findings note.
    Use prior search findings as context, but summarize only what this step learned from get_statements.
    If the tool output does not establish a relationship, say it remains unverified.
    Include:
    - Which inspected items are useful examples or anchors.
    - Relevant item statements, including subject QID, property PID, value QID or literal, and why each matters.
    - Candidate properties that appear correct, incorrect, or still uncertain.
    - What statement values or qualifiers should be inspected next.
    """  # noqa: E501

    STATEMENT_PROMPT = """\
    You are Step 3 of a Wikidata SPARQL generation workflow: inspect statement values.

    Your only tool is get_statement_values. Use it for entity-property pairs identified in Step 2.
    Look for qualifiers, ranks, deprecated values, references, and value modeling that could change the SPARQL.
    Inspect both positive examples and counterexamples when a property may include non-answer values.

    Write a statement-detail findings note.
    Use Steps 1-2 as context, but summarize only what this step learned from get_statement_values.
    If Steps 1-2 do not provide usable entity-property pairs, say what is missing instead of inventing pairs.
    Include:
    - Confirmed entity-property pairs and how their values are modeled.
    - Relevant qualifiers with qualifier PIDs and values.
    - Rank/deprecated-value concerns if they affect SPARQL.
    - Relationships or qualifiers that should or should not be used in the query.
    - What classes or hierarchy should be inspected next.
    """  # noqa: E501

    HIERARCHY_PROMPT = """\
    You are Step 4 of a Wikidata SPARQL generation workflow: inspect class hierarchy.

    Your only tool is get_instance_and_subclass_hierarchy. Use it to inspect candidate entities and classes from Steps 1-2.
    Decide which classes are safe to filter on, which are too broad or too narrow, and whether subclass expansion is needed.
    Compare broad and narrow classes for values that appeared in both examples and counterexamples.
    You may describe graph patterns such as using instance-of/subclass-of expansion.

    Write a class-hierarchy findings note.
    Use Steps 1-2 as context, but summarize only what this step learned from get_instance_and_subclass_hierarchy.
    Include:
    - Classes inspected and whether each is safe, too broad, too narrow, or uncertain.
    - Whether subclass expansion is needed and which class QIDs justify it.
    - Example or counterexample items that clarify class filtering.
    - Class/filter traps the SPARQL generator must avoid.
    """  # noqa: E501

    def __init__(self, model_name: str, mcp_url: str) -> None:
        """Configure the structure-inspection stage."""
        self.agent_factory = AgentFactory(model_name=model_name, mcp_url=mcp_url)

    async def run(self, state: dict) -> dict:
        """Run item, statement, and hierarchy inspection."""
        print("\n=== Step 2-4: Inspect Structure ===", flush=True)
        item_summary = await self.inspect_items(state)
        statement_summary = await self.inspect_statements(state, item_summary)
        class_summary = await self.inspect_hierarchy(state, item_summary)

        discovery_summary = self.build_discovery_summary(state, item_summary, statement_summary, class_summary)
        return {
            **state,
            "item_inspection_summary": item_summary,
            "statement_inspection_summary": statement_summary,
            "class_inspection_summary": class_summary,
            "discovery_summary": discovery_summary,
        }

    @staticmethod
    def build_discovery_summary(
        state: dict,
        item_summary: str,
        statement_summary: str,
        class_summary: str,
    ) -> str:
        """Bundle each discovery stage without interpreting model prose."""
        sections = [
            ("Step 1 Search Findings", state.get("search_summary", "")),
            ("Step 2 Item Statement Findings", item_summary),
            ("Step 3 Statement Detail Findings", statement_summary),
            ("Step 4 Class Hierarchy Findings", class_summary),
        ]
        return "\n\n".join(
            f"## {heading}\n{(summary or '').strip() or 'No findings reported by this stage.'}"
            for heading, summary in sections
        )

    async def inspect_items(self, state: dict) -> str:
        """Inspect candidate item statements."""
        print("\n=== Step 2: Inspect Items ===", flush=True)
        runner = await self.agent_factory.create(
            system_prompt=self.ITEM_PROMPT,
            tool_names=("get_statements",),
            tool_call_limit=6,
        )

        result = await runner.invoke_agent(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Question: {state['question']}\n\n"
                            f"Step 1 Search Summary:\n{state.get('search_summary', '')}\n\n"
                            "Use get_statements to inspect relevant candidate items before writing only the "
                            "Step 2 item-statement findings note."
                        ),
                    }
                ]
            },
            "inspect items",
        )
        return message_text(result)

    async def inspect_statements(self, state: dict, item_summary: str) -> str:
        """Inspect statement values and qualifiers for candidate pairs."""
        print("\n=== Step 3: Inspect Statements ===", flush=True)
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
                            f"Question: {state['question']}\n\n"
                            f"Step 1 Search Summary:\n{state.get('search_summary', '')}\n\n"
                            f"Step 2 Item Summary:\n{item_summary}\n\n"
                            "Use get_statement_values to inspect relevant entity-property pairs before writing "
                            "only the Step 3 statement-detail findings note."
                        ),
                    }
                ]
            },
            "inspect statements",
        )
        return message_text(result)

    async def inspect_hierarchy(self, state: dict, item_summary: str) -> str:
        """Inspect class hierarchy and produce its stage findings."""
        print("\n=== Step 4: Inspect Class Hierarchy ===", flush=True)
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
                            f"Question: {state['question']}\n\n"
                            f"Step 1 Search Summary:\n{state.get('search_summary', '')}\n\n"
                            f"Step 2 Item Summary:\n{item_summary}\n\n"
                            "Use get_instance_and_subclass_hierarchy to inspect relevant candidate entities "
                            "and classes before writing only the Step 4 class-hierarchy findings note."
                        ),
                    }
                ]
            },
            "inspect hierarchy",
        )
        return message_text(result)
