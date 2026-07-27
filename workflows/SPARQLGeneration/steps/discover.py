"""Step 1: search for relevant Wikidata entities and properties."""

from steps.agent import AgentFactory
from steps.workflow_utils import message_text, tool_result_to_text


class DiscoverStep:
    """Search for candidate Wikidata identifiers without inspecting structure."""

    TOOL_NAMES = ("search_items", "search_properties")
    TOOL_CALL_LIMIT = 3
    SYSTEM_PROMPT = """\
    You are Step 1 of a Wikidata SPARQL generation workflow: search only.

    Your only job is to search for candidate Wikidata items and properties that may be relevant to the user's question.
    Your only tools are search_items and search_properties. Use them to find candidate QIDs and PIDs.
    Select relevant candidates and identify clearly unrelated candidates.

    Treat search results as labels and descriptions only.
    Mark possible examples, counterexamples, properties, and classes as candidates to inspect later.
    Keep relationships, counts, classes, and modeling claims tentative unless the exact fact appears in search text.

    Write only a Step 1 search findings note containing:
    - Relevant candidate items with QIDs, labels, and why each may matter.
    - Relevant candidate properties with PIDs, labels, and why each may matter.
    - Candidate example and counterexample items to inspect later, with why each should be inspected.
    - Rejected or distractor candidates in a separate section.
    - "No grounded findings for this step" when search gives no useful question-relevant candidates.
    """  # noqa: E501

    def __init__(self, model_name: str, mcp_url: str) -> None:
        """Configure the discovery step."""
        self.agent_factory = AgentFactory(model_name=model_name, mcp_url=mcp_url)

    async def run(self, state: dict) -> dict:
        """Run search and return a prose search findings note."""
        print("\n=== Step 1: Search Candidates ===", flush=True)
        runner = await self.agent_factory.create(
            system_prompt=self.SYSTEM_PROMPT,
            tool_names=self.TOOL_NAMES,
            tool_call_limit=self.TOOL_CALL_LIMIT,
        )

        print(f"  [tool-call] search_items args={{'query': {state['question']!r}}}", flush=True)
        initial_item_result = tool_result_to_text(
            await runner.tools["search_items"].ainvoke({"query": state["question"]})
        )
        print(f"  [tool-call] search_properties args={{'query': {state['question']!r}}}", flush=True)
        initial_property_result = tool_result_to_text(
            await runner.tools["search_properties"].ainvoke({"query": state["question"]})
        )

        result = await runner.invoke_agent(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Question: `{state['question']}`\n\n"
                            "Initial Wikidata searches were already performed\n"
                            "search_items result:\n"
                            f"{initial_item_result}\n\n"
                            "search_properties result:\n"
                            f"{initial_property_result}\n\n"
                            "Search further only if needed, then write only the Step 1 search findings note."
                        ),
                    }
                ]
            }
        )
        search_summary = message_text(result)

        return {
            **state,
            "search_summary": search_summary,
            "discovery_summary": search_summary,
        }
