"""Step 1: search for relevant Wikidata entities and properties."""

from steps.agent import AgentFactory
from steps.workflow_utils import message_text, tool_result_to_text


class DiscoverStep:
    """Search for candidate Wikidata identifiers without inspecting structure."""

    TOOL_NAMES = ("search_items", "search_properties")
    TOOL_CALL_LIMIT = 4
    SYSTEM_PROMPT = """\
    You are Step 1 of a Wikidata SPARQL generation workflow: search only.

    Your only job is to search for candidate Wikidata items and properties that may be relevant to the user's question.
    Use search_items and search_properties.
    Select relevant candidates and reject obvious false positives.

    Treat search results as labels and descriptions only. Do not state that an item has a relationship, count, class,
    example value, or counterexample value unless that exact fact appears in the search result text.
    If a candidate might be an example or counterexample, say it is a candidate to inspect later.
    Do not infer how Wikidata models the answer.
    Do not decide which property or class the SPARQL query should use.

    Write only a Step 1 search findings note. This is not the final discovery summary.
    Include only candidates learned from search results:
    - Relevant candidate items with QIDs, labels, and why each may matter.
    - Relevant candidate properties with PIDs, labels, and why each may matter.
    - Candidate example and counterexample items to inspect later, without claiming unverified relationships or counts.
    - Rejected or distractor candidates in a separate section.
    """  # noqa: E501

    def __init__(self, model_name: str, mcp_url: str) -> None:
        """Configure the discovery step."""
        self.agent_factory = AgentFactory(model_name=model_name, mcp_url=mcp_url)

    async def run(self, state: dict) -> dict:
        """Run search and return a prose candidate summary."""
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
                            "Search further only if needed, then write the Step 1 candidate summary."
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
