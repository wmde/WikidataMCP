"""Step 1: discover relevant Wikidata entities and properties."""

from steps.agent import AgentFactory


class DiscoverStep:
    """Discover and verify candidate Wikidata identifiers."""

    TOOL_NAMES = (
        "search_items",
        "search_properties",
        "get_statements",
        "get_statement_values",
        "get_instance_and_subclass_hierarchy"
    )
    TOOL_CALL_LIMIT = 3
    SYSTEM_PROMPT = """You are an evidence-gathering agent, not a problem-solving or query-writing step. You goal is to explore Wikidata and write a summary of found information to prepare the next agent to write a SPARQL query.

    Allowed actions:
    - Use the provided Wikidata tools to search and inspect relevant entities, properties, statements, and hierarchy.
    - Write a summary of information grounded in tool results.

    Report information useful for a later agent to write SPARQL:
    - Useful entities: QIDs needed as fixed values, anchors, or domain concepts in the query.
    - Useful relationships: QID and PID pairs needed as graph edges or filters.
    - Useful classes: QIDs from instance-of/subclass-of evidence that help restrict or filter results.
    - Example items: concrete Wikidata items with QIDs that show how this domain is modeled, including their relevant statements and relationships.

    If information is missing, use the tools to search or inspect more.
    Do not perform actions outside the provided tools or describe hypothetical procedures.
    Do not ask the user follow-up questions.
    Do not answer the original question directly or write a SPARQL query.
    """  # noqa: E501

    def __init__(self, model_name: str, mcp_url: str) -> None:
        """Configure the discovery step."""
        self.agent_factory = AgentFactory(model_name=model_name, mcp_url=mcp_url)

    async def run(self, state: dict) -> dict:
        """Run discovery and return the verified workflow state."""
        print("\n=== Step 1: Discover ===", flush=True)
        search_messages = await self.run_search(state)
        inspect_messages = await self.run_inspect(state, search_messages)
        state = await self.run_summary(state, inspect_messages)
        return state

    async def run_search(self, state: dict) -> dict:
        """Search tools only."""
        print("\n=== Step 1.1: Search ===", flush=True)
        runner = await self.agent_factory.create(
            system_prompt=self.SYSTEM_PROMPT,
            tool_names=("search_items", "search_properties"),
            tool_call_limit=2,
        )

        print(f"  [tool-call] search_items args={{'query': {state['question']!r}}}", flush=True)
        initial_item_result = await runner.tools["search_items"].ainvoke({"query": state["question"]})
        print(f"  [tool-call] search_properties args={{'query': {state['question']!r}}}", flush=True)
        initial_property_result = await runner.tools["search_properties"].ainvoke({"query": state["question"]})

        initial_item_result = initial_item_result
        initial_property_result = initial_property_result
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
                            "Continue gathering grounded Wikidata evidence for generating the SPARQL query."
                        ),
                    }
                ]
            }
        )

        return result['messages']


    async def run_inspect(self, state: dict, search_messages: list) -> dict:
        """Inspect tools only."""
        print("\n=== Step 1.2: Inspect ===", flush=True)
        runner = await self.agent_factory.create(
            system_prompt=self.SYSTEM_PROMPT,
            tool_names=("get_statements", "get_statement_values", "get_instance_and_subclass_hierarchy"),
            tool_call_limit=5,
        )

        result = await runner.invoke_agent(
            {
                "messages": [
                    *search_messages,
                    {
                        "role": "user",
                        "content": (
                            f"Question: `{state['question']}`\n"
                            "Continue gathering grounded Wikidata evidence for generating the SPARQL query."
                        ),
                    }
                ]
            }
        )

        return result['messages']

    async def run_verification(self, state: dict) -> dict:
        """Verify the grounded evidence."""
        print("\n=== Step 1.4: Verification ===", flush=True)

        messages = []
        tool_prompts = {
            "get_statements": "Inspect each entity mentioned in the discovery summary and enrich the summary with missing or incorrect relationships and entity IDs.",  # noqa: E501
            "get_statement_values": "Inspect each relationship mentioned in the discovery summary and enrich the summary with missing or incorrect relationships and entity IDs.",  # noqa: E501
            "get_instance_and_subclass_hierarchy": "Inspect the instance and subclass hierarchy of entities to find the correct entity class to filter on."  # noqa: E501
        }
        for tool_name, prompt in tool_prompts.items():
            runner = await self.agent_factory.create(
                system_prompt=self.SYSTEM_PROMPT,
                tool_names=(tool_name,),
                tool_call_limit=5,
            )

            result = await runner.invoke_agent(
                {
                    "messages": [
                        *messages,
                        {
                            "role": "user",
                            "content": (
                                f"Question: `{state['question']}`\n"
                                f"Discovery Summary: {state['discovery_summary']}\n"
                                f"{prompt}"
                            ),
                        }
                    ]
                }
            )
            messages = result['messages']

        return result['messages']

    async def run_summary(self, state: dict, inspect_messages: list) -> dict:
        """Summarize the grounded evidence."""
        print("\n=== Step 1.3: Summarize ===", flush=True)
        runner = await self.agent_factory.create(system_prompt=self.SYSTEM_PROMPT)
        result = await runner.invoke_agent(
            {
                "messages": [
                    *inspect_messages,
                    {
                        "role": "user",
                        "content": (
                            f"Question: {state['question']}\n"
                            "Summarize the grounded Wikidata evidence needed to generate the SPARQL query."
                        ),
                    },
                ]
            }
        )

        return {
            **state,
            'discovery_summary': result['messages'][-1].content
        }
