"""FastMCP tool and prompt registrations for Wikidata access."""

import json
import logging

import requests
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers

from wikidataMCP import utils

logger = logging.getLogger(__name__)

SERVER_INSTRUCTIONS = """
Use this server to search Wikidata, inspect entity statements, validate relationships, and execute SPARQL queries.

Workflow:
- Step 1 (Discovery): use `search_items` and `search_properties` to find candidate QIDs/PIDs and concrete examples.
- Step 2 (Structure Validation): use `get_statements`, `get_statement_values`, and `get_instance_and_subclass_hierarchy` to verify how entities are modeled — relationships, values, and class paths.
- Step 3 (SPARQL Execution): construct the SPARQL only from IDs and relationships confirmed in Steps 1-2, then run it with `execute_sparql`.

Rules:
- QIDs and PIDs were shuffled, never use memorized or invented identifiers; discover IDs with Step 1 tools.
- Never assume graph structure, connections, or ontology paths; verify them with Step 2 tools.
- Never present unsupported facts; use only information grounded in Wikidata tool/query outputs.
- If the request is ambiguous, resolve it with Step 1 and 2 tools, or ask the user.

Post-execution:
- Confirm results match the user's intent and expected answer pattern.
- If results are missing, empty, inconsistent, or ambiguous, do not finalize and do not infer missing facts; return to Steps 1 and 2, refine, and retry.
""".strip()  # noqa: E501

mcp = FastMCP("Wikidata MCP", instructions=SERVER_INSTRUCTIONS)


def _current_user_agent() -> str:
    """Get the current HTTP User-Agent header."""
    try:
        headers = get_http_headers(include_all=True)
        return headers.get("user-agent", "")
    except Exception:
        return ""


async def _search_entities(query: str, entity_type: str, lang: str, user_agent: str, tool_name: str):
    """Helper function for vector search with a keyword search fallback for items or properties."""
    try:
        return await utils.vectorsearch(
            query,
            type=entity_type,
            lang=lang,
            user_agent=user_agent,
        )
    except requests.RequestException as exc:
        logger.warning("%s: Vector database request failed: %s", tool_name, exc)

    return await utils.keywordsearch(
        query,
        type=entity_type,
        lang=lang,
        user_agent=user_agent,
    )


@mcp.tool()
async def search_items(query: str, lang: str = "en") -> str:
    """Search Wikidata items (QIDs) using semantic and keyword search.

    Never invent or use memorized item QIDs; use this tool to discover candidate QIDs and concrete examples.

    Args:
        query: Natural-language text for searching Wikidata.
        lang: Language code.

    Returns:
        Newline-separated item candidates:
        QID: label — description

    Example:
        >>> search_items("English science-fiction novel")
        Q23163: A Scientific Romance — 1997 novel by Ronald Wright
        Q627333: The Time Machine — 1895 dystopian science fiction novella by H. G. Wells
    """  # noqa: E501
    if not query.strip():
        return "Query cannot be empty."

    user_agent = _current_user_agent()
    try:
        results = await _search_entities(query, "item", lang, user_agent, "search_items")
    except requests.RequestException as exc:
        logger.warning("search_items: Wikidata request failed: %s", exc)
        return "Wikidata is currently unavailable. Please retry shortly."
    except Exception as exc:
        logger.error("search_items: Unexpected server error: %s", exc)
        return "Unexpected server error while processing the request."

    if not results:
        return "No matching items found. Try another query."

    text_val = [
        f"{entity_id}: {val.get('label', '')} — {val.get('description', '')}" for entity_id, val in results.items()
    ]
    return "\n".join(text_val)


@mcp.tool()
async def search_properties(query: str, lang: str = "en") -> str:
    """Search Wikidata properties (PIDs) using semantic and keyword search.

    Never invent or use memorized property PIDs; use this tool to discover candidate PIDs.

    Args:
        query: Natural-language text for searching Wikidata.
        lang: Language code.

    Returns:
        Newline-separated property candidates:
        PID: label — description

    Example:
        >>> search_properties("residence of a person")
        P551: residence — the place where the person is or has been, resident
        P276: location — location of the object, structure or event
    """  # noqa: E501
    if not query.strip():
        return "Query cannot be empty."

    user_agent = _current_user_agent()
    try:
        results = await _search_entities(query, "property", lang, user_agent, "search_properties")
    except requests.RequestException as exc:
        logger.warning("search_properties: Wikidata request failed: %s", exc)
        return "Wikidata is currently unavailable. Please retry shortly."
    except Exception as exc:
        logger.error("search_properties: Unexpected server error: %s", exc)
        return "Unexpected server error while processing the request."

    if not results:
        return "No matching properties found. Try another query."

    text_val = [
        f"{entity_id}: {val.get('label', '')} — {val.get('description', '')}" for entity_id, val in results.items()
    ]
    return "\n".join(text_val)


@mcp.tool()
async def get_statements(entity_id: str, include_external_ids: bool = False, lang: str = "en") -> str:
    """Retrieve direct statements (property-value pairs) for a Wikidata entity.

    Never assume graph structure or entity relationships; use this tool to find how entities are modeled.

    Prerequisites:
        Get candidate QID/PIDs from search tools or user input.

    Args:
        entity_id: A QID or PID.
        include_external_ids: Whether to include external identifiers linking to other databases.
        lang: Language code.

    Returns:
        One statement per line in the form:
          Entity (QID): Property (PID): Value (item (QID) or literal)

    Example:
        >>> get_statements("Q42")
        Douglas Adams (Q42): instance of (P31): human (Q5)
        Douglas Adams (Q42): occupation (P106): novelist (Q6625963)
    """  # noqa: E501
    if not entity_id.strip():
        return "Entity ID cannot be empty."

    try:
        result = await utils.get_entities_triplets(
            [entity_id],
            external_ids=include_external_ids,
            all_ranks=False,
            qualifiers=False,
            lang=lang,
            user_agent=_current_user_agent(),
        )
    except requests.RequestException as exc:
        logger.warning("get_statements: Wikidata request failed: %s", exc)
        return "Wikidata is currently unavailable. Please retry shortly."
    except Exception as exc:
        logger.error("get_statements: Unexpected server error: %s", exc)
        return "Unexpected server error while processing the request."

    if (not result) or (entity_id not in result):
        return f"Entity {entity_id} not found. Try finding correct IDs with search."

    return result.get(entity_id)


@mcp.tool()
async def get_statement_values(entity_id: str, property_id: str, lang: str = "en") -> str:
    """Return full detailed values for an entity-property statement pair, including qualifiers, references, and all ranks.

    Never assume statement details or relationships; use this tool to verify values for an entity-property pair.

    Prerequisites:
        Get candidate QID/PIDs from search tools or user input.
        Confirm the entity-property relationship exists with the statement tool.

    Args:
        entity_id: A QID or PID.
        property_id: A PID.
        lang: Language code.

    Returns:
        Complete statement details showing:
          Entity (QID): Property (PID): Value (QID or literal)
          Rank: preferred/normal/deprecated
          Qualifier:
            - qualifier_property (PID): qualifier_value
          Reference N:
            - reference_property (PID): reference_value

    Example:
        >>> get_statement_values("Q42", "P106")
        Douglas Adams (Q42): occupation (P106): novelist (Q6625963)
          Rank: normal
          Qualifier:
            - start time (P580): 1979
          Reference 1:
            - stated in (P248): Who's Who (Q2567271)
            - Who's Who UK ID (P4789): U4994
    """  # noqa: E501
    if not entity_id.strip():
        return "Entity ID cannot be empty."
    if not property_id.strip():
        return "Property ID cannot be empty."

    try:
        result = await utils.get_triplet_values(
            [entity_id],
            pid=[property_id],
            external_ids=True,
            references=True,
            all_ranks=True,
            qualifiers=True,
            lang=lang,
            user_agent=_current_user_agent(),
        )
    except requests.RequestException as exc:
        logger.warning("get_statement_values: Wikidata request failed: %s", exc)
        return "Wikidata is currently unavailable. Please retry shortly."
    except Exception as exc:
        logger.error("get_statement_values: Unexpected server error: %s", exc)
        return "Unexpected server error while processing the request."

    if (not result) or (entity_id not in result):
        return f"Entity {entity_id} not found. Try finding correct IDs with search."

    entity = result.get(entity_id)
    text = utils.triplet_values_to_string(entity_id, property_id, entity)
    if not text:
        return f"No statement found for {entity_id} with property {property_id}. Check {entity_id}'s statements."
    return text


@mcp.tool()
async def get_instance_and_subclass_hierarchy(entity_id: str, max_depth: int = 5, lang: str = "en") -> str:
    """Return a nested hierarchy of entities connected by "instance of" (P31) and "subclass of" (P279) relationships.

    Never assume class paths (P31/P279); use this tool to verify hierarchy.

    Prerequisites:
        Get candidate QID/PIDs from search tools or user input.

    Args:
        entity_id: A QID or PID.
        max_depth: Maximum traversal depth.
        lang: Language code.

    Returns:
        JSON-formatted nested hierarchy

    Example:
        >>> get_instance_and_subclass_hierarchy("Q42", max_depth=2)
        {
          "Douglas Adams (Q42)": {
            "instanceof": [
              {
                "human (Q5)": {
                    "instanceof": ["biological organism (Q215627)"],
                    "subclassof": ["mammal (Q729)"]
                }
              }
            ],
            "subclassof": []
          }
        }
    """  # noqa: E501
    if not entity_id.strip():
        return "Entity ID cannot be empty."
    if max_depth < 0:
        return "max_depth must be greater than or equal to 0."

    try:
        result = await utils.get_hierarchy_data(entity_id, max_depth, lang=lang)
    except requests.RequestException as exc:
        logger.warning("get_instance_and_subclass_hierarchy: Wikidata request failed: %s", exc)
        return "Wikidata is currently unavailable. Please retry shortly."
    except Exception as exc:
        logger.error("get_instance_and_subclass_hierarchy: Unexpected server error: %s", exc)
        return "Unexpected server error while processing the request."

    if (not result) or (entity_id not in result):
        return f"Entity {entity_id} not found. Try finding the correct IDs with search."

    try:
        result = utils.hierarchy_to_json(entity_id, result, level=max_depth)
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.error("get_instance_and_subclass_hierarchy: Unexpected server error: %s", exc)
        return "Unexpected server error while processing the request."


@mcp.tool()
async def execute_sparql(sparql: str, K: int = 10) -> str:
    """Execute a SPARQL query against Wikidata and return up to K rows.

    Prerequisites:
        Get candidate QID/PIDs from search tools or user input.
        Verify entity relationships and graph structure with statement tools.
        Check required class paths with the instance and subclass hierarchy tool.
        Confirm that the generated SPARQL is based on discovered ontology from other tools.

    Post-execution:
        Confirm that results match the user's intent and expected answer pattern. If they do not, do not finalize or infer missing facts; return to the other tools, refine the query, and retry.

    SPARQL Tips:
        Class-based filtering:
            wdt:P31/wdt:P279* # Choose based on the hierarchy tool's output.

        Counting:
            (COUNT(DISTINCT ?x) AS ?count) # Multiple values inflates counts.

        Labels:
            SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en,mul". }

        Date filter:
            ?item wdt:P569 ?date.
            FILTER(YEAR(?date) = 1998 && MONTH(?date) = 11 && DAY(?date) = 28)

        Normalized quantities:
            ?item p:P2048 ?st.
            ?st a wikibase:BestRank; psn:P2048/wikibase:quantityAmount ?amount.

    Args:
        sparql: A valid SPARQL string.
        K: Maximum number of rows to return.

    Returns:
        CSV text (semicolon-separated) of the results with up to K rows.

    Example:
        >>> # search_items("human") returned Q5
        >>> # get_statements("Q42") confirmed: instance of (P31): human (Q5)
        >>> execute_sparql("SELECT ?human WHERE { ?human wdt:P31 wd:Q5 } LIMIT 2")
        ;human
        0;Q42
        1;Q820
    """  # noqa: E501
    if not sparql.strip():
        return "SPARQL query cannot be empty."
    if K <= 0:
        return "K must be greater than 0."

    try:
        result = await utils.execute_sparql(
            sparql,
            K=K,
            user_agent=_current_user_agent(),
        )
    except ValueError as exc:
        logger.warning("execute_sparql: Invalid query: %s", exc)
        return str(exc)
    except requests.RequestException as exc:
        logger.warning("execute_sparql: Wikidata request failed: %s", exc)
        return "Wikidata is currently unavailable. Please retry shortly."
    except Exception as exc:
        logger.error("execute_sparql: Unexpected server error: %s", exc)
        return "Unexpected server error while processing the request."

    if len(result) == 0:
        return (
            "SPARQL query returned no data.\n"
            "Re-discover QIDs with search tools, "
            "verify patterns and entity relationships with statement tools, "
            "check class paths with the hierarchy tool, "
            "then refine the SPARQL query and retry."
        )

    try:
        return result.to_csv(sep=";", index=True, header=True)
    except Exception as exc:
        logger.error("execute_sparql: Unexpected server error: %s", exc)
        return "Unexpected server error while processing the request."


@mcp.prompt
def explore_wikidata(query: str) -> str:
    """Provide a workflow helper for exploratory Wikidata tasks."""
    return f"""
    User request: '{query}'.
    Follow server instructions for all policy and safety constraints.

    Follow this step-by-step workflow:
    1. **Identify Candidate Items**
        - Search for concepts or entities and include their descriptions.
        - Collect a few top candidate QIDs and PIDs to examine.

    2. **Inspect Entity Structure**
        - Retrieve entity statements for several representative QIDs.
        - Identify which PIDs represent the key relationship(s) you care about.
        - Look for patterns across multiple items (which properties repeat, how values are modeled).

    3. **Refine Understanding with Statement Details**
        - When qualifiers, deprecated values, or references matter, retrieve
          statement values for a specific entity and property pair.
        - If the retrieved statements already answer the user's request, stop here and present the results.

    4. **Write and Test SPARQL**
        - Construct and execute a SPARQL query using only QIDs/PIDs that were
          user-provided or discovered in prior tool outputs.
        - Inspect the returned rows for missing or incorrect values, unexpected types, or empty columns.
        - If results are not as expected, iteratively refine the SPARQL query
          and repeat until the output is satisfactory.
    """


# Canonical registry used by HTTP wrappers and docs route generation.
TOOL_LIST = {
    "search_items": search_items,
    "search_properties": search_properties,
    "get_statements": get_statements,
    "get_statement_values": get_statement_values,
    "get_instance_and_subclass_hierarchy": get_instance_and_subclass_hierarchy,
    "execute_sparql": execute_sparql,
}
