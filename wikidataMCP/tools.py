"""FastMCP tool and prompt registrations for Wikidata access."""

import json
import traceback

import requests
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers

from wikidataMCP import utils

SERVER_INSTRUCTIONS = """
Use this server to search Wikidata, inspect entity statements, validate relationships, and execute SPARQL queries.

Rules:
- QIDs and PIDs may be shuffled.
- Never rely on memorized identifiers.
- Never invent QIDs or PIDs; discover them with Step 1 tools.
- Never assume graph structure, connections, or ontology paths.
- Never present unsupported facts; use only information grounded in Wikidata
  tool/query outputs.

Workflow:
- Step 1 (Discovery): use `search_items` and `search_properties` first to find
  candidate QIDs/PIDs and concrete examples.
- Step 2 (Structure Validation): use `get_statements`,
  `get_statement_values`, or `get_instance_and_subclass_hierarchy` to verify
  relationships, statement details, and hierarchy paths before SPARQL.
- Step 3 (SPARQL Execution): use `execute_sparql` only after Steps 1 and 2 are
  complete, and only with QIDs/PIDs that are user-provided or confirmed by
  prior tool outputs.

Post-execution:
- Confirm results match the user's intent and expected answer pattern.
- If results are missing, empty, inconsistent, or ambiguous, do not finalize
  and do not infer missing facts; return to Steps 1 and 2, refine, and retry.
""".strip()

mcp = FastMCP("Wikidata MCP", instructions=SERVER_INSTRUCTIONS)


def _current_user_agent() -> str:
    try:
        headers = get_http_headers(include_all=True)
        return headers.get("user-agent", "")
    except Exception:
        return ""


@mcp.tool()
async def search_items(query: str, lang: str = "en") -> str:
    """Search Wikidata items (QIDs) using semantic and keyword search.

    Rule:
    - Never invent or use memorized item IDs; use this tool to discover them.

    Workflow:
    - Step 1 (Discovery): run this first to discover candidate QIDs and
      concrete examples.
    - Prerequisite: none.

    Args:
        query: Natural-language text for searching Wikidata.
        lang: Language code.

    Returns:
        Newline-separated results in the form:
        QID: label — description

    Example:
        >>> search_items("English science-fiction novel")
        Q23163: A Scientific Romance — 1997 novel by Ronald Wright
        Q627333: The Time Machine — 1895 dystopian science fiction novella by H. G. Wells
    """
    if not query.strip():
        return "Query cannot be empty."

    user_agent = _current_user_agent()
    try:
        results = await utils.vectorsearch(
            query,
            lang=lang,
            user_agent=user_agent,
        )
    except Exception:
        try:
            results = await utils.keywordsearch(
                query,
                type="item",
                lang=lang,
                user_agent=user_agent,
            )
        except requests.RequestException:
            traceback.print_exc()
            return "Wikidata is currently unavailable. Please retry shortly."
        except Exception:
            traceback.print_exc()
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

    Rule:
    - Never invent or use memorized property IDs; use this tool to discover
      them.

    Workflow:
    - Step 1 (Discovery): run this first to discover candidate PIDs and
      concrete examples.
    - Prerequisite: none.

    Args:
        query: Natural-language text for searching Wikidata.
        lang: Language code.

    Returns:
        Newline-separated results in the form:
        PID: label — description

    Example:
        >>> search_properties("residence of a person")
        P551: residence — the place where the person is or has been, resident
        P276: location — location of the object, structure or event
    """
    if not query.strip():
        return "Query cannot be empty."

    user_agent = _current_user_agent()
    try:
        results = await utils.vectorsearch(
            query,
            type="property",
            lang=lang,
            user_agent=user_agent,
        )
    except Exception:
        try:
            results = await utils.keywordsearch(
                query,
                type="property",
                lang=lang,
                user_agent=user_agent,
            )
        except requests.RequestException:
            traceback.print_exc()
            return "Wikidata is currently unavailable. Please retry shortly."
        except Exception:
            traceback.print_exc()
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

    Rule:
    - Never assume graph structure or entity connections; use this tool to
      verify direct relationships.

    Workflow:
    - Step 2 (Structure Validation): run this after obtaining a QID/PID from
      search or user input, and before SPARQL.
    - This tool does not include deprecated values, qualifiers, or references.

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
    """
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
    except requests.RequestException:
        traceback.print_exc()
        return "Wikidata is currently unavailable. Please retry shortly."
    except Exception:
        traceback.print_exc()
        return "Unexpected server error while processing the request."

    if (not result) or (entity_id not in result):
        return f"Entity {entity_id} not found. Try finding correct IDs with search."

    return result.get(entity_id)


@mcp.tool()
async def get_statement_values(entity_id: str, property_id: str, lang: str = "en") -> str:
    """Return full values for an entity-property statement pair.

    Rule:
    - Never assume statement details or connections; use this tool to verify
      full values for an entity-property pair.

    Workflow:
    - Step 2 (Structure Validation): run this after obtaining an entity ID and
      property ID from search or user input, and before SPARQL when detailed
      statement values matter.

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
    """
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
    except requests.RequestException:
        traceback.print_exc()
        return "Wikidata is currently unavailable. Please retry shortly."
    except Exception:
        traceback.print_exc()
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

    Rule:
    - Never assume ontology paths (P31/P279); use this tool to verify class
      hierarchy traversal.

    Workflow:
    - Step 2 (Structure Validation): run this after obtaining a QID/PID from
      search or user input, and before SPARQL.

    Args:
        entity_id: A QID or PID.
        max_depth: Maximum traversal depth.
        lang: Language code.

    Returns:
        JSON-formatted nested hierarchy data

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
    """
    if not entity_id.strip():
        return "Entity ID cannot be empty."
    if max_depth < 0:
        return "max_depth must be greater than or equal to 0."

    try:
        result = await utils.get_hierarchy_data(entity_id, max_depth, lang=lang)
    except requests.RequestException:
        traceback.print_exc()
        return "Wikidata is currently unavailable. Please retry shortly."
    except Exception:
        traceback.print_exc()
        return "Unexpected server error while processing the request."

    if (not result) or (entity_id not in result):
        return f"Entity {entity_id} not found. Try finding the correct IDs with search."

    try:
        result = utils.hierarchy_to_json(entity_id, result, level=max_depth)
        return json.dumps(result, indent=2)
    except Exception:
        traceback.print_exc()
        return "Unexpected server error while processing the request."


@mcp.tool()
async def execute_sparql(sparql: str, K: int = 10) -> str:
    """Execute a SPARQL query against Wikidata and return up to K rows.

    Rule:
    - Never present unsupported facts; use only information grounded in the
      Wikidata query results.

    Workflow:
    - Step 3 (SPARQL Execution): run this only after Step 1 (Discovery) and
      Step 2 (Structure Validation) are complete.
    - Use only QIDs/PIDs that are user-provided or confirmed by prior tool
      outputs.

    Post-execution:
    - Confirm that results match the user's intent and expected answer pattern.
    - If they do not, do not finalize or infer missing facts; return to Step 1
      (Discovery) and Step 2 (Structure Validation), refine the query, and
      retry.

    SPARQL Tips:
        • For class-based filtering, use a property path:
            wdt:P31/wdt:P279*
            This expands both instance-of and subclass-of relationships.
            Examine hierarchy links first before applying this pattern.

        • Add the label service to display readable names:
            SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en,mul". }

        • Filtering by date:
            ?item wdt:P569 ?date.
            FILTER(YEAR(?date) = 1998 && MONTH(?date) = 11 && DAY(?date) = 28)

        • Getting normalized quantity values:
            ?item p:P2048 ?statement.
            ?statement a wikibase:BestRank;
                psn:P2048 ?valueNode.
            ?valueNode wikibase:quantityUnit wd:Q11573;
                wikibase:quantityAmount ?amount.
            This ensures all values are normalized and comparable across entities.

    Args:
        sparql: A valid SPARQL string.
        K: Maximum number of rows to return.

    Returns:
        CSV text (semicolon-separated) of the results with up to K rows.

    Example:
        >>> execute_sparql("SELECT ?human WHERE { ?human wdt:P31 wd:Q5 } LIMIT 2")
        ;human
        0;Q42
        1;Q820
    """
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
    except ValueError as e:
        traceback.print_exc()
        return str(e)
    except requests.RequestException:
        traceback.print_exc()
        return "Wikidata is currently unavailable. Please retry shortly."
    except Exception:
        traceback.print_exc()
        return "Unexpected server error while processing the request."

    if len(result) == 0:
        return (
            "SPARQL query returned no data.\n"
            "Return to Step 1 (Discovery) and Step 2 (Structure Validation), refine the query, and retry."
        )

    try:
        return result.to_csv(sep=";", index=True, header=True)
    except Exception:
        traceback.print_exc()
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
