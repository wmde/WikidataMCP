from fastmcp import FastMCP, Context
from fastmcp.server.dependencies import get_http_headers
from wikidataMCP import utils
import os
import json
import requests
import time
from .logger import Logger

mcp = FastMCP("Wikidata MCP")

VECTOR_ENABLED = os.environ.get("VECTOR_ENABLED", "true").lower() == "true"

def _current_user_agent() -> str:
    try:
        headers = get_http_headers(include_all=True)
        return headers.get("user-agent", "")
    except Exception:
        return ""

def _format_search_results(results: dict, entity_type: str) -> str:
    if not results:
        return f"No matching Wikidata {entity_type}s found."

    text_val = [
        f"{entity_id}: {val.get('label', '')} — {val.get('description', '')}"
        for entity_id, val in results.items()
    ]
    return "\n".join(text_val)


# Enable vector search
if VECTOR_ENABLED:

    @mcp.tool()
    async def search_items(query: str, lang: str = 'en') -> str:
        """Search Wikidata items (QIDs) using vector and keyword search.
        Find conceptually similar Wikidata items from a natural-language query. Matches are based on meaning and exact words.

        Args:
            query: Natural-language description of the concept to find.
            lang: Language code for the search (default: 'en').

        Returns:
            Newline-separated results in the form:
                QID: label — description

        Example:
            >>> search_items("English science-fiction novel")
            Q23163: A Scientific Romance — 1997 novel by Ronald Wright
            Q627333: The Time Machine — 1895 dystopian science fiction novella by H. G. Wells
        """
        start_time = time.time()
        user_agent = _current_user_agent()

        try:
            if not query.strip():
                return "Query cannot be empty."

            try:
                results = await utils.vectorsearch(
                    query,
                    lang=lang,
                    user_agent=user_agent,
                )
            except requests.RequestException:
                try:
                    results = await utils.keywordsearch(
                        query,
                        type="item",
                        lang=lang,
                        user_agent=user_agent,
                    )
                except requests.RequestException:
                    return "Wikidata is currently unavailable. Please retry shortly."
            except Exception:
                try:
                    results = await utils.keywordsearch(
                        query,
                        type="item",
                        lang=lang,
                        user_agent=user_agent,
                    )
                except requests.RequestException:
                    return "Wikidata is currently unavailable. Please retry shortly."
                except Exception:
                    return "Unexpected server error while processing the request."

            return _format_search_results(results, "item")

        finally:
            Logger.add_request_async(
                toolname="search_items",
                start_time=start_time,
                parameters={"query": query, "lang": lang},
                user_agent=user_agent,
            )


    @mcp.tool()
    async def search_properties(query: str, lang: str = 'en') -> str:
        """Search Wikidata properties (PIDs) using vector and keyword search.
        Find relevant Wikidata properties from a natural-language description of the relationship you need. Matches are based on meaning and exact words.

        Args:
            query: Natural-language description of the concept to find.
            lang: Language code for the search (default: 'en').

        Returns:
            Newline-separated results in the form:
                PID: label — description

        Example:
            >>> search_properties("residence of a person")
            P551: residence — the place where the person is or has been, resident
            P276: location — location of the object, structure or event
        """
        start_time = time.time()
        user_agent = _current_user_agent()

        try:
            if not query.strip():
                return "Query cannot be empty."

            try:
                results = await utils.vectorsearch(
                    query,
                    type="property",
                    lang=lang,
                    user_agent=user_agent,
                )
            except requests.RequestException:
                try:
                    results = await utils.keywordsearch(
                        query,
                        type="property",
                        lang=lang,
                        user_agent=user_agent,
                    )
                except requests.RequestException:
                    return "Wikidata is currently unavailable. Please retry shortly."
            except Exception:
                try:
                    results = await utils.keywordsearch(
                        query,
                        type="property",
                        lang=lang,
                        user_agent=user_agent,
                    )
                except requests.RequestException:
                    return "Wikidata is currently unavailable. Please retry shortly."
                except Exception:
                    return "Unexpected server error while processing the request."

            return _format_search_results(results, "property")
        finally:
            Logger.add_request_async(
                toolname="search_properties",
                start_time=start_time,
                parameters={"query": query, "lang": lang},
                user_agent=user_agent,
            )

else:

    @mcp.tool()
    async def search_items(query: str, lang: str = 'en') -> str:
        """Search Wikidata items (QIDs) with exact text matching.
        Looks up items by label/alias or literal phrases expected to appear in Wikidata. Useful when you already know the entity you're looking for.

        Args:
            query: Label, alias, or phrase expected to appear verbatim.
            lang: Language code for the search (default: 'en').

        Returns:
            Newline-separated lines in the form:
            QID: label — description

        Example:
            >>> search_items("Douglas Adams")
            Q42: Douglas Adams — English science fiction writer and humorist
            Q28421831: Douglas Adams — American environmental engineer
        """
        start_time = time.time()
        user_agent = _current_user_agent()

        try:
            if not query.strip():
                return "Query cannot be empty."

            try:
                results = await utils.keywordsearch(
                    query,
                    type="item",
                    lang=lang,
                    user_agent=user_agent,
                )
            except requests.RequestException:
                return "Wikidata is currently unavailable. Please retry shortly."
            except Exception:
                return "Unexpected server error while processing the request."

            return _format_search_results(results, "item")
        finally:
            Logger.add_request_async(
                toolname="search_items",
                start_time=start_time,
                parameters={"query": query, "lang": lang},
                user_agent=user_agent,
            )


    @mcp.tool()
    async def search_properties(query: str, lang: str = 'en') -> str:
        """Search Wikidata properties (PIDs) with exact text matching.
        Looks up properties by label/alias or literal phrases expected to appear in Wikidata. Useful when expected property name is already known.

        Args:
            query: Label, alias, or phrase expected to appear verbatim.
            lang: Language code for the search (default: 'en').

        Returns:
            Newline-separated lines in the form:
            PID: label — description

        Example:
            >>> search_properties("residence")
            P551: residence — the place where the person is or has been, resident
            P276: location — location of the object, structure or event
        """
        start_time = time.time()
        user_agent = _current_user_agent()

        try:
            if not query.strip():
                return "Query cannot be empty."

            try:
                results = await utils.keywordsearch(
                    query,
                    type="property",
                    lang=lang,
                    user_agent=user_agent,
                )
            except requests.RequestException:
                return "Wikidata is currently unavailable. Please retry shortly."
            except Exception:
                return "Unexpected server error while processing the request."

            return _format_search_results(results, "property")
        finally:
            Logger.add_request_async(
                toolname="search_properties",
                start_time=start_time,
                parameters={"query": query, "lang": lang},
                user_agent=user_agent,
            )


@mcp.tool()
async def get_statements(entity_id: str,
                        include_external_ids: bool = False,
                        lang: str = 'en') -> str:
    """Return the direct statements (property-value pairs) of an entity. Expose all direct graph connections of a Wikidata entity to inspect its factual context. This tool does not include deprecated values, qualifiers, or references (use `get_statement_values` instead).

    Args:
        entity_id: A QID or PID such as "Q42" or "P31".
        include_external_ids: Whether to include external identifiers linking to other databases.
        lang: Language code for labels and descriptions (default: 'en').

    Returns:
        One statement per line in the form:
          Entity (QID): Property (PID): Value (item (QID) or literal)

    Example:
        >>> get_statements("Q42")
        Douglas Adams (Q42): instance of (P31): human (Q5)
        Douglas Adams (Q42): occupation (P106): novelist (Q6625963)
    """

    start_time = time.time()
    user_agent = _current_user_agent()

    try:
        if not entity_id.strip():
            return "Entity ID cannot be empty."

        try:
            result = await utils.get_entities_triplets(
                [entity_id],
                external_ids=include_external_ids,
                all_ranks=False,
                qualifiers=False,
                lang=lang,
                user_agent=user_agent,
            )
        except requests.RequestException:
            return "Wikidata is currently unavailable. Please retry shortly."
        except Exception:
            return "Unexpected server error while processing the request."

        if not result:
            return f"Entity {entity_id} not found"

        return result.get(entity_id, f"Entity {entity_id} not found")
    finally:
        Logger.add_request_async(
            toolname="get_statements",
            start_time=start_time,
            parameters={
                "entity_id": entity_id,
                "include_external_ids": include_external_ids,
                "lang": lang,
            },
            user_agent=user_agent,
        )


@mcp.tool()
async def get_statement_values(entity_id: str,
                           property_id: str,
                           lang: str = 'en') -> str:
    """Get all values for a specific statement (entity-property pair), including all qualifiers, ranks and references. Returns complete statement information including deprecated values and reference data that are excluded from `get_statements`.

    Args:
        entity_id: A QID or PID such as "Q42" or "P31".
        property_id: A PID such as "P31".
        lang: Language code for labels and descriptions (default: 'en').

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

    start_time = time.time()
    user_agent = _current_user_agent()

    try:
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
                user_agent=user_agent,
            )
        except requests.RequestException:
            return "Wikidata is currently unavailable. Please retry shortly."
        except Exception:
            return "Unexpected server error while processing the request."

        if not result:
            return f"Entity {entity_id} not found"

        entity = result.get(entity_id)
        if not entity:
            return f"Entity {entity_id} not found"

        text = utils.triplet_values_to_string(entity_id, property_id, entity)
        if not text:
            return f"No statement found for {entity_id} with property {property_id}"
        return text
    finally:
        Logger.add_request_async(
            toolname="get_statement_values",
            start_time=start_time,
            parameters={
                "entity_id": entity_id,
                "property_id": property_id,
                "lang": lang,
            },
            user_agent=user_agent,
        )


@mcp.tool()
async def get_instance_and_subclass_hierarchy(entity_id: str,
                            max_depth: int = 5,
                            lang: str = 'en') -> str:
    """Expose the hierarchical context of a Wikidata entity to inspect its ontological placement. This tool retrieves hierarchical relationships based on "instance of" (P31) and "subclass of" (P279) properties.

    Args:
        entity_id: A QID or PID such as "Q42" or "P31".
        max_depth: Maximum depth of the hierarchy to retrieve. Defaults to 5.
        lang: Language code for labels and descriptions (default: 'en').

    Returns:
        JSON-formatted hierarchical data showing the entity's placement in the ontology.

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

    start_time = time.time()
    user_agent = _current_user_agent()

    try:
        if not entity_id.strip():
            return "Entity ID cannot be empty."

        try:
            result = await utils.get_hierarchy_data(entity_id, max_depth, lang=lang)
        except requests.RequestException:
            return "Wikidata is currently unavailable. Please retry shortly."
        except Exception:
            return "Unexpected server error while processing the request."

        if not result or entity_id not in result:
            return f"Entity {entity_id} not found"

        try:
            result = utils.hierarchy_to_json(entity_id, result, level=max_depth)
            return json.dumps(result, indent=2)
        except Exception:
            return "Unexpected server error while processing the request."
    finally:
        Logger.add_request_async(
            toolname="get_instance_and_subclass_hierarchy",
            start_time=start_time,
            parameters={"entity_id": entity_id, "max_depth": max_depth, "lang": lang},
            user_agent=user_agent,
        )


@mcp.tool()
async def execute_sparql(sparql: str, K: int = 10) -> str:
    """Execute a SPARQL query against Wikidata and return up to K rows as CSV.

    IMPORTANT: All QIDs (items) and PIDs (properties) are randomly shuffled, so you cannot rely on any prior knowledge of Wikidata identifiers or schema. The only way to retrieve information is by using the provided search and get tools prior to executing SPARQL.

    Tips:
        • Use the search and entity tools first to discover relevant QIDs and PIDs before writing a SPARQL query.

        • For class-based filtering, use:
            wdt:P31/wdt:P279*
            This expands both instance-of and subclass-of relationships.
            Use the get_instance_and_subclass_hierarchy tool to verify which class ID to filter on.

        • Add the label service to display readable names instead of QIDs:
            SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en,mul". }

        • Filtering by date:
            ?item wdt:P569 ?date.
            FILTER(YEAR(?date) = 1998 && MONTH(?date) = 11 && DAY(?date) = 28)
            This example filters by exact day.

        • Getting normalized quantity values:
            ?item p:P2048 ?statement. # P2048 = height
            ?statement a wikibase:BestRank;
                psn:P2048 ?valueNode.
            ?valueNode wikibase:quantityUnit wd:Q11573; # unit in metres
                wikibase:quantityAmount ?height.
            This ensures all values are normalized and comparable across items.

    Args:
        sparql: A valid SPARQL string.
        K: Maximum number of rows to return.

    Returns:
        CSV text (semicolon-separated) of the results with up to K rows. On error, returns the error message.

    Example:
        >>> execute_sparql("SELECT ?human WHERE { ?human wdt:P31 wd:Q5 } LIMIT 2")
        ;human
        0;Q42
        1;Q820
    """

    start_time = time.time()
    user_agent = _current_user_agent()

    try:
        if not sparql.strip():
            return "SPARQL query cannot be empty."

        try:
            result = await utils.execute_sparql(
                sparql,
                K=K,
                user_agent=user_agent,
            )
        except ValueError as e:
            return str(e)
        except requests.RequestException:
            return "Wikidata is currently unavailable. Please retry shortly."
        except Exception:
            return "Unexpected server error while processing the request."

        try:
            return result.to_csv(sep=';', index=True, header=True)
        except Exception:
            return "Unexpected server error while processing the request."
    finally:
        Logger.add_request_async(
            toolname="execute_sparql",
            start_time=start_time,
            parameters={"sparql": sparql, "K": K},
            user_agent=user_agent,
        )


@mcp.prompt
def explore_wikidata(query: str) -> str:
    """Instruct the model to explore Wikidata without assumptions."""

    return f"""
    You are an assistant that explores Wikidata on behalf of the user.
    The user's request is: '{query}'.

    IMPORTANT: All QIDs (items) and PIDs (properties) are randomly shuffled, so you cannot rely on any prior knowledge of Wikidata identifiers or schema. The only way to retrieve information is by using the provided tools.

    Follow this step-by-step workflow:
    1. **Identify Candidate Items**
        - Search for concepts or entities and include their descriptions.
        - Collect a few top candidate QIDs and PIDs to examine.

    2. **Inspect Entity Structure**
        - Retrieve entity statements for several representative QIDs.
        - Identify which PIDs represent the key relationship(s) you care about.
        - Look for patterns across multiple items (which properties repeat, how values are modeled).

    3. **Refine Understanding with Statement Details**
        - When qualifiers, deprecated values, or references matter, retrieve statement values for a specific entity and property pair.
        - If the retrieved statements already answer the user's request, stop here and present the results.

    4. **Write and Test SPARQL**
        - Construct and execute a SPARQL query using the discovered QIDs & PIDs.
        - Inspect the returned rows for missing or incorrect values, unexpected types, or empty columns.
        - If the results are not as expected, iteratively refine the SPARQL query and repeat until the results are satisfactory.
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
