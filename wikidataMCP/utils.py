"""Helper utilities for Wikidata API, textifier, and SPARQL access."""

import os
import re

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

VECTOR_SEARCH_URI = os.environ.get("VECTOR_SEARCH_URI", "https://wd-vectordb.wmcloud.org")
TEXTIFER_URI = os.environ.get("TEXTIFER_URI", "https://wd-textify.wmcloud.org")
WD_API_URI = os.environ.get("WD_API_URI", "https://www.wikidata.org/w/api.php")
WD_QUERY_URI = os.environ.get("WD_QUERY_URI", "https://query.wikidata.org/sparql")
USER_AGENT = os.environ.get("USER_AGENT", "Wikidata MCP Client (embedding@wikimedia.de)")

REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "15"))

SESSION = requests.Session()
adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
SESSION.mount("http://", adapter)
SESSION.mount("https://", adapter)


async def keywordsearch(query: str, type: str = "item", limit: int = 10, lang: str = "en", user_agent="") -> list:
    """Searches for Wikidata items or properties that match the input text.

    Args:
        query (str): The text to search for in Wikidata items or properties.
        type (str, optional): Type of entity to search for. One of
            "item" or "property". Defaults to "item".
        limit (int, optional): Maximum number of results to return. Defaults to 10.
        lang (str, optional): Language code used for labels and descriptions.
            Defaults to "en".
        user_agent (str, optional): Caller-provided suffix appended to
            the service User-Agent. Defaults to "".

    Returns:
        list: Matching entities with identifier, label, and description fields.
    """
    params = {
        "action": "wbsearchentities",
        "type": type,
        "search": query,
        "limit": limit,
        "language": lang,
        "format": "json",
        "origin": "*",
    }
    response = SESSION.get(
        WD_API_URI,
        params=params,
        headers={"User-Agent": f"{USER_AGENT} ({user_agent})"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    response_dict_search = response.json().get("search", {})

    item_dict = {
        x["id"]: {
            "label": x.get("display", {}).get("label", {}).get("value", ""),
            "description": x.get("display", {}).get("description", {}).get("value", ""),
        }
        for x in response_dict_search
    }
    return item_dict


def vectorsearch_verify_apikey(x_api_key: str) -> bool:
    """Verifies if the provided API key is valid for vector search.

    Args:
        x_api_key (str): API key for accessing the vector database.

    Returns:
        bool: True if the API key is valid, False otherwise.
    """
    try:
        if not x_api_key:
            x_api_key = ""

        response = SESSION.get(
            f"{VECTOR_SEARCH_URI}/item/query/?query=",
            headers={
                "x-api-secret": x_api_key,
                "User-Agent": USER_AGENT,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        return response.status_code != 401
    except Exception:
        return False


async def vectorsearch(
    query: str, x_api_key: str, type: str = "item", limit: int = 10, lang: str = "en", user_agent=""
) -> list:
    """Searches for Wikidata items or properties similar to the input text using a vector database.

    Args:
        query (str): The text to search for in Wikidata items or properties.
        x_api_key (str): API key for accessing the vector database.
        type (str, optional): Type of entity to search for. One of
            "item" or "property". Defaults to "item".
        limit (int, optional): Maximum number of results to return. Defaults to 10.
        lang (str, optional): Language code used when resolving labels.
            Defaults to "en".
        user_agent (str, optional): Caller-provided suffix appended to
            the service User-Agent. Defaults to "".

    Returns:
        list: Matching entities with identifier, label, and description fields.
    """
    id_name = "QID" if type == "item" else "PID"

    response = SESSION.get(
        f"{VECTOR_SEARCH_URI}/{type}/query/",
        params={
            "query": query,
            "k": limit,
        },
        headers={"x-api-secret": x_api_key, "User-Agent": f"{USER_AGENT} ({user_agent})"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    vectordb_result = response.json()

    ids = [x[id_name] for x in vectordb_result]
    entities_dict = await get_entities_labels_and_descriptions(ids, lang=lang)
    return entities_dict


async def execute_sparql(sparql_query: str, K: int = 10, user_agent="") -> pd.DataFrame:
    """Execute a SPARQL query on Wikidata.

    Args:
        sparql_query (str): The SPARQL query to execute.
        K (int, optional): Maximum number of rows to keep in output.
            Defaults to 10.
        user_agent (str, optional): Caller-provided suffix appended to
            the service User-Agent. Defaults to "".

    Returns:
        pandas.DataFrame: A cleaned dataframe of the results.
    """
    result = SESSION.get(
        WD_QUERY_URI,
        params={
            "query": sparql_query,
            "format": "json",
        },
        headers={"User-Agent": f"{USER_AGENT} ({user_agent})"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if result.status_code == 400:
        error_message = result.text.split("	at ")[0]
        raise ValueError(error_message)
    result.raise_for_status()

    result_bindings = result.json()["results"]["bindings"]
    df = pd.json_normalize(result_bindings)

    value_cols = {c: c.split(".")[0] for c in df.columns if c.endswith(".value")}
    df = df[list(value_cols)].rename(columns=value_cols)

    def shorten(val: str) -> str:
        if not isinstance(val, str):
            return val
        URI_RE = re.compile(r"http://www\.wikidata\.org/entity/([A-Z]\d+)$")
        match = URI_RE.match(val)
        return match.group(1) if match else val

    df = df.applymap(shorten)
    df = df.head(K)
    return df


def get_lang_specific(data, langs=["en", "mul"]) -> str:
    """Return the first non-empty label/description value for preferred languages."""
    for lang in langs:
        if lang in data:
            if data[lang].get("value"):
                return data[lang].get("value")
    return ""


async def get_entities_labels_and_descriptions(ids, lang="en") -> dict:
    """Fetches labels and descriptions for a list of Wikidata entity IDs.

    Args:
        ids (list[str]): List of Wikidata entity IDs (QIDs or PIDs).
        lang (str, optional): Language code available on Wikidata. Default to en.

    Returns:
        dict: A dictionary mapping entity IDs to WikidataEntity objects with labels and descriptions.
    """
    if not ids:
        return {}

    entities_data = {}

    # Wikidata API has a limit on the number of IDs per request,
    # typically 50 for wbgetentities.
    for chunk_idx in range(0, len(ids), 50):
        ids_chunk = ids[chunk_idx : chunk_idx + 50]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(ids_chunk),
            "languages": lang + "|mul|en",
            "props": "labels|descriptions",
            "format": "json",
            "origin": "*",
        }
        response = SESSION.get(
            WD_API_URI,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        chunk_data = response.json().get("entities", {})
        entities_data = entities_data | chunk_data

    entities_dict = {
        id: {
            "label": get_lang_specific(val.get("labels", {}), langs=[lang, "mul", "en"]),
            "description": get_lang_specific(val.get("descriptions", {}), langs=[lang, "mul", "en"]),
        }
        for id, val in entities_data.items()
    }
    return entities_dict


async def get_entities_triplets(
    ids: list[str],
    external_ids: bool = False,
    all_ranks: bool = False,
    qualifiers: bool = True,
    lang: str = "en",
    user_agent="",
) -> dict:
    """Fetches triplet representations for claims of a list of Wikidata entity IDs.

    Args:
        ids (list[str]): A list of Wikidata entity IDs to fetch triplet data for.
        external_ids (bool, optional): Whether to include external identifiers
            linking to other databases. Defaults to False.
        all_ranks (bool, optional): Whether to include all statement ranks
            (preferred, normal, deprecated). Defaults to False.
        qualifiers (bool, optional): Whether to include qualifiers in output.
            Defaults to True.
        lang (str, optional): Language code available on Wikidata. Default to en.
        user_agent (str, optional): Caller-provided suffix appended to
            the service User-Agent. Defaults to "".

    Returns:
        dict: A dictionary where keys are entity IDs and values are their RDF triplet representations as strings.
    """
    if not ids:
        return {}

    params = {
        "id": ",".join(ids),
        "external_ids": external_ids,
        "all_ranks": all_ranks,
        "qualifiers": qualifiers,
        "lang": lang,
        "format": "triplet",
    }
    response = SESSION.get(
        TEXTIFER_URI,
        params=params,
        headers={"User-Agent": f"{USER_AGENT} ({user_agent})"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    info = response.json()

    return info


async def get_claims(qid: str, pid: str, lang: str = "en") -> dict:
    """Fetches claim values for a given Wikidata QID and PID.

    Args:
        qid (str): The Wikidata QID to fetch claim data for.
        pid (str): The Wikidata PID to fetch claim data for.
        lang (str, optional): Language code available on Wikidata. Default to en.

    Returns:
        dict: A dictionary where keys are entity IDs and values are their RDF triplet representations as strings.
    """
    if not qid or not pid:
        return {}

    params = {
        "action": "wbgetclaims",
        "entity": qid,
        "property": pid,
        "format": "json",
        "origin": "*",
    }
    response = SESSION.get(
        WD_API_URI,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    entities_data = response.json().get("claims", {})

    claim_values = []
    for claim in entities_data.get(pid, []):
        mainsnak = claim.get("mainsnak", {})
        datavalue = mainsnak.get("datavalue", {})
        if "value" in datavalue:
            claim_values.append(datavalue["value"])
    return claim_values


async def get_triplet_values(
    ids: list[str],
    pid: list[str],
    external_ids: bool = False,
    all_ranks: bool = False,
    references: bool = False,
    qualifiers: bool = True,
    lang: str = "en",
    user_agent="",
) -> dict:
    """Fetches triplet representations for claims of a list of Wikidata entity IDs.

    Args:
        ids (list[str]): A list of Wikidata entity IDs to fetch triplet data for.
        pid (list[str]): Property IDs used to filter claims.
        external_ids (bool, optional): Whether to include external identifiers
            linking to other databases. Defaults to False.
        all_ranks (bool, optional): Whether to include all statement ranks
            (preferred, normal, deprecated). Defaults to False.
        references (bool, optional): Whether to retrieve references. Default to False.
        qualifiers (bool, optional): Whether to retrieve qualifiers.
            Defaults to True.
        lang (str, optional): Language code available on Wikidata. Default to en.
        user_agent (str, optional): Caller-provided suffix appended to
            the service User-Agent. Defaults to "".

    Returns:
        dict: A dictionary where keys are entity IDs and values are the triplet data as JSON.
    """
    if not ids:
        return {}

    params = {
        "id": ",".join(ids),
        "external_ids": external_ids,
        "all_ranks": all_ranks,
        "references": references,
        "qualifiers": qualifiers,
        "lang": lang,
        "pid": ",".join(pid),
        "format": "json",
    }
    response = SESSION.get(
        TEXTIFER_URI,
        params=params,
        headers={"User-Agent": f"{USER_AGENT} ({user_agent})"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    info = response.json()

    return info


async def get_hierarchy_data(qid: str, max_depth: int = 5, lang: str = "en") -> dict:
    """Fetches hierarchical data for a given Wikidata QID.

    Args:
        qid (str): The Wikidata QID to fetch hierarchical data for.
        max_depth (int, optional): Maximum depth of the hierarchy to retrieve. Defaults to 5.
        lang (str, optional): Language code available on Wikidata. Default to en.

    Returns:
        dict: A dictionary representing the hierarchical data.
    """
    qids = [qid]
    hierarchical_data = {}
    label_data = {}
    level = 0

    while qids and level <= max_depth:
        new_qids = set()

        current_data = await get_triplet_values(qids, pid=["P31", "P279"], lang=lang)

        for qid in qids:
            if qid not in current_data:
                continue

            instanceof = [c["values"] for c in current_data[qid]["claims"] if c["PID"] == "P31"]
            instanceof = [v["value"] for v in instanceof[0]] if instanceof else []

            subclassof = [c["values"] for c in current_data[qid]["claims"] if c["PID"] == "P279"]
            subclassof = [v["value"] for v in subclassof[0]] if subclassof else []

            instanceof_qids = [v.get("QID", v.get("PID")) for v in instanceof]
            subclassof_qids = [v.get("QID", v.get("PID")) for v in subclassof]

            hierarchical_data[qid] = {"instanceof": instanceof_qids, "subclassof": subclassof_qids}

            new_qids = new_qids | set(instanceof_qids) | set(subclassof_qids)

            for v in instanceof + subclassof:
                if "QID" in v:
                    label_data[v["QID"]] = v.get("label", "")
                elif "PID" in v:
                    label_data[v["PID"]] = v.get("label", "")
            label_data[qid] = current_data[qid].get("label", "")

        qids = new_qids - set(hierarchical_data.keys()) - set({None})
        level += 1

    qids = list(hierarchical_data.keys())
    for qid, label in label_data.items():
        if qid in hierarchical_data:
            hierarchical_data[qid]["label"] = label

    return hierarchical_data


def hierarchy_to_json(qid, data, level=5):
    """Convert hierarchy data to a nested JSON-serializable structure."""
    if level <= 0:
        return f"{data[qid]['label']} ({qid})"

    return {
        f"{data[qid]['label']} ({qid})": {
            "instance of (P31)": [
                hierarchy_to_json(i_qid, data, level - 1) for i_qid in data[qid]["instanceof"] if (i_qid in data)
            ],
            "subclass of (P279)": [
                hierarchy_to_json(i_qid, data, level - 1) for i_qid in data[qid]["subclassof"] if (i_qid in data)
            ],
        }
    }


def stringify(value) -> str:
    """Convert structured value objects into a readable string representation."""
    if isinstance(value, dict):
        if "values" in value:
            return ", ".join([stringify(v.get("value", {})) for v in value["values"]])
        if "value" in value:
            return stringify(value["value"])
        if "string" in value:
            return value["string"]
        if "QID" in value:
            return f"{value.get('label')} ({value.get('QID')})"
        if "PID" in value:
            return f"{value.get('label')} ({value.get('PID')})"
        if "amount" in value:
            return f"{value.get('amount')} {value.get('unit', '')}".strip()
    return str(value)


def triplet_values_to_string(entity_id: str, property_id: str, entity: dict) -> str:
    """Converts triplet values of a Wikidata statement into a human-readable string format.

    Args:
        entity_id (str): The Wikidata entity ID (QID).
        property_id (str): The Wikidata property ID (PID).
        entity (dict): The triplet data of the entity.

    Returns:
        str: A formatted string representing the triplet values, qualifiers, and references.
    """
    claims = entity.get("claims")
    if not claims:
        return None

    output = ""
    for claim in claims:
        for claim_value in claim.get("values", []):
            if output:
                output += "\n"

            output += f"{entity['label']} ({entity_id}): "
            claim_pid = claim.get("PID", property_id)
            output += f"{claim['property_label']} ({claim_pid}): "
            output += f"{stringify(claim_value['value'])}\n"

            output += f"  Rank: {claim_value.get('rank', 'normal')}\n"

            qualifiers = claim_value.get("qualifiers", [])
            if qualifiers:
                output += "  Qualifier:\n"
                for qualifier in qualifiers:
                    output += f"    - {qualifier['property_label']} ({qualifier['PID']}): "
                    output += stringify(qualifier)
                    output += "\n"

            references = claim_value.get("references", [])
            if references:
                i = 1
                for reference in references:
                    output += f"  Reference {i}:\n"
                    for reference_claim in reference:
                        output += f"    - {reference_claim['property_label']} ({reference_claim['PID']}): "
                        output += stringify(reference_claim)
                        output += "\n"
                    i += 1
    return output.strip()
