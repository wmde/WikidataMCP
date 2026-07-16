"""Small mechanical helpers for the SPARQL generation workflow."""

from __future__ import annotations

import json
import re
from typing import Any

READ_ONLY_QUERY_PATTERN = re.compile(r"^\s*(?:PREFIX\s+\w+:\s*<[^>]+>\s*)*(SELECT|ASK|CONSTRUCT|DESCRIBE)\b", re.I)
UPDATE_KEYWORDS = (
    "INSERT",
    "DELETE",
    "LOAD",
    "CLEAR",
    "CREATE",
    "DROP",
    "MOVE",
    "COPY",
    "ADD",
)


def tool_result_to_text(result: Any) -> str:
    """Convert LangChain/MCP tool output into plain prompt text."""
    if isinstance(result, tuple) and result:
        result = result[0]
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        blocks = [
            block["text"]
            for block in result
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        ]
        if blocks:
            return "\n".join(blocks)
    return json.dumps(result, ensure_ascii=False, default=str)


def message_text(result: dict) -> str:
    """Return the final model message content from an agent result."""
    messages = result.get("messages", [])
    for message in reversed(messages):
        if getattr(message, "type", None) != "ai":
            continue
        content = getattr(message, "content", "")
        if not content:
            continue
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()
    for message in reversed(messages):
        if getattr(message, "type", None) in {"tool", "human", "system"}:
            continue
        content = getattr(message, "content", "")
        if not content:
            continue
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()
    return ""


def is_read_only_sparql(sparql: str) -> bool:
    """Return whether the query appears to be a read-only SPARQL query."""
    if not READ_ONLY_QUERY_PATTERN.search(sparql or ""):
        return False
    upper = sparql.upper()
    return not any(re.search(rf"\b{keyword}\b", upper) for keyword in UPDATE_KEYWORDS)


def compact_text(text: str, max_chars: int = 8000) -> str:
    """Keep prompt inserts bounded while preserving the start and end."""
    if len(text or "") <= max_chars:
        return text or ""
    half = max_chars // 2
    return f"{text[:half]}\n\n[...truncated...]\n\n{text[-half:]}"


def format_sparql_bindings(bindings: list[dict[str, Any]], max_rows: int = 20) -> str:
    """Format Wikidata JSON bindings into compact prompt text."""
    if not bindings:
        return "SPARQL query returned no data."

    lines: list[str] = []
    for index, row in enumerate(bindings[:max_rows], start=1):
        cells: list[str] = []
        for name, value in row.items():
            raw_value = value.get("value", "") if isinstance(value, dict) else str(value)
            raw_value = raw_value.replace("http://www.wikidata.org/entity/", "")
            cells.append(f"{name}={raw_value}")
        lines.append(f"{index}. " + "; ".join(cells))
    if len(bindings) > max_rows:
        lines.append(f"... {len(bindings) - max_rows} more rows not shown")
    return "\n".join(lines)
