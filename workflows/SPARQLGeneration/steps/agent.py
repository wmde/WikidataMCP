"""Reusable abstract base for SPARQL workflow agents."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from pydantic import BaseModel


class AgentStep(ABC):
    """Base lifecycle for model-backed workflow steps."""

    SYSTEM_PROMPT: ClassVar[str]
    OUTPUT_MODEL: ClassVar[type[BaseModel] | None] = None
    TOOL_NAMES: ClassVar[tuple[str, ...]] = ()
    TOOL_CALL_LIMIT: ClassVar[int | None] = None

    def __init__(self, model_name: str, mcp_url: str | None = None) -> None:
        """Configure shared model and MCP settings."""
        self.model_name = model_name
        self.mcp_url = mcp_url
        self.tools: dict[str, Any] = {}
        self.agent: Any | None = None

    async def setup(self) -> None:
        """Load required MCP tools and create the structured agent once."""
        if self.agent is not None:
            return

        # Preapre tools
        if self.TOOL_NAMES:
            if self.mcp_url is None:
                raise ValueError(f"{type(self).__name__} requires an MCP URL.")
            client = MultiServerMCPClient({"wikidata": {"transport": "streamable_http", "url": self.mcp_url}})
            available_tools = {tool.name: tool for tool in await client.get_tools()}
            missing = set(self.TOOL_NAMES) - available_tools.keys()
            if missing:
                raise ValueError(f"MCP server is missing tools: {', '.join(sorted(missing))}")
            self.tools = {name: available_tools[name] for name in self.TOOL_NAMES}

        middleware = []
        if self.TOOL_CALL_LIMIT is not None:
            middleware.extend(
                ToolCallLimitMiddleware(tool_name=tool_name, run_limit=self.TOOL_CALL_LIMIT, exit_behavior="continue")
                for tool_name in self.TOOL_NAMES
            )

        # creat agent
        agent_kwargs = {
            "model": ChatOllama(model=self.model_name, temperature=0),
            "tools": list(self.tools.values()),
            "system_prompt": self.SYSTEM_PROMPT,
            "middleware": middleware or [],
        }
        if self.OUTPUT_MODEL:
            agent_kwargs["response_format"] = ToolStrategy(self.OUTPUT_MODEL)
        self.agent = create_agent(**agent_kwargs)

    def remove_tools(self) -> Any:
        """Setup the agent without tools."""
        agent_kwargs = {
            "model": ChatOllama(model=self.model_name, temperature=0),
            "system_prompt": self.SYSTEM_PROMPT,
        }
        if self.OUTPUT_MODEL:
            agent_kwargs["response_format"] = ToolStrategy(self.OUTPUT_MODEL)
        self.agent = create_agent(**agent_kwargs)

    async def invoke_agent(self, payload: dict) -> dict:
        """Invoke this step's agent while streaming model text."""
        if self.agent is None:
            raise ValueError(f"{type(self).__name__} agent is not configured.")

        result = {"messages": list(payload.get("messages", []))}
        streamed_text = False
        seen_tool_calls = set()
        seen_tool_results = set()
        seen_model_usage = set()

        async for mode, chunk in self.agent.astream(payload, stream_mode=["messages", "updates"]):
            if mode == "messages":
                message, _ = chunk
                content = getattr(message, "content", "")
                if content:
                    if not streamed_text:
                        print("    [model-stream]", flush=True)
                        streamed_text = True
                    print(content, end="", flush=True)
                for tool_call_chunk in getattr(message, "tool_call_chunks", []) or []:
                    call_key = (
                        tool_call_chunk.get("id"),
                        tool_call_chunk.get("index"),
                        tool_call_chunk.get("name"),
                    )
                    if call_key in seen_tool_calls or not tool_call_chunk.get("name"):
                        continue
                    seen_tool_calls.add(call_key)
                    if streamed_text:
                        print(flush=True)
                        streamed_text = False
                    print(
                        f"    [tool-call-stream] {tool_call_chunk.get('name')} "
                        f"args={tool_call_chunk.get('args') or ''}",
                        flush=True,
                    )
            elif mode == "updates":
                self.print_model_usage(chunk, seen_model_usage)
                self.print_tool_updates(chunk, seen_tool_calls, seen_tool_results)
                self.apply_agent_update(result, chunk)

        if streamed_text:
            print(flush=True)
        print(
            f"  [agent-result]: messages={len(result.get('messages', []))} "
            f"structured_response={result.get('structured_response') is not None}",
            flush=True,
        )
        return result

    def print_model_usage(self, update: dict, seen_model_usage: set) -> None:
        """Print token usage reported by completed model calls."""
        for node_update in update.values():
            if not isinstance(node_update, dict):
                continue
            for message in node_update.get("messages", []):
                usage = getattr(message, "usage_metadata", None) or {}
                metadata = getattr(message, "response_metadata", None) or {}
                prompt_tokens = usage.get("input_tokens") or metadata.get("prompt_eval_count")
                output_tokens = usage.get("output_tokens") or metadata.get("eval_count")
                total_tokens = usage.get("total_tokens")
                if total_tokens is None and prompt_tokens is not None and output_tokens is not None:
                    total_tokens = prompt_tokens + output_tokens
                if prompt_tokens is None and output_tokens is None and total_tokens is None:
                    continue

                usage_key = getattr(message, "id", None) or (
                    prompt_tokens,
                    output_tokens,
                    total_tokens,
                    metadata.get("total_duration"),
                )
                if usage_key in seen_model_usage:
                    continue
                seen_model_usage.add(usage_key)
                print(
                    "    [model-usage] "
                    f"prompt_tokens={prompt_tokens} "
                    f"output_tokens={output_tokens} "
                    f"total_tokens={total_tokens}",
                    flush=True,
                )

    def print_tool_updates(self, update: dict, seen_tool_calls: set, seen_tool_results: set) -> None:
        """Print completed tool calls and tool results from streamed updates."""
        for node_update in update.values():
            if not isinstance(node_update, dict):
                continue
            for message in node_update.get("messages", []):
                for tool_call in getattr(message, "tool_calls", []) or []:
                    call_key = tool_call.get("id") or (tool_call.get("name"), str(tool_call.get("args")))
                    if call_key in seen_tool_calls:
                        continue
                    seen_tool_calls.add(call_key)
                    print(
                        f"    [tool-call] {tool_call.get('name')} args={tool_call.get('args')}",
                        flush=True,
                    )

                if getattr(message, "type", None) != "tool":
                    continue
                result_key = getattr(message, "id", None) or getattr(message, "tool_call_id", None)
                if result_key in seen_tool_results:
                    continue
                seen_tool_results.add(result_key)
                name = getattr(message, "name", None) or "tool"
                status = getattr(message, "status", None) or "done"
                content = getattr(message, "content", "")
                print(f"    [tool-result] {name} status={status}", flush=True)
                if content:
                    print(content, flush=True)

    def apply_agent_update(self, result: dict, update: dict) -> None:
        """Merge streamed agent updates into a final result dict."""
        for node_update in update.values():
            if not isinstance(node_update, dict):
                continue
            result.setdefault("messages", []).extend(node_update.get("messages", []))
            if "structured_response" in node_update:
                result["structured_response"] = node_update["structured_response"]

    @abstractmethod
    async def run(self, state: dict) -> dict:
        """Execute the workflow step."""
