from typing import Any
import inspect

from fastapi import FastAPI, HTTPException, Query
from markdown2 import markdown
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.templating import Jinja2Templates
import uvicorn

from fastmcp import Context
from fastmcp.tools.tool import FunctionTool
from wikidataMCP import tools


templates = Jinja2Templates(directory="templates")
mcp = tools.mcp
mcp_app = mcp.http_app(path="/")


app = FastAPI(
    title="Wikidata Tool API",
    description="Auto-generated HTTP routes for all FastMCP tools.",
    version="0.1.0",
    lifespan=mcp_app.lifespan,
)


@app.middleware("http")
async def normalize_mcp_root_path(request: Request, call_next):
    # Accept both /mcp and /mcp/ without relying on client-side redirect handling.
    if request.scope.get("path") == "/mcp":
        request.scope["path"] = "/mcp/"
    return await call_next(request)


def _build_endpoint_signature(fn: Any) -> inspect.Signature:
    params: list[inspect.Parameter] = []
    signature = inspect.signature(fn)

    for param_name, param in signature.parameters.items():
        if param.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue
        if param.annotation is Context:
            continue

        annotation = (
            param.annotation if param.annotation is not inspect.Parameter.empty else Any
        )
        if param.default is inspect.Parameter.empty:
            query_default = Query(...)
        else:
            query_default = Query(param.default)

        params.append(
            inspect.Parameter(
                name=param_name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=query_default,
                annotation=annotation,
            )
        )

    return inspect.Signature(parameters=params)


def _register_tool_routes() -> None:
    def make_endpoint(fn: Any, tool_name: str, endpoint_signature: inspect.Signature):
        async def endpoint(**kwargs):
            try:
                if inspect.iscoroutinefunction(fn):
                    result = await fn(**kwargs)
                else:
                    result = fn(**kwargs)
            except TypeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid arguments: {e}") from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Tool execution failed: {e}") from e

            return {"tool": tool_name, "result": result, "arguments": kwargs}

        endpoint.__signature__ = endpoint_signature
        return endpoint

    for tool_name, tool_obj in tools.TOOL_LIST.items():
        if not isinstance(tool_obj, FunctionTool):
            continue

        fn = tool_obj.fn
        endpoint_signature = _build_endpoint_signature(fn)
        route_path = f"/tool/{tool_name}"
        summary = f"Run tool: {tool_name}"
        description = inspect.cleandoc(tool_obj.description).replace("\n", "  \n")
        endpoint_name = f"api_tool_{tool_name}"

        endpoint = make_endpoint(fn, tool_name, endpoint_signature)
        endpoint.__name__ = endpoint_name
        app.post(
            route_path,
            tags=["tools"],
            summary=summary,
            description=description,
        )(endpoint)


@app.middleware("http")
async def normalize_mcp_path(request: Request, call_next):
    if request.url.path == "/mcp":
        scope = dict(request.scope)
        scope["path"] = "/mcp/"
        request = Request(scope, request.receive)
    return await call_next(request)


@app.get("/", include_in_schema=False)
async def home(request: Request):
    prompt = await mcp.get_prompt("explore_wikidata")
    prompt_rendered = await prompt.render({"query": "[User Prompt]"})
    prompt_html = markdown(prompt_rendered[0].content.text)

    return templates.TemplateResponse(
        request,
        "docs.html",
        {
            "tools": tools.TOOL_LIST.keys(),
            "prompt": prompt_html,
        },
    )


@app.get("/health", tags=["meta"])
async def health():
    return JSONResponse({"status": "ok"})


_register_tool_routes()
app.mount("/mcp", mcp_app)


if __name__ == "__main__":
    # Run: uv run python main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
