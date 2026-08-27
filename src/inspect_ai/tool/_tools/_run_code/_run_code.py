from collections.abc import Sequence
from typing import Literal

import anyio

from inspect_ai._util.content import (
    Content,
    ContentText,
)
from inspect_ai._util.error import pip_dependency_error
from inspect_ai.model._chat_message import ChatMessageUser
from inspect_ai.model._tokens import count_text_tokens as local_estimate_tokens

from ....model._model import Model, get_model
from ..._tool import Tool, tool
from ..._tool_def import ToolDef
from .._execute import code_viewer
from ._run_code_executor import (
    MontyRunCodeExecutor,
    RunCodeExecutor,
    RunCodeResult,
)

TRUNCATION_MARKER = "..."

PYTHON_TYPES = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
    "null": "None",
}


def _tool_defs(tools: Sequence[Tool] | None) -> list[ToolDef]:
    """Convert allowed tools into ToolDef objects."""
    return [ToolDef(tool) for tool in tools or []]


def _tool_signature(tool_def: ToolDef) -> str:
    """Return a compact signature for an allowlisted tool."""
    parameters = tool_def.parameters
    if parameters is None or parameters.properties is None:
        return f"{tool_def.name}()"

    args: list[str] = []
    required = set(parameters.required or [])

    for name, schema in parameters.properties.items():
        typ = PYTHON_TYPES.get(str(schema.type), "Any")
        default = "" if name in required else f" = {schema.default!r}"
        args.append(f"{name}: {typ}{default}")

    return f"await {tool_def.name}({', '.join(args)})"


def _resolve_executor(
    executor: RunCodeExecutor | Literal["monty"],
    *,
    tool_defs: list[ToolDef],
    max_inner_tool_calls: int | None,
) -> RunCodeExecutor:
    """Resolve a run_code executor name or custom executor."""
    if isinstance(executor, str):
        if executor == "monty":
            try:
                import pydantic_monty  # noqa: F401
            except ImportError:
                raise pip_dependency_error("run_code", ["pydantic-monty"])
            return MontyRunCodeExecutor(
                tool_defs=tool_defs,
                max_inner_tool_calls=max_inner_tool_calls,
            )
        raise ValueError(f"Unknown run_code executor: {executor}")

    return executor


def _format_run_code_result(
    result: RunCodeResult,
    *,
    include_tool_call_trace: bool,
) -> list[Content]:
    """Format a run_code result for the model."""
    output = result.error if result.error else result.output

    content: list[Content] = (
        [ContentText(text=output)] if isinstance(output, str) else list(output)
    )

    if not include_tool_call_trace or not result.inner_tool_call_trace:
        return content

    trace_lines = ["", "Inner tool calls:"]
    for trace_entry in result.inner_tool_call_trace:
        status = "error" if trace_entry.error else "ok"
        trace_lines.append(f"- {trace_entry.name}: {status}")

        if trace_entry.args_preview != "()":
            trace_lines.append(f"  args: {trace_entry.args_preview}")
        if trace_entry.kwargs_preview != "{}":
            trace_lines.append(f"  kwargs: {trace_entry.kwargs_preview}")

        if trace_entry.error:
            trace_lines.append(f"  error: {trace_entry.error}")
        elif trace_entry.result_preview is not None:
            trace_lines.append(f"  result: {trace_entry.result_preview}")

    content.append(ContentText(text="\n".join(trace_lines)))
    return content


async def _fit_fallback_text(
    candidates: list[str], remaining: int, model: Model
) -> str | None:
    """Try candidate strings in order, return the first that fits remaining tokens.

    Candidates ordered from most to least informative — the first
    one whose token count fits the budget wins. Returns None if none fit.
    """
    for candidate in candidates:
        if await model.count_tokens(candidate) <= remaining:
            return candidate
    return None


async def _truncate_text_to_tokens(
    text: str, max_output_tokens: int, model: Model
) -> str:
    """Truncate text to fit within max_tokens, appending a marker."""
    if max_output_tokens <= 0:
        return ""

    suffix = await _fit_fallback_text(
        ["... [truncated: output exceeded token budget]", TRUNCATION_MARKER],
        max_output_tokens,
        model,
    )
    if suffix is None:
        return ""
    suffix_tokens = await model.count_tokens(suffix)

    remaining = max_output_tokens - suffix_tokens

    # non-network estimate used only to pick a starting point for the search
    full_local_tokens = local_estimate_tokens(text)
    ratio = len(text) / full_local_tokens if full_local_tokens else 1
    start = max(0, min(len(text), int(remaining * ratio)))

    start_tokens = await model.count_tokens(text[:start])

    if start_tokens <= remaining:
        lo, hi = start, len(text)
        best = start
    else:
        lo, hi = 0, start
        best = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        candidate_tokens = await model.count_tokens(text[:mid])
        if candidate_tokens <= remaining:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    result = text[:best] + suffix

    # boundary effect guard: count(prefix) + count(suffix) isn't guaranteed to
    # equal count(prefix + suffix) — a BPE merge can occur across the join.
    while best > 0 and await model.count_tokens(result) > max_output_tokens:
        best -= 1
        result = text[:best] + suffix
    return result


async def _truncate_content(
    content: list[Content], max_tokens: int | None
) -> list[Content]:
    """Truncate content to fit within a token budget.

    Processes items in consecutive order.
    Once an item doesn't fit, it is truncated or replaced with a fallback,
    and no subsequent items are included.

    """
    if max_tokens is None:
        return content

    model = get_model()
    result: list[Content] = []
    remaining = max_tokens

    for item in content:
        if remaining <= 0:
            break

        if isinstance(item, ContentText):
            token_cost = await model.count_tokens(item.text)
            if token_cost <= remaining:
                result.append(item)
                remaining -= token_cost
            else:
                text = await _truncate_text_to_tokens(item.text, remaining, model)
                if text:
                    result.append(ContentText(text=text))
                remaining = 0
        else:
            token_cost = await model.count_tokens([ChatMessageUser(content=[item])])
            if token_cost <= remaining:
                result.append(item)
                remaining -= token_cost
            else:
                placeholder_text = (
                    f"[{type(item).__name__} omitted: exceeded token budget]"
                )
                fallback = await _fit_fallback_text(
                    [placeholder_text, TRUNCATION_MARKER], remaining, model
                )
                if fallback is None:
                    break
                result.append(ContentText(text=fallback))
                remaining = 0
    return result


def _tool_interface_description(tool_defs: list[ToolDef]) -> str:
    """Describe the tools that will eventually be callable from run_code."""
    if not tool_defs:
        return (
            "No inner tools are currently available. "
            "The code can only use the Python execution environment."
        )

    lines = [
        "The code may call the following allowlisted tools as async functions.",
        "Use `await` when calling them:",
        "",
    ]

    for tool_def in tool_defs:
        lines.append(f"- `{_tool_signature(tool_def)}`: {tool_def.description}")

    return "\n".join(lines)


def _validate_tool_names(tool_defs: list[ToolDef]) -> None:
    """Check that allowlisted tools have distinct names.

    Each tool becomes an external function in the runtime namespace, so
    duplicate names would silently shadow one another.

    Raises:
        ValueError: If more than one allowlisted tool has the same name.
    """
    seen: set[str] = set()

    for tool_def in tool_defs:
        if tool_def.name in seen:
            raise ValueError(f"Duplicate run_code inner tool name: {tool_def.name}")
        seen.add(tool_def.name)


def _run_code_usage_description(tool_defs: list[ToolDef]) -> str:
    """Return model-facing instructions for using run_code."""
    lines = [
        "Write Python code to solve the task.",
        "The code is executed by Pydantic Monty, which supports only a restricted Python subset. Do not define classes.",
        "Use ordinary functions, variables, loops, conditionals, comprehensions, and async/await.",
        "Only a limited set of standard-library imports is available, such as asyncio, json, re, math, and datetime. Tool calls must be awaited.",
        "The final expression is returned as the run_code result.",
        "",
        "Tool results may contain non-text content (images, audio, video, documents) as opaque "
        "binary/media objects, not strings:",
        "- If a tool returns a list mixing text and media content, INDEX INTO the list "
        "(e.g. result[1]) to extract the specific item before passing it to another tool "
        "- Do NOT use str(), repr(), f-strings, or manual byte-level parsing/decoding on them.",
        "- Pass or return them as-is (directly, or inside plain lists/dicts) so they reach the "
        "final result as real media content.",
        "that expects a single typed Content object — do not pass the whole list.",
        "",
    ]

    if tool_defs:
        lines.extend(
            [
                "You may call the tools listed below, but ONLY from within the Python code passed to run_code.",
                "Do NOT call them directly as regular tools — they are only available inside the Python execution environment.",
                "",
                "Use `await` when calling these tools, or `asyncio.gather(...)` to run multiple calls concurrently.",
                "",
                "Example:",
                "```python",
                "import asyncio",
                "",
                "results = await asyncio.gather(",
                '    tool_name(arg="value"),',
                '    another_tool(arg="value"),',
                ")",
                "results",
                "```",
                "",
                "",
                "Example — indexing into a mixed list before passing to a typed tool:",
                "```python",
                "screenshot = await fake_screenshot()",
                "# screenshot is [ContentText, ContentImage] — index to get the image alone",
                "metadata = await image_metadata(screenshot[1])",
                "```",
                _tool_interface_description(tool_defs),
            ]
        )
    else:
        lines.extend(
            [
                "No inner tools are available.",
                "The code can only use the Python execution environment.",
            ]
        )

    return "\n".join(lines)


@tool(viewer=code_viewer("python", "code"))
def run_code(
    tools: Sequence[Tool] | None = None,
    timeout: float | None = None,
    executor: RunCodeExecutor | Literal["monty"] = "monty",
    max_inner_tool_calls: int | None = None,
    include_tool_call_trace: bool = False,
    max_output_tokens: int | None = None,
) -> Tool:
    """Run Python code that can orchestrate selected tools.

    Args:
        tools: Tools that code executed by run_code may call.
        timeout: Maximum execution time in seconds.
        executor: Executor used to run code. Use "monty" for the Pydantic Monty-backed executor,
            or pass a custom RunCodeExecutor for tests / alternative backends.
        max_inner_tool_calls: Maximum number of allowlisted tool calls from inside run_code.
        include_tool_call_trace: Whether to include a compact trace of inner tool calls in the result.
        max_output_tokens: Maximum number of tokens returned by run_code. If None, output is not truncated.
    """
    tool_defs = _tool_defs(tools)
    _validate_tool_names(tool_defs)
    usage_description = _run_code_usage_description(tool_defs)
    executor = _resolve_executor(
        executor,
        tool_defs=tool_defs,
        max_inner_tool_calls=max_inner_tool_calls,
    )

    async def execute(code: str) -> list[Content]:
        """Run Python code.

        Args:
            code: Python code to execute.
        """
        try:
            with anyio.fail_after(timeout):
                result = await executor.execute(code)
        except TimeoutError:
            return [
                ContentText(
                    text=f"run_code execution timed out after {timeout} seconds."
                )
            ]

        formatted = _format_run_code_result(
            result,
            include_tool_call_trace=include_tool_call_trace,
        )

        return await _truncate_content(formatted, max_output_tokens)

    return ToolDef(
        execute,
        name="run_code",
        description=(
            "Run Python code that can orchestrate selected tools.\n\n"
            f"{usage_description}"
        ),
    ).as_tool()
