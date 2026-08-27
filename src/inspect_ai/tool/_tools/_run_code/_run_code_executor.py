from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, get_args

from pydantic import TypeAdapter, ValidationError

from inspect_ai._util.content import (
    Content,
    ContentAudio,
    ContentDocument,
    ContentImage,
    ContentText,
    ContentVideo,
)
from inspect_ai._util.error import pip_dependency_error

from ..._tool import ToolError
from ..._tool_def import ToolDef
from ._bridge import (
    RunCodeInnerToolCallTraceEntry,
    RunCodeToolBridge,
)

_content_item_adapter: TypeAdapter[Content] = TypeAdapter(Content)

_content_classes: tuple[type, ...] = get_args(Content)

_content_type_to_class: dict[str, type[Content]] = {
    "text": ContentText,
    "image": ContentImage,
    "audio": ContentAudio,
    "video": ContentVideo,
    "document": ContentDocument,
}


def _looks_like_content(value: dict[str, Any]) -> bool:
    content_type = value.get("type")
    if not isinstance(content_type, str):
        return False
    content_cls = _content_type_to_class.get(content_type)
    if content_cls is None:
        return False
    try:
        content_cls.model_validate(value)
    except ValidationError:
        return False
    return True


_CONTENT_PLACEHOLDER = "<content:{index}>"


def _extract_content(value: Any, extracted: list[Content]) -> Any:
    """Walk a raw (still-serialized) value, pulling out any Content-shaped dicts into `extracted` and leaving a placeholder string in their place.

    Returns a JSON-serializable skeleton with placeholders substituted for
    any Content found at any depth.
    """
    if isinstance(value, dict) and _looks_like_content(value):
        item = _content_item_adapter.validate_python(value)
        placeholder = _CONTENT_PLACEHOLDER.format(index=len(extracted))
        extracted.append(item)
        return placeholder
    if isinstance(value, (list, tuple)):
        return [_extract_content(v, extracted) for v in value]
    if isinstance(value, dict):
        return {k: _extract_content(v, extracted) for k, v in value.items()}
    return value


def _reconstruct_content(value: Any) -> list[Content]:
    """Restore a run_code return value into model-safe Content blocks.

    Three shapes, in order of preference:
      - The whole value is a single serialized Content dict -> Content.
      - The whole value is a flat list of serialized Content dicts -> that
        list, reconstructed in place.
      - Anything else (scalars, structured data, or Content nested/mixed
        inside a larger structure (e.g. an asyncio.gather result mixing
        text, numbers, and Content)) is rendered as JSON text, with any
        Content found at any depth extracted out and appended as separate,
        real Content blocks, so images/documents never end up
        base64-embedded in text the model has to parse itself. The text
        carries a `<content:N>` placeholder at each extraction point.
    """
    if isinstance(value, dict) and _looks_like_content(value):
        return [_content_item_adapter.validate_python(value)]

    if (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(isinstance(v, dict) and _looks_like_content(v) for v in value)
    ):
        return [_content_item_adapter.validate_python(v) for v in value]

    extracted: list[Content] = []
    skeleton = _extract_content(value, extracted)
    text = skeleton if isinstance(skeleton, str) else json.dumps(skeleton)
    return [ContentText(text=text), *extracted]


@dataclass
class RunCodeResult:
    """Result of a run_code execution."""

    output: list[Content]
    error: str | None = None
    inner_tool_call_trace: list[RunCodeInnerToolCallTraceEntry] = field(
        default_factory=list
    )


class RunCodeExecutor(Protocol):
    """Executor for run_code."""

    async def execute(self, code: str) -> RunCodeResult:
        """Execute code.

        Args:
            code: Python code to execute.

        Returns:
            Result of the code execution.
        """
        ...


class MontyRunCodeExecutor:
    """Run code using Pydantic Monty."""

    def __init__(
        self,
        tool_defs: list[ToolDef] | None = None,
        *,
        max_inner_tool_calls: int | None = None,
    ) -> None:
        self.tool_defs = tool_defs or []
        self.max_tool_calls = max_inner_tool_calls

    async def execute(self, code: str) -> RunCodeResult:
        bridge = RunCodeToolBridge(
            self.tool_defs,
            max_inner_tool_calls=self.max_tool_calls,
        )

        try:
            result = await self._run(bridge, code)
        except BaseException:
            # Monty rebuilds an inner tool's exception from its own type list, so
            # a terminate reaches this point as a MontyError.
            bridge.raise_pending_control_flow()
            raise

        # the generated code can catch a terminate with a bare except, in which
        # case the run "succeeds" and this is the only place the signal is left
        bridge.raise_pending_control_flow()
        return result

    async def _run(self, bridge: RunCodeToolBridge, code: str) -> RunCodeResult:
        try:
            import pydantic_monty
            from pydantic_monty import MontyError
        except ImportError:
            raise pip_dependency_error("run_code", ["pydantic-monty"])

        try:
            monty = pydantic_monty.Monty(
                code,
                script_name="run_code.py",
                type_check=False,
            )
            output = await monty.run_async(
                external_functions=bridge.external_functions(),
            )

            contents: list[Content] = []

            if output is not None:
                contents.extend(_reconstruct_content(output))
            return RunCodeResult(
                output=contents,
                inner_tool_call_trace=bridge.call_trace,
            )
        except MontyError as exc:
            raise ToolError(str(exc))
        except Exception as exc:
            return RunCodeResult(
                output=[],
                error=str(exc),
                inner_tool_call_trace=bridge.call_trace,
            )
