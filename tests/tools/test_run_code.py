import json

import anyio
import pytest
from test_helpers.utils import skip_if_trio

from inspect_ai._util.content import ContentDocument, ContentImage, ContentText
from inspect_ai._util.exception import TerminateSampleError
from inspect_ai._util.registry import registry_info
from inspect_ai.approval import ApprovalPolicy, approval, auto_approver
from inspect_ai.model._chat_message import ChatMessage
from inspect_ai.tool import Tool, ToolDef, ToolError, run_code
from inspect_ai.tool._tool_call import ToolCall
from inspect_ai.tool._tools._run_code._bridge import (
    RunCodeInnerToolCallTraceEntry,
    RunCodeMaxToolCallsExceededError,
    RunCodeToolBridge,
    _content_to_runtime_value,
    _preview,
    _tool_call_arguments,
)
from inspect_ai.tool._tools._run_code._run_code import (
    TRUNCATION_MARKER,
    _format_run_code_result,
    _run_code_usage_description,
    _tool_defs,
    _tool_interface_description,
    _tool_signature,
    _validate_tool_names,
)
from inspect_ai.tool._tools._run_code._run_code_executor import (
    RunCodeResult,
    _looks_like_content,
    _reconstruct_content,
)


def test_run_code_tool_constructs():
    tool = run_code()
    assert callable(tool)


def test_run_code_tool_def_has_name():
    tool_def = ToolDef(run_code())
    assert tool_def.name == "run_code"


def test_run_code_accepts_empty_tools_list():
    tool = run_code(tools=[])
    assert callable(tool)


def test_run_code_viewer_renders_code():
    viewer = registry_info(run_code()).metadata["viewer"]

    view = viewer(ToolCall(id="1", function="run_code", arguments={"code": "1 + 1"}))

    assert view.call is not None
    assert "```python\n1 + 1\n```" in view.call.content


def dummy_tool() -> Tool:
    async def execute(value: str) -> str:
        """Echo a value.

        Args:
            value: Value to echo.
        """
        return value

    return ToolDef(
        execute,
        name="dummy_tool",
        description="Echo a value.",
    ).as_tool()


def test_run_code_accepts_wrapped_tools():
    tool = run_code(tools=[dummy_tool()])
    assert callable(tool)


def test_run_code_normalizes_wrapped_tools():
    tool_defs = _tool_defs([dummy_tool()])
    assert len(tool_defs) == 1
    assert tool_defs[0].name == "dummy_tool"


def test_tool_signature_includes_parameter_schema():
    tool_defs = _tool_defs([dummy_tool()])

    signature = _tool_signature(tool_defs[0])

    assert signature == "await dummy_tool(value: str)"


def test_tool_interface_description_without_tools():
    description = _tool_interface_description([])

    assert "No inner tools" in description


def test_tool_interface_description_with_tool():
    tool_defs = _tool_defs([dummy_tool()])

    description = _tool_interface_description(tool_defs)

    assert "await dummy_tool(value: str)" in description
    assert "Echo a value." in description


def test_run_code_description_mentions_wrapped_tool():
    tool = run_code(tools=[dummy_tool()])
    tool_def = ToolDef(tool)

    assert "Use `await`" in tool_def.description
    assert "await dummy_tool(value: str)" in tool_def.description
    assert "Echo a value." in tool_def.description


class FakeRunCodeExecutor:
    async def execute(self, code: str) -> RunCodeResult:
        return RunCodeResult(output=[ContentText(text=f"executed: {code}")])


@pytest.mark.anyio
async def test_run_code_uses_injected_executor():
    tool = run_code(executor=FakeRunCodeExecutor())

    result = await tool(code="x = 1")

    assert result == [ContentText(text="executed: x = 1")]


class ErrorRunCodeExecutor:
    async def execute(self, code: str) -> RunCodeResult:
        return RunCodeResult(output=[ContentText(text="")], error="boom")


@pytest.mark.anyio
async def test_run_code_returns_executor_error():
    tool = run_code(executor=ErrorRunCodeExecutor())

    result = await tool(code="raise Exception()")

    assert result == [ContentText(text="boom")]


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_executes_simple_code_with_monty():
    pytest.importorskip("pydantic_monty")

    tool = run_code(executor="monty")
    result = await tool(code="1 + 1")

    assert "2" in result[0].text


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_raises_tool_error_on_monty_syntax_error():
    pytest.importorskip("pydantic_monty")

    tool = run_code(executor="monty")

    with pytest.raises(ToolError) as exc_info:
        await tool(code="def foo(:")

    assert "parameter" in str(exc_info.value).lower()


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_raises_tool_error_on_monty_runtime_error():
    pytest.importorskip("pydantic_monty")

    tool = run_code(executor="monty")

    with pytest.raises(ToolError) as exc_info:
        await tool(code="1/0")

    assert "ZeroDivisionError" in str(exc_info.value)


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_returns_falsy_results():
    pytest.importorskip("pydantic_monty")

    tool = run_code(executor="monty")

    for code, expected in [
        ("0", "0"),
        ("1 - 1", "0"),
        ("False", "false"),
        ("[]", "[]"),
        ("{}", "{}"),
    ]:
        result = await tool(code=code)
        assert result
        assert result[0].text == expected


@pytest.mark.anyio
async def test_external_functions_call_wrapped_tool():
    tool_defs = _tool_defs([dummy_tool()])
    bridge = RunCodeToolBridge(tool_defs)
    external_functions = bridge.external_functions()

    assert "dummy_tool" in external_functions

    result = await external_functions["dummy_tool"]("hello")

    assert result == "hello"


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_can_call_wrapped_tool_with_monty():
    pytest.importorskip("pydantic_monty")

    tool = run_code(tools=[dummy_tool()], executor="monty")

    result = await tool(code='await dummy_tool("hello")')

    print(result)

    assert "hello" in result[0].text


@pytest.mark.anyio
async def test_run_code_bridge_records_inner_tool_call():
    bridge = RunCodeToolBridge(_tool_defs([dummy_tool()]))

    external_functions = bridge.external_functions()
    result = await external_functions["dummy_tool"]("hello")

    assert result == "hello"
    assert len(bridge.call_trace) == 1
    assert bridge.call_trace[0].name == "dummy_tool"
    assert bridge.call_trace[0].args_preview == "('hello',)"
    assert bridge.call_trace[0].kwargs_preview == "{}"
    assert "hello" in bridge.call_trace[0].result_preview
    assert bridge.call_trace[0].error is None


@pytest.mark.anyio
async def test_run_code_bridge_enforces_max_tool_calls():
    bridge = RunCodeToolBridge(
        _tool_defs([dummy_tool()]),
        max_inner_tool_calls=1,
    )

    external_functions = bridge.external_functions()
    result = await external_functions["dummy_tool"]("first")
    assert result == "first"

    with pytest.raises(
        RuntimeError, match="Maximum run_code inner tool calls exceeded"
    ):
        await external_functions["dummy_tool"]("second")

    assert len(bridge.call_trace) == 1


def failing_tool() -> Tool:
    async def execute(value: str) -> str:
        """Fail.

        Args:
            value: Ignored value.
        """
        raise RuntimeError("inner boom")

    return ToolDef(
        execute,
        name="failing_tool",
        description="Always fails.",
    ).as_tool()


@pytest.mark.anyio
async def test_run_code_bridge_records_inner_tool_error():
    bridge = RunCodeToolBridge(_tool_defs([failing_tool()]))

    external_functions = bridge.external_functions()

    with pytest.raises(RuntimeError, match="inner boom"):
        await external_functions["failing_tool"]("x")

    assert len(bridge.call_trace) == 1
    assert bridge.call_trace[0].name == "failing_tool"
    assert bridge.call_trace[0].error == "inner boom"


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_monty_raises_tool_error_on_max_tool_calls():
    pytest.importorskip("pydantic_monty")

    tool = run_code(
        tools=[dummy_tool()],
        executor="monty",
        max_inner_tool_calls=1,
    )

    with pytest.raises(ToolError) as exc_info:
        await tool(
            code="""
await dummy_tool("first")
await dummy_tool("second")
"""
        )

    assert "Maximum run_code inner tool calls exceeded" in str(exc_info.value)


def test_format_run_code_result_without_trace():
    result = RunCodeResult(
        output=[ContentText(text="hello")],
        inner_tool_call_trace=[
            RunCodeInnerToolCallTraceEntry(
                name="dummy_tool",
                args_preview="('x',)",
                kwargs_preview="{}",
                result_preview="'x'",
            )
        ],
    )

    formatted = _format_run_code_result(
        result,
        include_tool_call_trace=False,
    )

    assert formatted == [ContentText(text="hello")]


def test_format_run_code_result_with_trace():
    result = RunCodeResult(
        output=[ContentText(text="hello")],
        inner_tool_call_trace=[
            RunCodeInnerToolCallTraceEntry(
                name="dummy_tool",
                args_preview="('x',)",
                kwargs_preview="{}",
                result_preview="'x'",
            )
        ],
    )

    formatted = _format_run_code_result(
        result,
        include_tool_call_trace=True,
    )

    assert "hello" in formatted[0].text
    assert "Inner tool calls:" in formatted[1].text
    assert "- dummy_tool: ok" in formatted[1].text


def test_format_run_code_result_with_error_trace():
    result = RunCodeResult(
        output=[],
        error="boom",
        inner_tool_call_trace=[
            RunCodeInnerToolCallTraceEntry(name="dummy_tool", error="inner boom")
        ],
    )

    formatted = _format_run_code_result(
        result,
        include_tool_call_trace=True,
    )

    assert "boom" in formatted[0].text
    assert "- dummy_tool: error" in formatted[1].text
    assert "inner boom" in formatted[1].text


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_can_include_inner_tool_trace_with_monty():
    pytest.importorskip("pydantic_monty")

    tool = run_code(
        tools=[dummy_tool()],
        executor="monty",
        include_tool_call_trace=True,
    )

    result = await tool(code='await dummy_tool("hello")')

    assert "hello" in result[0].text
    assert "Inner tool calls:" in result[1].text
    assert "- dummy_tool: ok" in result[1].text


class SlowRunCodeExecutor:
    async def execute(self, code: str) -> RunCodeResult:
        await anyio.sleep(1)
        return RunCodeResult(output=[ContentText(text="finished")])


@pytest.mark.anyio
async def test_run_code_enforces_timeout():
    tool = run_code(
        executor=SlowRunCodeExecutor(),
        timeout=0.01,
    )

    result = await tool(code="slow")

    assert "timed out" in result[0].text


SHORT_TEXT = "short test text"
LONG_TEXT = " ".join([SHORT_TEXT] * 100)
MOCK_BASE64_IMAGE = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8"
    "AAAAASUVORK5CYII="
)
chunk = "AAAA"
LARGE_PADDING = chunk * 250_000
LARGE_MOCK_BASE64_IMAGE = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8"
    "AAAAASUVORK5CYII=" + LARGE_PADDING
)


class LargeOutputRunCodeExecutor:
    async def execute(self, code: str) -> RunCodeResult:
        return RunCodeResult(output=[ContentText(text=LONG_TEXT)])


class LargeContentBlocksRunCodeExecutor:
    async def execute(self, code: str) -> RunCodeResult:
        return RunCodeResult(
            output=[ContentText(text=SHORT_TEXT), ContentText(text=LONG_TEXT)]
        )


class MixedContentRunCodeExecutor:
    async def execute(self, code: str) -> RunCodeResult:
        return RunCodeResult(
            output=[ContentText(text=SHORT_TEXT), ContentImage(image=MOCK_BASE64_IMAGE)]
        )


class LargeMixedContentRunCodeExecutor:
    async def execute(self, code: str) -> RunCodeResult:
        return RunCodeResult(
            output=[
                ContentText(text=LONG_TEXT),
                ContentImage(image=LARGE_MOCK_BASE64_IMAGE),
            ]
        )


class FakeModel:
    """Deterministic stand-in for count_tokens, no network/API calls."""

    async def count_tokens(self, input: str | list[ChatMessage]) -> int:
        if isinstance(input, str):
            return len(input.split())
        # non-text content branch — content itself doesn't matter for
        # truncation-of-text tests, just needs to be deterministic
        return 20


@pytest.fixture
def fake_model_get_model(monkeypatch: pytest.MonkeyPatch) -> FakeModel:
    model = FakeModel()
    monkeypatch.setattr(
        "inspect_ai.tool._tools._run_code._run_code.get_model",
        lambda *args, **kwargs: model,
    )
    return model


@pytest.mark.anyio
async def test_run_code_truncates_output(fake_model_get_model):
    max_tokens = 20
    tool = run_code(
        executor=LargeOutputRunCodeExecutor(),
        max_output_tokens=max_tokens,
    )

    result = await tool(code="large")
    output_text = result[0].text

    truncated_tokens = await fake_model_get_model.count_tokens(output_text)

    assert truncated_tokens <= max_tokens
    assert TRUNCATION_MARKER in output_text
    assert output_text.startswith(SHORT_TEXT.split()[0])


@pytest.mark.anyio
async def test_run_code_does_not_truncate_output_by_default():
    tool = run_code(
        executor=LargeOutputRunCodeExecutor(),
        max_output_tokens=None,
    )

    result = await tool(code="ignored")
    output_text = result[0].text

    assert output_text == LONG_TEXT


@pytest.mark.anyio
async def test_run_code_content_fits_budget_no_truncation(fake_model_get_model):
    max_tokens = len(LONG_TEXT.split())
    tool = run_code(
        executor=LargeOutputRunCodeExecutor(),
        max_output_tokens=max_tokens,
    )

    result = await tool(code="large")
    output_text = result[0].text

    truncated_tokens = await fake_model_get_model.count_tokens(output_text)

    assert truncated_tokens == max_tokens
    assert output_text == LONG_TEXT


@pytest.mark.anyio
async def test_truncate_text_content_stays_within_limit_across_text_blocks(
    fake_model_get_model,
):
    max_tokens = 50
    tool = run_code(
        executor=LargeContentBlocksRunCodeExecutor(),
        max_output_tokens=max_tokens,
    )

    result = await tool(code="large_blocks")
    total_tokens = 0

    assert len(result) == 2

    for i, item in enumerate(result):
        truncated_tokens = await fake_model_get_model.count_tokens(item.text)
        total_tokens += truncated_tokens

        if i == len(result) - 1:
            assert TRUNCATION_MARKER in item.text
        else:
            assert TRUNCATION_MARKER not in item.text

        assert total_tokens <= max_tokens


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_model_get_model")
async def test_truncate_content_keeps_image_that_fits():
    max_tokens = 3 + 20
    tool = run_code(
        executor=MixedContentRunCodeExecutor(),
        max_output_tokens=max_tokens,
    )

    result = await tool(code="mixed")

    assert len(result) == 2
    assert isinstance(result[0], ContentText) and result[0].text == SHORT_TEXT
    assert isinstance(result[1], ContentImage) and result[1].image == MOCK_BASE64_IMAGE


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_model_get_model")
async def test_truncate_content_replaces_image_with_placeholder():
    max_tokens = 3 + 10
    tool = run_code(
        executor=MixedContentRunCodeExecutor(),
        max_output_tokens=max_tokens,
    )

    result = await tool(code="mixed")

    assert len(result) == 2
    assert isinstance(result[1], ContentText)
    assert "exceeded token budget" in result[1].text


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_model_get_model")
async def test_truncate_content_replaces_image_with_fallback():
    max_tokens = 5
    tool = run_code(
        executor=MixedContentRunCodeExecutor(),
        max_output_tokens=max_tokens,
    )

    result = await tool(code="mixed")

    assert len(result) == 2
    assert isinstance(result[1], ContentText)
    assert TRUNCATION_MARKER in result[1].text


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_model_get_model")
async def test_truncate_content_drops_item_when_budget_exhausted():
    max_tokens = 10
    tool = run_code(
        executor=LargeMixedContentRunCodeExecutor(),
        max_output_tokens=max_tokens,
    )

    result = await tool(code="large_mixed")

    assert len(result) == 1


def test_run_code_preview_truncates_long_values():
    preview = _preview("x" * 100, max_chars=30)

    assert len(preview) <= 30
    assert "truncated" in preview


def test_run_code_preview_handles_bad_repr():
    class BadRepr:
        def __repr__(self) -> str:
            raise RuntimeError("bad repr")

    preview = _preview(BadRepr())

    assert "unrepresentable" in preview


def test_run_code_rejects_duplicate_wrapped_tool_names():
    tool_defs = _tool_defs([dummy_tool(), dummy_tool()])

    with pytest.raises(ValueError, match="Duplicate run_code inner tool name"):
        _validate_tool_names(tool_defs)


def test_run_code_rejects_duplicate_wrapped_tool_names_at_construction():
    with pytest.raises(ValueError, match="Duplicate run_code inner tool name"):
        run_code(tools=[dummy_tool(), dummy_tool()])


def test_run_code_usage_description_without_tools():
    description = _run_code_usage_description([])

    assert "Write Python code" in description
    assert "No inner tools are available" in description


def test_run_code_usage_description_with_tools_mentions_await():
    tool_defs = _tool_defs([dummy_tool()])

    description = _run_code_usage_description(tool_defs)

    assert "Use `await`" in description
    assert "await dummy_tool(value: str)" in description
    assert "Echo a value." in description


def second_dummy_tool() -> Tool:
    async def execute(value: str) -> str:
        """Echo a second value.

        Args:
            value: Value to echo.
        """
        return f"second:{value}"

    return ToolDef(
        execute,
        name="second_dummy_tool",
        description="Echo a second value.",
    ).as_tool()


@pytest.mark.anyio
async def test_run_code_bridge_can_call_multiple_wrapped_tools():
    bridge = RunCodeToolBridge(_tool_defs([dummy_tool(), second_dummy_tool()]))

    external_functions = bridge.external_functions()

    result_1 = await external_functions["dummy_tool"]("a")
    result_2 = await external_functions["second_dummy_tool"]("b")

    assert result_1 == "a"
    assert result_2 == "second:b"

    assert len(bridge.call_trace) == 2
    assert bridge.call_trace[0].name == "dummy_tool"
    assert bridge.call_trace[1].name == "second_dummy_tool"


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_can_call_multiple_wrapped_tools_with_monty():
    pytest.importorskip("pydantic_monty")

    tool = run_code(
        tools=[dummy_tool(), second_dummy_tool()],
        executor="monty",
    )

    result = await tool(
        code="""
a = await dummy_tool("x")
b = await second_dummy_tool("y")
[a, b]
"""
    )

    assert "x" in result[0].text
    assert "second:y" in result[0].text


def add_numbers_tool() -> Tool:
    async def execute(a: int, b: int) -> int:
        """Add two integers.

        Args:
            a: First integer.
            b: Second integer.
        """
        return a + b

    return ToolDef(
        execute,
        name="add_numbers",
        description="Add two integers.",
    ).as_tool()


@pytest.mark.anyio
async def test_external_functions_preserve_scalar_return_type():
    bridge = RunCodeToolBridge(_tool_defs([add_numbers_tool()]))
    external_functions = bridge.external_functions()

    result = await external_functions["add_numbers"](2, 3)

    assert result == 5
    assert isinstance(result, int)


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_chains_typed_tool_results_with_monty():
    pytest.importorskip("pydantic_monty")

    # the first result feeds the second call; the int must survive the
    # Monty boundary so the second call validates against the int schema
    tool = run_code(tools=[add_numbers_tool()], executor="monty")

    result = await tool(
        code="""
x = await add_numbers(a=1, b=2)
await add_numbers(a=x, b=40)
"""
    )

    assert result[0].text == "43"


def list_channels_tool() -> Tool:
    async def execute() -> list:
        """List channel names.

        Returns:
            Channel names.
        """
        return ["general", "random", "engineering"]

    return ToolDef(
        execute, name="list_channels", description="List channels."
    ).as_tool()


def transactions_tool() -> Tool:
    async def execute() -> list:
        """List transactions.

        Returns:
            Transactions.
        """
        return [{"id": 1, "amount": 10}, {"id": 2, "amount": 32}]

    return ToolDef(
        execute, name="get_transactions", description="List transactions."
    ).as_tool()


@pytest.mark.anyio
async def test_external_functions_preserve_structured_return_type():
    bridge = RunCodeToolBridge(_tool_defs([list_channels_tool()]))
    external_functions = bridge.external_functions()

    result = await external_functions["list_channels"]()

    assert result == ["general", "random", "engineering"]
    assert isinstance(result, list)


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_iterates_structured_tool_result_with_monty():
    pytest.importorskip("pydantic_monty")

    tool = run_code(tools=[transactions_tool()], executor="monty")

    result = await tool(
        code="""
txs = await get_transactions()
total = sum(t["amount"] for t in txs)
f"{len(txs)} txs total={total}"
"""
    )

    assert result[0].text == "2 txs total=42"


def test_project_result_raises_on_unprojectable_value():
    # Neither scalar, Content, nor JSON-serializable: raise instead of
    # degrading to text.
    bridge = RunCodeToolBridge([])
    circular: dict = {}
    circular["self"] = circular

    with pytest.raises(ToolError):
        bridge._project_result(circular, "demo_tool")


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_can_call_wrapped_tools_with_asyncio_gather():
    pytest.importorskip("pydantic_monty")

    tool = run_code(
        tools=[dummy_tool(), second_dummy_tool()],
        executor="monty",
        include_tool_call_trace=True,
    )

    result = await tool(
        code="""
import asyncio

results = await asyncio.gather(
    dummy_tool("x"),
    second_dummy_tool("y"),
)
results
"""
    )

    assert "x" in result[0].text
    assert "second:y" in result[0].text
    assert "Inner tool calls:" in result[1].text
    assert "dummy_tool" in result[1].text
    assert "second_dummy_tool" in result[1].text


def test_run_code_usage_description_mentions_asyncio_gather():
    tool_defs = _tool_defs([dummy_tool()])
    description = _run_code_usage_description(tool_defs)

    assert "asyncio.gather" in description


def typed_count_tool(calls: list[int]) -> Tool:
    async def execute(count: int) -> str:
        """Echo a count.

        Args:
            count: Count to echo.
        """
        calls.append(count)
        return f"count:{count}"

    return ToolDef(
        execute,
        name="typed_count_tool",
        description="Echo a typed count.",
    ).as_tool()


@pytest.mark.anyio
async def test_run_code_bridge_uses_inspect_argument_validation():
    calls: list[int] = []
    bridge = RunCodeToolBridge(_tool_defs([typed_count_tool(calls)]))

    external_functions = bridge.external_functions()
    result = await external_functions["typed_count_tool"]("not-an-int")

    assert calls == []
    assert isinstance(result, str)
    assert result
    assert "validation errors parsing tool input arguments" in result
    assert "not-an-int" in result
    assert "integer" in result


def tool_error_tool() -> Tool:
    async def execute(value: str) -> str:
        """Raise a ToolError.

        Args:
            value: Ignored value.
        """
        raise ToolError("bad inner input")

    return ToolDef(
        execute,
        name="tool_error_tool",
        description="Always raises ToolError.",
    ).as_tool()


@pytest.mark.anyio
async def test_run_code_bridge_surfaces_inner_tool_error():
    bridge = RunCodeToolBridge(_tool_defs([tool_error_tool()]))

    external_functions = bridge.external_functions()
    result = await external_functions["tool_error_tool"]("x")

    assert isinstance(result, str)
    assert "bad inner input" in result


@pytest.mark.anyio
async def test_run_code_bridge_uses_inspect_approval_for_inner_tool_calls():
    calls: list[str] = []

    def approval_probe_tool() -> Tool:
        async def execute(value: str) -> str:
            """Record a value.

            Args:
                value: Value to record.
            """
            calls.append(value)
            return f"approved:{value}"

        return ToolDef(
            execute,
            name="approval_probe_tool",
            description="Tool used to test approval.",
        ).as_tool()

    bridge = RunCodeToolBridge(_tool_defs([approval_probe_tool()]))

    external_functions = bridge.external_functions()

    with approval(
        [
            ApprovalPolicy(
                approver=auto_approver(decision="reject"),
                tools="approval_probe_tool",
            )
        ]
    ):
        result = await external_functions["approval_probe_tool"]("secret")

    assert calls == []
    assert isinstance(result, str)
    assert "approval: Automatic decision." in result


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_monty_uses_inspect_approval_for_inner_tool_calls():
    pytest.importorskip("pydantic_monty")

    calls: list[str] = []

    def approval_probe_tool() -> Tool:
        async def execute(value: str) -> str:
            """Record a value.

            Args:
                value: Value to record.
            """
            calls.append(value)
            return f"approved:{value}"

        return ToolDef(
            execute,
            name="approval_probe_tool",
            description="Tool used to test approval.",
        ).as_tool()

    tool = run_code(
        tools=[approval_probe_tool()],
        executor="monty",
    )

    with approval(
        [
            ApprovalPolicy(
                approver=auto_approver(decision="reject"),
                tools="approval_probe_tool",
            )
        ]
    ):
        result = await tool(code='await approval_probe_tool("secret")')

    assert calls == []
    assert isinstance(result, list)
    assert result[0].text == "approval: Automatic decision."


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_monty_runs_inner_tool_when_approval_allows_it():
    pytest.importorskip("pydantic_monty")

    calls: list[str] = []

    def approval_probe_tool() -> Tool:
        async def execute(value: str) -> str:
            """Record a value.

            Args:
                value: Value to record.
            """
            calls.append(value)
            return f"approved:{value}"

        return ToolDef(
            execute,
            name="approval_probe_tool",
            description="Tool used to test approval.",
        ).as_tool()

    tool = run_code(
        tools=[approval_probe_tool()],
        executor="monty",
    )

    with approval(
        [
            ApprovalPolicy(
                approver=auto_approver(decision="approve"),
                tools="approval_probe_tool",
            )
        ]
    ):
        result = await tool(code='await approval_probe_tool("secret")')

    assert calls == ["secret"]
    assert result[0].text == "approved:secret"


@pytest.mark.anyio
async def test_run_code_bridge_propagates_terminate_sample_error():
    bridge = RunCodeToolBridge(_tool_defs([dummy_tool()]))
    external_functions = bridge.external_functions()

    with approval(
        [
            ApprovalPolicy(
                approver=auto_approver(decision="terminate"),
                tools="dummy_tool",
            )
        ]
    ):
        with pytest.raises(TerminateSampleError):
            await external_functions["dummy_tool"]("secret")


@pytest.mark.anyio
async def test_run_code_bridge_refuses_calls_after_terminate():
    bridge = RunCodeToolBridge(_tool_defs([dummy_tool()]))
    external_functions = bridge.external_functions()

    with approval(
        [
            ApprovalPolicy(
                approver=auto_approver(decision="terminate"),
                tools="dummy_tool",
            )
        ]
    ):
        with pytest.raises(TerminateSampleError):
            await external_functions["dummy_tool"]("secret")

    # the policy is gone, but the terminate still stands
    with pytest.raises(TerminateSampleError):
        await external_functions["dummy_tool"]("again")


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_terminate_survives_swallowing_code():
    pytest.importorskip("pydantic_monty")

    tool = run_code(tools=[dummy_tool()], executor="monty")

    with approval(
        [
            ApprovalPolicy(
                approver=auto_approver(decision="terminate"),
                tools="dummy_tool",
            )
        ]
    ):
        with pytest.raises(TerminateSampleError):
            await tool(
                code=(
                    "try:\n"
                    '    await dummy_tool("secret")\n'
                    '    result = "called"\n'
                    "except Exception:\n"
                    '    result = "swallowed"\n'
                    "result\n"
                )
            )


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_cancellation_survives_swallowing_code():
    pytest.importorskip("pydantic_monty")

    cancelled_exc_class = anyio.get_cancelled_exc_class()

    def cancelling_tool() -> Tool:
        async def execute(value: str) -> str:
            """Cancel the caller.

            Args:
                value: Ignored.
            """
            raise cancelled_exc_class()

        return ToolDef(
            execute,
            name="cancelling_tool",
            description="Cancels the caller.",
        ).as_tool()

    tool = run_code(tools=[cancelling_tool()], executor="monty")

    with pytest.raises(cancelled_exc_class):
        await tool(
            code=(
                "try:\n"
                '    await cancelling_tool("x")\n'
                '    result = "called"\n'
                "except BaseException:\n"
                '    result = "swallowed"\n'
                "result\n"
            )
        )


def file_not_found_tool() -> Tool:
    async def execute(path: str) -> str:
        """Raise FileNotFoundError.

        Args:
            path: Path to open.
        """
        raise FileNotFoundError(2, "No such file or directory", path)

    return ToolDef(
        execute,
        name="file_not_found_tool",
        description="Always raises FileNotFoundError.",
    ).as_tool()


@pytest.mark.anyio
async def test_run_code_bridge_converts_recoverable_tool_errors():
    bridge = RunCodeToolBridge(_tool_defs([file_not_found_tool()]))
    external_functions = bridge.external_functions()

    result = await external_functions["file_not_found_tool"]("missing.txt")

    assert isinstance(result, str)
    assert result.startswith("file_not_found: ")
    assert "missing.txt" in result


@pytest.mark.anyio
async def test_run_code_bridge_raises_custom_error_on_max_tool_calls():
    bridge = RunCodeToolBridge(
        _tool_defs([dummy_tool()]),
        max_inner_tool_calls=1,
    )
    external_functions = bridge.external_functions()

    await external_functions["dummy_tool"]("first")

    with pytest.raises(RunCodeMaxToolCallsExceededError) as exc_info:
        await external_functions["dummy_tool"]("second")

    assert exc_info.value.max_tool_calls == 1
    assert "Maximum run_code inner tool calls exceeded: 1" in str(exc_info.value)


@pytest.mark.anyio
async def test_run_code_bridge_finalizes_inner_tool_event():
    from inspect_ai.event._tool import ToolEvent
    from inspect_ai.log._transcript import transcript

    before = len(transcript().events)

    bridge = RunCodeToolBridge(_tool_defs([add_numbers_tool()]))
    external_functions = bridge.external_functions()

    result = await external_functions["add_numbers"](1, 2)

    assert result == 3

    events = [
        event
        for event in transcript().events[before:]
        if isinstance(event, ToolEvent) and event.function == "add_numbers"
    ]

    assert events

    event = events[-1]
    assert event.pending is not True
    assert event.completed is not None
    assert event.working_time is not None
    assert event.result == "3"


@pytest.mark.anyio
async def test_run_code_bridge_truncates_inner_tool_event_result():
    from inspect_ai.event._tool import ToolEvent
    from inspect_ai.log._transcript import transcript

    def long_output_tool() -> Tool:
        async def execute(value: str) -> str:
            """Return a long string.

            Args:
                value: Value to repeat.
            """
            return value * (32 * 1024)

        return ToolDef(
            execute,
            name="long_output_tool",
            description="Returns a long string.",
        ).as_tool()

    before = len(transcript().events)

    bridge = RunCodeToolBridge(_tool_defs([long_output_tool()]))
    external_functions = bridge.external_functions()

    result = await external_functions["long_output_tool"]("x")

    # the code gets the whole result, the transcript gets a bounded one
    assert len(result) == 32 * 1024

    event = [
        event
        for event in transcript().events[before:]
        if isinstance(event, ToolEvent) and event.function == "long_output_tool"
    ][-1]

    assert event.truncated == (32 * 1024, 16 * 1024)
    assert "too long to be displayed" in event.result


def test_looks_like_content_false_positive_on_bare_type_key():
    value = {"type": "text", "value": 42}
    assert _looks_like_content(value) is False


@pytest.mark.parametrize(
    "value",
    [
        {"type": "text", "text": "hello"},
        {"type": "image", "image": "base64data"},
        {"type": "audio", "audio": "base64data", "format": "wav"},
        {"type": "video", "video": "base64data", "format": "mp4"},
        {"type": "document", "document": "base64data"},
    ],
)
def test_looks_like_content_true_positive(value):
    assert _looks_like_content(value) is True


def test_looks_like_content_false_positive_missing_required_field():
    value = {"type": "audio", "audio": "base64data"}  # missing "format"
    assert _looks_like_content(value) is False


def test_looks_like_content_false_positive_wrong_field_type():
    value = {"type": "text", "text": 12345}
    assert _looks_like_content(value) is False


def test_looks_like_content_unknown_type():
    value = {"type": "something_else", "text": "hello"}
    assert _looks_like_content(value) is False


def test_looks_like_content_document_optional_fields_defaulted():
    value = {"type": "document", "document": "base64data"}
    assert _looks_like_content(value) is True


def test_single_content_text_collapses_to_plain_string():
    content = [ContentText(text="hello")]
    result = _content_to_runtime_value(content, preserve_list_shape=False)
    assert result == "hello"
    assert isinstance(result, str)


def test_multi_content_does_not_collapse():
    content = [ContentText(text="a"), ContentText(text="b")]
    result = _content_to_runtime_value(content, preserve_list_shape=True)
    assert isinstance(result, list)


def test_single_content_text_roundtrips_through_reconstruct():
    content = [ContentText(text="hello")]
    projected = _content_to_runtime_value(content, preserve_list_shape=False)
    reconstructed = _reconstruct_content(projected)
    assert reconstructed == [ContentText(text="hello")]


def test_projected_text_passed_as_str_argument_to_next_tool():
    content = [ContentText(text="hello")]
    projected = _content_to_runtime_value(content, preserve_list_shape=False)

    def next_tool(message: str) -> str:
        return message.upper()

    tool_def = ToolDef(next_tool, name="next_tool")
    args = _tool_call_arguments(tool_def, (projected,), {})
    assert args == {"message": "hello"}


def test_projected_text_remains_plain_string_for_next_tool():
    content = [ContentText(text="hello")]
    projected = _content_to_runtime_value(content, preserve_list_shape=False)

    def next_tool(payload: ContentText) -> str:
        return payload.text

    tool_def = ToolDef(next_tool, name="next_tool")
    args = _tool_call_arguments(tool_def, (projected,), {})
    assert isinstance(args["payload"], str)
    assert not isinstance(args["payload"], ContentText)


def test_single_content_image_with_preserved_list_shape_stays_array():
    content = [ContentImage(image=MOCK_BASE64_IMAGE)]

    result = _content_to_runtime_value(
        content,
        preserve_list_shape=True,
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["type"] == "image"
    assert result[0]["image"] == MOCK_BASE64_IMAGE


def test_single_content_image_is_serialized_as_single_object():
    content = [ContentImage(image=MOCK_BASE64_IMAGE)]

    result = _content_to_runtime_value(
        content,
        preserve_list_shape=False,
    )

    assert isinstance(result, dict)
    assert result["type"] == "image"
    assert result["image"] == MOCK_BASE64_IMAGE


def image_tool() -> Tool:
    async def execute() -> ContentImage:
        """Return an image.

        Returns:
            An image.
        """
        return ContentImage(image=MOCK_BASE64_IMAGE)

    return ToolDef(
        execute,
        name="image_tool",
        description="Returns an image.",
    ).as_tool()


def image_list_tool() -> Tool:
    async def execute() -> list[ContentImage]:
        """Return a list of images.

        Returns:
            A list of images.
        """
        return [ContentImage(image=MOCK_BASE64_IMAGE)]

    return ToolDef(
        execute,
        name="image_list_tool",
        description="Returns a list of images.",
    ).as_tool()


def image_receiving_tool(received: list) -> Tool:
    async def execute(image: ContentImage) -> str:
        """Receive an image.

        Args:
            image: Image to receive.
        """
        received.append(image)
        return "received"

    return ToolDef(
        execute,
        name="image_receiving_tool",
        description="Receives an image.",
    ).as_tool()


def image_list_receiving_tool(received: list) -> Tool:
    async def execute(images: list[ContentImage]) -> str:
        """Receive a list of images.

        Args:
            images: Images to receive.
        """
        received.append(images)
        return f"received:{len(images)}"

    return ToolDef(
        execute,
        name="image_list_receiving_tool",
        description="Receives a list of images.",
    ).as_tool()


def empty_image_list_tool() -> Tool:
    async def execute() -> list[ContentImage]:
        """Return an empty list of images.

        Returns:
            An empty list.
        """
        return []

    return ToolDef(
        execute,
        name="empty_image_list_tool",
        description="Returns an empty list of images.",
    ).as_tool()


def test_is_content_result_false_for_empty_list():
    bridge = RunCodeToolBridge([])
    assert bridge._is_content_result([]) is False


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_discarded_image_not_in_final_result():
    pytest.importorskip("pydantic_monty")

    tool = run_code(tools=[image_tool()], executor="monty")

    result = await tool(
        code="""
await image_tool()
"done"
"""
    )

    assert "done" in result[0].text
    assert not any(isinstance(item, ContentImage) for item in result)


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_returned_image_reaches_final_result():
    pytest.importorskip("pydantic_monty")

    tool = run_code(tools=[image_tool()], executor="monty")

    result = await tool(
        code="""
img = await image_tool()
img
"""
    )

    image_items = [item for item in result if isinstance(item, ContentImage)]
    assert len(image_items) == 1
    assert image_items[0].image == MOCK_BASE64_IMAGE


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_passes_image_between_tools():
    received: list = []
    tool = run_code(
        tools=[image_tool(), image_receiving_tool(received)],
        executor="monty",
    )

    result = await tool(
        code="""
img = await image_tool()
await image_receiving_tool(img)
"""
    )

    assert "received" in result[0].text
    assert len(received) == 1
    assert isinstance(received[0], ContentImage)
    assert received[0].image == MOCK_BASE64_IMAGE


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_passes_single_image_list_between_tools():
    pytest.importorskip("pydantic_monty")

    received: list = []
    tool = run_code(
        tools=[image_tool(), image_list_receiving_tool(received)],
        executor="monty",
    )

    result = await tool(
        code="""
img = await image_tool()
await image_list_receiving_tool([img])
"""
    )

    assert "received:1" in result[0].text
    assert len(received) == 1
    assert len(received[0]) == 1
    assert isinstance(received[0][0], ContentImage)


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_single_element_list_result_preserves_array_shape():
    pytest.importorskip("pydantic_monty")

    received: list = []
    tool = run_code(
        tools=[image_list_tool(), image_list_receiving_tool(received)],
        executor="monty",
    )

    result = await tool(
        code="""
imgs = await image_list_tool()
await image_list_receiving_tool(imgs)
"""
    )

    assert "received:1" in result[0].text
    assert len(received) == 1
    assert len(received[0]) == 1
    assert isinstance(received[0][0], ContentImage)


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_empty_content_list_result():
    pytest.importorskip("pydantic_monty")

    received: list = []
    tool = run_code(
        tools=[empty_image_list_tool(), image_list_receiving_tool(received)],
        executor="monty",
    )

    result = await tool(
        code="""
imgs = await empty_image_list_tool()
await image_list_receiving_tool(imgs)
"""
    )

    assert "received:0" in result[0].text
    assert len(received) == 1
    assert received[0] == []


def test_reconstruct_content_flat_list_unchanged():
    value = [
        {"type": "text", "text": "hello"},
        {"type": "image", "image": MOCK_BASE64_IMAGE},
    ]
    result = _reconstruct_content(value)
    assert result == [
        ContentText(text="hello"),
        ContentImage(image=MOCK_BASE64_IMAGE),
    ]


def test_reconstruct_content_plain_structured_data_no_content():
    value = [{"id": 1, "amount": 10}, {"id": 2, "amount": 32}]
    result = _reconstruct_content(value)
    assert len(result) == 1
    assert isinstance(result[0], ContentText)
    assert json.loads(result[0].text) == value


def test_reconstruct_content_extracts_nested_mixed_content():
    value = [
        [
            {"type": "text", "text": "screenshot taken successfully"},
            {"type": "image", "image": MOCK_BASE64_IMAGE},
        ],
        "Page title for https://example.com: Example Domain",
        [
            {"type": "text", "text": "document read successfully"},
            {"type": "document", "document": "data:application/pdf;base64,AAAA"},
        ],
    ]

    result = _reconstruct_content(value)

    assert isinstance(result[0], ContentText)
    skeleton = json.loads(result[0].text)
    assert skeleton[1] == "Page title for https://example.com: Example Domain"
    assert skeleton[0][0].startswith("<content:")
    assert skeleton[0][1].startswith("<content:")

    extracted = result[1:]
    assert len(extracted) == 4
    assert extracted[0] == ContentText(text="screenshot taken successfully")
    assert isinstance(extracted[1], ContentImage)
    assert extracted[2] == ContentText(text="document read successfully")
    assert isinstance(extracted[3], ContentDocument)


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_gather_mixed_content_and_text_with_monty():
    pytest.importorskip("pydantic_monty")

    tool = run_code(tools=[image_tool(), dummy_tool()], executor="monty")

    result = await tool(
        code="""
import asyncio

results = await asyncio.gather(
    image_tool(),
    dummy_tool("hello"),
)
results
"""
    )

    image_items = [item for item in result if isinstance(item, ContentImage)]
    assert len(image_items) == 1
    assert image_items[0].image == MOCK_BASE64_IMAGE


@pytest.mark.anyio
@skip_if_trio  # pydantic-monty runs on asyncio
async def test_run_code_empty_image_list_result():
    pytest.importorskip("pydantic_monty")

    tool = run_code(
        tools=[empty_image_list_tool()],
        executor="monty",
    )

    result = await tool(
        code="""
imgs = await empty_image_list_tool()
imgs
"""
    )

    assert len(result) == 1
    assert result[0].text == "[]"


def test_reconstruct_content_extracts_bare_content_dict_among_list_siblings():
    value = [
        [{"type": "text", "text": "ok"}, {"type": "image", "image": "AAAA"}],
        "title string",
        "plain string",
        {"type": "image", "image": "AAAA"},
    ]
    result = _reconstruct_content(value)

    skeleton = json.loads(result[0].text)
    assert skeleton[3].startswith("<content:")
    assert "AAAA" not in result[0].text  # no raw base64 leaking into the text skeleton

    extracted = result[1:]
    assert len(extracted) == 3
    assert isinstance(extracted[-1], ContentImage)


def test_reconstruct_content_handles_tuple_shape():
    value = (
        [{"type": "text", "text": "ok"}, {"type": "image", "image": "AAAA"}],
        "title string",
        "error text",
    )
    result = _reconstruct_content(value)

    assert "AAAA" not in result[0].text
    extracted = result[1:]
    assert len(extracted) == 2
    assert isinstance(extracted[-1], ContentImage)
