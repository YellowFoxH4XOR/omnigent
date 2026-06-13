"""Unit tests for native-worker YOLO ``terminal_launch_args`` derivation.

Nessie's native sub-agent workers (claude-native / codex-native) launch
in a headless pane where no human can answer an approval prompt, so they
declare full-bypass intent in their bundle. The server translates that
declaration into the per-session ``terminal_launch_args`` the runner
appends to the claude / codex argv.

These tests exercise the pure translation helper
``_derive_terminal_launch_args_from_spec`` directly with real
:class:`AgentSpec` / :class:`ExecutorSpec` objects, including the
string-coerced config values the spec parser actually produces (it
stringifies every ``executor.config`` value, so ``yolo: true`` becomes
``"True"``).
"""

from __future__ import annotations

import pytest

from omnigent.server.routes.sessions import _derive_terminal_launch_args_from_spec
from omnigent.spec.types import AgentSpec, ExecutorSpec


def _spec_with_config(config: dict[str, str]) -> AgentSpec:
    """
    Build a minimal sub-agent spec carrying a given ``executor.config``.

    :param config: The ``executor.config`` mapping, e.g.
        ``{"harness": "claude-native", "permission_mode": "bypassPermissions"}``.
        Values are plain strings to mirror what the spec parser produces
        (it coerces every config value to ``str``).
    :returns: An :class:`AgentSpec` whose executor carries *config*.
    """
    return AgentSpec(
        spec_version=1,
        name="impl",
        executor=ExecutorSpec(type="omnigent", config=config),
    )


def test_claude_native_permission_mode_translates_to_flag() -> None:
    """
    claude-native + ``permission_mode`` -> ``--permission-mode <value>``.

    A failure here means the YOLO claude worker would launch with no
    permission flag and stall on the first Edit/Write ApprovalCard. The
    value must be passed through verbatim (``bypassPermissions``), proving
    the worker bundle's declared bypass reached the runner argv.
    """
    spec = _spec_with_config({"harness": "claude-native", "permission_mode": "bypassPermissions"})
    assert _derive_terminal_launch_args_from_spec(spec) == [
        "--permission-mode",
        "bypassPermissions",
    ]


def test_claude_native_permission_mode_obeys_arg_length_bound() -> None:
    """
    Spec-derived ``permission_mode`` is bounded like request-supplied args.

    The value comes from an uploaded bundle, not directly from the create
    request body, but it still becomes a persisted CLI argument. A failure
    here means a bundle config value could bypass the route's
    ``terminal_launch_args`` length cap and produce an oversized row or
    launch command.
    """
    # _validate_terminal_launch_args caps each entry at 4096 bytes/chars.
    spec = _spec_with_config({"harness": "claude-native", "permission_mode": "x" * 4097})
    with pytest.raises(ValueError, match="terminal_launch_args entry exceeds"):
        _derive_terminal_launch_args_from_spec(spec)


def test_codex_native_yolo_string_true_translates_to_bypass_flag() -> None:
    """
    codex-native + ``yolo`` (string ``"True"``) -> the codex bypass flag.

    The spec parser stringifies ``yolo: true`` into ``"True"``, so this is
    the value the server actually sees in production. A failure means the
    codex worker would launch in its default approval-prompting mode and
    hang headless. The exact flag string must match codex's
    ``--dangerously-bypass-approvals-and-sandbox``.
    """
    spec = _spec_with_config({"harness": "codex-native", "yolo": "True"})
    assert _derive_terminal_launch_args_from_spec(spec) == [
        "--dangerously-bypass-approvals-and-sandbox",
    ]


def test_native_spec_without_yolo_field_returns_none() -> None:
    """
    A native sub-agent that declares no bypass field gets no args.

    Both native harnesses must return ``None`` when their respective
    field is absent — otherwise a plain native sub-agent would silently
    inherit YOLO. ``None`` (not ``[]``) is the contract the create path
    treats as "leave terminal_launch_args unset".
    """
    claude = _spec_with_config({"harness": "claude-native"})
    codex = _spec_with_config({"harness": "codex-native"})
    # No permission_mode / yolo declared -> nothing to translate.
    assert _derive_terminal_launch_args_from_spec(claude) is None
    assert _derive_terminal_launch_args_from_spec(codex) is None


def test_codex_native_yolo_false_string_returns_none() -> None:
    """
    ``yolo: false`` (string ``"False"``) must NOT enable bypass.

    This guards the ``bool("False") is True`` trap: a naive truthiness
    check on the parser's stringified value would enable YOLO for an
    explicit opt-out. A failure here means an agent that set
    ``yolo: false`` would still launch with the dangerous bypass flag.
    """
    spec = _spec_with_config({"harness": "codex-native", "yolo": "False"})
    assert _derive_terminal_launch_args_from_spec(spec) is None


@pytest.mark.parametrize(
    "harness",
    ["claude-sdk", "codex", "openai-agents", "databricks_supervisor"],
)
def test_non_native_harness_with_bypass_fields_is_ignored(harness: str) -> None:
    """
    Non-native harnesses never get terminal args, even with bypass fields.

    ``terminal_launch_args`` is a native-terminal (claude/codex TUI)
    concept; a claude-sdk worker sets bypass via the SDK ``permissionMode``
    spawn env, not a CLI flag. Translating these fields for a non-native
    harness would emit a flag the runner has no terminal to apply it to.
    A failure means the harness gate leaked. Both bypass fields are set to
    prove neither branch fires for a non-native harness.
    """
    spec = _spec_with_config(
        {"harness": harness, "permission_mode": "bypassPermissions", "yolo": "True"}
    )
    assert _derive_terminal_launch_args_from_spec(spec) is None
