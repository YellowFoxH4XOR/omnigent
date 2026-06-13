"""Phase 0 characterization test — sub-agent visibility in overview.

Uses ``examples/coding_supervisor.yaml`` so the REPL hosts an
``openai-agents`` supervisor with ``claude_worker`` and
``codex_worker`` session-tools. Sends a prompt that explicitly
instructs the supervisor to use the chosen worker to print a
short message, waits for the sub-agent spawn to land in the
supervisor's tool stream, hits ``Ctrl+G``, cycles to the
sub-agent target with ``Tab``, and asserts the overview pane
renders ``Session ID:``, ``Executor:``, and ``Messages:`` lines
for the sub-agent. Parametrized so each wrapped harness
(claude-sdk, codex) is exercised as the sub-agent worker.

**What breaks if this fails:**
- The ``sys_session_send`` builtin's output JSON drops the
  ``conversation_id`` field — the REPL's overview target
  registration keys on it, so a missing ``conversation_id``
  produces a target with no session reference and an empty
  pane. **This test is specifically designed to catch that
  regression**: the ``Session ID:`` line comes from
  ``target.session.id`` which is None without a valid
  ``conversation_id``.
- ``_collect_overview_targets`` stops including managed agent
  sessions (``Session._agent_sessions``), so the sub-agent
  target is missing from the sidebar.
- ``_render_overview_managed_session_text`` regresses so the
  ``Executor:`` / ``Messages:`` metadata lines are dropped.
- The wrapped harness invocation inside the sub-agent fails
  (missing CLI, PAT profile invalid), so the worker never
  comes up and the overview never gets a second target.

Design reference: ``designs/OMNIGENT_INTEGRATION.md`` §Phase 0
REPL pexpect suite — "Sub-agent visibility in overview".
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which
from typing import Any

import pytest

from tests._model_pools import resolve_model
from tests.e2e._harness_probes import HARNESS_HARNESS_MODELS, HARNESS_IDS
from tests.e2e.omnigent._pexpect_harness import (
    clean_exit,
    spawn_omnigent_run,
    strip_ansi,
    submit_prompt,
    wait_for_ready,
)
from tests.e2e.omnigent._repl_test_helpers import drain_for
from tests.e2e.omnigent._snapshot import compare_snapshot

# Supervisor model — top-level openai-agents harness uses GPT;
# sub-agents override via their own executor blocks in the YAML.
_SUPERVISOR_MODEL = resolve_model("databricks-gpt-5-4", key=__name__)
_SUPERVISOR_HARNESS = "openai-agents"

# Mapping from harness id to the YAML's worker tool name. The
# coding_supervisor.yaml fixture defines ``claude_worker``
# (claude-sdk) and ``codex_worker`` (codex); other harnesses
# don't have a worker tool defined in this fixture. The pi row
# of the harness probe matrix (added in 4d / commit 9e0f540)
# falls into the "no worker tool" bucket below — the test
# skips pi cleanly until/unless a ``pi_worker`` AgentTool is
# added to the example YAML.
_WORKER_TOOL_BY_HARNESS: dict[str, str] = {
    "claude-sdk": "claude_worker",
    "codex": "codex_worker",
}

# Per-harness substring set the rendered Executor: line might
# print. Prompt-toolkit column overwrites can collapse the
# hyphen ("claude-sdk" → "claudesdk", etc.), so we tolerate
# both spellings.
_EXECUTOR_MARKERS_BY_HARNESS: dict[str, tuple[str, ...]] = {
    "claude-sdk": ("claude-sdk", "claudesdk"),
    "codex": ("codex",),
}

# The user_message we send to the worker. Its appearance in the
# pane proves the managed session's ``items`` list populated
# (fails if the Messages: line's count-source is empty or
# missing).
_SUBAGENT_MESSAGE_CONTENT = "say hello"

_SPAWN_TIMEOUT = 60.0
_BOOT_TIMEOUT = 60.0
_RUNNING_TIMEOUT = 30.0
# Sub-agent spawns add significant latency: the supervisor must
# decide to call the tool, wait for the claude-sdk harness to
# boot (CLI + MCP bridge), and process at least one turn. 240s
# keeps headroom for cold-start delays.
_COMPLETION_TIMEOUT = 240.0
_EXIT_TIMEOUT = 15.0
_OVERVIEW_DRAIN_TIMEOUT = 6.0
_EXPECT_SUBAGENT_TIMEOUT = 30.0


def _check_worker_harness_available(harness: str, omnigent_python: Path) -> None:
    """
    Fail loud if the worker harness's prerequisites are missing.

    Mirrors per-harness availability checks elsewhere in the
    suite. claude-sdk needs the SDK package + ``claude`` CLI;
    codex needs the ``codex`` CLI on PATH.

    :param harness: The worker harness identifier under test.
    :param omnigent_python: The subprocess interpreter.
    """
    if harness == "claude-sdk":
        probe = subprocess.run(
            [
                str(omnigent_python),
                "-c",
                "import importlib.util, sys; "
                "sys.exit(0 if importlib.util.find_spec('claude_agent_sdk') else 1)",
            ],
            capture_output=True,
        )
        if probe.returncode != 0 or which("claude") is None:
            pytest.fail(
                "claude-sdk prerequisites missing: need both the "
                "'claude_agent_sdk' Python package and the 'claude' "
                "CLI binary on PATH."
            )
    elif harness == "codex":
        if which("codex") is None:
            pytest.fail(
                "codex prerequisite missing: the 'codex' CLI binary "
                "must be installed on PATH (install via "
                "'npm i -g @openai/codex')."
            )


@pytest.mark.parametrize("harness,model", HARNESS_HARNESS_MODELS, ids=HARNESS_IDS)
def test_repl_overview_subagent_visibility(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
    patched_databrickscfg: None,
    harness: str,
    model: str,
) -> None:
    """
    Spawn a supervisor that delegates to a sub-agent worker, open
    the overview, cycle to the sub-agent target, and verify
    its metadata lines render. Parametrized across each wrapped
    harness so the sidebar code path is verified for every
    sub-agent harness.

    :param omnigent_python: Interpreter with omnigent +
        openai-agents + the worker harness's SDK installed.
    :param omnigent_repo_root: Working directory for the
        subprocess.
    :param omnigent_credentials_env: Env vars with
        ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` /
        ``DATABRICKS_CONFIG_PROFILE`` populated.
    :param patched_databrickscfg: Rewrites ``~/.databrickscfg``
        to PAT form for the test and restores on teardown.
        Required because the worker harnesses read the cfg
        directly and don't honor env-var PATs.
    :param harness: The worker harness identifier from
        :data:`HARNESS_HARNESS_MODELS`. Selects the matching
        ``<harness>_worker`` tool from the YAML fixture.
    :param model: The model identifier for the worker harness.
        Unused at the CLI level (the YAML's worker block already
        pins the model per harness) — accepted to match the
        :data:`HARNESS_HARNESS_MODELS` parametrize shape.
    """
    if harness not in _WORKER_TOOL_BY_HARNESS:
        # ``coding_supervisor.yaml`` only defines worker tools for
        # claude-sdk and codex. Other harnesses (currently just
        # ``pi``) skip cleanly rather than KeyError.
        pytest.skip(
            f"{harness!r} has no <harness>_worker tool in "
            f"tests/resources/examples/coding_supervisor.yaml; this test requires the "
            f"YAML to declare an AgentTool for the harness."
        )
    _check_worker_harness_available(harness, omnigent_python)
    worker_tool = _WORKER_TOOL_BY_HARNESS[harness]
    worker_label_prefix = f"{worker_tool}:"
    executor_markers = _EXECUTOR_MARKERS_BY_HARNESS[harness]
    # Explicit instruction to use the worker tool — the YAML's
    # system prompt already gives the supervisor a bias toward
    # delegation, but we need to force a spawn, not a direct
    # reply. Mentioning ``sys_session_send`` and the worker tool
    # by name removes ambiguity.
    user_prompt = (
        f"Delegate to {worker_tool}. Call sys_session_send with "
        f"tool={worker_tool}, session=demo, and input='say hello'. "
        f"Do not answer inline. After the worker replies, relay its "
        f"message verbatim."
    )
    yaml_path = omnigent_repo_root / "tests" / "resources" / "examples" / "coding_supervisor.yaml"

    child = spawn_omnigent_run(
        omnigent_python=omnigent_python,
        yaml_path=yaml_path,
        model=_SUPERVISOR_MODEL,
        harness=_SUPERVISOR_HARNESS,
        env=omnigent_credentials_env,
        cwd=omnigent_repo_root,
        timeout=_SPAWN_TIMEOUT,
    )
    try:
        wait_for_ready(child, timeout=_BOOT_TIMEOUT)
        submit_prompt(child, user_prompt)
        # Wait for the tool-call line that represents the
        # ``sys_session_send`` invocation for the worker.
        # Once this tool call starts, the ManagedAgentSession
        # is registered in ``session._agent_sessions`` — the
        # exact dict ``_collect_overview_targets`` scans. We
        # wait for this rather than ``await_turn_complete``
        # because the supervisor YAML sets ``async: true``,
        # meaning the supervisor's turn returns immediately
        # after dispatching the worker; the managed session
        # exists at the moment of dispatch.
        child.expect(
            rf"sys_session_send \({worker_tool}:",
            timeout=_COMPLETION_TIMEOUT,
        )
        # Open the overview.
        child.sendcontrol("g")
        # Wait for the overview to paint and drain follow-up
        # frames so the buffer has the sidebar + main session
        # pane.
        drain_for(child, _OVERVIEW_DRAIN_TIMEOUT)
        # Tab cycles to the next target. With the worker
        # sub-agent registered, Tab moves from "main" to
        # "<worker>:<key>". We wait for the sub-agent's label
        # to appear in the pane header.
        child.send("\t")
        # Drain post-Tab frames until the sub-agent's header
        # is rendered. Using expect on the label prefix rather
        # than a full-text match tolerates session-key
        # variation.
        child.expect(
            f"Session: {worker_label_prefix}",
            timeout=_EXPECT_SUBAGENT_TIMEOUT,
        )
        subagent_pane_tail = drain_for(child, _OVERVIEW_DRAIN_TIMEOUT)
        subagent_stripped = (
            strip_ansi(child.before or "")
            + f"Session: {worker_label_prefix}"
            + strip_ansi(subagent_pane_tail)
        )
        clean_exit(child, timeout=_EXIT_TIMEOUT)
        exit_code = child.exitstatus
    finally:
        if not child.closed:
            child.close(force=True)

    observed: dict[str, Any] = {
        "exit_code": exit_code,
        # The sidebar/header contains the sub-agent's label,
        # which combines the tool name and session key. Its
        # presence proves ``_collect_overview_targets`` saw the
        # ManagedAgentSession — the dict that's populated only
        # when ``conversation_id`` threads through correctly
        # from the spawn output JSON.
        "subagent_label_present": worker_label_prefix in subagent_stripped,
        # Harness identifier from the sub-agent's ExecutorSpec
        # rendered by ``_render_overview_managed_session_text``.
        # Matching tolerant variants (e.g. "claudesdk") allows
        # for prompt-toolkit's column-level overwrites on
        # narrow rows without weakening the claim.
        "subagent_executor_harness_rendered": any(
            marker in subagent_stripped for marker in executor_markers
        ),
        # The user_message we sent ("say hello") rendered as
        # part of the items list — proves the managed session's
        # items traveled end-to-end from the spawn call into the
        # overview pane. Absent if the Messages-source list was
        # empty or never wired.
        "subagent_user_message_rendered": _SUBAGENT_MESSAGE_CONTENT in subagent_stripped,
    }
    diffs = compare_snapshot("test_repl_overview_subagent_visibility", observed)
    assert diffs == [], (
        "Snapshot mismatch for sub-agent overview visibility:\n"
        + "\n".join(diffs)
        + f"\n\nsubagent stripped (last 2500):\n"
        f"{subagent_stripped[-2500:]}"
    )
