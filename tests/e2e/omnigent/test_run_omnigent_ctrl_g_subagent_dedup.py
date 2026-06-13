"""
End-to-end: a sub-agent spawned with ``sys_session_send`` and then
re-addressed with ``sys_session_send`` must appear as a SINGLE
row in the Ctrl+G overview, not two.

The child is identified by ``(sub_agent_type, session_key)`` —
e.g. ``codex_worker:tmp-list``. Whichever LLM-facing function
created or resumed it (``sys_session_send`` vs
``sys_session_send``) is orthogonal to child identity. Legacy
omnigent labels child sessions by the AgentTool name (the
type) because that's the child's actual tool. The Omnigent mode SSE
bridge must match.

**What breaks if this fails:**

- The sidebar grows one row per turn that addresses a child,
  drowning the user in per-call entries that all refer to the
  same conversation.
- The ``Session ID`` line inside each pane mismatches between
  rows that really point at the same child.
- Dedupe logic in :func:`_extract_sub_agent_refs` regresses to
  keying by the CALLING function name instead of the child's
  (type, session_key).
"""

from __future__ import annotations

from pathlib import Path

from tests._model_pools import resolve_model
from tests.e2e.omnigent._pexpect_harness import (
    await_turn_complete,
    clean_exit,
    spawn_omnigent_run,
    strip_ansi,
    submit_prompt,
    wait_for_ready,
)

_YAML_REL = "tests/resources/examples/coding_supervisor.yaml"
_MODEL = resolve_model("databricks-gpt-5-mini", key=__name__)
_HARNESS = "openai-agents"

# Two turns: the first spawns a named codex_worker, the second
# re-addresses the SAME name with sys_session_send. A
# well-behaved bridge merges these into one sidebar row.
_SPAWN_PROMPT = (
    "Spawn a codex_worker sub-agent named 'dedup' and ask it to print the literal text 'first'."
)
_SEND_PROMPT = (
    "Send another message to the codex_worker sub-agent named "
    "'dedup' asking it to print the literal text 'second'. Use "
    "sys_session_send — do NOT spawn a new worker."
)
# The sidebar label for a codex_worker child is
# ``codex_worker:<session_key>``. ``codex_worker:dedup`` (18
# chars) fits cleanly in the 22-char sidebar column.
_EXPECTED_SIDEBAR_LABEL = "codex_worker:dedup"
# The two calling-function names that MUST NOT appear in the
# sidebar as row prefixes — they'd indicate the old per-call
# labeling regressed. Each is prefixed with an icon + space in
# the rendered sidebar; we match on the trailing ``:dedup``
# token which is unique per-row.
_FORBIDDEN_ROW_PREFIXES = (
    "sys_session_send:dedup",
    "sys_session_send:dedup",
)

_SPAWN_TIMEOUT = 60.0
# Cold-boot of ``coding_supervisor.yaml`` under Omnigent mode —
# spawns the in-process Omnigent server (FastAPI + uvicorn + DBOS +
# alembic) and registers supervisor + two sub-agents. Original
# 30s flaked on cold DBOS dbs; 120s matches the supervisor-
# driven tests in test_run_omnigent_quiet_startup /
# test_run_omnigent_coding_supervisor.
_BOOT_TIMEOUT = 120.0
_RUNNING_TIMEOUT = 20.0
_COMPLETION_TIMEOUT = 120.0
_EXIT_TIMEOUT = 15.0
_OVERVIEW_DRAIN_TIMEOUT = 10.0


def test_run_omnigent_ctrl_g_deduplicates_subagent_across_spawn_and_send(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
    databricks_workspace: tuple[str, str],
) -> None:
    """
    After one spawn + one send to the same sub-agent name, the
    Ctrl+G sidebar shows exactly one row for that child.

    :param omnigent_python: Shared interpreter fixture.
    :param omnigent_repo_root: Subprocess cwd.
    :param omnigent_credentials_env: Env with PAT + profile.
    """
    yaml_path = omnigent_repo_root / _YAML_REL

    child = spawn_omnigent_run(
        omnigent_python=omnigent_python,
        yaml_path=yaml_path,
        model=_MODEL,
        harness=_HARNESS,
        env=omnigent_credentials_env,
        cwd=omnigent_repo_root,
        timeout=_SPAWN_TIMEOUT,
    )
    try:
        wait_for_ready(child, timeout=_BOOT_TIMEOUT)

        submit_prompt(child, _SPAWN_PROMPT)
        spawn_turn = await_turn_complete(
            child,
            running_timeout=_RUNNING_TIMEOUT,
            completion_timeout=_COMPLETION_TIMEOUT,
        )
        assert "sys_session_send" in spawn_turn.stripped, (
            f"Supervisor did not call sys_session_send on turn 1 — "
            f"the dedup check below would be meaningless. "
            f"stripped tail:\n{spawn_turn.stripped[-2000:]}"
        )

        submit_prompt(child, _SEND_PROMPT)
        send_turn = await_turn_complete(
            child,
            running_timeout=_RUNNING_TIMEOUT,
            completion_timeout=_COMPLETION_TIMEOUT,
        )
        assert "sys_session_send" in send_turn.stripped, (
            f"Supervisor did not call sys_session_send on turn 2 "
            f"— it may have spawned a new child instead, which "
            f"would make the dedup check below test the wrong "
            f"thing. stripped tail:\n{send_turn.stripped[-2000:]}"
        )

        # Open the overview. Expect the dedup'd label
        # ``codex_worker:dedup`` directly — the bridge must key
        # managed sessions by ``(sub_agent_type, session_key)``
        # so spawn + send collapse into one row.
        child.sendcontrol("g")
        child.expect(_EXPECTED_SIDEBAR_LABEL, timeout=_OVERVIEW_DRAIN_TIMEOUT)
        # After ``child.expect`` returns, pexpect guarantees both
        # ``.before`` (pre-match text) and ``.after`` (the match itself)
        # are populated strings — the ``Any | None`` type in pexpect's
        # stubs only models the pre-first-match state.
        assert child.before is not None and child.after is not None, (
            "child.expect populated no before/after text"
        )
        overview_raw = child.before + child.after
        overview_stripped = strip_ansi(overview_raw)

        clean_exit(child, timeout=_EXIT_TIMEOUT)
    finally:
        if not child.closed:
            child.close(force=True)

    # The dedup'd label must be present (positive assertion is
    # redundant with the ``expect`` above but survives edits to
    # the drain strategy).
    assert _EXPECTED_SIDEBAR_LABEL in overview_stripped, (
        f"Sidebar missing dedup'd row {_EXPECTED_SIDEBAR_LABEL!r}. "
        f"Overview tail:\n{overview_stripped[-2000:]}"
    )
    # The per-call labels MUST NOT appear as sidebar rows. A
    # regression in ``_extract_sub_agent_refs`` (keying by the
    # calling function name instead of the child's
    # ``(sub_agent_type, session_key)``) would emit these.
    leaked = [p for p in _FORBIDDEN_ROW_PREFIXES if p in overview_stripped]
    assert not leaked, (
        f"Sidebar shows per-call labels {leaked} instead of the "
        f"dedup'd ``codex_worker:dedup`` row. A single child "
        f"addressed via sys_session_send then sys_session_send "
        f"must collapse into one sidebar row, keyed by the "
        f"child's (type, name) — not by the LLM-facing function "
        f"that created or resumed it. "
        f"Overview tail:\n{overview_stripped[-2000:]}"
    )
