"""End-to-end coverage for inline tool-call + result streaming under
the Omnigent REPL with the ``claude-sdk`` harness.

Drives a real interactive REPL via the e2e pexpect harness against
``tests/resources/agents/sys-terminal-test/sys-terminal-test.yaml``
and asserts the user-visible ordering invariant:

    Each AP-side ``sys_terminal_*`` tool's call line AND its result
    panel appear BEFORE the agent's final assistant text —
    interleaved with the response, not bunched at end-of-turn.

Uses ``sys_terminal_launch`` (an AP-side built-in) — that's the
exact harness path the user hit. Three categories of tool
behavior under claude-sdk:

- **Built-in Bash**: handled inside the Claude SDK process. The
  harness adapter's ``ToolCallComplete`` branch emits a paired
  ``function_call_output`` SSE event INLINE as each Bash result
  lands, even without the inline-streaming fixes. Built-in tools don't
  exercise ``_dispatch_action_required``, so they don't catch the
  regression.
- **YAML function tools with `callable:`** (e.g. ``get_current_time``):
  registered as MCP tools in the SDK with a Python callback. The
  callback can short-circuit native — also doesn't go through
  ``_dispatch_action_required``.
- **AP-side built-ins** (``sys_terminal_*``, ``sys_session_*``):
  the SDK's MCP callback awaits ``ctx.dispatch_tool`` (action_required
  emission) → outer ``_dispatch_action_required`` runs the tool on
  the Omnigent server (where the terminal state lives) → PATCHes back.
  THIS is the path the inline ``function_call_output`` emit targets.
  Without the fix, every result panel bunches at the
  ``response.completed`` flush — exactly the user-reported bug
  (``sys_terminal_*`` calls in the databricks_coding_agent
  transcript).

**What breakage would surface here:**

- ``omnigent/runtime/harnesses/_client_executor.py:_translate_omnigent_event``
  reverts to buffering function_call events without yielding a
  :class:`ToolCallInProgress` — every ⏵ line then renders only at the
  ``response.completed`` flush, so all call lines appear AFTER any
  assistant text the LLM streamed alongside them. The "all ⏵ before
  Done." assertion below catches this.
- ``_dispatch_action_required`` stops emitting the inline
  ``response.output_item.done`` ``function_call_output`` event — every
  result panel then waits for the late ``ToolCallObserved`` flush, so
  result panels bunch at end-of-turn. The "result panels appear before
  Done. text" assertion catches this.
- ``omnigent_client._stream.BlockStream`` reverts to deferring
  ``ToolResultBlock`` yields until the next ``TextDelta`` /
  ``ResponseCreated`` — same end-of-turn bunching even if the SSE
  events arrive inline. The same assertions catch this.
- The new ``seen_call_ids`` / ``seen_result_call_ids`` dedupes regress
  to clearing on ``TextDelta`` — duplicate ⏵ or result-panel renders
  surface as the per-tool counts going above 1.

Test integrity:

The iron-rule audit walked through reverting each of those production
changes individually and confirmed the corresponding assertion below
fails loud. See the inline tool-streaming notes in
``designs/RUN_OMNIGENT_REPL_PARITY.md``.

Why claude-sdk specifically:

The bug surfaced under the claude-sdk harness because that's the path
where every spec-declared tool round-trips through the AP-side
``the harness HTTP client._dispatch_action_required`` pipeline. ``codex``
and ``pi`` use the same code path, so this single-harness test pins
the contract for all three. Adding the matrix would burn 3x CI time
without distinguishing coverage.
"""

from __future__ import annotations

from pathlib import Path

import pexpect

from tests._model_pools import resolve_model
from tests.e2e.omnigent._example_helpers import require_claude_sdk
from tests.e2e.omnigent._pexpect_harness import (
    clean_exit,
    spawn_omnigent_run,
    strip_ansi,
    submit_prompt,
)
from tests.e2e.omnigent._repl_test_helpers import drain_for

# Same model / harness combination the gap was reported against. Using
# the project's standard claude-sdk e2e config keeps test prerequisites
# (Databricks profile, claude binary on PATH) consistent with the rest
# of the suite.
_MODEL = resolve_model("databricks-claude-opus-4-6", key=__name__)
_HARNESS = "claude-sdk"

# Three calls to ``sys_terminal_launch`` — the AP-side built-in
# whose result lives on the Omnigent server (not in the harness
# subprocess). The SDK's MCP wrapper invokes
# ``ctx.dispatch_tool`` for it, emitting an ``action_required``
# function_call SSE event; the outer ``_dispatch_action_required``
# runs the tool, PATCHes back, and emits the inline
# ``function_call_output`` via the inline emit. THIS is the path
# the inline emit fixed. Without the inline publish, the result panels
# bunch at the ``response.completed`` flush.
_PROMPT = (
    "Use sys_terminal_launch exactly 3 times to start the bash "
    "terminal under sessions s1, s2, and s3 (one per call). "
    "Then say done-streaming. Do not use sys_terminal_send, "
    "sys_terminal_read, sys_terminal_close, or any other tool."
)
_DONE_MARKER = "done-streaming"
# Number of tool invocations the prompt requests. Three is enough
# to make all-bunched-at-end visually distinct from inline
# rendering.
_EXPECTED_CALL_COUNT = 3
# Bare tool name as it renders in the ⏵ call line.
_TOOL_RENDER_PREFIX = "⏵ sys_terminal_launch"

# The completion timeout has to swallow Databricks gateway warmup +
# Claude SDK CLI cold start + three Bash invocations. Other claude-sdk
# REPL tests in this directory use 240 s for a single-turn prompt; +
# the per-tool overhead, 360 s leaves headroom on slow CI runners
# without hiding hangs.
_BOOT_TIMEOUT = 90.0
_RESPONSE_TIMEOUT = 360.0


def test_repl_inline_tool_call_and_result_streaming(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
    databricks_workspace: tuple[str, str],
) -> None:
    """
    Three Bash calls render with each ⏵ followed (eventually) by its
    result panel, all before the agent's "done-streaming" text.

    Iron rule: revert ANY of the inline-streaming production changes and this
    test fails. Specifically:

    - Revert ``ToolCallInProgress`` emission in ``_translate_omnigent_event``
      → ⏵ lines render only at ``response.completed`` flush, AFTER the
      ``done-streaming`` text → the "all ⏵ before Done text" assertion
      fails because the ⏵ positions sit AFTER ``done-streaming``.
    - Revert the inline emit of ``function_call_output``
      in ``_dispatch_action_required`` → result panels render only at
      flush time, AFTER ``done-streaming`` → the "result panels before
      Done text" assertion fails.
    - Revert ``BlockStream``'s immediate-yield-on-ToolResult → result
      panels stay in ``pending_tools`` until the next ``ResponseCreated``
      sweep, which doesn't fire until the NEXT user turn → result
      panels never render in this turn at all → the per-marker count
      check fails (count == 0).
    """
    require_claude_sdk()

    yaml_path = (
        omnigent_repo_root
        / "tests"
        / "resources"
        / "agents"
        / "sys-terminal-test"
        / "sys-terminal-test.yaml"
    )
    assert yaml_path.exists(), (
        f"expected the sys-terminal test yaml at {yaml_path}; the "
        f"e2e test depends on this fixture for the claude-sdk + "
        f"AP-side ``sys_terminal_*`` configuration."
    )

    child = spawn_omnigent_run(
        omnigent_python=omnigent_python,
        yaml_path=yaml_path,
        model=_MODEL,
        harness=_HARNESS,
        env=omnigent_credentials_env,
        cwd=omnigent_repo_root,
        timeout=_RESPONSE_TIMEOUT,
    )
    try:
        # Wait for the ❯ prompt marker. The bottom-toolbar
        # ``state: sleeping`` line that ``wait_for_ready`` matches
        # depends on prompt-toolkit's CPR probe completing — under
        # pexpect that races with the welcome banner and sometimes
        # never paints. The ❯ marker is what the user actually sees
        # before typing, so it's the right signal.
        child.expect(r"❯ ", timeout=_BOOT_TIMEOUT)

        submit_prompt(child, _PROMPT)

        # Two-phase expect: wait for the assistant marker first so
        # the ``done-streaming`` match below catches the AGENT's
        # text, not the user's echoed prompt (the prompt itself
        # contains the marker — prompt-toolkit echoes typed input
        # back through the PTY, so a single ``expect(_DONE_MARKER)``
        # would race the agent's response and match the echo).
        # The YAML's ``name:`` is ``sys-terminal-test``; humanized
        # to ``sys terminal test`` in the assistant marker.
        child.expect(r"◆ sys.terminal.test", timeout=_BOOT_TIMEOUT)
        # Now wait for the agent to actually emit ``done-streaming``.
        try:
            child.expect(_DONE_MARKER, timeout=_RESPONSE_TIMEOUT)
        except pexpect.TIMEOUT:
            # Capture what we got so the test log shows what the
            # agent actually emitted before the timeout.
            print("=== TIMEOUT — captured so far ===")
            print(strip_ansi(child.before or "")[:5000])
            print("=== END ===")
            raise

        # Drain the trailing render frames so the result panel that
        # may have arrived just before ``done-streaming`` (parallel
        # dispatch can complete after the LLM emits the final text)
        # gets captured. 4s = ~80 prompt-toolkit refresh ticks; long
        # enough to flush the post-text panels, short enough to keep
        # a hung test from hanging the suite.
        captured_after = drain_for(child, total_timeout=4.0)

        # ``child.before`` carries everything from the previous
        # ``expect`` (the ❯ prompt match) up to the current match —
        # i.e. the entire response from the prompt submission to the
        # ``done-streaming`` marker. Plus the drained tail.
        full_capture = (child.before or "") + (child.after or "") + captured_after
        rendered = strip_ansi(full_capture)
        print("=== POSITIONS:")
        print(f"  call rfind={rendered.rfind(_TOOL_RENDER_PREFIX)}")
        print(f"  panel rfind={rendered.rfind('╭─ ')}")
        print(f"  done find={rendered.find(_DONE_MARKER, 200)}")
        print(f"=== TAIL:\n{rendered[-1500:]}")
    finally:
        try:
            clean_exit(child, timeout=10.0)
        except (pexpect.TIMEOUT, pexpect.EOF, OSError):
            child.close(force=True)

    # ── Per-marker presence and dedup ──────────────────────
    #
    # Every marker appears exactly once as a ⏵ Bash call line and
    # exactly once as a result panel. More than one of either
    # would mean a duplicate render (the original symptom).
    # Zero would mean the call line or result panel didn't render
    # at all (the bunching-at-flush regression).
    # ⏵ sys_terminal_launch(...) call lines — one per invocation, total 3.
    # The ⏵ prefix is unique to the call line render — the user's
    # typed prompt and the agent's narration don't include it. So
    # ``⏵ sys_terminal_launch`` matches exactly the call-line renders.
    call_line_total = rendered.count("⏵ sys_terminal_launch")
    assert call_line_total == _EXPECTED_CALL_COUNT, (
        f"Expected exactly {_EXPECTED_CALL_COUNT} ``⏵ sys_terminal_launch`` "
        f"call lines (one per request from the prompt); got "
        f"{call_line_total}. If <{_EXPECTED_CALL_COUNT}, the agent "
        f"didn't call the tool the right number of times — adjust "
        f"the prompt phrasing or model. If >{_EXPECTED_CALL_COUNT}, "
        f"the dedup at ``BlockStream`` (``seen_call_ids``) "
        f"regressed and the late ``ToolCallObserved`` flush is "
        f"re-rendering call lines."
    )

    # Three result panels — one per invocation. The ``╭─ `` opener
    # of a Rich Panel is unique to result-block rendering; user
    # prompts and agent narration don't draw box-drawing characters.
    panel_count = rendered.count("╭─ ")
    assert panel_count == _EXPECTED_CALL_COUNT, (
        f"Expected exactly {_EXPECTED_CALL_COUNT} result panels "
        f"(one per ``⏵ sys_terminal_launch`` invocation); got {panel_count}. "
        f"If <{_EXPECTED_CALL_COUNT}, result panels didn't render "
        f"for some calls — either the inline emit of "
        f"``function_call_output`` in ``_dispatch_action_required`` "
        f"regressed, or ``BlockStream`` reverted to deferring "
        f"``ToolResultBlock`` until ``ResponseCreated`` (which "
        f"fires only on the next user turn). If "
        f">{_EXPECTED_CALL_COUNT}, the ``seen_result_call_ids`` "
        f"dedup regressed and the late ``ToolCallObserved`` "
        f"flush re-rendered the panels."
    )

    # ── Inline ordering: results BEFORE final assistant text ──
    #
    # The decisive invariant: each result panel must appear BEFORE
    # the agent's final ``done-streaming`` text. Currently with the
    # bug, results bunch at ``response.completed`` flush — which
    # fires AFTER all assistant text deltas. So bunched results
    # would render AFTER ``done-streaming``.
    #
    # With the fix:
    # - Each ⏵ Bash line renders inline (via ``ToolCallInProgress``).
    # - Each result panel renders the moment its dispatch returns
    #   (via the inline emit of ``function_call_output``).
    # - Both paths fire DURING the agent's response stream, before
    #   the final ``done-streaming`` text the LLM emits last.
    done_idx = rendered.find(_DONE_MARKER)
    assert done_idx >= 0, (
        f"Couldn't find ``{_DONE_MARKER}`` in the captured output — "
        f"the agent didn't reach its final text marker. Check the "
        f"prompt or the harness CLI; this assertion can't proceed "
        f"without an end-of-turn anchor."
    )
    # ── Decisive ordering: LAST call line + LAST panel BEFORE done ──
    #
    # Stronger than first-only assertions: even if the FIRST call
    # rendered inline but later ones bunched, this catches the
    # bunching. ``rfind`` returns the position of the LAST
    # occurrence. If the LAST call line and the LAST result panel
    # both appear before ``done-streaming``, then ALL of them do
    # (positions are monotonic in a single rendered stream).
    last_call_idx = rendered.rfind("⏵ sys_terminal_launch")
    assert 0 <= last_call_idx < done_idx, (
        f"Last ``⏵ sys_terminal_launch`` call line (at index {last_call_idx}) "
        f"must appear BEFORE ``{_DONE_MARKER}`` (at index "
        f"{done_idx}). If last_call_idx > done_idx, at least one "
        f"call line rendered AFTER the agent's final text — the "
        f"``ToolCallInProgress`` inline emission regressed and "
        f"call lines bunch at the ``response.completed`` flush, "
        f"which fires after all text deltas."
    )
    last_panel_idx = rendered.rfind("╭─ ")
    assert 0 <= last_panel_idx < done_idx, (
        f"Last result panel (at index {last_panel_idx}) must "
        f"appear BEFORE ``{_DONE_MARKER}`` (at index {done_idx}). "
        f"If last_panel_idx > done_idx, at least one result panel "
        f"rendered AFTER the agent's final text — either the "
        f"inline emit of ``function_call_output`` in "
        f"``_dispatch_action_required`` regressed, or "
        f"``BlockStream`` reverted to deferring "
        f"``ToolResultBlock`` until the next text/response "
        f"transition. THIS IS THE DECISIVE ASSERTION for gap "
        f"#49 — the user-visible bug was that result panels "
        f"rendered after final text."
    )
