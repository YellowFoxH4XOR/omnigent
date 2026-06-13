"""Phase 0 characterization test — mid-turn cancellation re-arms the REPL.

Submits a long-running prompt that produces visible streaming
text, then issues the REPL's documented mid-turn cancellation
mechanism. Asserts (a) the REPL stays alive (does NOT exit)
after the cancellation, and (b) a follow-up prompt is accepted
and produces a new assistant response — proving the streaming
consumer re-armed for the next turn instead of getting stuck
in a half-cancelled state.

**About Ctrl+C in the current REPL:** the Omnigent REPL today
binds ``c-c`` to ``event.app.exit(result=None)`` (cli.py
``_interrupt``) — Ctrl+C *exits* the REPL rather than
cancelling the current turn. The design doc's pexpect spec
("send Ctrl+C, assert the REPL stays alive") describes the
target behavior under the SSE bridge that lands in phase 4 of
the integration; characterizing it against today's REPL would
require asserting on exit, which doesn't catch the regression
the design names ("SSE consumer doesn't re-arm after
cancellation"). Instead this test exercises the REPL's actual
documented cancellation path — the ``/cancel`` slash command
in ``_submit_input`` (cli.py line 2167) which calls
``session.cancel_current_turn()`` — because that's the path
that has the same shape as the future Ctrl+C behavior:
in-flight turn is cancelled, the REPL stays alive, and the
next prompt re-uses the same streaming consumer. When phase 4
swaps Ctrl+C from ``app.exit`` to a cancellation call, this
test should be re-pointed to send Ctrl+C and the same
assertions will continue to hold.

**What breaks if this fails:**
- ``Session.cancel_current_turn`` regresses so the in-flight
  turn doesn't actually stop (status bar wouldn't return to
  ``state: sleeping``).
- The REPL's stream consumer (the ``_render_one_turn`` loop
  in cli.py) fails to re-arm after cancellation — would
  manifest as the follow-up prompt never reaching ``running``
  or never returning to ``sleeping``. **This is the regression
  the design identified as the highest-priority interrupt
  test.**
- ``_submit_input``'s ``/cancel`` branch (lines 2167-2186)
  changes shape so cancellation is silently dropped.

Design reference: ``designs/OMNIGENT_INTEGRATION.md`` §Phase 0
REPL pexpect suite — "Ctrl+C interrupt mid-stream".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests._model_pools import resolve_model
from tests.e2e.omnigent._pexpect_harness import (
    STATE_RUNNING,
    STATE_SLEEPING,
    await_turn_complete,
    clean_exit,
    spawn_omnigent_run,
    submit_prompt,
    wait_for_ready,
)
from tests.e2e.omnigent._snapshot import compare_snapshot

# openai-agents top-level harness — supports turn cancellation
# (supports_turn_cancellation == True for streaming-capable
# harnesses), which the ``/cancel`` slash command requires.
_MODEL = resolve_model("databricks-gpt-5-mini", key=__name__)
_HARNESS = "openai-agents"

# A prompt that produces visibly-long streaming output so the
# cancellation lands while the turn is mid-flight rather than
# right after the assistant finishes. Counting forces many
# tokens; "slowly" nudges the model toward verbose, evenly-
# paced output.
_LONG_PROMPT = (
    "Count slowly from 1 to 100. Print one number per line, "
    "with a short verbal description after each number "
    "explaining what the number could mean. Take your time."
)
_FOLLOW_UP_PROMPT = "say hi"

# Cancellation status line emitted by ``_submit_input`` when
# ``cancel_current_turn`` succeeds. Matching this proves the
# cancellation request was accepted by the session, not just
# silently dropped.
_CANCEL_STATUS_REQUESTED = "Cancellation requested."

_SPAWN_TIMEOUT = 60.0
_BOOT_TIMEOUT = 30.0
_RUNNING_TIMEOUT = 20.0
# Initial turn must be long-lived enough for cancellation to
# land mid-stream. Setting a generous ceiling lets a slow LLM
# still hit the cancel path; if the turn finishes too fast we
# still verify cancellation was attempted via the status line.
_INITIAL_RUNNING_BUDGET = 30.0
_CANCEL_ACK_TIMEOUT = 30.0
_FOLLOWUP_RUNNING_TIMEOUT = 30.0
_FOLLOWUP_COMPLETION_TIMEOUT = 60.0
_EXIT_TIMEOUT = 15.0


def test_repl_cancel_re_arms_for_next_turn(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
) -> None:
    """
    Submit a long prompt, ``/cancel`` it mid-stream, then
    submit a follow-up and verify it completes — proving the
    REPL stayed alive AND the streaming consumer re-armed.

    :param omnigent_python: Interpreter with omnigent +
        openai-agents installed.
    :param omnigent_repo_root: Working directory for the
        subprocess.
    :param omnigent_credentials_env: Env vars with
        ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` /
        ``DATABRICKS_CONFIG_PROFILE`` populated.
    """
    yaml_path = omnigent_repo_root / "tests" / "resources" / "examples" / "hello_world.yaml"

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
        submit_prompt(child, _LONG_PROMPT)
        # Wait for the turn to actually start streaming — the
        # ``running`` transition marks the moment the executor
        # has accepted the prompt and is producing output. Only
        # after this is cancellation meaningful (cancelling a
        # not-yet-running turn would produce ``not_running``).
        child.expect(STATE_RUNNING, timeout=_INITIAL_RUNNING_BUDGET)
        # Submit ``/cancel`` via the slash-command path. This
        # is the REPL's documented mid-turn cancel: cli.py
        # ``_submit_input`` recognises it before the empty-text
        # check and calls ``session.cancel_current_turn()``.
        # Use the harness's submit_prompt rather than direct
        # writes so the CR semantics match the rest of the
        # suite.
        submit_prompt(child, "/cancel")
        # The ``Cancellation requested.`` status line is the
        # observable proof the cancel call returned a
        # ``cancelled`` status. Its absence within the budget
        # would mean either the cancel was silently dropped or
        # the session reported a non-cancellable state — both
        # failures the design test was designed to catch.
        child.expect(_CANCEL_STATUS_REQUESTED, timeout=_CANCEL_ACK_TIMEOUT)
        # After cancellation the session transitions back to
        # ``sleeping``. Wait for that to settle before
        # submitting the follow-up so we don't race an in-
        # flight stream-close with the new prompt.
        child.expect(STATE_SLEEPING, timeout=_CANCEL_ACK_TIMEOUT)
        # Follow-up prompt — proves the input area still
        # accepts text and the streaming consumer re-armed.
        # If the consumer were stuck after cancellation, the
        # follow-up would never transition to ``running`` (or
        # never return to ``sleeping``).
        submit_prompt(child, _FOLLOW_UP_PROMPT)
        followup_turn = await_turn_complete(
            child,
            running_timeout=_FOLLOWUP_RUNNING_TIMEOUT,
            completion_timeout=_FOLLOWUP_COMPLETION_TIMEOUT,
        )
        clean_exit(child, timeout=_EXIT_TIMEOUT)
        exit_code = child.exitstatus
    finally:
        if not child.closed:
            child.close(force=True)

    observed: dict[str, Any] = {
        "exit_code": exit_code,
        # The follow-up turn must produce assistant output —
        # the "Agent>" banner is emitted by
        # ``_format_assistant_message`` only when the model
        # actually returned text. Its absence after a
        # successful cancellation means the consumer didn't
        # re-arm.
        "follow_up_assistant_response_rendered": "Agent>" in followup_turn.stripped,
        # Follow-up's user-prompt echo must also be present —
        # proves the input area accepted submission for the
        # second turn (not just the cancellation).
        "follow_up_user_banner_rendered": "You>" in followup_turn.stripped,
    }
    diffs = compare_snapshot("test_repl_ctrl_c_interrupt", observed)
    assert diffs == [], (
        "Snapshot mismatch for cancellation re-arm:\n"
        + "\n".join(diffs)
        + f"\n\nfollow-up turn stripped (last 2000):\n"
        f"{followup_turn.stripped[-2000:]}"
    )
