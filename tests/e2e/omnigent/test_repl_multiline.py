"""Phase 0 characterization test — multi-line input via Ctrl+J.

Drives the REPL under pexpect, types the first half of a
prompt, sends ``Ctrl+J`` to insert a newline mid-input, types the
second half, and finally submits with Enter. Asserts the full
multi-line message reached the agent by looking for BOTH halves
in the rendered ``You>`` banner that the REPL echoes to
scrollback before streaming the assistant response.

**What breaks if this fails:**
- ``omnigent.cli`` removes the ``@kb.add("c-j", ...)`` binding
  that maps Ctrl+J to ``insert_text("\\n")`` — multi-line
  composition is a core REPL affordance.
- ``_format_user_message`` stops preserving interior newlines
  (folds the message into one line), so multi-line input still
  submits but is no longer faithfully echoed.
- The prompt-toolkit submit binding regresses to fire on the
  *first* newline rather than the explicit Enter keypress —
  would truncate the prompt at the Ctrl+J boundary.

Design reference: ``designs/OMNIGENT_INTEGRATION.md`` §Phase 0
REPL pexpect suite — "Multi-line input".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests._model_pools import resolve_model
from tests.e2e.omnigent._pexpect_harness import (
    await_turn_complete,
    clean_exit,
    spawn_omnigent_run,
    strip_ansi,
    wait_for_ready,
)
from tests.e2e.omnigent._snapshot import compare_snapshot

# openai-agents is used because it doesn't require the
# ``~/.databrickscfg`` patch — the env vars the credentials
# fixture sets are sufficient for this harness.
_MODEL = resolve_model("databricks-gpt-5-mini", key=__name__)
_HARNESS = "openai-agents"

# Two distinguishable halves so the assertion survives ANSI
# wrapping and prompt-toolkit's redraw minimization. Both strings
# must appear in the rendered buffer between the prompt
# submission and the assistant's reply — proves Ctrl+J inserted
# the newline instead of eating the input.
_FIRST_LINE = "line-one-alpha"
_SECOND_LINE = "line-two-beta"

_SPAWN_TIMEOUT = 60.0
_BOOT_TIMEOUT = 30.0
_RUNNING_TIMEOUT = 20.0
_COMPLETION_TIMEOUT = 60.0
_EXIT_TIMEOUT = 15.0


def test_repl_multiline_ctrl_j_insert(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
) -> None:
    """
    Compose a two-line prompt using Ctrl+J and submit with Enter.

    The harness's ``submit_prompt`` helper always appends a CR and
    finalizes the input; this test intentionally does NOT use it
    because the whole point is splitting the send with a Ctrl+J in
    the middle. Instead it types the first half, sends a Ctrl+J,
    types the second half, then sends a plain CR to submit.

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
        # Type the first line, insert a newline via Ctrl+J, type
        # the second line, then submit with CR. Using
        # sendcontrol("j") rather than sending the raw byte so
        # the key event reaches prompt-toolkit's binding registry
        # as the canonical "c-j" it filters on.
        child.send(_FIRST_LINE)
        child.sendcontrol("j")
        child.send(_SECOND_LINE)
        child.send("\r")
        turn = await_turn_complete(
            child,
            running_timeout=_RUNNING_TIMEOUT,
            completion_timeout=_COMPLETION_TIMEOUT,
        )
        clean_exit(child, timeout=_EXIT_TIMEOUT)
        exit_code = child.exitstatus
    finally:
        if not child.closed:
            child.close(force=True)

    # Merge with the post-exit drain so the assertion survives
    # whichever render frame the echo happens to land in.
    combined_stripped = turn.stripped + "\n" + strip_ansi(child.before or "")

    observed: dict[str, Any] = {
        "exit_code": exit_code,
        # Both halves must appear — confirms Ctrl+J kept the
        # first half in the input buffer instead of submitting
        # early, AND Enter eventually submitted the combined
        # text.
        "first_line_present": _FIRST_LINE in combined_stripped,
        "second_line_present": _SECOND_LINE in combined_stripped,
        # The "You>" banner is the REPL's deterministic echo of
        # a submitted prompt. If only the input buffer were shown
        # (no submission), this would be absent — the banner
        # proves ``_submit_input`` actually ran with the full
        # multi-line text.
        "user_banner_present": "You>" in combined_stripped,
        "agent_banner_present": "Agent>" in combined_stripped,
    }
    diffs = compare_snapshot("test_repl_multiline", observed)
    assert diffs == [], (
        "Snapshot mismatch for multi-line Ctrl+J input:\n"
        + "\n".join(diffs)
        + f"\n\nstripped buffer (last 2000):\n"
        f"{combined_stripped[-2000:]}"
    )
