"""Phase 0 characterization test — Ctrl+G debug overview toggle.

Submits one prompt so the session has at least one message,
hits ``Ctrl+G`` to open the debug overview, asserts the
sidebar + overview pane paints (``Session: main`` header +
``debug:`` footer hints), then hits ``q`` (which the REPL
binds alongside Esc in overview mode) to return to main mode
and asserts the normal status bar is back.

Tab/shift-tab cycling between multiple overview targets is
exercised by the sub-agent and terminal overview tests where
more than one target actually exists. With only the main
session, Tab wraps to itself and prompt-toolkit may suppress
the frame as a no-op redraw — so cycling is intentionally not
part of this test.

We use ``q`` rather than Esc for the close because prompt-
toolkit waits for the escape-sequence timeout (~100 ms) before
dispatching a bare Esc, which introduces flake risk; ``q`` is
an equivalent binding with immediate dispatch. The Esc binding
still ships and is exercised indirectly by the full keybinding
suite via ``@kb.add("escape", filter=overview_mode_filter)``.

**What breaks if this fails:**
- ``omnigent.cli`` removes or reorders the
  ``@kb.add("c-g")`` / overview-mode bindings.
- ``_collect_overview_targets`` stops producing a ``main``
  target, so the sidebar renders empty and the test's anchor
  disappears.
- ``_overview_footer_fragments`` changes its hint text so the
  footer-detection substring no longer matches.
- The ``Esc`` overview-mode binding stops returning the layout
  to main mode (the status bar wouldn't reappear).

Design reference: ``designs/OMNIGENT_INTEGRATION.md`` §Phase 0
REPL pexpect suite — "Ctrl+G debug overview".
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
    submit_prompt,
    wait_for_ready,
)
from tests.e2e.omnigent._repl_test_helpers import drain_for
from tests.e2e.omnigent._snapshot import compare_snapshot

_MODEL = resolve_model("databricks-gpt-5-mini", key=__name__)
_HARNESS = "openai-agents"
_PROMPT = "say ok"

# Substrings that identify overview mode. The sidebar prints the
# target label ("main" for the top-level session) and the
# overview pane paints "Session: main" followed by "Session ID:
# ...". The footer prints a hint line starting with "debug:".
_OVERVIEW_SESSION_HEADER = "Session: main"
_OVERVIEW_FOOTER_HINT = "debug:"

_SPAWN_TIMEOUT = 60.0
_BOOT_TIMEOUT = 30.0
_RUNNING_TIMEOUT = 20.0
_COMPLETION_TIMEOUT = 60.0
_EXIT_TIMEOUT = 15.0
_OVERVIEW_DRAIN_TIMEOUT = 5.0


def test_repl_ctrl_g_overview_toggle(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
) -> None:
    """
    Toggle into the debug overview with Ctrl+G and back out
    with Esc.

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
        submit_prompt(child, _PROMPT)
        await_turn_complete(
            child,
            running_timeout=_RUNNING_TIMEOUT,
            completion_timeout=_COMPLETION_TIMEOUT,
        )
        # Open the debug overview. The binding schedules an
        # async ``_toggle_overview`` via create_background_task,
        # so we wait for the overview pane to paint "Session:
        # main" — that's the earliest moment at which overview
        # mode is definitively active.
        child.sendcontrol("g")
        child.expect(_OVERVIEW_SESSION_HEADER, timeout=_OVERVIEW_DRAIN_TIMEOUT)
        # Drain any trailing overview-frame bytes so tab_drain
        # only captures Tab-triggered output. The accumulated
        # overview_stripped is the pre-expect buffer plus the
        # matched header plus any follow-up frames (footer,
        # sidebar refinement) captured by the short drain.
        overview_tail = drain_for(child, 1.0)
        overview_stripped = (
            strip_ansi(child.before or "") + _OVERVIEW_SESSION_HEADER + strip_ansi(overview_tail)
        )
        # Exit overview with 'q'. The REPL binds both Esc and
        # q to close-overview in the overview-mode filter. We
        # send 'q' rather than Esc because prompt-toolkit's key
        # parser waits for the escape-sequence timeout (~100 ms
        # by default) before registering a bare Esc, which
        # introduces nondeterminism on slower systems. 'q' has
        # no such timer — it dispatches immediately.
        child.send("q")
        # On close-overview, the main layout's status bar
        # repaints with ``state: sleeping``. Its reappearance
        # after the overview-drain completed (where the status
        # bar was hidden) is proof we returned to main mode.
        escape_frame_drain = drain_for(child, _OVERVIEW_DRAIN_TIMEOUT)
        escape_drain = strip_ansi(escape_frame_drain)
        clean_exit(child, timeout=_EXIT_TIMEOUT)
        exit_code = child.exitstatus
    finally:
        if not child.closed:
            child.close(force=True)

    observed: dict[str, Any] = {
        "exit_code": exit_code,
        # "Session: main" is painted by
        # ``_render_overview_session_text`` at the top of the
        # overview pane for the main target. Its presence
        # proves the layout flipped into overview mode AND the
        # main target was selected.
        "overview_session_header_present": _OVERVIEW_SESSION_HEADER in overview_stripped,
        # The overview footer hint is painted by
        # ``_overview_footer_fragments``. Proves the overview
        # layout (sidebar + pane + footer) is active, not just
        # a partial render.
        "overview_footer_hint_present": _OVERVIEW_FOOTER_HINT in overview_stripped,
        # The main-mode status bar uses the ``state: sleeping``
        # substring. Its reappearance after Esc proves the
        # overview layout is gone and main mode is active
        # again.
        "main_mode_restored_after_esc": "state: sleeping" in escape_drain,
    }
    diffs = compare_snapshot("test_repl_ctrl_g_overview", observed)
    assert diffs == [], (
        "Snapshot mismatch for Ctrl+G overview toggle:\n"
        + "\n".join(diffs)
        + f"\n\noverview stripped (last 2000):\n"
        f"{overview_stripped[-2000:]}"
        f"\n\nescape stripped (last 1000):\n"
        f"{escape_drain[-1000:]}"
    )
