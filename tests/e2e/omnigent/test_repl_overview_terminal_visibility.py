"""Phase 0 characterization test — terminal visibility in overview.

Uses ``examples/terminal_workers.yaml`` so the REPL hosts a
terminal-supervisor with ``worker`` and ``shell`` terminal
definitions. Sends a prompt that instructs the supervisor to
launch a terminal session, waits for the spawn to land,
opens the overview, cycles to the terminal target, and asserts
the overview pane renders the tmux ``attach`` instruction line.

**What breaks if this fails:**
- ``_collect_overview_targets`` stops including terminal
  instances from ``Session._terminal_instances``.
- ``_render_overview_terminal_text`` regresses so the
  ``Attach: tmux -S <socket> attach -t <target>`` line is
  dropped.
- ``_terminal_attach_command`` stops composing the
  ``tmux -S ... attach`` string correctly.
- The ``sys_terminal_launch`` tool path silently fails to
  register the instance in ``session._terminal_instances``.

Design reference: ``designs/OMNIGENT_INTEGRATION.md`` §Phase 0
REPL pexpect suite — "Terminal visibility in overview".
"""

from __future__ import annotations

from pathlib import Path
from shutil import which
from typing import Any

import pytest

from tests._model_pools import resolve_model
from tests.e2e.omnigent._pexpect_harness import (
    clean_exit,
    spawn_omnigent_run,
    strip_ansi,
    submit_prompt,
    wait_for_ready,
)
from tests.e2e.omnigent._repl_test_helpers import drain_for
from tests.e2e.omnigent._snapshot import compare_snapshot

# Supervisor model — top-level open-responses harness pairs
# with databricks-gpt-5-4 per the YAML's defaults.
_MODEL = resolve_model("databricks-gpt-5-4", key=__name__)
_HARNESS = "open-responses"

# The YAML lists a "worker" terminal backed by ``isaac`` and a
# "shell" terminal backed by ``bash``. We ask for a specific
# terminal ("shell") because bash is always present; isaac is
# installed on dev machines but would need a skip-guard on CI
# if used. Instructing the supervisor to ``sys_terminal_launch``
# the shell makes the spawn deterministic.
_PROMPT = (
    'Call sys_terminal_launch(terminal="shell", session="probe") '
    "and then tell me you're done. Do not call any other tools."
)

# Target label produced by ``_collect_overview_targets`` for
# the shell terminal. Tool name is "shell", session key is
# "probe" per our prompt.
_TERMINAL_LABEL = "shell:probe"

# Markers that prove the pane rendered the terminal's attach
# instructions. Matching fragments of the ``tmux -S ... attach``
# command tolerates column-wrap artifacts from prompt-toolkit's
# painted layout (the sidebar and pane share rows).
_ATTACH_TMUX_MARKER = "tmux -S"
_ATTACH_VERB_MARKER = "attach"

_SPAWN_TIMEOUT = 60.0
_BOOT_TIMEOUT = 60.0
_RUNNING_TIMEOUT = 30.0
# Terminal launch is fast — spawning a bash under tmux is
# typically well under a second. 120 s is generous headroom
# that still bounds a runaway prompt.
_COMPLETION_TIMEOUT = 120.0
_EXIT_TIMEOUT = 15.0
_OVERVIEW_DRAIN_TIMEOUT = 6.0
_EXPECT_TERMINAL_TIMEOUT = 15.0


@pytest.fixture
def tmux_available() -> bool:
    """
    Skip-guard: terminal tools require ``tmux`` on PATH.

    :returns: True when ``tmux`` is available.
    """
    return which("tmux") is not None


def test_repl_overview_terminal_visibility(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
    tmux_available: bool,
) -> None:
    """
    Launch a terminal session through the supervisor, open the
    overview, cycle to the terminal target, and verify the
    tmux attach instructions render.

    :param omnigent_python: Interpreter with omnigent
        installed.
    :param omnigent_repo_root: Working directory for the
        subprocess.
    :param omnigent_credentials_env: Env vars with
        ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` /
        ``DATABRICKS_CONFIG_PROFILE`` populated.
    :param tmux_available: True when ``tmux`` is on PATH.
        Terminals use tmux unconditionally; if missing we fail
        loud per the phase 0 design's "no silent skips" rule.
    """
    if not tmux_available:
        pytest.fail(
            "tmux binary not found on PATH — terminal-tool tests "
            "require tmux to be installed (``brew install tmux``)."
        )

    yaml_path = omnigent_repo_root / "tests" / "resources" / "examples" / "terminal_workers.yaml"

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
        # Wait for the ``sys_terminal_launch`` tool call line.
        # Once it appears, the supervisor has invoked the
        # launch — ``_terminal_instances[("shell", "probe")]``
        # is registered (the dict
        # ``_collect_overview_targets`` scans). Note: unlike
        # ``sys_session_send``, the REPL's
        # ``_tool_display_name`` does NOT format
        # ``sys_terminal_launch`` with its args, so we match
        # the bare tool name. The ``• sys_terminal_launch
        # (<ms>)`` completion line then guarantees the launch
        # actually finished (so the instance is fully wired,
        # not mid-spawn).
        child.expect(
            r"• sys_terminal_launch \(\d+ms\)",
            timeout=_COMPLETION_TIMEOUT,
        )
        # Drain the completion line so the subsequent overview
        # render isn't masked by tool-call tail bytes.
        drain_for(child, 2.0)
        # Open the overview.
        child.sendcontrol("g")
        drain_for(child, _OVERVIEW_DRAIN_TIMEOUT)
        # Tab cycles through targets: main → shell:probe. The
        # overview pane then paints the terminal-specific
        # metadata via ``_render_overview_terminal_text``.
        child.send("\t")
        # Wait for the terminal pane's "Terminal:" header to
        # render. With only "main" and the shell terminal as
        # targets, one Tab lands us on the terminal. Note: the
        # terminal pane uses ``Terminal: <name>:<key>`` (not
        # ``Session:`` like managed agent sessions) — that's
        # the literal ``_render_overview_terminal_text`` line
        # 1691.
        child.expect(f"Terminal: {_TERMINAL_LABEL}", timeout=_EXPECT_TERMINAL_TIMEOUT)
        terminal_pane_tail = drain_for(child, _OVERVIEW_DRAIN_TIMEOUT)
        terminal_stripped = (
            strip_ansi(child.before or "")
            + f"Terminal: {_TERMINAL_LABEL}"
            + strip_ansi(terminal_pane_tail)
        )
        clean_exit(child, timeout=_EXIT_TIMEOUT)
        exit_code = child.exitstatus
    finally:
        if not child.closed:
            child.close(force=True)

    observed: dict[str, Any] = {
        "exit_code": exit_code,
        # Sidebar/header must carry the shell terminal's label.
        # Absent if ``_terminal_instances`` was empty at the
        # overview-target-collection moment.
        "terminal_label_present": _TERMINAL_LABEL in terminal_stripped,
        # The ``Attach:`` line produced by
        # ``_terminal_attach_command`` embeds ``tmux -S`` — the
        # socket-prefix flag. Its presence is the defining
        # signal that attach instructions were generated and
        # rendered.
        "tmux_socket_flag_rendered": _ATTACH_TMUX_MARKER in terminal_stripped,
        # The ``attach`` verb from the same composed command.
        # Pairing this with ``tmux -S`` nails the check to the
        # attach-instructions line rather than to some passing
        # mention of tmux elsewhere.
        "attach_verb_rendered": _ATTACH_VERB_MARKER in terminal_stripped,
    }
    diffs = compare_snapshot("test_repl_overview_terminal_visibility", observed)
    assert diffs == [], (
        "Snapshot mismatch for terminal overview visibility:\n"
        + "\n".join(diffs)
        + f"\n\nterminal stripped (last 2500):\n"
        f"{terminal_stripped[-2500:]}"
    )
