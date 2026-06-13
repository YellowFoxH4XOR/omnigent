"""
End-to-end: ``omnigent run`` starts the REPL without
leaking omnigent server boot chatter onto the terminal.

Under ``omnigent chat``, the server runs as a subprocess with
``stdout=stderr=DEVNULL`` so DBOS / alembic / mlflow init
messages never reach the user's terminal. Under Omnigent mode, the
server runs **in-process** via ``httpx.ASGITransport`` — every
logger in the process shares the REPL's terminal. Without
suppression the user sees ~30 lines of DBOS init banners,
sqlite schema migration progress, and a DBOS Conductor URL
advisory pushing the prompt-toolkit frame off-screen.

**What breaks if this fails:**

- :func:`omnigent.cli._quiet_omnigent_server_logging` (or its
  successor) stops suppressing DBOS / alembic / mlflow loggers.
- Someone reintroduces a logger or ``print`` on the Omnigent mode
  boot path that bypasses the suppression.
- A new dependency with a chatty default logger gets pulled in
  during ``create_app`` and isn't added to the quiet list.

The test grep-matches substrings every affected user would see —
it doesn't care about exact formatter output, just that no
``DBOS`` / ``alembic`` / ``Applying DBOS SQLite`` banner
leaked through.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from tests._model_pools import resolve_model
from tests.e2e.omnigent._pexpect_harness import (
    clean_exit,
    spawn_omnigent_run,
    strip_ansi,
    wait_for_ready,
)
from tests.e2e.omnigent._repl_test_helpers import drain_for

_YAML_REL = "tests/resources/examples/coding_supervisor.yaml"
_MODEL = resolve_model("databricks-gpt-5-mini", key=__name__)
_HARNESS = "openai-agents"

# Substrings that appear in the DBOS / alembic / mlflow boot
# output (see the user's reported output). Any one of these
# showing up in the pre-REPL buffer is a regression; the
# suppression must cover all four source libraries.
_FORBIDDEN_BOOT_MARKERS: tuple[str, ...] = (
    # DBOS's custom formatter: "10:58:52 [    INFO] (dbos:...)"
    # — anchor on "(dbos:" so we catch every record regardless
    # of timestamp.
    "(dbos:",
    # DBOS's one-shot "initializing" banner.
    "Initializing DBOS",
    # The "Applying DBOS SQLite system database schema
    # migration N" block — 17 lines in the user's output.
    "Applying DBOS SQLite",
    # DBOS Conductor URL advisory — the most egregious because
    # it's an outbound-URL suggestion in what's supposed to be
    # an interactive REPL.
    "console.dbos.dev",
    # alembic's migration runner: "INFO [alembic.runtime...]"
    "alembic.runtime",
)

_SPAWN_TIMEOUT = 60.0
# Cold-boot of ``coding_supervisor.yaml`` under Omnigent mode —
# spawns the in-process Omnigent server (FastAPI + uvicorn + DBOS +
# alembic) and registers the supervisor + two sub-agents on
# a fresh DBOS db. Without Omnigent mode the same YAML boots in
# <10s; the in-process server adds ~30-60s. 120s keeps the
# test from flaking on cold starts while still surfacing
# genuine regressions (the failure mode fixed earlier was a
# SyntaxError within seconds, not a slow boot).
_BOOT_TIMEOUT = 120.0
_EXIT_TIMEOUT = 15.0
# After the REPL reaches idle, drain briefly to flush any late
# boot output that arrived after ``state: sleeping`` — e.g. a
# background DBOS thread's init line. Should be empty; we still
# capture to assert.
_POST_READY_DRAIN = 1.5


def test_run_omnigent_startup_does_not_leak_server_logs(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
    databricks_workspace: tuple[str, str],
) -> None:
    """
    Boot the REPL under Omnigent mode and verify none of the
    omnigent server's chatty init loggers leak to the
    terminal before the REPL takes over.

    :param omnigent_python: Shared interpreter fixture.
    :param omnigent_repo_root: Subprocess cwd.
    :param omnigent_credentials_env: Env with PAT + profile
        populated.
    :param databricks_workspace: ``(profile, host)`` from the
        active pytest ``--profile`` option — requested so the
        workspace the test fixtures expect is validated up front.
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
        # Capture EVERYTHING from process start through the
        # first ``state: sleeping`` marker. ``child.before``
        # holds the entire pre-match stream so any leaked boot
        # message is visible here.
        wait_for_ready(child, timeout=_BOOT_TIMEOUT)
        # After wait_for_ready succeeds, pexpect guarantees ``child.before``
        # contains the pre-match stream (pexpect types it ``Any | None``;
        # it is only ``None`` before any match runs).
        assert child.before is not None, "wait_for_ready populated no pre-match text"
        pre_ready = child.before
        # Background threads (DBOS queue listener, etc.) may
        # flush late log lines after the REPL paints. Catch
        # those too — the user still sees them.
        post_ready_tail = drain_for(child, _POST_READY_DRAIN)

        clean_exit(child, timeout=_EXIT_TIMEOUT)
    finally:
        if not child.closed:
            child.close(force=True)

    combined_stripped = strip_ansi(pre_ready) + strip_ansi(post_ready_tail)

    leaked = [marker for marker in _FORBIDDEN_BOOT_MARKERS if marker in combined_stripped]
    assert not leaked, (
        f"``omnigent run --omnigent`` leaked server boot output onto "
        f"the terminal. Leaked markers: {leaked}. Under ``omnigent "
        f"chat``, the server runs as a subprocess with stdout+"
        f"stderr=DEVNULL so these never reach the user; the "
        f"``--omnigent`` path must match that quiet behavior so the "
        f"REPL's prompt-toolkit frame isn't pushed off-screen by "
        f"~30 lines of DBOS / alembic init noise. "
        f"Combined stripped output (last 4000 chars):\n"
        f"{combined_stripped[-4000:]}"
    )


def test_run_prompt_mode_is_headless_for_local_agent(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    databricks_workspace: tuple[str, str],
    tmp_path: Path,
) -> None:
    """
    ``omnigent run <agent> -p`` sends one Omnigent turn, prints the
    assistant text, and exits without starting the Rich REPL UI.

    This is the e2e regression for the headless prompt-mode bug. It deliberately
    matches the reported local command shape,
    ``examples/databricks_coding_agent.yaml -p`` — with Databricks
    routing coming from the global config's ``auth:`` block (the
    ``--profile`` CLI flag was removed from the omnigent CLI).
    """
    yaml_path = omnigent_repo_root / "examples" / "databricks_coding_agent.yaml"
    profile, _ = databricks_workspace
    prompt = "hi"
    argv = [
        str(omnigent_python),
        "-m",
        "omnigent",
        "run",
        str(yaml_path),
        "-p",
        prompt,
    ]

    # The removed ``--profile`` flag is replaced by an ``auth:`` block in
    # an isolated ``OMNIGENT_CONFIG_HOME`` — the supported source of
    # Databricks model/gateway routing for spawned CLIs.
    config_home = tmp_path / "omnigent-config"
    config_home.mkdir()
    (config_home / "config.yaml").write_text(
        f"auth:\n  type: databricks\n  profile: {profile}\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["OMNIGENT_CONFIG_HOME"] = str(config_home)
    env["DATABRICKS_CONFIG_PROFILE"] = profile

    result = subprocess.run(
        argv,
        env=env,
        cwd=str(omnigent_repo_root),
        capture_output=True,
        text=True,
        timeout=180,
    )
    stripped = strip_ansi(result.stdout + result.stderr)

    assert result.returncode == 0, (
        f"headless prompt command failed; stdout/stderr:\n{stripped[-4000:]}"
    )
    assert "What can I help you with" in stripped, (
        "headless prompt command did not print assistant output; stdout/stderr:\n"
        f"{stripped[-4000:]}"
    )
    forbidden = [
        "Traceback (most recent call last)",
        "AttributeError:",
        "╭",
        "╰",
        "state: sleeping",
        "Use /help",
    ]
    leaked = [item for item in forbidden if item in stripped]
    assert not leaked, (
        "``omnigent run -p`` should be headless and must not "
        f"print tracebacks or REPL UI. Leaked markers: {leaked}. "
        f"Output tail:\n{stripped[-4000:]}"
    )
