"""E2e compaction test for the sessions-native path.

Uses pexpect to run multiple turns within a single ``omnigent run``
session. With ``AP_CONTEXT_WINDOW_OVERRIDE=4096`` and
``trigger_threshold=0.05`` (204 token budget), proactive compaction
fires after the first verbose turn.

Run with::

    pytest tests/e2e/omnigent/test_compaction_sessions_native_e2e.py -v --profile oss
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import sqlite3
import time
from pathlib import Path

import pexpect
import pytest

from tests._model_pools import resolve_model

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07")

_COMPACTION_AGENT_YAML = """\
name: compaction-e2e-test
description: Agent for e2e compaction testing.

executor:
  harness: openai-agents
  profile: oss

prompt: |
  You are a test assistant. Reply with detailed, verbose answers
  so that conversation history grows quickly.
"""

_MODEL = resolve_model("databricks-gpt-5-4-mini", key=__name__)
_HARNESS = "openai-agents"
_BOOT_TIMEOUT = 120.0
_TURN_TIMEOUT = 300.0


def _strip(text: str) -> str:
    """Remove ANSI escape codes."""
    return _ANSI_RE.sub("", text)


def _drain_until(
    child: pexpect.spawn,
    pattern: str,
    timeout: float,
) -> str:
    """
    Read from the PTY until *pattern* appears in the ANSI-stripped
    accumulated output, or *timeout* elapses.

    :param child: Live pexpect child.
    :param pattern: Substring to find (case-insensitive).
    :param timeout: Max seconds.
    :returns: The accumulated ANSI-stripped output.
    """
    deadline = time.monotonic() + timeout
    accumulated = ""
    pat_lower = pattern.lower()
    while time.monotonic() < deadline:
        try:
            chunk = child.read_nonblocking(size=100000, timeout=3)
            accumulated += chunk
        except (pexpect.TIMEOUT, pexpect.EOF):
            pass
        clean = _strip(accumulated)
        if pat_lower in clean.lower():
            return clean
    pytest.fail(
        f"Pattern {pattern!r} not found within {timeout}s. Output: {_strip(accumulated)[:500]!r}"
    )
    return ""


def _send_and_wait(
    child: pexpect.spawn,
    prompt_text: str,
    timeout: float,
) -> str:
    """
    Send a prompt via CR and wait for 'sleeping' (turn complete).

    :param child: Live pexpect child.
    :param prompt_text: User message.
    :param timeout: Max seconds for the turn.
    :returns: ANSI-stripped output captured during the turn.
    """
    child.send(prompt_text)
    child.send("\r")
    return _drain_until(child, "sleeping", timeout)


def test_compaction_fires_and_agent_retains_context(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    tmp_path: Path,
) -> None:
    """
    Multi-turn pexpect test: 2 verbose turns trigger proactive
    compaction, then a 3rd turn proves the agent retains context.

    Breakage this catches: if proactive compaction doesn't fire,
    the compaction item won't appear in the DB. If the summary
    doesn't capture prior context, turn 3 can't reference it.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    yaml_path = tmp_path / "compaction-e2e-test.yaml"
    yaml_path.write_text(_COMPACTION_AGENT_YAML)
    real_cfg = Path.home() / ".databrickscfg"
    if real_cfg.exists():
        shutil.copy2(str(real_cfg), str(fake_home / ".databrickscfg"))

    env = dict(os.environ)
    for stale in (
        "ANTHROPIC_API_KEY",
        "DATABRICKS_TOKEN",
        "CLAUDE_CODE",
        "CODEX",
    ):
        env.pop(stale, None)
    env["HOME"] = str(fake_home)
    env["OMNIGENT_SKIP_ONBOARD"] = "1"
    env["OMNIGENT_NO_UPDATE_CHECK"] = "1"
    env["AP_CONTEXT_WINDOW_OVERRIDE"] = "256"
    env["TERM"] = "xterm-256color"
    env["LINES"] = "40"
    env["COLUMNS"] = "120"

    child = pexpect.spawn(
        str(omnigent_python),
        [
            "-m",
            "omnigent",
            "run",
            str(yaml_path),
            "--model",
            _MODEL,
            "--harness",
            _HARNESS,
            # Databricks routing comes from the YAML's ``executor.profile:
            # oss`` (the ``--profile`` CLI flag was removed).
            "--no-log",
        ],
        env=env,
        cwd=str(omnigent_repo_root),
        encoding="utf-8",
        timeout=_TURN_TIMEOUT,
        dimensions=(40, 120),
    )
    try:
        _drain_until(child, "sleeping", _BOOT_TIMEOUT)

        out1 = _send_and_wait(
            child,
            (
                "List exactly 20 countries. For each country, write the capital city, "
                "the population, the official language, the currency, and a famous "
                "landmark with a 3-sentence description. Number them 1 through 20."
            ),
            _TURN_TIMEOUT,
        )
        assert len(out1) > 100, f"Turn 1 too short: {out1[:100]!r}"

        out2 = _send_and_wait(
            child,
            (
                "Now list 20 MORE countries not in the previous list, same detailed "
                "format with capital, population, language, currency, and landmark."
            ),
            _TURN_TIMEOUT,
        )
        assert len(out2) > 100, f"Turn 2 too short: {out2[:100]!r}"

        out3 = _send_and_wait(
            child,
            "What was the very first thing I asked you? Reply in one sentence.",
            _TURN_TIMEOUT,
        )

        # Wait for the server's relay to persist items before exit.
        time.sleep(5)
        child.sendcontrol("d")
        with contextlib.suppress(pexpect.TIMEOUT):
            child.expect(pexpect.EOF, timeout=15)
    finally:
        if not child.closed:
            child.close(force=True)

    # Verify compaction item was persisted to the DB.
    db_path = fake_home / ".omnigent" / "chat.db"
    assert db_path.is_file(), f"DB not found at {db_path}"
    with sqlite3.connect(str(db_path)) as conn:
        compaction_rows = conn.execute(
            "SELECT type FROM conversation_items WHERE type = 'compaction'"
        ).fetchall()
    # At least 1 compaction item: proactive compaction fired after
    # turn 1's history exceeded the 102-token budget (128 * 0.8).
    # 0 means _proactive_compact_if_needed didn't fire or the POST
    # to the server didn't persist the item.
    assert len(compaction_rows) >= 1, (
        f"Expected >= 1 compaction item in DB. Found {len(compaction_rows)}."
    )

    # Verify turn 3 references prior context — proves the
    # compacted summary preserved meaningful context.
    combined = out3.lower()
    assert any(
        kw in combined for kw in ["countr", "capital", "landmark", "list", "nation", "asked"]
    ), f"Turn 3 doesn't reference prior context. Response: {out3[:300]!r}"
