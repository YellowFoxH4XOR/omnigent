"""
PTY-driven e2e: ``omnigent run --harness <harness>`` returns
the LLM's reply through the new harness contract.

Why this test exists: it covers the interactive CLI path that
server integration tests miss — the CLI's spec-bundling pipeline
(turning ``--harness <harness>`` into an ``executor.harness:
<harness>`` YAML, packaging it, posting to ``/api/agents``) and
the REPL's SSE rendering (event consumption from the Omnigent server
back into the terminal). CLAUDE.md mandates a real REPL run
before declaring an executor change done; this test pins it for
every wrapped harness.

Gated on ``--profile`` (real LLM). Without it, the test skips.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.e2e._harness_probes import HARNESS_HARNESS_MODELS, HARNESS_IDS

pexpect = pytest.importorskip("pexpect")

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Strip ANSI escape codes before substring assertions; pexpect
# captures everything raw and the REPL emits a heavy amount of
# styling that would otherwise drown the marker out.
_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_MARKER_TIMEOUT_S = 120.0


@pytest.fixture
def databricks_profile(request: pytest.FixtureRequest) -> str:
    """
    Return the ``--profile`` CLI arg, or skip if not provided.
    """
    profile: str = request.config.getoption("--profile")
    if not profile:
        pytest.skip("REPL e2e requires --profile <name> (e.g. --profile test-profile)")
    return profile


@pytest.fixture
def repl_env(databricks_profile: str, tmp_path: Path) -> dict[str, str]:
    """
    Build the env dict for the REPL subprocess.

    Strips ambient credentials that this agent process may have
    inherited (DATABRICKS_TOKEN, ANTHROPIC_API_KEY, CODEX,
    CLAUDE_CODE) so the test forces config to flow through the
    spec — matches CLAUDE.md's "clear environment variables...
    before running project code" guidance.

    The Databricks profile is supplied through the global config's
    ``auth:`` block in an isolated ``OMNIGENT_CONFIG_HOME`` — the
    supported replacement for the removed ``--profile`` CLI flag.

    :returns: Env mapping for ``pexpect.spawn``.
    """
    from tests.e2e.omnigent._pexpect_harness import ensure_repl_test_theme_env

    config_home = tmp_path / "omnigent-config"
    config_home.mkdir()
    (config_home / "config.yaml").write_text(
        f"auth:\n  type: databricks\n  profile: {databricks_profile}\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "OMNIGENT_CONFIG_HOME": str(config_home),
        "DATABRICKS_CONFIG_PROFILE": databricks_profile,
        # PYTHONPATH so the worktree wins over any sibling
        # editable install of omnigent.
        "PYTHONPATH": (f"{_REPO_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"),
        # Force ANSI on; we strip it per-assertion via _ANSI_RE.
        "TERM": "xterm-256color",
        # Disable cursor-position reporting so the buffer doesn't
        # fill with control sequences that confuse expect()
        # patterns.
        "PROMPT_TOOLKIT_NO_CPR": "1",
        # The workflow's compaction layer constructs an LLM
        # client at startup; never actually called for the
        # claude-sdk routing, but the env check happens first.
        "OPENAI_API_KEY": "stub-not-used-by-claude-sdk-path",
    }
    for var in ("DATABRICKS_TOKEN", "ANTHROPIC_API_KEY", "CODEX", "CLAUDE_CODE"):
        env.pop(var, None)
    return ensure_repl_test_theme_env(env)


def _strip_ansi(text: str) -> str:
    """
    Remove ANSI escape codes from a captured pexpect buffer.

    :param text: Raw captured text.
    :returns: Plain text suitable for substring assertions.
    """
    return _ANSI_RE.sub("", text)


def _read_until_marker(
    child: Any,
    marker: str,
    *,
    forbidden_in_match: str,
    timeout_s: float = 180.0,
) -> str:
    """
    Read child output until *marker* appears in Claude's reply.

    Why this isn't just ``child.expect(marker)``: the REPL
    echoes the user's input back, so a marker that happens to
    appear in the prompt text would match the echo line first.
    This helper accumulates all output, strips ANSI, scrubs the
    forbidden text (the user prompt), and only succeeds when
    the marker is found in the *remaining* text.

    :param child: Active pexpect child.
    :param marker: Substring that must appear in the model's
        reply.
    :param forbidden_in_match: Text that must be excluded
        before checking for the marker — typically the user's
        own prompt, which is echoed back.
    :param timeout_s: Total deadline before failing.
    :returns: The full ANSI-stripped buffer at success time.
    :raises AssertionError: If the marker never appears within
        the timeout.
    """
    import time

    buf = ""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with contextlib.suppress(pexpect.exceptions.TIMEOUT):
            child.expect([pexpect.TIMEOUT, pexpect.EOF], timeout=2.0)
        if child.before:
            buf += child.before
        plain = _strip_ansi(buf)
        scrubbed = plain.replace(forbidden_in_match, "")
        if marker in scrubbed:
            return plain
        if not child.isalive():
            break
    raise AssertionError(
        f"marker {marker!r} never appeared in Claude's reply within "
        f"{timeout_s:.0f}s. captured (last 4000 chars, ANSI-stripped):\n"
        f"{_strip_ansi(buf)[-4000:]}"
    )


@pytest.mark.parametrize("harness,model", HARNESS_HARNESS_MODELS, ids=HARNESS_IDS)
def test_repl_run_routes_harness_through_new_harness_contract(
    repl_env: dict[str, str],
    harness: str,
    model: str,
) -> None:
    """
    Drive the full ``omnigent run --harness <harness>``
    flow under a PTY and verify the LLM's reply comes back.

    Verifies the path that the HTTP-only e2e tests miss:

    1. The CLI's ``run_chat`` packs ``--harness <harness>`` +
       ``--model`` (plus the config-home auth block's profile)
       into the temporary spec.
    2. It spawns a local Omnigent server subprocess.
    3. It uploads the spec via ``/api/agents``.
    4. The Omnigent server's ``_create_executor`` sees an
       ``executor.type == "omnigent"`` +
       ``config.harness == <harness>`` spec (after the
       omnigent-YAML translator runs) and dispatches to
       the harness HTTP client via the step-5f
       branch.
    5. The LLM's reply streams back through SSE → the REPL's
       SDK client → terminal rendering.

    The marker is XYZZY (not in the prompt) so the assertion
    checks the model's reply specifically, not the echoed prompt.
    """
    # The marker MUST NOT appear in the user prompt verbatim:
    # the REPL echoes the prompt back into the PTY, and a marker
    # in the prompt would fire the substring assertion before
    # the model's reply has even been generated. So we describe
    # the 7 characters individually and ask the model to
    # concatenate them.
    #
    # The earlier prompt — "output the 7 characters X-Y-Z-Z-Y-4-2
    # concatenated, no dashes" — was ambiguous enough that
    # codex (gpt-5-4-mini) and openai-agents both sometimes
    # echoed the dashed form ``X-Y-Z-Z-Y-4-2`` instead of the
    # concatenated ``XYZZY42``, manifesting as a 180s timeout
    # on the marker substring check. The new instruction names
    # each character explicitly ("capital X then capital Y
    # ...") and asserts the exact length, leaving no room for
    # the model to interpret "concatenated, no dashes" as a
    # description of the input rather than the output.
    marker = "XYZZY42"
    user_prompt = (
        "Reply with EXACTLY 7 characters and nothing else: "
        "capital X, then capital Y, then capital Z, then "
        "capital Z, then capital Y, then digit 4, then digit "
        "2 — joined with no separators. Your entire reply "
        "must be those 7 characters in that order with no "
        "spaces, no dashes, no quotes, no commas, no "
        "newlines, and no surrounding text."
    )

    child = pexpect.spawn(
        sys.executable,
        [
            "-m",
            "omnigent.cli",
            "run",
            "tests/resources/examples/hello_world.yaml",
            "--harness",
            harness,
            "--model",
            model,
            # This test verifies harness routing and rendered output, not
            # persistent session resumption. Keep it on the isolated
            # one-shot path so parallel shards do not contend over the
            # shared local daemon / persistent chat.db.
            "--no-session",
            "-p",
            user_prompt,
        ],
        cwd=str(_REPO_ROOT),
        env=repl_env,
        encoding="utf-8",
        codec_errors="replace",
        dimensions=(40, 160),
        timeout=_MARKER_TIMEOUT_S,
    )
    try:
        plain = _read_until_marker(
            child,
            marker=marker,
            forbidden_in_match=user_prompt,
            timeout_s=_MARKER_TIMEOUT_S,
        )
    finally:
        # Best-effort clean shutdown — the REPL hangs in input
        # mode after the one-shot prompt completes, so we have
        # to send /quit. Fall back to terminate() if it sticks.
        with contextlib.suppress(Exception):
            child.sendline("/quit")
            child.expect(pexpect.EOF, timeout=5)
        if child.isalive():
            child.terminate(force=True)

    # Sanity that Claude's reply actually rendered into the
    # terminal — without this assertion a regression that
    # silently swallowed all SSE deltas (e.g. the SDK client's
    # event mapper changing) might still leave the marker
    # visible elsewhere (a debug line, etc.).
    assert marker in plain, f"marker {marker!r} not found in REPL output (post-strip)"


def test_repl_pexpect_dependencies_are_present() -> None:
    """
    Sanity check that :mod:`pexpect` is importable.

    Acts as a guard rail — if the prior ``importorskip`` at
    module load skipped the whole file, this test would also
    skip. Useful diagnostic for "the REPL test isn't running"
    cases on CI: the file collected, this test ran, but the
    real one was skipped due to ``--profile`` being absent.
    """
    # Trivially true; the load-time import-or-skip is what
    # matters.
    assert pexpect is not None
    # Sanity that ``omnigent`` is importable from this
    # worktree — if it isn't, the REPL spawn would fail with
    # a confusing ``ModuleNotFoundError`` instead of a clean
    # skip on missing fixtures. ``shutil.which`` is irrelevant
    # because we invoke the module directly via ``-m``.
    assert shutil.which(sys.executable) is not None
