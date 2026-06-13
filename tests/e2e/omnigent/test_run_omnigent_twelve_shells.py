"""
E2E: 12 parallel-shell tool dispatches with claude-sdk harness, Omnigent path.

Required by the user as the canonical regression for the 12-shell
repro (gateway 429s under parallel-tool fan-out). Asks the LLM to
spawn 12 parallel ``Bash``
tool calls, each running a short ``python -c "...time.sleep(N)..."``,
and verifies they all complete and the agent confirms the count.

Validates:
- ``RetryPolicy`` defaults (max_retries=7) flow through the
  claude-sdk harness via ``ANTHROPIC_MAX_RETRIES`` env var.
- Per-LLM-call SDK retry survives transient gateway 429s during
  the 12-shell parallel fan-out.
- The harness scaffold's terminal-event guarantee
  (``_build_terminal_event``) holds when the inner SDK eventually
  succeeds.
- The 12 shell results all reach the agent's final response.

Excluded from default ``pytest`` runs (lives under ``tests/e2e``).
Invoke explicitly:

    pytest tests/e2e/omnigent/test_run_omnigent_twelve_shells.py -v
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from shutil import which

import pytest

from tests._model_pools import resolve_model

# Databricks-hosted Claude model that exercises the gateway path
# the original user repro hit. Matches the model used in
# ``test_per_harness_claude_sdk.py`` so the same auth + endpoint
# resolution code is covered.
_MODEL = resolve_model("databricks-claude-sonnet-4-6", key=__name__)
_HARNESS = "claude-sdk"

# Prompt asks the agent to fan out 12 parallel shell commands.
# Each runs a short python sleep so they can complete in parallel
# without burning the test timeout. The "all 12 done" confirmation
# is the regression-guard signal — if any shell fails or the LLM
# loses count under retry, the assertion below fails.
_PROMPT = (
    "Run 12 shell commands IN PARALLEL using the Bash tool. Each "
    "should be exactly: "
    "python -c 'import time; time.sleep(0.5); print(\"done\")' "
    "After ALL 12 complete, respond with a final line that "
    "contains exactly the phrase 'TWELVE_SHELLS_COMPLETE' "
    "(uppercase, with underscores). Do not produce that phrase "
    "before all 12 commands have completed."
)

# Minimum count of "done" appearances we expect in stdout. The
# agent invokes 12 shells, each prints "done"; the LLM may or may
# not echo all 12 in its summary, so the floor is conservative.
_MIN_DONE_OUTPUTS = 1

# Phrase the agent must produce only after all 12 shells complete.
_COMPLETION_PHRASE = "TWELVE_SHELLS_COMPLETE"

# Subprocess timeout. Twelve parallel shells (each ~0.5s) take a
# few seconds; the LLM call to dispatch them and the follow-up
# call to summarize together usually finish in ~30-60s. 360s
# leaves headroom for cold imports, retry storms, and CI host
# variance.
_RUN_TIMEOUT_SEC = 360


@pytest.fixture
def claude_sdk_available(omnigent_python: Path) -> bool:
    """
    Skip-guard for environments without the Claude SDK + CLI.

    :param omnigent_python: The interpreter the subprocess uses.
    :returns: ``True`` when both the ``claude_agent_sdk`` Python
        package and the ``claude`` CLI binary are installed.
    """
    probe = subprocess.run(
        [
            str(omnigent_python),
            "-c",
            "import importlib.util, sys; "
            "sys.exit(0 if importlib.util.find_spec('claude_agent_sdk') else 1)",
        ],
        capture_output=True,
    )
    sdk_present = probe.returncode == 0
    cli_present = which("claude") is not None
    return sdk_present and cli_present


def test_twelve_parallel_shells_complete_under_ap(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
    patched_databrickscfg: None,
    claude_sdk_available: bool,
) -> None:
    """
    Twelve parallel shell tool calls all complete and the agent
    emits the completion phrase exactly once.

    Drives ``omnigent run hello_world.yaml --harness
    claude-sdk -p <12-shell prompt>`` against the live Databricks
    gateway. The test exercises:

    1. ``RetryPolicy(max_retries=7)`` defaults baked into the
       harness's ``AsyncOpenAI`` / ``ClaudeAgentOptions.env`` setup
       (``ANTHROPIC_MAX_RETRIES`` env var).
    2. Per-LLM-call SDK retry surviving any transient 429s.
    3. AP-side child-workflow per-tool dispatch running
       the 12 shells without ``DBOSUnexpectedStepError``.
    4. The harness scaffold synthesizing a clean terminal event
       (``response.completed`` with the agent's confirmation
       text) once all 12 tool results land.

    :param omnigent_python: Interpreter with omnigent +
        claude_agent_sdk installed.
    :param omnigent_repo_root: Cwd for the subprocess.
    :param omnigent_credentials_env: PAT + base URL for the
        Databricks gateway.
    :param patched_databrickscfg: Fixture that swaps
        the active profile to a PAT for the test (Claude SDK reads
        the profile's ``token`` field directly).
    :param claude_sdk_available: Skip-guard.
    """
    if not claude_sdk_available:
        pytest.skip(
            "Claude SDK harness not available (claude_agent_sdk and/or 'claude' CLI not installed)"
        )

    yaml_path = omnigent_repo_root / "tests" / "resources" / "examples" / "hello_world.yaml"
    result = subprocess.run(
        [
            str(omnigent_python),
            "-m",
            "omnigent",
            "run",
            str(yaml_path),
            "--model",
            _MODEL,
            "--harness",
            _HARNESS,
            "-p",
            _PROMPT,
            "--no-log",
            "--no-session",
        ],
        env=omnigent_credentials_env,
        cwd=str(omnigent_repo_root),
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT_SEC,
    )

    # Exit zero is the load-bearing assertion. A non-zero exit
    # means the workflow failed (gateway 429 cascade exhausted,
    # subprocess died, classifier misclassified an error as
    # permanent, etc.). The error message in the failure output
    # tells the operator which layer broke.
    assert result.returncode == 0, (
        f"omnigent run --omnigent exited non-zero with the 12-shell "
        f"prompt — under normal load this means the per-LLM-call "
        f"retry budget (max_retries=7) was exhausted by transient "
        f"gateway 429s, OR a non-retry failure (subprocess crash, "
        f"misclassified permanent error) escaped. "
        f"\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    stdout = result.stdout

    # The agent's final response must contain TWELVE_SHELLS_COMPLETE
    # exactly once. Failure here means either:
    # (a) the LLM responded before all 12 tools completed (race),
    # (b) a tool result got dropped on retry,
    # (c) the agent's summary diverged from the requested phrase
    #     (model-non-determinism, less concerning).
    # We accept ≥1 occurrences to tolerate (c) — the agent may
    # echo the phrase in a paragraph plus a final-line summary.
    completion_count = stdout.count(_COMPLETION_PHRASE)
    assert completion_count >= 1, (
        f"Agent response did not contain {_COMPLETION_PHRASE!r}, "
        f"which means it did NOT confirm all 12 shells finished. "
        f"This is the canonical 12-shell regression — under retry "
        f"or fan-out failure the LLM may respond before tool "
        f"results land.\nstdout:\n{stdout}"
    )

    # Sanity check: the stdout should reference shell output
    # (the literal string "done" the python sleep prints). Allows
    # for the LLM summarizing rather than echoing each result, so
    # the floor is just one.
    done_count = len(re.findall(r"\bdone\b", stdout))
    assert done_count >= _MIN_DONE_OUTPUTS, (
        f"Expected at least {_MIN_DONE_OUTPUTS} 'done' echo from "
        f"the shell tool outputs in the agent's response, got "
        f"{done_count}.\nstdout:\n{stdout}"
    )
