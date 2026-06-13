"""Phase 0 characterization test — pi harness, one-shot prompt.

Runs ``omnigent run hello_world.yaml --harness pi --model
<model> -p "..."`` as a real subprocess and snapshots structural
observations (exit code, stderr cleanliness, assistant text
length). Captured against current Omnigent; re-run unchanged
in later phases to prove the integration preserves behavior for
the pi harness.

**What breaks if this fails:**
- Omnigent' ``PiExecutor`` regresses (the ``pi --mode rpc``
  subprocess lifecycle, the JSONL event protocol, the TCP
  ``_ToolServer`` that proxies tool calls back to Python, or
  the generated JavaScript extension that registers
  Omnigent tools with ``pi.registerTool()``).
- The ``pi`` CLI binary disappears from PATH or its
  ``--mode rpc`` subcommand changes its startup contract.
- The Databricks credentials resolution regresses — ``PiExecutor``
  reads ``~/.databrickscfg`` directly to generate the temporary
  ``models.json`` that Pi picks up via ``PI_CODING_AGENT_DIR``.
- ``omnigent.cli._run_agent`` for the ``-p`` one-shot path
  stops printing assistant text to stdout on turn complete.

Design reference: ``designs/OMNIGENT_INTEGRATION.md`` §Phase 0
per-harness suite.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests._model_pools import resolve_model
from tests.e2e._harness_probes import cli_unavailable_reason
from tests.e2e.omnigent._snapshot import compare_snapshot

# Model + harness are hardcoded because the test name advertises
# "pi harness". Pi's Databricks integration generates a
# ``models.json`` with OpenAI/Anthropic providers based on the
# model name's prefix; ``databricks-gpt-5-4-mini`` routes through
# the OpenAI-Responses provider which is the best-tested path.
_MODEL = resolve_model("databricks-gpt-5-4-mini", key=__name__)
_HARNESS = "pi"
_PROMPT = "say hi in 5 words"

# Minimum assistant-text length. Anything longer than "hi" proves
# the turn produced a real model reply rather than an empty
# response or a pure error banner.
_MIN_ASSISTANT_CHARS = 4

# Subprocess timeout. Pi spawns a JS subprocess with its own
# init path and registers tools via the generated extension
# before accepting the first turn — slower than openai-agents,
# comparable to claude-sdk. 180s matches the other slow-harness
# tests.
_RUN_TIMEOUT_SEC = 180

_pytest_pi_unavailable = cli_unavailable_reason("pi")
pytestmark = pytest.mark.skipif(
    _pytest_pi_unavailable is not None,
    reason=(
        "pi harness e2e requires a runnable 'pi' CLI; "
        f"{_pytest_pi_unavailable}. Install/fix Pi to run this test."
    ),
)


def test_per_harness_pi_one_shot(
    omnigent_repo_root: Path,
    omnigent_python: Path,
    omnigent_credentials_env: dict[str, str],
    patched_databrickscfg: None,
) -> None:
    """
    ``omnigent run hello_world.yaml --harness pi -p <prompt>``
    exits 0 and emits a non-trivial assistant reply.

    Uses ``patched_databrickscfg`` because ``PiExecutor`` reads
    ``~/.databrickscfg`` directly to build its temporary
    ``models.json`` provider config — OAuth-profile tokens
    silently 403 Pi's model requests. Same workaround as
    claude-sdk/codex; disappears once the ``databricks-sdk``
    rewrite lands.

    :param omnigent_python: Interpreter with omnigent
        installed and importable.
    :param omnigent_repo_root: Cwd for the subprocess so the
        YAML spec and example tool modules resolve on sys.path.
    :param omnigent_credentials_env: Env vars with
        ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` /
        ``DATABRICKS_CONFIG_PROFILE`` populated from
        ``--llm-api-key``.
    :param patched_databrickscfg: Fixture that rewrites
        ``~/.databrickscfg`` to PAT form for the test and
        restores it on teardown.
    """
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

    observed: dict[str, Any] = {
        "exit_code": result.returncode,
        "stderr_is_clean": result.stderr.strip() == "",
        # Trimmed because whitespace around LLM output is noisy
        # and not something we want the snapshot comparator to
        # trip on.
        "assistant_text": result.stdout.strip(),
    }

    # Full stderr surfaced on failure so CI logs show WHY the run
    # went wrong — stderr here is opaque unless we dump it.
    diffs = compare_snapshot("test_per_harness_pi", observed)
    assert diffs == [], (
        "Snapshot mismatch for pi run:\n"
        + "\n".join(diffs)
        + f"\n\nstdout:\n{result.stdout!r}\n\nstderr:\n{result.stderr!r}"
    )
    # Separate assertion so a length regression names the length
    # check directly instead of being buried in the snapshot diff.
    assert len(observed["assistant_text"]) >= _MIN_ASSISTANT_CHARS, (
        f"Pi assistant text shorter than {_MIN_ASSISTANT_CHARS} "
        f"chars; got {observed['assistant_text']!r}"
    )
