"""Phase 0 characterization test — ``agent_with_policies.yaml``.

Runs ``tests/resources/examples/agent_with_policies.yaml`` through
``omnigent run`` with a prompt and snapshots the structural
evidence that the policy engine intercepts and produces a
``[Denied by policy: <reason>]`` marker on stdout.

**What the policy enforces** (read from the YAML):
- ``block_canada_input`` is a ``prompt`` policy on ``input``.
  It is the FIRST policy evaluated on every user prompt.
  Its ``executor.model`` is an OpenAI-style Databricks model
  served by the workspace gateway's Responses API.
- ``block_canada_output`` is a ``prompt`` policy on ``output``.
- ``block_long_sleep`` is a ``function`` policy on ``tool_call``.

**Removing the policy would change behavior as follows:** if
``block_canada_input`` were removed, prompts would flow through
to the base model and produce ordinary assistant replies. With
the policy in place, Omnigent routes EVERY input prompt
through the policy's ``executor`` before calling the base model;
the runtime's ``_build_deny_sentinel`` writes a
``[Denied by policy: <reason>]`` string to stdout in place of the
suppressed model turn. The YAML pins the reason text to
``"Canada-related topics are denied."``, so the test asserts on
that literal substring to prove the ``block_canada_input``
policy specifically fired (rather than any other DENY).

**What breaks if this fails:**
- Omnigent' policy engine regresses (``omnigent.policies``
  ``PolicyEngine`` wiring, prompt-policy ``request`` phase, or the
  ``PolicyAction.DENY`` → ``[Denied by policy: <reason>]``
  stdout path in ``omnigent.runtime.workflow``).
- YAML spec parsing regresses on the ``policies:`` block (the
  ``type: prompt`` + ``on: [request]`` mapping or the nested
  ``executor:`` override).
- The prompt-policy evaluator drops the ``reason`` field from
  the verdict, breaking the substring fingerprint.

Design reference: ``designs/OMNIGENT_INTEGRATION.md`` §Phase 0
YAML→agent characterization (policy-bearing YAML slot).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which
from typing import Any

import pytest

from tests.e2e._harness_probes import HARNESS_HARNESS_MODELS, HARNESS_IDS
from tests.e2e.omnigent._snapshot import compare_snapshot

# Prompt engineered to match the ``block_canada_input`` policy's
# DENY criteria: "mentions Canada or any city in Canada".
# The YAML's policy-example comment lists
# ``"Name major cities in Canada."`` as an exemplar, so we use
# a close cousin to land deterministically on a DENY verdict.
_PROMPT = "Name the provinces of Canada."

# The runtime emits ``[Denied by policy: <reason>]`` on a DENY
# (see ``_build_deny_sentinel`` in ``omnigent/runtime/workflow.py``).
# We snapshot the reason text — the YAML pins it to "Canada-related
# topics are denied." — so the assertion proves a DENY from
# ``block_canada_input`` specifically rather than any other policy.
_DENIED_MARKER = "[Denied by policy: Canada-related topics are denied"

_RUN_TIMEOUT_SEC = 240  # two LLM roundtrips: input-policy judge
# and then the full turn would normally follow (but doesn't,
# because the judge denies first). 240s keeps headroom for both
# calls on cold starts.


def _check_harness_available(harness: str, omnigent_python: Path) -> None:
    """
    Fail loud if the parametrized harness's prerequisites are missing.

    Mirrors the per-harness availability checks in
    ``test_per_harness_claude_sdk.py`` and
    ``test_per_harness_codex.py``. Following CLAUDE.md rule 30
    we fail rather than silently skip so missing prerequisites
    stay visible.

    :param harness: The harness identifier under test.
    :param omnigent_python: The subprocess interpreter — used
        to probe Python-package availability.
    """
    if harness == "claude-sdk":
        probe = subprocess.run(
            [
                str(omnigent_python),
                "-c",
                "import importlib.util, sys; "
                "sys.exit(0 if importlib.util.find_spec('claude_agent_sdk') else 1)",
            ],
            capture_output=True,
        )
        if probe.returncode != 0 or which("claude") is None:
            pytest.fail(
                "claude-sdk harness prerequisites missing: both the "
                "'claude_agent_sdk' Python package and the 'claude' CLI "
                "binary must be present on PATH."
            )
    elif harness == "codex":
        if which("codex") is None:
            pytest.fail(
                "codex harness prerequisite missing: the 'codex' CLI "
                "binary must be installed on PATH (install via "
                "'npm i -g @openai/codex')."
            )


@pytest.mark.parametrize("harness,model", HARNESS_HARNESS_MODELS, ids=HARNESS_IDS)
def test_yaml_policies_blocks_canada_input(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
    patched_databrickscfg: None,
    harness: str,
    model: str,
) -> None:
    """
    ``omnigent run agent_with_policies.yaml --harness <harness>
    -p "Name the provinces of Canada."`` exits 0 and stdout
    contains ``[Denied by policy: Canada-related topics are denied``.

    Exit-0 with the DENY-reason marker is the documented contract
    for an input-policy denial: the turn completed cleanly, the
    ``block_canada_input`` policy intercepted the prompt before
    it reached the base model, and the runtime surfaced the
    denial-reason sentinel on stdout in place of the suppressed
    turn. Exit-nonzero would mean an unexpected CLI-level error;
    a missing marker would mean either the policy silently failed
    open OR a different policy fired (both are regressions).
    Parametrized so each wrapped harness exercises the policy gate.

    :param omnigent_python: Interpreter with omnigent
        installed and importable.
    :param omnigent_repo_root: Cwd for the subprocess so the
        YAML's relative ``callable:`` and ``runner:`` dotted
        paths (``tests.resources.examples._shared.tool_functions.*``) resolve on
        sys.path.
    :param omnigent_credentials_env: Env vars with
        ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` /
        ``DATABRICKS_CONFIG_PROFILE`` populated from
        ``--llm-api-key``.
    :param patched_databrickscfg: Rewrites the dogfood profile
        in ``~/.databrickscfg`` to PAT form for the test (claude
        and codex harnesses both read the file directly and
        OAuth profiles 403).
    :param harness: The harness identifier from
        :data:`HARNESS_HARNESS_MODELS`.
    :param model: The harness-routed model identifier from
        :data:`HARNESS_HARNESS_MODELS`.
    """
    _check_harness_available(harness, omnigent_python)
    yaml_path = (
        omnigent_repo_root / "tests" / "resources" / "examples" / "agent_with_policies.yaml"
    )

    result = subprocess.run(
        [
            str(omnigent_python),
            "-m",
            "omnigent",
            "run",
            str(yaml_path),
            "--harness",
            harness,
            "--model",
            model,
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
        # Belt-and-braces against any interactive prompt the child
        # might emit (e.g. onboarding's missing-profile Y/n). The
        # env var ``OMNIGENT_SKIP_ONBOARD=1`` in
        # ``omnigent_credentials_env`` is the primary defense;
        # DEVNULL stdin guarantees ``sys.stdin.isatty()`` is
        # False so any future prompts also short-circuit.
        stdin=subprocess.DEVNULL,
    )

    observed: dict[str, Any] = {
        "exit_code": result.returncode,
        # We deliberately do NOT assert stderr is clean — policy
        # evaluation logs to stderr in some configurations, and
        # the phase-0 observation here is the stdout-visible
        # ``[Denied`` marker, not stderr quietness.
        "stdout": result.stdout,
    }

    diffs = compare_snapshot("test_yaml_policies", observed)
    assert diffs == [], (
        "Snapshot mismatch for agent_with_policies run:\n"
        + "\n".join(diffs)
        + f"\n\nstdout:\n{result.stdout!r}\n\nstderr:\n{result.stderr!r}"
    )
    # Belt-and-braces: the snapshot's ``contains`` comparator
    # already checks this, but naming the assertion explicitly
    # makes a failure message self-explanatory if the snapshot
    # file is ever accidentally deleted.
    assert _DENIED_MARKER in result.stdout, (
        f"Expected policy-denial marker {_DENIED_MARKER!r} in "
        f"stdout — ``block_canada_input`` should have blocked "
        f"the prompt.\n\nstdout:\n{result.stdout!r}"
    )
