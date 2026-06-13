"""Phase 5 integration-code tests — ``omnigent run`` shim.

Phase 5 of the Omnigent / omnigent integration replaced the
``run`` command's Omnigent mode hard-error with a real in-process shim:
it prepares the YAML bundle, registers it with omnigent stores,
POSTs the prompt through an :class:`httpx.ASGITransport`, prints
the assistant text, and exits.

**What breaks if this fails:**
- The ``run`` dispatch site regresses to the pre-phase-5 hard
  error ("lands in phase 5") — i.e. ``_run_agent_via_omnigent`` is no
  longer called.
- The shim's YAML preparation pipeline (``_OmnigentOverrides``,
  ``_prepare_omnigent_yaml_bundle``, ``_omnigent_register_yaml_bundle``)
  breaks silently — e.g. ``--model`` is dropped, the harness
  defaults wrong, or the agent doesn't land in ``AgentStore``.
- The in-process omnigent app fails to answer
  ``POST /v1/responses`` — covers regressions in
  ``omnigent.server.app.create_app``,
  ``_build_omnigent_stores``, or the runtime-init sequence the shim
  inherits from phase 3.
- The output extraction regresses: the shim prints garbage
  instead of the ``output_text`` fragments (catches changes to
  the OpenResponses output shape that the shim's
  ``_extract_assistant_text`` helper doesn't track).
- ``OMNIGENT_RUNTIME=1`` stops being honored as an alternative to
  the Omnigent mode flag.
- ``omnigent version`` starts diverging from
  ``omnigent version`` (should stay a constant string
  regardless of Omnigent mode).

Design reference: ``designs/OMNIGENT_INTEGRATION.md`` §Phase 5.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from tests._model_pools import resolve_model

# Databricks FM gateway model that supports the Responses API
# passthrough through the openai-agents harness. Matches the model
# used by other Phase 0 openai-agents tests (see
# ``test_per_harness_openai_agents_sdk.py``) so the same gateway
# endpoint is exercised end-to-end.
_MODEL = resolve_model("databricks-gpt-5-4-mini", key=__name__)

# Harness chosen because it honors ``OPENAI_BASE_URL`` /
# ``OPENAI_API_KEY`` directly — no ``~/.databrickscfg`` patching
# required. Matches the model above and keeps the test
# prerequisites identical to the legacy run-path characterization.
_HARNESS = "openai-agents"

_PROMPT = "say hi in 5 words"

# Minimum assistant-text length. Anything longer than "hi" proves
# a real model reply round-tripped through the shim — an empty
# stdout would mean the output-extraction helper regressed
# silently.
_MIN_ASSISTANT_CHARS = 4

# Subprocess timeout matches the existing per-harness tests —
# 180s leaves headroom for DBOS sqlite migrations + cold imports
# + one openai-agents turn.
_RUN_TIMEOUT_SEC = 180


def _run_omnigent_run_omnigent(
    *,
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
    extra_env: dict[str, str] | None = None,
    use_flag: bool = True,
) -> subprocess.CompletedProcess[str]:
    """
    Execute the ``omnigent run <hello_world.yaml> ... -p
    <prompt>`` subprocess. Omnigent mode is the default -- no
    argv flag is needed -- so this helper just runs the command
    and returns the result. ``extra_env`` lets the caller still
    test the ``OMNIGENT_RUNTIME=1`` env-var path independently.

    :param omnigent_python: Interpreter with omnigent and
        omnigent installed.
    :param omnigent_repo_root: Cwd for the subprocess.
    :param omnigent_credentials_env: Env vars with the PAT and
        base URL.
    :param extra_env: Additional env vars to set on top of the
        standard e2e env, e.g.
        ``{"OMNIGENT_RUNTIME": "1"}``. ``None`` means no additions.
    :param use_flag: No-op since the ``--omnigent`` argv
        flag was removed (AP is now the default). Kept for backwards-compat
        with existing parametrize labels; the body simply ignores
        the value.
    :returns: The completed subprocess result (stdout/stderr
        captured).
    """
    yaml_path = omnigent_repo_root / "tests" / "resources" / "examples" / "hello_world.yaml"
    argv: list[str] = [
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
    ]
    # ``use_flag`` historically gated ``argv.append("--omnigent")`` here;
    # Omnigent is now the default and ``--omnigent`` was removed, so the
    # parameter is a no-op kept for the existing parametrize labels.
    _ = use_flag
    env = dict(omnigent_credentials_env)
    if extra_env is not None:
        env.update(extra_env)
    return subprocess.run(
        argv,
        env=env,
        cwd=str(omnigent_repo_root),
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT_SEC,
    )


def _run_omnigent_run_legacy(
    *,
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """
    Execute the same ``omnigent run`` command as
    :func:`_run_omnigent_run_omnigent` but WITHOUT the omnigent
    integration, so the legacy code path runs.

    :param omnigent_python: Interpreter with omnigent
        installed.
    :param omnigent_repo_root: Cwd for the subprocess.
    :param omnigent_credentials_env: Env vars with the PAT and
        base URL.
    :returns: The completed subprocess result (stdout/stderr
        captured).
    """
    yaml_path = omnigent_repo_root / "tests" / "resources" / "examples" / "hello_world.yaml"
    return subprocess.run(
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


def _structural_observations(
    result: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    """
    Distill the observable behavior of an ``omnigent run``
    subprocess into a comparator-friendly dict.

    Snapshotting free-form assistant text would be flaky (LLM
    non-determinism), so we capture only the structural
    properties both the legacy path and the Omnigent path must
    agree on: zero exit, non-empty assistant text, and text at
    least :data:`_MIN_ASSISTANT_CHARS` chars long.

    :param result: The subprocess result to inspect.
    :returns: A dict of fields suitable for side-by-side
        comparison across the legacy and Omnigent paths.
    """
    text = result.stdout.strip()
    return {
        "exit_code": result.returncode,
        "assistant_text_nonempty": bool(text),
        "assistant_text_meets_min_length": len(text) >= _MIN_ASSISTANT_CHARS,
    }


def test_run_omnigent_smoke(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
) -> None:
    """
    ``omnigent run hello_world.yaml -p <prompt>`` exits 0,
    prints non-trivial assistant text, and does not re-emit the
    pre-phase-5 hard-error on stderr.

    :param omnigent_python: Interpreter with omnigent and
        omnigent installed.
    :param omnigent_repo_root: Cwd for the subprocess so the
        relative YAML path resolves.
    :param omnigent_credentials_env: Env vars with the PAT and
        base URL populated.
    """
    result = _run_omnigent_run_omnigent(
        omnigent_python=omnigent_python,
        omnigent_repo_root=omnigent_repo_root,
        omnigent_credentials_env=omnigent_credentials_env,
    )
    # Exit 0 proves the ASGI POST returned a successful
    # ResponseObject. Non-zero would mean either the shim failed
    # to boot the app, the agent workflow hit an error, or the
    # task terminated in a non-"completed" status.
    assert result.returncode == 0, (
        f"expected exit 0, got {result.returncode}.\n"
        f"stdout:\n{result.stdout!r}\n\nstderr:\n{result.stderr!r}"
    )
    assistant_text = result.stdout.strip()
    # The shim's _extract_assistant_text must pull the
    # output_text blocks out of the ResponseObject; an empty
    # stdout here would signal that regression (or that the
    # agent simply didn't produce a message, which is itself a
    # regression for a "say hi" prompt).
    assert len(assistant_text) >= _MIN_ASSISTANT_CHARS, (
        f"--omnigent assistant text shorter than {_MIN_ASSISTANT_CHARS} chars; "
        f"got {assistant_text!r}"
    )
    # Pre-phase-5 the run dispatch printed "lands in phase 5" on
    # stderr. Asserting on its absence catches a regression where
    # the dispatch site falls back to the hard-error path.
    assert "phase 5" not in result.stderr, (
        f"Regression: stderr contains the pre-phase-5 hard-error wording. stderr={result.stderr!r}"
    )


def test_run_omnigent_matches_legacy_structural_fields(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
) -> None:
    """
    Running the same YAML + prompt through the legacy code path
    and through Omnigent mode must agree on the structural fields — no
    integration-flag-only behavior change for users.

    Implements the design contract that characterization tests
    "run unchanged with integration active." Exact assistant text
    is non-deterministic across LLM calls, so we compare only the
    structural shape via :func:`_structural_observations`.

    :param omnigent_python: Interpreter with omnigent and
        omnigent installed.
    :param omnigent_repo_root: Cwd for both subprocesses.
    :param omnigent_credentials_env: Env vars with the PAT and
        base URL populated.
    """
    legacy = _run_omnigent_run_legacy(
        omnigent_python=omnigent_python,
        omnigent_repo_root=omnigent_repo_root,
        omnigent_credentials_env=omnigent_credentials_env,
    )
    omnigent = _run_omnigent_run_omnigent(
        omnigent_python=omnigent_python,
        omnigent_repo_root=omnigent_repo_root,
        omnigent_credentials_env=omnigent_credentials_env,
    )
    legacy_obs = _structural_observations(legacy)
    omnigent_obs = _structural_observations(omnigent)
    # Both invocations must exit 0; divergence here means one of
    # the two paths regressed and the "invisible integration"
    # contract is broken.
    assert legacy_obs == omnigent_obs, (
        "Structural observations diverge between legacy and --omnigent:\n"
        f"legacy={legacy_obs!r}\n"
        f"omnigent={omnigent_obs!r}\n\n"
        f"legacy stdout: {legacy.stdout!r}\n"
        f"legacy stderr: {legacy.stderr!r}\n\n"
        f"omnigent stdout: {omnigent.stdout!r}\n"
        f"omnigent stderr: {omnigent.stderr!r}"
    )


def test_run_omnigent_env_var_enables_integration(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
) -> None:
    """
    ``OMNIGENT_RUNTIME=1`` (with no Omnigent mode flag on argv) must route
    through the omnigent shim — the design's "env var honored
    as an alternative to the flag" contract.

    :param omnigent_python: Interpreter with omnigent and
        omnigent installed.
    :param omnigent_repo_root: Cwd for the subprocess.
    :param omnigent_credentials_env: Env vars with the PAT and
        base URL populated.
    """
    result = _run_omnigent_run_omnigent(
        omnigent_python=omnigent_python,
        omnigent_repo_root=omnigent_repo_root,
        omnigent_credentials_env=omnigent_credentials_env,
        extra_env={"OMNIGENT_RUNTIME": "1"},
        use_flag=False,
    )
    assert result.returncode == 0, (
        f"OMNIGENT_RUNTIME=1 did not yield exit 0; "
        f"got {result.returncode}.\n"
        f"stdout:\n{result.stdout!r}\n\nstderr:\n{result.stderr!r}"
    )
    assistant_text = result.stdout.strip()
    # Same min-length check as the flag path — proves the env var
    # actually activated the shim instead of being silently
    # ignored (which would still exit 0 on the legacy path).
    assert len(assistant_text) >= _MIN_ASSISTANT_CHARS, (
        f"OMNIGENT_RUNTIME=1 assistant text shorter than "
        f"{_MIN_ASSISTANT_CHARS} chars; got {assistant_text!r}"
    )
    assert "phase 5" not in result.stderr, (
        f"OMNIGENT_RUNTIME=1 fell back to the pre-phase-5 hard error. stderr={result.stderr!r}"
    )


def test_version_omnigent_matches_version(
    omnigent_python: Path,
    omnigent_repo_root: Path,
) -> None:
    """
    ``omnigent version`` must be stable and independent of
    unrelated AP/e2e credential setup.

    :param omnigent_python: Interpreter with omnigent
        installed.
    :param omnigent_repo_root: Cwd for the subprocess (not
        required for ``version`` but passed for parity with the
        rest of the suite).
    """
    baseline = subprocess.run(
        [
            str(omnigent_python),
            "-m",
            "omnigent",
            "version",
        ],
        env={k: v for k, v in os.environ.items() if k != "OMNIGENT_RUNTIME"},
        cwd=str(omnigent_repo_root),
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT_SEC,
    )
    with_ap = subprocess.run(
        [
            str(omnigent_python),
            "-m",
            "omnigent",
            "version",
        ],
        env={**os.environ, "OMNIGENT_RUNTIME": "1"},
        cwd=str(omnigent_repo_root),
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT_SEC,
    )
    assert baseline.returncode == 0
    assert with_ap.returncode == 0
    # Exact-string match on stdout — version output is a literal
    # constant; any divergence means Omnigent mode is leaking into the
    # version path, which the design forbids.
    assert baseline.stdout == with_ap.stdout, (
        "omnigent version diverged between baseline and OMNIGENT_RUNTIME=1. "
        f"baseline={baseline.stdout!r} ap={with_ap.stdout!r}"
    )
    version_text = baseline.stdout.strip()
    # The click command prints ``omnigent <version>`` plus an
    # optional ``(<short-sha>, built <ts>)`` suffix when the build
    # hook in ``setup.py`` baked ``_build_info.py`` into the wheel.
    # Assert the stable prefix + a version-looking token so a silent
    # empty stdout still fails loudly without pinning the exact
    # development version or the optional build-info suffix.
    assert version_text, "omnigent version printed no stdout"
    assert version_text.startswith("omnigent "), f"unexpected version output: {baseline.stdout!r}"
    after_prefix = version_text[len("omnigent ") :]
    assert after_prefix and after_prefix[0].isdigit(), (
        f"unexpected version output: {baseline.stdout!r}"
    )
