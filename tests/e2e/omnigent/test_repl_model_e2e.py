"""E2E: /model command in the real Omnigent REPL under pexpect.

Mirrors :mod:`tests.e2e.omnigent.test_repl_effort_e2e`. Drives
``/model`` against a real ``omnigent run`` REPL spawned with the
test's configured Databricks profile (resolved via the
``databricks_workspace`` fixture from environment) and asserts the
slash-command surface — show / set / show-after-set / reset —
matches the design's contract end-to-end.
"""

from __future__ import annotations

from pathlib import Path

from tests.e2e.omnigent._pexpect_harness import (
    clean_exit,
    spawn_omnigent_run,
    submit_prompt,
)

_MODEL = "databricks-gpt-5-mini"
_HARNESS = "openai-agents"
# Override target. Same family as _MODEL but a different size — the
# REPL only needs to confirm the slash-command surface; it doesn't
# actually invoke the gateway, so any non-empty model id distinct
# from the spawn one is sufficient for the show/set assertions below.
_OVERRIDE_MODEL = "databricks-gpt-5-4-mini"
_SPAWN_TIMEOUT = 90.0
_BOOT_TIMEOUT = 60.0
_EXIT_TIMEOUT = 15.0


def _submit_slash_command(child, text: str) -> None:
    """Submit a slash command under prompt-toolkit/pexpect."""
    submit_prompt(child, text)


def test_repl_model_command_show_set_reset(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
    databricks_workspace: tuple[str, str],
) -> None:
    """Drive /model through its full state machine in a real REPL.

    Asserts each transition:

    1. Initial ``/model`` echoes ``(agent default)``.
    2. ``/model <name>`` confirms ``model set to <name>``.
    3. Subsequent ``/model`` echoes the override (i.e. session
       state actually persisted between commands).
    4. ``/model default`` confirms reset to ``agent default``.

    Matches :mod:`test_repl_effort_e2e`'s coverage shape so a parity
    regression in either surfaces here. Doesn't drive a real LLM
    turn (would require model-availability gating beyond what the
    slash command itself tests); the workflow- and harness-level
    tests cover propagation into actual model calls.
    """
    yaml_path = omnigent_repo_root / "tests" / "resources" / "examples" / "hello_world.yaml"
    env = dict(omnigent_credentials_env)
    env["PYTHONPATH"] = f"{omnigent_repo_root}:{omnigent_repo_root / 'sdks' / 'python-client'}" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    child = spawn_omnigent_run(
        omnigent_python=omnigent_python,
        yaml_path=yaml_path,
        model=_MODEL,
        harness=_HARNESS,
        env=env,
        cwd=omnigent_repo_root,
        timeout=_SPAWN_TIMEOUT,
    )
    try:
        # Match the visible prompt marker rather than the bottom-
        # toolbar state text: under pexpect the prompt-toolkit CPR
        # handshake can suppress ``state: sleeping`` even when the
        # REPL is ready.
        child.expect(r"❯ ", timeout=_BOOT_TIMEOUT)

        _submit_slash_command(child, "/model")
        # ``(agent default)`` is the canonical phrase the no-override
        # show branch emits — search the slash-command handler for
        # the matching string if you rename it.
        child.expect(r"model: \(agent default\)", timeout=10)
        # Usage hint is always printed alongside the show line.
        child.expect("usage: /model", timeout=10)

        _submit_slash_command(child, f"/model {_OVERRIDE_MODEL}")
        child.expect(f"model set to {_OVERRIDE_MODEL}", timeout=10)

        # Same drain trick test_repl_effort_e2e.py uses: nudge a
        # fresh prompt to confirm the input buffer is clear before
        # firing the next slash command.
        child.send("\r")
        child.expect(r"❯ ", timeout=10)

        _submit_slash_command(child, "/model")
        child.expect(f"model: {_OVERRIDE_MODEL}", timeout=10)

        _submit_slash_command(child, "/model default")
        child.expect("model reset to agent default", timeout=10)

        clean_exit(child, timeout=_EXIT_TIMEOUT)
        assert child.exitstatus in (0, None)
        assert child.signalstatus is None
    finally:
        if not child.closed:
            child.close(force=True)
