"""End-to-end test for ``examples/agent_with_os_env.yaml``.

The example wires an ``os_env:`` block onto the agent and exposes
the built-in ``sys_os_read`` / ``sys_os_write`` / ``sys_os_edit`` /
``sys_os_shell`` tools. The YAML now ships with
``sandbox: type: none`` so it runs on macOS too — flip back to
``linux_bwrap`` on Linux to exercise the actual sandbox.

**What breaks if this fails:**
- The spec parser regresses on ``os_env.sandbox`` blocks.
- The ``sys_os_*`` builtin registration breaks for YAML-declared
  agents with an ``os_env:`` field.
"""

from __future__ import annotations

from pathlib import Path

from tests.e2e.omnigent._example_helpers import (
    assert_completed_one_shot,
    run_one_shot,
)


def test_agent_with_os_env_one_shot(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
) -> None:
    """
    ``omnigent run agent_with_os_env -p <prompt>`` completes
    cleanly and streams a reply.

    :param omnigent_python: Interpreter with omnigent +
        openai-agents installed.
    :param omnigent_repo_root: Repo root for subprocess cwd.
    :param omnigent_credentials_env: PAT + BASE_URL env.
    """
    result = run_one_shot(
        omnigent_python=omnigent_python,
        omnigent_repo_root=omnigent_repo_root,
        omnigent_credentials_env=omnigent_credentials_env,
        example_name="agent_with_os_env",
    )
    assert_completed_one_shot(result, "agent_with_os_env")
