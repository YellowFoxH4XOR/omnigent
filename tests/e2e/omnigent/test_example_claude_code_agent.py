"""End-to-end test for ``examples/claude_code_agent.yaml``.

The example pins ``executor.type: claude_sdk`` and exposes
Claude Code's built-in tools (Bash, Read, Edit, etc.) plus any
Omnigent tools declared in YAML (passed through as MCP tools).

**What breaks if this fails:**
- ``executor.type: claude_sdk`` spec translation regresses.
- The ``claude_sdk`` harness wiring loses its MCP-tool bridging
  (Omnigent tools declared in YAML stop reaching Claude).
- Harness-specific ``--model`` resolution changes.

Dependency: requires the ``claude-agent-sdk`` Python package. We
fail loud upfront via :func:`require_claude_sdk` rather than
letting the subprocess die with a mid-run ImportError.
"""

from __future__ import annotations

from pathlib import Path

from tests.e2e.omnigent._example_helpers import (
    assert_completed_one_shot,
    require_claude_sdk,
    run_one_shot,
)


def test_claude_code_agent_one_shot(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
) -> None:
    """
    Run the claude_code_agent YAML one-shot through the claude_sdk
    harness (pinned in the YAML — we pass ``harness=None`` so the
    spec wins).

    :param omnigent_python: Interpreter with omnigent +
        claude-agent-sdk installed.
    :param omnigent_repo_root: Repo root for subprocess cwd.
    :param omnigent_credentials_env: Credentials env (the SDK
        reads ~/.databrickscfg; our env fixture also provides
        PAT/BASE_URL for OAuth fallback).
    """
    require_claude_sdk()
    result = run_one_shot(
        omnigent_python=omnigent_python,
        omnigent_repo_root=omnigent_repo_root,
        omnigent_credentials_env=omnigent_credentials_env,
        example_name="claude_code_agent",
        harness=None,  # Let the YAML's executor.type pin win.
        model=None,
    )
    assert_completed_one_shot(result, "claude_code_agent")
