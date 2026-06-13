"""End-to-end test for ``examples/agents/agent_with_subagent_session``.

The example demonstrates the ``sys_session_*`` builtin tool family:
``sys_session_send`` / ``sys_session_get_history`` /
``sys_session_cancel_turn`` / ``sys_read_inbox``. A supervisor
agent delegates work to a persistent worker sub-agent via a named
session.

**What breaks if this fails:**
- ``tools.<name>.type: agent`` translation regresses (sub-agent
  spec no longer converts into an :class:`AgentTool`).
- The ``sys_session_*`` builtin registrations drop from the
  effective tool set when an agent declares sub-agents.
- Session dispatch wiring in the runtime loses the ability to
  route messages to a named session.

The prompt asks the supervisor to start worker session alpha and
run a trivial calculation, so the session tools fire during the
turn — a reply that doesn't mention the worker would mean the
supervisor handled it directly, bypassing the feature under test.
"""

from __future__ import annotations

from pathlib import Path

from tests.e2e.omnigent._example_helpers import (
    assert_completed_one_shot,
    run_one_shot,
)

_PROMPT = "Start worker session alpha and ask it to calculate 2 + 2."


def test_agent_with_subagent_session_one_shot(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
) -> None:
    """
    Run the subagent-session example one-shot and assert the run
    finishes cleanly. Sub-agent tool invocations land inside the
    captured stdout stream.

    :param omnigent_python: Interpreter with omnigent +
        openai-agents installed.
    :param omnigent_repo_root: Repo root for subprocess cwd.
    :param omnigent_credentials_env: PAT + BASE_URL env.
    """
    result = run_one_shot(
        omnigent_python=omnigent_python,
        omnigent_repo_root=omnigent_repo_root,
        omnigent_credentials_env=omnigent_credentials_env,
        example_name="agent_with_subagent_session",
        prompt=_PROMPT,
    )
    assert_completed_one_shot(result, "agent_with_subagent_session")
