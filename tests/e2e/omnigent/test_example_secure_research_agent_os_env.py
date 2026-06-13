"""End-to-end test for ``examples/agents/secure_research_agent_os_env``.

``secure_research_agent`` + an ``os_env:`` sandbox. Same fake
research tools as the non-sandbox variant.

The YAML now ships with ``sandbox: type: none`` so the example
(and this test) run on macOS. Swap in ``linux_bwrap`` on a
Linux host to actually exercise the bwrap write-path
restriction the example is designed to demonstrate.

**What breaks if this fails:**
- Spec parser regresses on a YAML combining ``tools:``,
  ``policies:``, AND ``os_env:`` in the same agent.
- The ``sys_os_*`` builtins stop auto-registering on agents
  that already declare non-trivial ``tools:``.
"""

from __future__ import annotations

from pathlib import Path

from tests.e2e.omnigent._example_helpers import (
    assert_completed_one_shot,
    run_one_shot,
)


def test_secure_research_agent_os_env_one_shot(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
) -> None:
    """
    Run the os_env variant one-shot cross-platform.

    :param omnigent_python: Interpreter with omnigent +
        openai-agents installed.
    :param omnigent_repo_root: Repo root for subprocess cwd.
    :param omnigent_credentials_env: PAT + BASE_URL env.
    """
    result = run_one_shot(
        omnigent_python=omnigent_python,
        omnigent_repo_root=omnigent_repo_root,
        omnigent_credentials_env=omnigent_credentials_env,
        example_name="secure_research_agent_os_env",
    )
    assert_completed_one_shot(result, "secure_research_agent_os_env")
