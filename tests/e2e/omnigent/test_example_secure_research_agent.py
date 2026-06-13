"""End-to-end test for ``examples/secure_research_agent.yaml``.

The example declares three tools backed by simple Python
callables (``web_search``, ``read_internal_doc``, ``run_shell``)
plus a :class:`FunctionPolicy` that gates which tools can fire
based on prompt content. No external services — runs entirely
from the venv.

**What breaks if this fails:**
- Spec parser regresses on mixed tool types (function +
  cancellable_function + a policy in the same YAML).
- The FunctionPolicy's tool-gating hook drops from the agent
  loop's tool-call phase.
- Dotted-callable resolution for agent-local Python modules
  (our per-agent duplicated ``tool_functions.py`` copy) breaks.
"""

from __future__ import annotations

from pathlib import Path

from tests.e2e.omnigent._example_helpers import (
    assert_completed_one_shot,
    run_one_shot,
)


def test_secure_research_agent_one_shot(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
) -> None:
    """
    Run the secure_research_agent one-shot. Fake tools mean no
    external network calls; the policy fires during the tool
    phase of the agent loop.

    :param omnigent_python: Interpreter with omnigent +
        openai-agents installed.
    :param omnigent_repo_root: Repo root for subprocess cwd.
    :param omnigent_credentials_env: PAT + BASE_URL env.
    """
    result = run_one_shot(
        omnigent_python=omnigent_python,
        omnigent_repo_root=omnigent_repo_root,
        omnigent_credentials_env=omnigent_credentials_env,
        example_name="secure_research_agent",
    )
    assert_completed_one_shot(result, "secure_research_agent")
