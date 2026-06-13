"""End-to-end test for ``examples/agent_with_uc_tools.yaml``.

The example registers Databricks Unity Catalog functions as tools
via the ``type: uc`` tool type. The UC tool resolver looks up the
function metadata against a Databricks workspace at registration
time — the function doesn't have to be *invoked* by the LLM for
the registration path to fire.

**What breaks if this fails:**
- Spec parser regresses on ``type: uc`` tool entries.
- UC metadata resolution at tool-registration time throws on a
  config shape the parser previously accepted.
- The workspace client resolution (profile-based) regresses.

The default prompt asks the agent for a short reply — the LLM
decides whether to invoke the UC tool, but either path exercises
the tool-registration code the unification changed.
"""

from __future__ import annotations

from pathlib import Path

from tests.e2e.omnigent._example_helpers import (
    assert_completed_one_shot,
    run_one_shot,
)


def test_agent_with_uc_tools_one_shot(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
) -> None:
    """
    ``omnigent run agent_with_uc_tools -p <prompt>`` completes
    cleanly. The UC tool registers at startup; the LLM reply
    confirms the session reached the turn loop.

    :param omnigent_python: Interpreter with omnigent +
        openai-agents + databricks-sdk installed.
    :param omnigent_repo_root: Repo root for subprocess cwd.
    :param omnigent_credentials_env: PAT + BASE_URL + profile env.
    """
    result = run_one_shot(
        omnigent_python=omnigent_python,
        omnigent_repo_root=omnigent_repo_root,
        omnigent_credentials_env=omnigent_credentials_env,
        example_name="agent_with_uc_tools",
    )
    assert_completed_one_shot(result, "agent_with_uc_tools")
