"""End-to-end test for ``examples/agents/coding_supervisor_with_forks``.

Supervisor + two worker sub-agents, each with a forked os_env
(hardlink-tree COW). The test is parametrized across the
wrapped harnesses so each one drives the supervisor + workers.

YAML has ``sandbox: type: none`` everywhere so the sandbox is
off; the fork mode itself works cross-platform. Running
end-to-end still requires the parametrized harness's outer
CLI binary on PATH — when missing we fail loud (CLAUDE.md
rule 30 forbids silent skips).

**What breaks if this fails:**
- Sub-agent ``os_env.fork`` propagation regresses.
- Per-worker harness specification is lost during spec translation.
- The ``sys_session_*`` + forked-env combination stops wiring
  the symlinks under ``.sessions/<worker>/`` that the supervisor
  reads to diff worker output.
"""

from __future__ import annotations

from pathlib import Path
from shutil import which

import pytest

from tests.e2e._harness_probes import HARNESS_HARNESS_MODELS, HARNESS_IDS
from tests.e2e.omnigent._example_helpers import (
    assert_completed_one_shot,
    require_claude_sdk,
    require_codex_cli,
    run_one_shot,
)


@pytest.mark.parametrize("harness,model", HARNESS_HARNESS_MODELS, ids=HARNESS_IDS)
def test_coding_supervisor_with_forks_one_shot(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
    harness: str,
    model: str,
) -> None:
    """
    Run the forked coding-supervisor one-shot. The CLI's
    ``--harness`` / ``--model`` flags override every executor
    block in the YAML so the parametrized harness drives both
    the supervisor and its forked workers.

    :param omnigent_python: Interpreter with omnigent +
        the harness's SDK installed.
    :param omnigent_repo_root: Repo root for subprocess cwd.
    :param omnigent_credentials_env: PAT + BASE_URL env.
    :param harness: The harness identifier from
        :data:`HARNESS_HARNESS_MODELS`.
    :param model: The harness-routed model identifier.
    """
    if harness == "claude-sdk":
        require_claude_sdk()
        if which("claude") is None:
            pytest.fail(
                "claude-sdk harness prerequisite missing: the 'claude' "
                "CLI binary must be installed on PATH."
            )
    elif harness == "codex":
        require_codex_cli()
    result = run_one_shot(
        omnigent_python=omnigent_python,
        omnigent_repo_root=omnigent_repo_root,
        omnigent_credentials_env=omnigent_credentials_env,
        example_name="coding_supervisor_with_forks",
        harness=harness,
        model=model,
    )
    assert_completed_one_shot(result, "coding_supervisor_with_forks")
