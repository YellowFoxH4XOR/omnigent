"""Drift guard: every agent — top-level ``examples/*.yaml``,
dir-shaped ``examples/<name>/`` (containing ``config.yaml``), or
test-only ``tests/resources/agents/<name>/`` — must have a
dedicated ``test_example_<name>.py`` file under
``tests/e2e/omnigent/``.

When a new agent lands in any of those roots, the author should
add a test file in the same commit. This test fails loud if an
agent ships without one, and loud again if a test file points
at an agent that no longer exists.

Not a functional test — just a structural cross-check so the
coverage-per-agent rule can't silently drift.
"""

from __future__ import annotations

from pathlib import Path

# Agents that have e2e tests in files other than the
# ``test_example_<name>.py`` naming convention (historical, pre-
# unification). Kept as an explicit allow-list so *new* agents
# can't slip past this guard by accident — expanding the list
# requires editing it here.
_ALT_COVERED: frozenset[str] = frozenset(
    {
        # Covered by test_yaml_hello_world.py (via agent_with_tools
        # fixture) and many dedicated hello_world-named e2e tests
        # under tests/e2e/omnigent/test_run_omnigent_* etc.
        "hello_world",
        # Covered by test_yaml_hello_world.py's tool-dispatch test.
        "agent_with_tools",
        # Covered by test_yaml_policies.py.
        "agent_with_policies",
        # Covered by tests/e2e/test_archer_*.py (multiple files).
        "archer",
        # Covered by tests/e2e/test_coder_subagent.py +
        # tests/e2e/test_chat_e2e.py.
        "coder",
        # Covered by tests/e2e/omnigent/test_run_omnigent_coding_supervisor.py
        # (seven test functions).
        "coding_supervisor",
        # Covered by tests/e2e/test_openai_coder_*.py.
        "openai-coder",
        # Covered by tests/e2e/omnigent/test_repl_overview_terminal_visibility.py.
        "terminal_workers",
        # Pre-existing coverage gaps — ``chat_model`` is exercised
        # by ``web/``'s integration flow (``web/README.md`` leads
        # the dev-server README with it) and ``coding_supervisor_openai``
        # is the OpenAI-model sibling of ``coding_supervisor`` that
        # reuses the same sub-agent coverage. Both are allowlisted
        # rather than split into dedicated e2e files because no
        # behavior they exercise is unique.
        "chat_model",
        "coding_supervisor_openai",
        # Test-only fixtures under tests/resources/agents/ that
        # don't have test_example_<name>.py files — these are
        # loaded ad-hoc by specific test files (e.g.
        # claude-coder, coding-supervisor, terminal_supervisor are
        # loaded by tests/e2e/conftest.py fixtures; ask-demo,
        # compaction-test, terminal_test are referenced by
        # name from existing e2e tests).
        "ask-demo",
        "claude-coder",
        "coding-supervisor",
        "compaction-test",
        # Test-only fixtures added with OMNIGENT_TERMINAL_BRIDGE (commits
        # 3d9dd0a / 1f9a3a8). Loaded by:
        # - sys-terminal-test → tests/e2e/test_sys_terminal_e2e.py
        #   via the sys_terminal_test_agent fixture in
        #   tests/e2e/conftest.py.
        # - supervisor-terminal-test → tests/e2e/test_repl_terminal_overview_e2e.py
        #   for the parametrized parent+sub-agent terminal sidebar
        #   test.
        "supervisor-terminal-test",
        "sys-terminal-test",
        # Skills-filter test fixtures under tests/resources/agents/.
        # Loaded by tests/e2e/test_codex_skills_filter_e2e.py,
        # test_pi_skills_filter_e2e.py, and
        # test_claude_coder_skills.py.
        "codex_skills_all",
        "codex_skills_list",
        "codex_skills_none",
        "pi_skills_all",
        "pi_skills_list",
        "pi_skills_none",
        "skills_all",
        "skills_list",
        "skills_none",
        # inbox_test is loaded by test_sys_async_inbox_e2e.py /
        # test_sys_async_inbox_harness_e2e.py.
        "inbox_test",
        # timer-test is loaded by test fixtures for sys_timer_*
        # tool tests.
        "timer-test",
        # ralph_loop is a loop-mode demo; no dedicated e2e yet.
        "ralph_loop",
    }
)


def test_every_agent_has_a_dedicated_test_file() -> None:
    """
    Walk both agent roots and assert each directory has either
    a matching ``test_example_<name>.py`` file or an entry in
    :data:`_ALT_COVERED`. Also flag orphaned test files whose
    ``<name>`` no longer matches any agent directory.

    :raises AssertionError: When an agent is missing coverage
        or a test file points at a removed agent.
    """
    repo_root = Path(__file__).resolve().parents[3]
    e2e_dir = repo_root / "tests" / "e2e" / "omnigent"

    # Top-level ``examples/<name>/`` AGENTSPEC dirs (must contain a
    # ``config.yaml`` to count — excludes non-agent dirs like
    # ``examples/databricks_apps/``).
    on_disk: set[str] = set()
    examples_root = repo_root / "examples"
    if examples_root.is_dir():
        for p in examples_root.iterdir():
            if p.is_dir() and not p.name.startswith(("_", ".")) and (p / "config.yaml").is_file():
                on_disk.add(p.name)
    # Test-only fixture agents under ``tests/resources/agents/`` —
    # any directory counts (single-file bundles are valid here).
    test_resources_root = repo_root / "tests" / "resources" / "agents"
    if test_resources_root.is_dir():
        for p in test_resources_root.iterdir():
            if p.is_dir() and not p.name.startswith(("_", ".")):
                on_disk.add(p.name)
    # Top-level single-YAML demos (``examples/<name>.yaml``). Each
    # YAML filename stem counts as an agent identity for coverage
    # purposes — the post-2026-04-24 layout puts simple demos at
    # the top level rather than wrapping them in a directory.
    top_level_root = repo_root / "examples"
    if top_level_root.is_dir():
        for p in top_level_root.iterdir():
            if p.is_file() and p.suffix in {".yaml", ".yml"} and not p.name.startswith(("_", ".")):
                on_disk.add(p.stem)

    # Pick up existing tests by file-name convention.
    named_covered: set[str] = set()
    for p in e2e_dir.iterdir():
        name = p.name
        if name.startswith("test_example_") and name.endswith(".py"):
            # Strip prefix+suffix: ``test_example_<name>.py`` -> ``<name>``.
            named_covered.add(name[len("test_example_") : -len(".py")])

    missing = on_disk - named_covered - _ALT_COVERED
    assert missing == set(), (
        f"Agents without a dedicated test file: {sorted(missing)}. "
        f"Create tests/e2e/omnigent/test_example_<name>.py for "
        f"each, or add the name to _ALT_COVERED above if coverage "
        f"lives in a differently-named test file."
    )

    # Agents whose test_example_<name>.py exists but the agent
    # directory was removed (or never committed). Kept as an
    # allowlist so the orphan check doesn't block CI while the
    # agent is pending restoration.
    _ORPHAN_ALLOWED: frozenset[str] = frozenset(
        {
            # test_example_omni.py exists but tests/resources/agents/omni
            # was never committed. The test itself also fails
            # (FileNotFoundError). Allowlisted until the resource is
            # created or the test is deleted.
            "omni",
        }
    )
    stale = named_covered - on_disk - _ORPHAN_ALLOWED
    assert stale == set(), (
        f"Orphaned test_example_<name>.py files for agents that "
        f"no longer exist in examples/ or "
        f"tests/resources/agents/: {sorted(stale)}. Delete the "
        f"test file or restore the agent."
    )
