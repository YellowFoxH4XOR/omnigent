"""E2E tests for ``omnigent config --global`` defaults (CUJ 4).

Unit-level coverage of the config command and its loaders lives in
``tests/cli/test_cli.py`` (CliRunner, in-process, monkeypatched globals).
The gap these tests close: do the same contracts hold across a real
subprocess boundary — one ``omnigent config`` invocation writes the
file under a real ``$HOME``, the next invocation reads it back from the
filesystem? That's what catches file-format drift, YAML escaping bugs,
and env-isolation gaps that in-process Click tests can't see.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_RUN_TIMEOUT_SEC = 180


def _bare_env(home: Path, omnigent_repo_root: Path) -> dict[str, str]:
    """
    Build a minimal subprocess env that doesn't need LLM credentials.

    Used by the config-command-only tests (tests 1 and 2). Skips
    ``omnigent_credentials_env`` because writing/listing config files
    never calls the gateway — keeping these tests cred-free lets them
    run on a developer laptop without a Databricks PAT.

    :param home: Directory to use as ``$HOME`` so
        ``~/.omnigent/config.yaml`` lands under test isolation.
    :param omnigent_repo_root: Worktree root, prepended onto
        ``PYTHONPATH`` so the subprocess imports the worktree's
        omnigent and not the installed package. Mirrors the
        pattern in ``omnigent_credentials_env``.
    :returns: Env dict suitable for ``subprocess.run(env=...)``.
    """
    existing_pp = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join(p for p in (str(omnigent_repo_root), existing_pp) if p)
    return {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": pythonpath,
        "OMNIGENT_SKIP_ONBOARD": "1",
        "OMNIGENT_NO_UPDATE_CHECK": "1",
    }


def _run_omnigent(
    *,
    omnigent_python: Path,
    omnigent_repo_root: Path,
    env: dict[str, str],
    args: list[str],
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """
    Spawn ``python -m omnigent <args>`` with the given env.

    :param omnigent_python: Interpreter from the fixture.
    :param omnigent_repo_root: Cwd so module resolution and YAML
        callable-imports work.
    :param env: Subprocess env. Caller is responsible for credentials.
    :param args: Argv tail after ``-m omnigent``.
    :param stdin: Optional stdin payload (used to drive interactive
        modes from headless tests).
    :returns: The completed subprocess result.
    """
    return subprocess.run(
        [str(omnigent_python), "-m", "omnigent", *args],
        env=env,
        cwd=str(omnigent_repo_root),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT_SEC,
    )


def test_global_config_write_then_list_roundtrips(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    tmp_path: Path,
) -> None:
    """
    ``config set --global KEY=VALUE`` writes the file; ``config list``
    reads it back. ``config unset`` removes the key.

    Catches file-format drift (YAML escaping, key ordering),
    ``config list`` output regressions, and any subprocess-boundary bug
    where the write succeeds but the next process can't parse it.
    """
    home = tmp_path / "home"
    home.mkdir()
    env = _bare_env(home, omnigent_repo_root)

    write = _run_omnigent(
        omnigent_python=omnigent_python,
        omnigent_repo_root=omnigent_repo_root,
        env=env,
        args=[
            "config",
            "set",
            "--global",
            "default_agent=tests/resources/examples/hello_world.yaml",
            "model=databricks-claude-sonnet-4-6",
            "server=https://example.databricks.com",
        ],
    )
    assert write.returncode == 0, (
        f"config set --global write failed: stdout={write.stdout!r} stderr={write.stderr!r}"
    )

    # The config file must exist on disk after the write — the
    # subprocess didn't just print success and skip the I/O.
    config_path = home / ".omnigent" / "config.yaml"
    assert config_path.is_file(), f"Expected config at {config_path} after write; not found."

    listed = _run_omnigent(
        omnigent_python=omnigent_python,
        omnigent_repo_root=omnigent_repo_root,
        env=env,
        args=["config", "list"],
    )
    assert listed.returncode == 0, (
        f"config list failed: stdout={listed.stdout!r} stderr={listed.stderr!r}"
    )
    # All three keys must appear in --list output. The config
    # command resolves relative paths to absolute on write
    # (the saved defaults need to work from any cwd), so we check
    # the basename rather than the literal input string.
    assert "model=databricks-claude-sonnet-4-6" in listed.stdout, (
        f"model not in --list output; got {listed.stdout!r}"
    )
    assert "server=https://example.databricks.com" in listed.stdout, (
        f"server not in --list output; got {listed.stdout!r}"
    )
    assert "default_agent=" in listed.stdout, (
        f"default_agent not in --list output; got {listed.stdout!r}"
    )
    assert "hello_world.yaml" in listed.stdout, (
        f"hello_world.yaml not in --list output; got {listed.stdout!r}"
    )

    # --unset removes a single key without disturbing the others.
    unset = _run_omnigent(
        omnigent_python=omnigent_python,
        omnigent_repo_root=omnigent_repo_root,
        env=env,
        args=["config", "unset", "--global", "server"],
    )
    assert unset.returncode == 0, (
        f"config unset failed: stdout={unset.stdout!r} stderr={unset.stderr!r}"
    )

    listed_after = _run_omnigent(
        omnigent_python=omnigent_python,
        omnigent_repo_root=omnigent_repo_root,
        env=env,
        args=["config", "list"],
    )
    assert listed_after.returncode == 0
    assert "server=" not in listed_after.stdout, (
        f"server key should be gone after unset; got {listed_after.stdout!r}"
    )
    # The other two keys must still be there — --unset shouldn't
    # truncate the file.
    assert "model=databricks-claude-sonnet-4-6" in listed_after.stdout
    assert "default_agent=" in listed_after.stdout


def test_global_config_unknown_key_rejected_at_subprocess_boundary(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    tmp_path: Path,
) -> None:
    """
    ``config set --global bogus_key=foo`` exits non-zero with a clear error
    message. The validator must fire in the subprocess, not just in the
    in-process CliRunner harness.
    """
    home = tmp_path / "home"
    home.mkdir()
    env = _bare_env(home, omnigent_repo_root)

    result = _run_omnigent(
        omnigent_python=omnigent_python,
        omnigent_repo_root=omnigent_repo_root,
        env=env,
        args=["config", "set", "--global", "bogus_key=foo"],
    )
    assert result.returncode != 0, (
        f"Expected non-zero exit for unknown config key; got 0.\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # Error should name the offending key so the user can fix it.
    combined = result.stdout + result.stderr
    assert "bogus_key" in combined, (
        f"Expected the unknown key name in the error message; "
        f"got stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # And the config file should NOT have been written — invalid
    # writes must be transactional, not "write what you could and
    # then complain".
    config_path = home / ".omnigent" / "config.yaml"
    assert not config_path.exists() or "bogus_key" not in config_path.read_text(), (
        f"Invalid key was persisted to {config_path}; write should "
        f"have been rejected before touching the file."
    )


def test_global_config_default_agent_drives_bare_omnigent(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
    tmp_path: Path,
) -> None:
    """
    A ``default_agent`` set via ``config set --global`` is honored by
    bare ``omnigent -p ...`` (no AGENT arg) — proving the saved
    file actually flows into the run path on a separate subprocess
    invocation.

    What breaks if this fails: the file is written correctly but
    ``_load_effective_config`` doesn't reach the run/bare path on
    subprocess start (env-isolation issue, lazy loader regression,
    or a refactor that broke the bare-``omnigent`` shortcut
    documented in the README).
    """
    home = tmp_path / "home"
    home.mkdir()
    # Inherit creds + base URL from the fixture but override HOME so
    # the test's config file is isolated from any developer-local
    # ~/.omnigent.
    env = dict(omnigent_credentials_env)
    env["HOME"] = str(home)

    yaml_path = omnigent_repo_root / "tests" / "resources" / "examples" / "hello_world.yaml"
    write = _run_omnigent(
        omnigent_python=omnigent_python,
        omnigent_repo_root=omnigent_repo_root,
        env=env,
        args=[
            "config",
            "set",
            "--global",
            f"default_agent={yaml_path}",
            "harness=openai-agents",
            "model=databricks-gpt-5-4-mini",
        ],
    )
    assert write.returncode == 0, (
        f"config set --global write failed: stdout={write.stdout!r} stderr={write.stderr!r}"
    )

    # Bare ``omnigent -p PROMPT`` (no AGENT). With the global
    # default_agent set, this must resolve to hello_world.yaml and
    # produce a real assistant reply.
    run = _run_omnigent(
        omnigent_python=omnigent_python,
        omnigent_repo_root=omnigent_repo_root,
        env=env,
        args=["run", "-p", "say hi in 5 words", "--no-session", "--no-log"],
    )
    assert run.returncode == 0, (
        f"bare ``omnigent run`` with global default_agent failed: "
        f"stdout={run.stdout!r} stderr={run.stderr!r}\n"
        f"If exit != 0, the saved config file didn't reach the run "
        f"path — ``_load_effective_config`` is no longer being called "
        f"or the default_agent key isn't being threaded into the "
        f"target argument."
    )
    # Some non-trivial assistant reply must land in stdout — proves
    # the agent resolved from config actually ran a turn.
    assert len(run.stdout.strip()) >= 4, (
        f"Expected an assistant reply in stdout; got {run.stdout!r}"
    )
