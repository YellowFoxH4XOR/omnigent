"""Lightweight CLI update reminder.

Two install shapes are supported:

* **Dev clone** (``git clone … && uv sync`` / ``pip install -e .``): we
  walk up from this module to find a ``.git/`` directory, then run
  ``git fetch`` and ``rev-list`` to count how many commits ``HEAD`` is
  behind ``origin/main`` (or ``origin/master``). The result is cached
  to ``~/.omnigent/.update_check.json`` so the (potentially slow)
  ``git fetch`` only runs once per staleness window.

* **Installed wheel** (``uv tool install git+<repo>``,
  ``pip install <pkg>``, ``pipx install``, etc.): no clone is reachable
  on disk, so we read the install metadata that the installer wrote into
  the package's ``.dist-info/`` directory and nag when the install is
  more than ``_INSTALL_STALENESS_SECONDS`` old. The exact upgrade
  command we suggest is tailored to the installer (uv/pip/pipx) and to
  whether the install came from a VCS URL or a registry.

The dispatcher in ``maybe_show_update_notice`` picks the shape based on
whether a ``.git/`` directory is reachable from this module's path.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Imported only for type hints; the Rich import remains lazy at
    # runtime so importing this module stays cheap.
    from rich.console import Console

_ENV_SKIP = "OMNIGENT_NO_UPDATE_CHECK"
_CACHE_DIR = Path.home() / ".omnigent"
_CACHE_FILE = _CACHE_DIR / ".update_check.json"
_STALENESS_SECONDS = 4 * 60 * 60  # 4 hours
_GIT_TIMEOUT_SECONDS = 5
# Wheel-install staleness threshold — nag when the installed build is
# older than this. Chosen to match the user's "1 day stale" ask; nags
# print on every CLI invocation once the threshold is crossed, until
# the user reinstalls.
_INSTALL_STALENESS_SECONDS = 24 * 60 * 60  # 1 day
_DIST_NAME = "omnigent"
# The placeholder that PEP 610 tooling (pip, newer uv) writes into
# ``direct_url.json`` in place of a URL's userinfo. See
# ``_unredact_ssh_userinfo`` for why we have to repair it.
_REDACTED_USERINFO = "****"


@dataclass
class _CacheEntry:
    """Deserialized update-check cache.

    :param last_check_epoch: Unix timestamp of the last ``git fetch``
        check, e.g. ``1716100000.0``.
    :param commits_behind: Number of commits HEAD is behind
        the upstream branch, e.g. ``3``.
    :param head_sha: The HEAD commit hash at the time the cache was
        written, e.g. ``"abc123..."``.  Used to detect when the user
        has pulled (HEAD moves) so the stale count can be rechecked
        cheaply without a fresh ``git fetch``.
    :param kind: Which detection path wrote this entry —
        ``"clone"`` for the dev-clone (``git fetch``) path, or some
        other tag for future install shapes. Defaults to ``"clone"``
        so legacy caches written before this field existed are
        treated as clone caches. The wheel-install path does not
        cache (its dist-info reads are cheap), so no ``"wheel"`` tag
        is currently written.
    """

    last_check_epoch: float
    commits_behind: int
    head_sha: str = ""
    kind: str = "clone"


@dataclass
class _InstalledWheelInfo:
    """Metadata read from the installed package's ``.dist-info/``.

    Populated by ``_read_installed_wheel_info``. ``None`` when our
    distribution is not installed (e.g. running from the source tree
    without any ``pip``/``uv`` install).

    :param install_time_epoch: Unix timestamp for when the package
        was installed, e.g. ``1779311637.0``. Taken from
        ``uv_cache.json`` when available (most accurate; uv records
        the build/cache time), otherwise falls back to the
        ``.dist-info/`` directory's filesystem mtime (universal
        across installers).
    :param installer: The lowercase installer name from the
        ``INSTALLER`` file (PEP 376), e.g. ``"uv"`` / ``"pip"`` /
        ``"poetry"``. ``None`` if the file is missing or empty.
        Note: pipx delegates to pip so its ``INSTALLER`` reads
        ``"pip"``; we additionally check ``sys.prefix`` for the
        ``/pipx/venvs/`` pattern to detect it.
    :param vcs_url: The git URL recorded in ``direct_url.json``
        (PEP 610) when the install came from a VCS source, e.g.
        ``"git+https://github.com/omnigent-ai/omnigent.git"``.
        ``None`` for registry installs (no ``direct_url.json``) or
        URL installs without ``vcs_info``.
    :param commit_sha: The pinned commit SHA recorded by uv or pip
        at install time, e.g. ``"010cf77c3..."``. ``None`` for
        registry installs. Populated from ``direct_url.json``'s
        ``vcs_info.commit_id`` (PEP 610) or, failing that, from
        ``uv_cache.json``'s ``commit`` field.
    :param is_editable: ``True`` when ``direct_url.json`` records
        ``dir_info.editable``, i.e. ``pip install -e`` / editable
        ``uv tool install``. The wheel-check path bails for these.
    :param package_version: The installed package version from
        ``METADATA``, e.g. ``"0.1.0"``. Surfaced in the nag text.
    :param detected_installer: The effective installer label we use
        for picking the upgrade command — same as ``installer`` for
        most cases, but ``"pipx"`` when the pipx venv path heuristic
        fires even though ``INSTALLER`` says ``"pip"``.
    """

    install_time_epoch: float
    installer: str | None
    vcs_url: str | None
    commit_sha: str | None
    is_editable: bool
    package_version: str
    detected_installer: str | None


def maybe_show_update_notice() -> None:
    """Print an update reminder to stderr if this install is stale.

    Dispatches to either the dev-clone path (``.git/`` reachable from
    this module) or the installed-wheel path (``uv tool install`` / pip
    / pipx). Safe to call unconditionally — silently returns when the
    check is disabled by env var, when metadata is missing, when
    ``git`` is unavailable, or when any I/O error occurs.
    """
    if os.environ.get(_ENV_SKIP):
        return

    repo_root = _find_repo_root()
    if repo_root is not None:
        _run_dev_clone_check(repo_root)
    else:
        _run_installed_wheel_check()


def _run_dev_clone_check(repo_root: Path) -> None:
    """Run the dev-clone update check (``git fetch`` + ``rev-list``).

    Caches the commits-behind count to avoid running ``git fetch`` on
    every CLI invocation. Re-counts cheaply when ``HEAD`` has moved
    (the user ran ``git pull``). Any failure is swallowed — the check
    must never break the CLI.

    :param repo_root: Absolute path to the dev clone's Git repo root,
        e.g. ``Path("/Users/me/omnigent")``.
    """
    try:
        cached = _read_cache()
        if cached is not None and cached.kind == "clone" and not _is_stale(cached):
            behind = cached.commits_behind
            # If HEAD moved since the cache was written (e.g. the user
            # ran ``git pull``), do a cheap local recount — no fetch.
            if behind > 0 and cached.head_sha:
                cur_sha = _get_head_sha(repo_root)
                if cur_sha and cur_sha != cached.head_sha:
                    recounted = _local_rev_list_count(repo_root)
                    if recounted is not None:
                        behind = recounted
                        _write_cache(
                            _CacheEntry(
                                last_check_epoch=cached.last_check_epoch,
                                commits_behind=behind,
                                head_sha=cur_sha,
                            )
                        )
            if behind > 0:
                _print_notice(behind)
            return

        result = _run_check(repo_root)
        if result is None:
            # git unavailable or fetch/rev-list failed — record a
            # fresh timestamp so we don't retry on every invocation.
            _write_cache(_CacheEntry(last_check_epoch=time.time(), commits_behind=0, head_sha=""))
            return

        _write_cache(result)
        if result.commits_behind > 0:
            _print_notice(result.commits_behind)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        # Never let the update check break the CLI.
        pass


def _run_installed_wheel_check() -> None:
    """Run the installed-wheel update check (metadata-driven).

    Reads ``INSTALLER``, ``direct_url.json``, and ``uv_cache.json``
    from the package's ``.dist-info/`` directory to determine when
    the package was installed and which installer wrote it. Prints
    a nag when the install is older than ``_INSTALL_STALENESS_SECONDS``.

    No cache — the underlying dist-info reads are cheap (a few small
    files), and once the threshold is crossed the user should see the
    nag on every invocation until they upgrade.
    """
    try:
        info = _read_installed_wheel_info()
    except (OSError, ValueError, KeyError):
        # Metadata parsing failed — fail open.
        return
    if info is None:
        return
    if info.is_editable:
        # Editable install with no .git/ reachable — most likely a
        # ``pip install -e .`` outside the source tree. The clone path
        # would have caught this if .git/ were reachable; there's no
        # meaningful "install age" for an editable install, so bail.
        return

    age_seconds = int(time.time() - info.install_time_epoch)
    if age_seconds >= _INSTALL_STALENESS_SECONDS:
        _print_install_notice(info, age_seconds)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _find_repo_root() -> Path | None:
    """Locate the dev clone's repo root, or ``None`` for installed wheels.

    The check is deliberately narrow: the candidate must be the
    DIRECT parent of the ``omnigent/`` package, and must contain
    both ``.git/`` AND a ``pyproject.toml``. The layout we accept
    is exactly what a dev clone of this repo looks like::

        <repo>/.git/
        <repo>/pyproject.toml
        <repo>/omnigent/update_check.py    ← this file

    We do NOT walk further up the filesystem tree. The previous
    implementation did, which caused a real bug: when ``omnigent``
    is installed via ``uv tool install`` at
    ``~/.local/share/uv/tools/omnigent/…``, the unbounded walk-up
    matched ``~/.git/`` (a dotfiles repo, or any other unrelated
    git repo between $HOME and the install dir) and misclassified
    the install as a dev clone. The dispatcher then ran
    ``git fetch`` against the wrong repo and wrote
    ``kind: "clone"`` to the cache.

    :returns: The repo root ``Path`` for a dev clone, or ``None``
        for any installed-wheel scenario.
    """
    package_dir = Path(__file__).resolve().parent  # <candidate>/omnigent/
    candidate = package_dir.parent
    if (candidate / ".git").is_dir() and (candidate / "pyproject.toml").is_file():
        return candidate
    return None


def _read_cache() -> _CacheEntry | None:
    """Read the cached check result from disk.

    :returns: A ``_CacheEntry`` if the cache file exists and is valid
        JSON, otherwise ``None``.
    """
    try:
        raw = _CACHE_FILE.read_text()
        data = json.loads(raw)
        return _CacheEntry(
            last_check_epoch=float(data["last_check_epoch"]),
            commits_behind=int(data["commits_behind"]),
            head_sha=str(data.get("head_sha", "")),
            # Legacy caches (written before this field existed) get
            # the default ``"clone"`` — they were all clone caches.
            kind=str(data.get("kind", "clone")),
        )
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def _is_stale(entry: _CacheEntry) -> bool:
    """Return whether *entry* is older than the staleness threshold.

    :param entry: The cached check result.
    :returns: ``True`` if the cache should be refreshed.
    """
    return (time.time() - entry.last_check_epoch) >= _STALENESS_SECONDS


def _write_cache(entry: _CacheEntry) -> None:
    """Atomically write *entry* to the cache file.

    Uses write-to-tmpfile + ``Path.replace()`` so concurrent CLI
    invocations never see a half-written file.

    :param entry: The check result to persist.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "last_check_epoch": entry.last_check_epoch,
            "commits_behind": entry.commits_behind,
            "head_sha": entry.head_sha,
            "kind": entry.kind,
        }
    )
    # ``dir=`` on the same filesystem guarantees atomic replace.
    fd, tmp_path = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".tmp")
    closed = False
    try:
        os.write(fd, payload.encode())
        os.close(fd)
        closed = True
        Path(tmp_path).replace(_CACHE_FILE)
    except OSError:
        if not closed:
            with contextlib.suppress(OSError):
                os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def _run_check(repo_root: Path) -> _CacheEntry | None:
    """Fetch upstream and count how many commits HEAD is behind.

    Tries ``origin/main`` first; falls back to ``origin/master``.

    :param repo_root: Absolute path to the Git repository root,
        e.g. ``Path("/home/user/omnigent-2")``.
    :returns: A ``_CacheEntry`` with the result, or ``None`` if
        ``git`` is not available or both branches fail.
    """
    for branch in ("main", "master"):
        behind = _fetch_and_count(repo_root, branch)
        if behind is not None:
            return _CacheEntry(
                last_check_epoch=time.time(),
                commits_behind=behind,
                head_sha=_get_head_sha(repo_root) or "",
            )
    return None


def _fetch_and_count(repo_root: Path, branch: str) -> int | None:
    """Fetch ``origin/<branch>`` and return commits-behind count.

    :param repo_root: Absolute path to the Git repository root.
    :param branch: Remote branch name, e.g. ``"main"``.
    :returns: Number of commits HEAD is behind, or ``None`` on
        any failure (timeout, missing remote, missing branch).
    """
    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "fetch", "origin", branch, "--quiet"],
            timeout=_GIT_TIMEOUT_SECONDS,
            capture_output=True,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None

    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "rev-list",
                f"HEAD..origin/{branch}",
                "--count",
            ],
            timeout=_GIT_TIMEOUT_SECONDS,
            capture_output=True,
            check=True,
            text=True,
        )
        return int(result.stdout.strip())
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def _get_head_sha(repo_root: Path) -> str | None:
    """Return the current HEAD commit hash, or ``None`` on failure.

    :param repo_root: Absolute path to the Git repository root.
    :returns: The full SHA-1 hex string, or ``None``.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            timeout=_GIT_TIMEOUT_SECONDS,
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _local_rev_list_count(repo_root: Path) -> int | None:
    """Count commits HEAD is behind origin/main (or master), locally only.

    No ``git fetch`` — uses whatever ``origin/main`` ref is already
    available.  This is cheap and handles the post-pull case.

    :param repo_root: Absolute path to the Git repository root.
    :returns: Number of commits behind, or ``None`` on failure.
    """
    for branch in ("main", "master"):
        try:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "rev-list",
                    f"HEAD..origin/{branch}",
                    "--count",
                ],
                timeout=_GIT_TIMEOUT_SECONDS,
                capture_output=True,
                check=True,
                text=True,
            )
            return int(result.stdout.strip())
        except (subprocess.SubprocessError, OSError, ValueError):
            continue
    return None


def _print_notice(commits_behind: int) -> None:
    """Print the update notice to stderr.

    :param commits_behind: Number of commits the local clone is
        behind, e.g. ``3``.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console(stderr=True)
    body = Text.assemble(
        ("Update available", "bold yellow"),
        " — origin/main is ",
        (f"{commits_behind}", "bold"),
        " commit(s) ahead.\nRun ",
        ("git pull", "bold"),
        " to update.",
    )
    console.print(Panel(body, border_style="yellow", expand=False))


# ------------------------------------------------------------------
# Installed-wheel path
# ------------------------------------------------------------------


def _read_build_info() -> tuple[float, str] | None:
    """Return ``(build_time_epoch, commit_sha)`` from ``_build_info``.

    The ``omnigent/_build_info.py`` module is generated by
    ``setup.py``'s ``build_py`` override at wheel build time. It is
    gitignored, so source checkouts that have never been built
    won't have it on disk — in which case the import fails and we
    return ``None`` so the caller can fall back to other signals.

    Wrapped as a module-level helper (rather than an inline ``try
    import``) so tests can monkeypatch it without going through
    ``sys.modules`` to fake the import outcome.

    :returns: A ``(BUILD_TIME_EPOCH, COMMIT_SHA)`` pair, or ``None``
        when ``_build_info`` is unavailable. ``COMMIT_SHA`` may be
        the empty string (when ``setup.py`` ran in an environment
        without ``git``); the caller treats that as "no commit info".
    """
    try:
        from omnigent import _build_info  # type: ignore[attr-defined]
    except ImportError:
        return None
    try:
        ts = float(_build_info.BUILD_TIME_EPOCH)
        sha = str(_build_info.COMMIT_SHA)
    except (AttributeError, TypeError, ValueError):
        # Malformed _build_info.py (manually edited or corrupted).
        # Don't trust it — fall back to other signals.
        return None
    return ts, sha


def _get_distribution() -> importlib.metadata.Distribution | None:
    """Resolve our installed distribution, or ``None`` if not installed.

    Wrapped as a module-level helper so tests can monkeypatch it
    without walking through ``importlib.metadata``'s module
    singleton (see ``CLAUDE.md`` rule 14 on global-module-clobbering).

    :returns: The ``Distribution`` for the ``omnigent`` package,
        or ``None`` when not installed (e.g. running directly from a
        source tarball without ``pip install``).
    """
    try:
        return importlib.metadata.distribution(_DIST_NAME)
    except importlib.metadata.PackageNotFoundError:
        return None


def _unredact_ssh_userinfo(vcs_url: str) -> str:
    """Repair a redacted SSH user in a VCS URL read from ``direct_url.json``.

    PEP 610 tooling (pip, and newer uv) redacts the userinfo of a VCS
    URL before writing it to ``direct_url.json``: a bare username with
    no password — which for an SSH remote is just the login user — is
    rewritten to the marker ``****``. So an install from
    ``git+ssh://git@github.com/org/repo.git`` is recorded as
    ``git+ssh://****@github.com/org/repo.git``.

    The SSH user is not a secret (it is ``git`` for GitHub, GitLab,
    Bitbucket, and every other major host), but the redaction leaves the
    reinstall command unrunnable: SSH tries to authenticate as the
    literal user ``****`` and fails with ``Permission denied
    (publickey)``. The original user is irrecoverable from
    the metadata, so we restore the canonical ``git`` — the only SSH
    user those hosts accept — which makes both the command we display
    and the command we run match and actually work.

    Only SSH URLs whose userinfo is *exactly* the redaction marker (the
    no-password case) are repaired. A partially-redacted ``user:****@``
    form encodes a real password we cannot and must not reconstruct, and
    non-SSH URLs (HTTPS, ``file://``) pass through untouched.

    :param vcs_url: The normalized ``<vcs>+<scheme>://…`` reinstall URL,
        e.g. ``"git+ssh://****@github.com/omnigent-ai/omnigent.git"``.
    :returns: The same URL with a redacted SSH user restored to ``git``,
        e.g. ``"git+ssh://git@github.com/omnigent-ai/omnigent.git"``;
        the input unchanged when no redacted SSH user is present.
    """
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(vcs_url)
    # Scheme is e.g. ``ssh`` or ``git+ssh``. Only SSH carries a login
    # user we can safely default to ``git``; HTTPS userinfo is a real
    # credential.
    if not (parts.scheme == "ssh" or parts.scheme.endswith("+ssh")):
        return vcs_url
    # Repair only the whole-username redaction (no password). A
    # ``user:****`` form hides a real password — leave it alone.
    if parts.username != _REDACTED_USERINFO or parts.password is not None:
        return vcs_url

    new_netloc = parts.netloc.replace(f"{_REDACTED_USERINFO}@", "git@", 1)
    return urlunsplit(parts._replace(netloc=new_netloc))


def _read_installed_wheel_info() -> _InstalledWheelInfo | None:
    """Read the installed distribution's metadata for the wheel check.

    Parses three files in the ``.dist-info/`` directory:

    * ``INSTALLER`` (PEP 376) — the installer that wrote the package
      (``"uv"``, ``"pip"``, ``"poetry"``, ...).
    * ``direct_url.json`` (PEP 610) — present for direct-URL installs
      (``git+<url>``, wheel-by-URL, ``-e``); absent for plain registry
      installs. Carries ``vcs_info`` (commit SHA) for git installs and
      ``dir_info.editable`` for editable installs.
    * ``uv_cache.json`` (uv-specific) — carries the build/cache
      timestamp and pinned commit SHA. Present for any uv-built
      package; absent for pip/pipx/poetry installs.

    Install time falls back to the ``.dist-info/`` directory's mtime
    when ``uv_cache.json`` is absent — this works for any installer.

    :returns: An ``_InstalledWheelInfo`` populated from the
        available metadata, or ``None`` when the package isn't
        installed at all (e.g. running from a checkout without a
        registered distribution) or when no install-time signal
        can be recovered.
    """
    dist = _get_distribution()
    if dist is None:
        return None

    installer = _read_installer(dist)

    is_editable = False
    vcs_url: str | None = None
    commit_sha: str | None = None
    direct_url_raw = _safe_read_dist_file(dist, "direct_url.json")
    if direct_url_raw:
        try:
            data = json.loads(direct_url_raw)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            dir_info = data.get("dir_info")
            if isinstance(dir_info, dict) and dir_info.get("editable"):
                is_editable = True
            vcs_info = data.get("vcs_info")
            if isinstance(vcs_info, dict):
                url = data.get("url")
                cid = vcs_info.get("commit_id")
                if isinstance(url, str):
                    # Normalize to the ``git+<url>`` form pip/uv accept
                    # back as a reinstall target. ``direct_url.json``
                    # stores the bare ``url`` field; the ``vcs`` (e.g.
                    # ``"git"``) prefix lives in ``vcs_info``.
                    vcs = vcs_info.get("vcs", "git")
                    vcs_url = url if url.startswith(f"{vcs}+") else f"{vcs}+{url}"
                    # Repair an SSH user the installer redacted to
                    # ``****`` when it wrote direct_url.json — otherwise
                    # the reinstall command ssh's in as user ``****``
                    # and fails.
                    vcs_url = _unredact_ssh_userinfo(vcs_url)
                if isinstance(cid, str):
                    commit_sha = cid

    install_time_epoch: float | None = None
    # Tier 1 (preferred): build-baked _build_info.py. Written by
    # ``setup.py``'s build_py override at wheel build time; carries
    # the exact build moment and commit SHA. Works across every
    # installer (uv / pip / pipx / poetry / ...) because it travels
    # inside the wheel, not as installer-specific metadata.
    build_info = _read_build_info()
    if build_info is not None:
        ts, sha = build_info
        install_time_epoch = ts
        if commit_sha is None and sha:
            commit_sha = sha

    # Tier 2: uv_cache.json — uv-only, but more accurate than mtime
    # for uv installs. Only consulted when _build_info didn't set
    # install_time_epoch (i.e. source checkouts where the build
    # hook never ran, or wheels packaged without setup.py running).
    uv_cache_raw = _safe_read_dist_file(dist, "uv_cache.json")
    if uv_cache_raw:
        try:
            uv_data = json.loads(uv_cache_raw)
        except json.JSONDecodeError:
            uv_data = None
        if isinstance(uv_data, dict):
            if install_time_epoch is None:
                ts = uv_data.get("timestamp")
                if isinstance(ts, dict):
                    secs = ts.get("secs_since_epoch")
                    if isinstance(secs, (int, float)):
                        install_time_epoch = float(secs)
            if commit_sha is None:
                uv_commit = uv_data.get("commit")
                if isinstance(uv_commit, str):
                    commit_sha = uv_commit

    if install_time_epoch is None:
        # Fall back to the dist-info dir's mtime — set by the
        # installer when it wrote the files. Works for pip / pipx /
        # poetry, and for uv installs that for some reason lack
        # ``uv_cache.json``.
        dist_info_dir = _dist_info_dir(dist)
        if dist_info_dir is not None:
            try:
                install_time_epoch = dist_info_dir.stat().st_mtime
            except OSError:
                install_time_epoch = None

    if install_time_epoch is None:
        return None

    detected_installer = installer
    if installer == "pip" and _looks_like_pipx_install():
        detected_installer = "pipx"

    return _InstalledWheelInfo(
        install_time_epoch=install_time_epoch,
        installer=installer,
        vcs_url=vcs_url,
        commit_sha=commit_sha,
        is_editable=is_editable,
        package_version=dist.version,
        detected_installer=detected_installer,
    )


def _read_installer(dist: importlib.metadata.Distribution) -> str | None:
    """Return the lowercase installer name from ``INSTALLER``.

    :param dist: The distribution to read from.
    :returns: The installer name (e.g. ``"uv"``, ``"pip"``), or
        ``None`` when ``INSTALLER`` is missing, empty, or unreadable.
    """
    raw = _safe_read_dist_file(dist, "INSTALLER")
    if not raw:
        return None
    name = raw.strip().lower()
    return name or None


def _safe_read_dist_file(dist: importlib.metadata.Distribution, name: str) -> str | None:
    """Read a dist-info file by name, returning ``None`` on any error.

    ``Distribution.read_text`` already swallows ``FileNotFoundError``
    and friends, but the wrapper guards against transient permission
    errors and surprise ``OSError`` flavors so the caller never has
    to catch.

    :param dist: The distribution to read from.
    :param name: File name within the ``.dist-info/`` directory,
        e.g. ``"INSTALLER"`` or ``"direct_url.json"``.
    :returns: File contents as text, or ``None`` when the file is
        missing / unreadable / empty.
    """
    try:
        text = dist.read_text(name)
    except (OSError, UnicodeDecodeError):
        return None
    if text is None:
        return None
    return text


def _dist_info_dir(dist: importlib.metadata.Distribution) -> Path | None:
    """Return the ``.dist-info/`` directory for a path-based distribution.

    ``importlib.metadata.PathDistribution`` stores the directory path
    on its ``_path`` attribute. The underscore is unfortunate but
    that's the documented handle; ``Distribution`` itself defines no
    public accessor. Returns ``None`` for non-path distributions
    (e.g. zipped eggs), where ``stat()`` can't give us an mtime.

    :param dist: The distribution.
    :returns: Absolute path to the ``.dist-info/`` directory, or
        ``None`` when the distribution isn't a ``PathDistribution``.
    """
    raw = getattr(dist, "_path", None)
    if isinstance(raw, Path):
        return raw
    # Some Python builds store this as a ``zipp.Path``-like object;
    # ``str()`` of those is the on-disk path. Only return when the
    # resulting path actually exists on disk.
    if raw is not None:
        candidate = Path(str(raw))
        if candidate.is_dir():
            return candidate
    return None


def _looks_like_pipx_install() -> bool:
    """Detect whether the running interpreter lives inside a pipx venv.

    pipx delegates to pip, so ``INSTALLER`` reads ``"pip"`` — but pipx
    creates its venvs under ``~/.local/pipx/venvs/<pkg>/`` (or the
    platform equivalent). Checking ``sys.prefix`` for the ``pipx/venvs``
    segment is the standard heuristic and is what pipx itself uses for
    self-recognition.

    :returns: ``True`` when ``sys.prefix`` looks like a pipx venv.
    """
    return "pipx/venvs" in sys.prefix.replace(os.sep, "/")


@dataclass
class _UpgradeSuggestion:
    """A suggested upgrade command for the user's install shape.

    :param command: The shell command (or prose) we display in the
        nag panel, e.g. ``"uv tool upgrade omnigent"`` or
        ``"reinstall omnigent from your original source"``.
        Always populated.
    :param runnable: ``True`` when ``command`` is a real shell
        invocation we can execute via ``subprocess.run`` —
        i.e. the installer is one of uv / pip / pipx / poetry
        (or pip-as-fallback for an unknown installer with a known
        VCS URL). ``False`` for the unknown-installer prose
        fallbacks (``"reinstall X from ..."``) which exist to be
        read, not run; the interactive "run this now?" prompt is
        suppressed in that case.
    """

    command: str
    runnable: bool


def _build_upgrade_suggestion(info: _InstalledWheelInfo) -> _UpgradeSuggestion:
    """Build the right upgrade command for the user's install shape.

    Picks based on ``detected_installer`` (uv / pip / pipx / poetry /
    unknown) and whether ``direct_url.json`` recorded a VCS URL.

    :param info: Metadata from ``_read_installed_wheel_info``.
    :returns: A :class:`_UpgradeSuggestion` whose ``command`` is the
        line printed in the nag panel and whose ``runnable`` flag
        tells the caller whether the line is an actual invocation
        (so the interactive prompt is offered) or a prose fallback
        (so the prompt is suppressed).
    """
    installer = info.detected_installer or info.installer

    if info.vcs_url:
        # VCS install — we know the exact source URL.
        if installer == "uv":
            return _UpgradeSuggestion(
                command=f"uv tool install --reinstall {info.vcs_url}",
                runnable=True,
            )
        if installer == "pipx":
            # pipx tracks the original spec; ``reinstall`` re-pulls it.
            return _UpgradeSuggestion(command=f"pipx reinstall {_DIST_NAME}", runnable=True)
        if installer in ("pip", None):
            return _UpgradeSuggestion(
                command=f"pip install --force-reinstall {info.vcs_url}",
                runnable=True,
            )
        if installer == "poetry":
            return _UpgradeSuggestion(command=f"poetry add --force {info.vcs_url}", runnable=True)
        # Unknown installer with a known URL — fall through to a
        # generic suggestion that names the URL so the user can wire
        # it into their own tool. Not runnable.
        return _UpgradeSuggestion(
            command=f"reinstall {_DIST_NAME} from {info.vcs_url}", runnable=False
        )

    # Registry install — no VCS URL recorded.
    if installer == "uv":
        return _UpgradeSuggestion(command=f"uv tool upgrade {_DIST_NAME}", runnable=True)
    if installer == "pipx":
        return _UpgradeSuggestion(command=f"pipx upgrade {_DIST_NAME}", runnable=True)
    if installer == "pip":
        return _UpgradeSuggestion(command=f"pip install -U {_DIST_NAME}", runnable=True)
    if installer == "poetry":
        return _UpgradeSuggestion(command=f"poetry update {_DIST_NAME}", runnable=True)
    return _UpgradeSuggestion(
        command=f"reinstall {_DIST_NAME} from your original source",
        runnable=False,
    )


def _stdin_is_tty() -> bool:
    """Return ``True`` when ``sys.stdin`` is connected to a terminal.

    Wrapped as a module-level helper so tests can monkeypatch the
    answer directly without having to replace ``sys.stdin`` itself
    (replacing the stream would also break Rich's input-reading path
    used by the confirmation prompt).

    :returns: ``True`` for an interactive terminal, ``False`` for
        piped/redirected stdin (CI, ``< /dev/null``, etc.).
    """
    return sys.stdin.isatty()


def _prompt_yes_no(console: Console, question: str) -> bool:
    """Prompt the user with a yes/no Rich confirmation, defaulting to no.

    Wrapped as a module-level helper so tests can monkeypatch the
    answer without having to feed bytes through stdin. The Rich
    ``Confirm`` import is deferred so this module stays cheap to
    import.

    :param console: The Rich ``Console`` (stderr) used by the
        surrounding panel — passing it keeps the prompt visually
        attached to the nag rather than landing on a separate
        stream.
    :param question: Prompt text, e.g. ``"Run this now?"``.
    :returns: ``True`` for yes, ``False`` for no or an empty answer.
    """
    from rich.prompt import Confirm

    return Confirm.ask(question, default=False, console=console)


def _run_upgrade_command(command: str, console: Console) -> int:
    """Run the upgrade command in a foreground subprocess.

    The subprocess inherits stdin/stdout/stderr so the user sees
    installer progress (uv/pip output) live. We never pass
    ``shell=True``; the command is tokenized with ``shlex.split``.

    :param command: Shell-style command line, e.g.
        ``"uv tool upgrade omnigent"``.
    :param console: Rich console (stderr) used for the surrounding
        "Running:" / failure status lines, kept consistent with the
        panel above.
    :returns: The subprocess's exit code, or ``-1`` when the binary
        couldn't be started (binary missing from PATH, invalid
        command string). The caller treats any non-zero return as
        "upgrade failed" and falls back to the existing install.
    """
    import shlex

    console.print(f"[yellow]Running:[/yellow] {command}")
    try:
        result = subprocess.run(shlex.split(command), check=False)
    except (OSError, ValueError) as exc:
        # OSError: binary not on PATH. ValueError: shlex.split
        # rejected the command (extremely unlikely for our
        # generated commands, but the upgrade is best-effort).
        console.print(f"[red]Upgrade failed to start:[/red] {exc}")
        return -1
    return result.returncode


def _print_install_notice(info: _InstalledWheelInfo, age_seconds: int) -> None:
    """Print the wheel-install update notice and optionally run the upgrade.

    Always renders the Rich panel summarizing install age + the
    recommended upgrade command. When that command is *runnable*
    (i.e. the installer is one we have a recipe for) AND stdin is
    an interactive terminal, additionally prompts the user to run
    the upgrade now. On user confirmation: shells out via
    ``subprocess.run``, then exits the process regardless of
    outcome — ``sys.exit(0)`` on success, ``sys.exit(1)`` on
    failure. The exit on failure matters because the running
    interpreter still has the OLD modules loaded in memory; if we
    fell through into the user's original command after a failed
    upgrade, they would silently keep running the pre-upgrade
    code while believing they had just attempted to upgrade.
    Exiting forces them to re-invoke (which either picks up the
    new code on success, or surfaces the failure clearly so they
    can investigate).

    When the recommended command is NOT runnable (unknown
    installer fallback), the panel itself is the explanation and
    we deliberately stay silent afterwards — no prompt, no skip
    notice. When stdin is not a TTY (CI, piped, redirected) we
    print a single-line skip notice so the absence of a prompt is
    visible in logs.

    :param info: Metadata about the install, used to format the
        upgrade command.
    :param age_seconds: Seconds since install — gets rounded down to
        days for display. The minimum displayed value is ``1`` so we
        never print ``"0 day(s) old"`` even if the threshold logic
        ever calls us right at the boundary.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    days = max(1, age_seconds // 86400)
    suggestion = _build_upgrade_suggestion(info)
    console = Console(stderr=True)
    body = Text.assemble(
        ("Update check", "bold yellow"),
        " — installed build is ",
        (f"{days}", "bold"),
        " day(s) old.\nRun ",
        (suggestion.command, "bold"),
        " to update.",
    )
    console.print(Panel(body, border_style="yellow", expand=False))

    # Unknown installer → the "command" is prose, not an invocation.
    # Don't offer to run it and don't print a skip notice — the
    # panel's "reinstall X from your original source" line already
    # tells the user we can't help further.
    if not suggestion.runnable:
        return

    # Non-interactive stdin: prompting would block forever (CI,
    # background jobs) or EOF immediately (``< /dev/null``).
    # Surface the skip explicitly so a log reader can tell why no
    # prompt was offered when the panel above promised an action.
    if not _stdin_is_tty():
        console.print("[dim]Skipped interactive upgrade prompt — stdin is not a TTY.[/dim]")
        return

    if not _prompt_yes_no(console, "Run this now?"):
        return

    exit_code = _run_upgrade_command(suggestion.command, console)
    # Always exit — the running interpreter still has the
    # pre-upgrade modules loaded. Continuing into the user's
    # original command would either (a) silently run old code
    # after a "successful" upgrade or (b) silently run old code
    # while pretending the failed upgrade was a no-op. Both are
    # confusing; forcing a re-invoke is the only honest path.
    if exit_code == 0:
        console.print(
            "[green]Upgrade complete.[/green] Re-run your command to use the new version."
        )
        sys.exit(0)
    console.print(
        f"[red]Upgrade exited with status {exit_code}.[/red] "
        "Re-run your command to retry (or unset the upgrade prompt)."
    )
    sys.exit(1)
