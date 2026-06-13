"""Helpers for native-wrapper resume hints."""

from __future__ import annotations

import shlex

import click


def format_native_resume_command(
    *,
    native_command: str,
    session_id: str,
    server: str | None = None,
) -> str:
    """
    Build a copyable native-wrapper resume command.

    :param native_command: Native wrapper subcommand, e.g.
        ``"claude"``.
    :param session_id: Omnigent conversation id, e.g.
        ``"conv_abc123"``.
    :param server: Optional Omnigent server URL, e.g.
        ``"https://example.databricks.com"``.
    :returns: Shell-quoted command string, e.g.
        ``"omnigent claude --resume conv_abc123"``.
    """
    parts = ["omnigent", native_command]
    if server is not None:
        parts.extend(["--server", server])
    parts.extend(["--resume", session_id])
    return " ".join(shlex.quote(part) for part in parts)


def echo_native_resume_hint(
    *,
    native_command: str,
    session_id: str,
    server: str | None = None,
) -> None:
    """
    Print a copyable native-wrapper resume command to stderr.

    :param native_command: Native wrapper subcommand, e.g.
        ``"codex"``.
    :param session_id: Omnigent conversation id, e.g.
        ``"conv_abc123"``.
    :param server: Optional Omnigent server URL, e.g.
        ``"https://example.databricks.com"``.
    :returns: None.
    """
    command = format_native_resume_command(
        native_command=native_command,
        session_id=session_id,
        server=server,
    )
    click.echo(f"Resume with: {command}", err=True)
