"""Tests for native-wrapper resume hint formatting."""

from __future__ import annotations

from omnigent._native_resume_hint import format_native_resume_command


def test_format_native_resume_command_includes_remote_context() -> None:
    """
    Remote native-wrapper hints include enough context to copy/paste.

    The command must carry the wrapper name, Omnigent server, and
    Omnigent conversation id. If any of those fields are dropped, a
    user who launched against a non-default remote workspace cannot
    reliably resume the same conversation from the printed hint.
    There is no ``--profile`` part: the CLI flag was removed, so a
    hint containing it would tell the user to run a command that
    no longer parses.
    """
    command = format_native_resume_command(
        native_command="claude",
        server="https://example.databricks.com",
        session_id="conv_abc",
    )

    assert command == ("omnigent claude --server https://example.databricks.com --resume conv_abc")
