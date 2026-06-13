"""
Driver for the overflow-render regression test (test_overflow_render.py).

Streams a numbered markdown list whose total line count EXCEEDS the
PTY's viewport rows, so the streamed lines scroll past the top of the
viewport into scrollback. A buggy ``replace_streamed_text`` would
issue a cursor-up + erase that can't reach scrolled-off rows, leaving
raw markdown in scrollback while the rendered markdown also appears in
the viewport — visible duplication wherever the user can see scrollback.

Sister to ``_double_render_driver.py``, which exercises the
in-viewport replace path. Both drive a real :class:`TerminalHost` via
``host.run(handler)``.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys

from omnigent_client import BlockContext, TextChunk, TextDone
from omnigent_ui_sdk.terminal._formatter import RichBlockFormatter
from omnigent_ui_sdk.terminal._host import TerminalHost
from prompt_toolkit.application import get_app

WELCOME_HINTS = ["/help help", "Ctrl+O debug", "Esc cancel", "Ctrl+C exit"]


async def _drive(host: TerminalHost) -> None:
    """
    Drive the host with a long numbered list (intentionally exceeds
    the viewport). Splits via ``RichBlockFormatter`` exactly as the
    REPL would for a streamed text response.
    """
    await asyncio.sleep(0.5)
    fmt = RichBlockFormatter()
    ctx = BlockContext(agent=None, depth=0, turn=0)

    # Each item ends with a trailing space so the assertion
    # ``"description for item N "`` (with space) is unambiguous —
    # without it, "for item 1" would also match "for item 10",
    # "for item 11", etc., poisoning the count.
    chunks: list[str] = ["Items:\n"]
    for i in range(1, 30):
        chunks.append(f"{i:2}. **item{i}** — description for item {i} .\n")
    chunks.append("30. **item30** — description for item 30 .\n\nAll 30 items above.")
    full = "".join(chunks)

    for c in chunks:
        for item in fmt.format_text_chunk(TextChunk(text=c, ctx=ctx)):
            host.output(item)
        await asyncio.sleep(0)
    for item in fmt.format_text_done(TextDone(full_text=full, ctx=ctx)):
        host.output(item)
    await asyncio.sleep(1.0)
    with contextlib.suppress(Exception):
        get_app().exit(exception=EOFError())


async def _amain() -> None:
    host = TerminalHost(model_name="overflow_test", toolbar_hints=WELCOME_HINTS)
    driver_task = asyncio.create_task(_drive(host))  # noqa: RUF006, F841

    async def _handler(text: str) -> None:
        return None

    with contextlib.suppress(EOFError, KeyboardInterrupt):
        await host.run(_handler)


if __name__ == "__main__":
    asyncio.run(_amain())
    sys.exit(0)
