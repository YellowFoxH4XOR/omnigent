"""E2E for ``ToolCallInProgress.is_client_side`` plumbing: the inline
``response.output_item.done`` SSE event for a client-side tool carries
``status="action_required"``."""

from __future__ import annotations

import json
import threading
from typing import Any

import httpx

_SENTINEL = "DBX-SSE-STATUS-Q9X7K3M2-CANARY"


_LOOKUP_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_secret_token",
            "description": (
                "Look up the unguessable secret token for today. The "
                "user does not know it; you MUST call this tool to "
                "retrieve the token. Return the token verbatim."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


def _iter_sse(response: httpx.Response):
    """Yield decoded SSE event dicts from a streaming response."""
    buffer = ""
    for chunk in response.iter_text():
        buffer += chunk
        while "\n\n" in buffer:
            frame, _, buffer = buffer.partition("\n\n")
            data_line = next(
                (line for line in frame.splitlines() if line.startswith("data:")),
                None,
            )
            if data_line is None:
                continue
            payload = data_line[len("data:") :].strip()
            if payload == "[DONE]":
                return
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue


def test_client_side_tool_inline_sse_carries_action_required(
    live_server: str,
    http_client: httpx.Client,
    claude_coder_agent: str,
) -> None:
    """Stream a turn, assert the inline function_call SSE event has
    ``status="action_required"``, PATCH back, assert the sentinel
    round-trips into the text deltas.

    :param live_server: Base URL. Used to build a fresh PATCH client
        so the streaming connection doesn't head-of-line block.
    :param http_client: HTTP client pointed at the live server.
    :param claude_coder_agent: Uploaded claude-coder agent name.
    """
    patch_done = threading.Event()
    patch_status: dict[str, Any] = {}

    def _patch_in_thread(response_id: str, call_id: str) -> None:
        try:
            with httpx.Client(base_url=live_server, timeout=30) as patch_client:
                resp = patch_client.patch(
                    f"/v1/responses/{response_id}",
                    json={
                        "tool_results": [
                            {"call_id": call_id, "output": _SENTINEL},
                        ],
                    },
                )
                patch_status["status"] = resp.status_code
                patch_status["body"] = resp.text[:300]
        finally:
            patch_done.set()

    body = {
        "model": claude_coder_agent,
        "input": (
            "What is today's secret token? You MUST call the "
            "lookup_secret_token tool to retrieve it. Then include "
            "the token verbatim in your response. Do not "
            "paraphrase, summarize, or wrap it in any explanation. "
            "The token is opaque and unguessable; do not invent one."
        ),
        "stream": True,
        "tools": _LOOKUP_TOOLS,
    }

    response_id: str | None = None
    inline_function_call_status: str | None = None
    text_chunks: list[str] = []
    saw_completed = False

    with http_client.stream("POST", "/v1/responses", json=body, timeout=180.0) as response:
        assert response.status_code == 200, (
            f"POST returned {response.status_code}: {response.read()[:300]!r}"
        )
        for event in _iter_sse(response):
            etype = event.get("type")
            if etype == "response.created":
                response_id = event.get("response", {}).get("id")
            elif etype == "response.output_item.done":
                item = event.get("item") or {}
                if (
                    item.get("type") == "function_call"
                    and item.get("name") == "lookup_secret_token"
                ):
                    inline_function_call_status = item.get("status")
                    call_id = item.get("call_id")
                    if response_id and call_id and not patch_done.is_set():
                        threading.Thread(
                            target=_patch_in_thread,
                            args=(response_id, call_id),
                            daemon=True,
                        ).start()
            elif etype == "response.output_text.delta":
                delta = event.get("delta")
                if isinstance(delta, str):
                    text_chunks.append(delta)
            elif etype == "response.completed":
                saw_completed = True
                break
            elif etype == "response.failed":
                break

    assert inline_function_call_status == "action_required", (
        f"got status={inline_function_call_status!r}, expected 'action_required'"
    )
    assert patch_done.wait(timeout=15), "PATCH thread didn't finish within 15s"
    assert patch_status.get("status") == 200, (
        f"PATCH status={patch_status.get('status')!r} body={patch_status.get('body')!r}"
    )
    assert saw_completed, "stream did not reach response.completed"

    combined = "".join(text_chunks)
    assert _SENTINEL in combined, f"{_SENTINEL!r} missing from {combined[:600]!r}"
