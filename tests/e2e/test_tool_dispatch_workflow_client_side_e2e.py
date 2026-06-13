"""
E2E: request-supplied client-side tools route through
``tool_dispatch_workflow``'s ``_is_parent_client_side_tool``
branch instead of falling through to ``_execute_tool``'s
``unknown server-side tool`` error envelope.

Sabotage check: drop the new ``elif`` in
``omnigent/runtime/tool_dispatch_workflow.py`` and the final
``_SENTINEL in combined`` assertion fails — the LLM gets the
error envelope instead of the REPL output.

Invoke with::

    pytest tests/e2e/test_tool_dispatch_workflow_client_side_e2e.py \\
        --llm-api-key "$(cat /tmp/mykey)" -v
"""

from __future__ import annotations

from typing import Any

import httpx

from tests.e2e.conftest import poll_for_pending_tool_calls, poll_until_terminal

# Cryptographically unguessable so a hallucinating LLM cannot
# produce it without actually receiving the tool output.
_SENTINEL = "DBX-PR50-K9X7Q3M2W1R5T8-CLIENT-TOOL-CANARY"


_LOOKUP_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_secret_token",
            "description": (
                "Look up the unguessable secret token for today. The "
                "user does not know it; you MUST call this tool to "
                "retrieve the token. Return the token verbatim in "
                "your response."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


def test_harness_routes_client_side_tool_through_dispatch_workflow(
    http_client: httpx.Client,
    claude_coder_agent: str,
) -> None:
    """
    Claude SDK harness invokes a request-supplied client-side
    tool; the REPL-PATCHed sentinel must reach the LLM verbatim.

    :param http_client: HTTP client pointed at the live e2e server.
    :param claude_coder_agent: Uploaded claude-coder agent name.
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": claude_coder_agent,
            "input": (
                "What is today's secret token? You MUST call the "
                "lookup_secret_token tool to retrieve it. After you "
                "receive the token, your response MUST include the "
                "token verbatim — do not paraphrase, summarize, or "
                "wrap it in any explanation. The token is opaque "
                "and unguessable; do not invent one."
            ),
            "background": True,
            "tools": _LOOKUP_TOOLS,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]

    pending = poll_for_pending_tool_calls(http_client, response_id, timeout=120)
    lookup_calls = [fc for fc in pending if fc.get("name") == "lookup_secret_token"]
    assert lookup_calls, (
        f"Expected lookup_secret_token in pending; got "
        f"{[fc.get('name') for fc in pending]}. Empty means the LLM "
        f"never invoked the registered tool — upstream of the "
        f"dispatch fix under test."
    )

    patch_resp = http_client.patch(
        f"/v1/responses/{response_id}",
        json={
            "tool_results": [
                {
                    "call_id": lookup_calls[0]["call_id"],
                    "output": _SENTINEL,
                },
            ],
        },
    )
    assert patch_resp.status_code == 200, (
        f"PATCH /v1/responses/{response_id} returned "
        f"{patch_resp.status_code}: {patch_resp.text[:300]}"
    )

    body = poll_until_terminal(http_client, response_id, timeout=180)
    assert body["status"] == "completed", (
        f"Expected completed, got {body['status']}. Error: {body.get('error')}"
    )

    output_texts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") != "message" or item.get("role") != "assistant":
            continue
        for block in item.get("content", []):
            text = block.get("text")
            if text:
                output_texts.append(text)
    combined = "\n".join(output_texts)

    # Verbatim sentinel proves the REPL PATCH round-tripped through
    # the new dispatch-workflow branch back into the LLM's context.
    assert _SENTINEL in combined, (
        f"Expected sentinel {_SENTINEL!r} verbatim in agent's "
        f"final response, got: {combined[:600]!r}. Absence likely "
        f"means the new client-side branch in tool_dispatch_workflow "
        f"did not fire — the dispatch returned an "
        f"``unknown server-side tool`` envelope instead of parking "
        f"on the pending_tool_calls row."
    )
