"""E2E test: zero-downtime agent update.

Verifies that an in-flight request on the old agent version completes
successfully, and a new request after the update uses the new version
(observable via changed instructions that affect the response content).

Usage::

    pytest tests/e2e/test_agent_update.py \
        --llm-api-key $LLM_API_KEY -v
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from tests.e2e.conftest import (
    _rewrite_yaml_models,
    create_runner_bound_session,
    poll_session_until_terminal,
    send_user_message_to_session,
)
from tests.e2e.helpers import final_assistant_text

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Use compaction-test — a minimal agent (gpt-5.4 via openai-agents, no
# tools, no sub-agents) so the test runs fast and cheap. An OpenAI-family
# model is required: the openai-agents harness returns empty output when
# pointed at a Claude model on the Databricks gateway. Archer-style
# research agents make dozens of tool calls per request, which blows test
# timeouts without exercising anything specific to the update path.
_TEST_AGENT_DIR = _REPO_ROOT / "tests" / "resources" / "agents" / "compaction-test"
_TEST_AGENT_NAME = "compaction-test"

# Marker phrase injected into v2 instructions so we can verify
# the v2 response was produced by the updated spec.
_V2_MARKER = "ZEBRAFINCH"


def _upload_agent_with_id(
    client: httpx.Client,
    agent_dir: Path,
    databricks_workspace_host: str | None = None,
) -> dict[str, Any]:
    """
    Upload an agent bundle via multipart ``POST /v1/sessions`` and
    return the agent metadata (including ``id``) from the
    session-scoped agent endpoint.

    :param client: HTTP client pointed at the server.
    :param agent_dir: Path to the agent directory.
    :param databricks_workspace_host: Workspace host when --profile
        is set; triggers model name rewriting.
    :returns: The agent response JSON with ``id``, ``name``,
        ``version``, etc.
    """
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        with tarfile.open(tmp.name, "w:gz") as tar:
            for item in sorted(agent_dir.rglob("*")):
                if not item.is_file():
                    continue
                arcname = str(item.relative_to(agent_dir))
                if item.name == "config.yaml" and databricks_workspace_host is not None:
                    cfg = yaml.safe_load(item.read_text())
                    _rewrite_yaml_models(cfg)
                    data = yaml.dump(cfg).encode()
                    info = tarfile.TarInfo(name=arcname)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
                else:
                    tar.add(str(item), arcname=arcname)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            metadata = json.dumps({})
            resp = client.post(
                "/v1/sessions",
                data={"metadata": metadata},
                files={
                    "bundle": (
                        "agent.tar.gz",
                        f,
                        "application/gzip",
                    ),
                },
            )
        resp.raise_for_status()
        session_id = resp.json()["session_id"]
        agent_resp = client.get(f"/v1/sessions/{session_id}/agent")
        agent_resp.raise_for_status()
        agent_data: dict[str, Any] = agent_resp.json()
        # Stash the session_id so _update_agent can use the
        # session-scoped PUT endpoint.
        agent_data["_session_id"] = session_id
        return agent_data
    finally:
        os.unlink(tmp_path)


def _build_updated_bundle(
    agent_dir: Path,
    config_overrides: dict[str, Any],
    databricks_workspace_host: str | None = None,
) -> bytes:
    """
    Build a tarball from an agent directory with config.yaml
    fields overridden.

    Reads the original config.yaml, merges the overrides, and
    writes the modified config into the tarball. All other files
    are included as-is.

    :param agent_dir: Path to the original agent directory.
    :param config_overrides: Dict of fields to merge into
        config.yaml, e.g. ``{"description": "v2"}``.
    :returns: Raw bytes of the ``.tar.gz`` bundle.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for item in Path(agent_dir).rglob("*"):
            if not item.is_file():
                continue
            arcname = str(item.relative_to(agent_dir))
            if item.name == "config.yaml" and item.parent == agent_dir:
                # Override the root config.yaml
                config = yaml.safe_load(item.read_text())
                config.update(config_overrides)
                if databricks_workspace_host is not None:
                    _rewrite_yaml_models(config)
                data = yaml.dump(config).encode()
                info = tarfile.TarInfo(name=arcname)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            else:
                tar.add(str(item), arcname=arcname)
    return buf.getvalue()


def _update_agent(
    client: httpx.Client,
    session_id: str,
    bundle_bytes: bytes,
) -> dict[str, Any]:
    """
    PUT a new bundle to update an existing agent via the
    session-scoped agent endpoint.

    :param client: HTTP client pointed at the server.
    :param session_id: The session ID whose agent to update.
    :param bundle_bytes: Raw bytes of the new ``.tar.gz`` bundle.
    :returns: The updated agent response JSON.
    """
    resp = client.put(
        f"/v1/sessions/{session_id}/agent",
        files={
            "bundle": (
                "agent.tar.gz",
                bundle_bytes,
                "application/gzip",
            ),
        },
    )
    resp.raise_for_status()
    return resp.json()


def test_update_agent_zero_downtime(
    http_client: httpx.Client,
    live_runner_id: str,
    databricks_workspace_host: str | None,
) -> None:
    """
    Verifies that the update endpoint doesn't disrupt in-flight
    requests and that new requests use the updated spec.

    **What this test proves:**
    - The PUT endpoint succeeds while a background request is
      running (the server doesn't crash or deadlock).
    - A request created after the update uses the new spec
      (verified via a marker phrase in instructions).
    - The in-flight request completes without error.

    **What this test does NOT prove (inherent E2E limitation):**
    - It does not guarantee the v1 request was mid-LLM-call when
      the update happened. The gateway often finishes the turn in
      under 2s, so the PUT may land after v1 already completed.
      Either way the update must not error and v2 must pick up the
      new spec -- that is what we assert. True mid-execution
      concurrent update testing requires the mock LLM's blocking
      mechanism, which is not available with a real LLM.
    - The marker assertion depends on the LLM following the
      injected instruction. If the LLM ignores it, the test
      gives a false negative, not a false positive.

    Steps:
    1. Upload compaction-test agent (version 1).
    2. Send a verbose request (best-effort in-flight window).
    3. PUT a new bundle with modified instructions containing a
       marker phrase (version 2).
    4. Send a second request on a fresh v2 session.
    5. Both requests complete successfully.
    6. V1 response does NOT contain the marker.
    7. V2 response DOES contain the marker.
    8. Agent metadata shows version=2 and updated_at is set.
    """
    if databricks_workspace_host is None:
        pytest.skip(
            "agent-update e2e requires --profile: the openai-agents harness "
            "routes through the Databricks serving endpoint"
        )

    # Step 1: Upload compaction-test (v1) and bind the runner.
    # The multipart POST creates a session but doesn't bind a runner —
    # PATCH is required before sending any events.
    created = _upload_agent_with_id(
        http_client,
        _TEST_AGENT_DIR,
        databricks_workspace_host=databricks_workspace_host,
    )
    session_id = created["_session_id"]
    assert created["version"] == 1
    http_client.patch(
        f"/v1/sessions/{session_id}",
        json={"runner_id": live_runner_id},
    ).raise_for_status()

    # Step 2: Start a turn with verbose output so the session stays
    # running long enough to PUT the update.
    response_id_1 = send_user_message_to_session(
        http_client,
        session_id=session_id,
        content=(
            "Generate verbose output: write a detailed, multi-"
            "paragraph explanation of how photosynthesis works, "
            "covering the light-dependent reactions, the Calvin "
            "cycle, and the role of chlorophyll. At least 5 "
            "paragraphs."
        ),
    )

    # Best-effort: give the turn a moment to start. We do NOT skip if
    # it already finished -- the PUT-then-new-request path below is the
    # real regression guard and runs regardless. Catching the turn
    # mid-flight (status == "running") additionally exercises the
    # zero-downtime path, but the gateway often finishes the turn in
    # under 2s, so we treat that as a bonus, not a precondition.
    time.sleep(2)

    # Step 3: Update agent to v2 with marker in instructions. The v1
    # turn either is still running (in-flight update) or already
    # completed under the old spec; both leave v1 free of the marker.
    v2_bundle = _build_updated_bundle(
        _TEST_AGENT_DIR,
        {
            "description": "Updated compaction-test v2 for e2e test",
            "instructions": (
                f"You MUST include the word '{_V2_MARKER}' "
                f"somewhere in every response you give. This is "
                f"a mandatory requirement."
            ),
        },
        databricks_workspace_host=databricks_workspace_host,
    )
    updated = _update_agent(http_client, session_id, v2_bundle)
    assert updated["version"] == 2

    # Step 4: New session on v2 — ask something simple.
    session_id_2 = create_runner_bound_session(
        http_client,
        agent_name=_TEST_AGENT_NAME,
        runner_id=live_runner_id,
    )
    response_id_2 = send_user_message_to_session(
        http_client,
        session_id=session_id_2,
        content="What is 2+2? Answer briefly.",
    )

    # Step 5: Poll both to terminal state.
    body1 = poll_session_until_terminal(
        http_client, session_id=session_id, response_id=response_id_1, timeout=60
    )
    body2 = poll_session_until_terminal(
        http_client, session_id=session_id_2, response_id=response_id_2, timeout=60
    )

    # Both requests completed — neither was disrupted
    assert body1["status"] == "completed", (
        f"V1 request failed with status {body1['status']!r}. "
        f"The update should not disrupt in-flight requests. "
        f"Output: {body1.get('output', [])}"
    )
    assert body2["status"] == "completed", (
        f"V2 request failed with status {body2['status']!r}. Output: {body2.get('output', [])}"
    )

    # Step 6: V1 response should NOT contain the marker —
    # it was served by the old spec before the update.
    v1_text = final_assistant_text(body1)
    assert _V2_MARKER not in v1_text, (
        f"V1 response unexpectedly contains the v2 marker "
        f"'{_V2_MARKER}'. This means the in-flight request "
        f"picked up the new spec instead of using the one it "
        f"loaded at start. First 500 chars: {v1_text[:500]}"
    )

    # Step 7: V2 response MUST contain the marker — it was
    # served by the updated spec with the injected instruction.
    # NOTE: This assertion depends on the LLM following the
    # mandatory instruction. A false negative (test fails but
    # spec was loaded correctly) is possible if the LLM ignores
    # the instruction. A false positive is NOT possible — the
    # marker only exists in v2's instructions.
    v2_text = final_assistant_text(body2)
    assert _V2_MARKER in v2_text, (
        f"V2 response does NOT contain the marker "
        f"'{_V2_MARKER}'. This means the new request did not "
        f"use the updated spec's instructions. The cache swap "
        f"may have failed. First 500 chars: {v2_text[:500]}"
    )

    # Step 8: Agent metadata reflects the update
    agent_resp = http_client.get(f"/v1/sessions/{session_id}/agent")
    agent_resp.raise_for_status()
    agent = agent_resp.json()
    assert agent["version"] == 2
    assert agent["updated_at"] is not None
