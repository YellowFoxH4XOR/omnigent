"""E2E for the non-OpenAI ``web_search`` async-dispatch path.

Stubs Perplexity via the ``OMNIGENT_PERPLEXITY_BASE_URL`` override
so no real key is needed. Asserts: dispatch_async ran (function_call
present), and the stub's sentinel reaches the LLM via the
async_work_complete drain (text appears in some output item).
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.e2e.conftest import poll_until_terminal, upload_agent

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WEB_SEARCH_TEST_DIR = _REPO_ROOT / "tests" / "resources" / "agents" / "web-search-test"

_STUB_ANSWER = "Stubbed search result: the framework's regression fixture is wired correctly."
_FAKE_BEARER = "test-bearer-not-validated"


class _FakePerplexityHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        body = json.dumps(
            {
                "choices": [{"message": {"content": _STUB_ANSWER}}],
                "citations": ["https://example.invalid/regression-marker"],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# Module-level startup: live_server reads os.environ at fixture-
# setup time and pytest may resolve fixtures in any order, so the
# env var has to be set before any session fixture spawns.
_FAKE_SERVER = ThreadingHTTPServer(("127.0.0.1", 0), _FakePerplexityHandler)
threading.Thread(target=_FAKE_SERVER.serve_forever, daemon=True).start()
os.environ["OMNIGENT_PERPLEXITY_BASE_URL"] = (
    f"http://127.0.0.1:{_FAKE_SERVER.server_address[1]}/chat/completions"
)


def _materialize_with_resolved_env(src: Path, dst: Path) -> Path:
    """
    Copy the fixture and resolve its ``${VAR}`` references client-side.

    The server no longer expands ``${VAR}`` in uploaded (session-scoped)
    bundles against its own environment — that is the exfiltration
    vector this fixture used to rely on. Real clients resolve
    env vars before upload (``omnigent.cli._resolve_bundle_env_vars``), so
    this fixture mirrors that: the stubbed Perplexity key is a throwaway
    bearer (the fake server never validates it) and the OpenAI key is read
    from the client env, falling back to the same throwaway when running
    against the Databricks gateway (profile auth supersedes the api_key).

    :param src: The fixture agent directory containing ``${VAR}`` refs.
    :param dst: Destination directory for the resolved copy.
    :returns: *dst*, populated with the env-resolved ``config.yaml``.
    """
    shutil.copytree(src, dst)
    cfg = dst / "config.yaml"
    resolved = (
        cfg.read_text()
        .replace("${PERPLEXITY_API_KEY}", _FAKE_BEARER)
        .replace("${OPENAI_API_KEY}", os.environ.get("OPENAI_API_KEY", _FAKE_BEARER))
    )
    cfg.write_text(resolved)
    return dst


@pytest.fixture(scope="session")
def web_search_test_agent(
    http_client: httpx.Client,
    databricks_workspace_host: str | None,
    tmp_path_factory: pytest.TempPathFactory,
) -> str:
    staging = tmp_path_factory.mktemp("web-search-test-bundle")
    prepared = _materialize_with_resolved_env(_WEB_SEARCH_TEST_DIR, staging / "agent")
    return upload_agent(
        http_client,
        prepared,
        rewrite_model_for_databricks=databricks_workspace_host is not None,
    )


def _function_calls(body: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [
        item
        for item in body.get("output", [])
        if item.get("type") == "function_call" and item.get("name") == name
    ]


def _all_text(body: dict[str, Any]) -> str:
    """Concat every text block — assistant output_text AND drained user input_text."""
    parts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for block in item.get("content", []):
            text = block.get("text")
            if text:
                parts.append(text)
    return "\n".join(parts)


def test_web_search_async_dispatch_completes_on_non_openai_model(
    http_client: httpx.Client,
    web_search_test_agent: str,
) -> None:
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": web_search_test_agent,
            "input": (
                "Search the web for today's top news headline. Quote "
                "the search result verbatim. Issue exactly one "
                "web_search call."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]

    body = poll_until_terminal(http_client, response_id, timeout=300)

    assert body["status"] == "completed", (
        f"status={body['status']!r}, output={body.get('output', [])}"
    )
    assert _function_calls(body, "web_search"), (
        f"no web_search function_call — dispatch_async didn't run. "
        f"output types: {[i.get('type') for i in body.get('output', [])]}"
    )
    # Stub's sentinel surfaces via the async_work_complete drain as
    # a role=user input_text, NOT in the assistant's final text.
    assert _STUB_ANSWER in _all_text(body), (
        f"stub sentinel missing — drain did not deliver the result. "
        f"text: {_all_text(body)[:500]!r}"
    )
