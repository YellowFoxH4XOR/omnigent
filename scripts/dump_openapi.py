"""Generate and post-process the omnigent OpenAPI 3.2 document.

The omnigent server runs on FastAPI 0.135.x, which emits OpenAPI
3.1. OpenAPI 3.2 (released September 2025) introduced first-class
support for sequential media types — specifically, the
``itemSchema`` keyword for describing each item in a streaming
response on a ``text/event-stream`` content entry. We need 3.2's
``itemSchema`` so the SSE routes describe their per-event shape
correctly to consuming SDK / docs tooling.

This script:

1. Imports :func:`omnigent.server.app.create_app` and instantiates
   it against in-memory store stubs (no DB needed).
2. Calls ``app.openapi()`` to get the FastAPI-generated 3.1 dict.
3. Bumps the top-level ``openapi`` field to ``"3.2.0"``.
4. Materializes the :data:`ServerStreamEvent` discriminated union as
   a top-level entry under ``components.schemas`` so SSE responses
   can ``$ref`` it.
5. Rewrites the ``text/event-stream`` content entries on the SSE
   routes to use the OAS 3.2 ``itemSchema`` keyword in place of
   FastAPI's 3.1 ``schema`` keyword.
6. Writes the result to ``openapi.json`` at the repo root.

Run with no arguments to (re)generate the file. Pass ``--check``
in CI to verify the on-disk file is up to date — non-zero exit
means a developer changed the spec without regenerating.

Usage::

    python scripts/dump_openapi.py             # write openapi.json
    python scripts/dump_openapi.py --check     # exit 1 if drifted
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# DBOS's ``compute_app_version`` calls ``hashlib.md5()`` without
# ``usedforsecurity=False`` for a non-security content hash, which
# raises ``ValueError`` on FIPS-enabled hosts. Patch md5 here BEFORE
# any DBOS import so both this script and the drift test
# (``tests/server/test_openapi_drift.py``, which imports
# ``generate_spec`` from this module) work on FIPS hosts. The flag is
# the standard Python 3.9+ way to opt non-security md5 calls out of
# the FIPS gate; on non-FIPS hosts it's a harmless no-op.
_orig_md5 = hashlib.md5


def _fips_safe_md5(*args: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("usedforsecurity", False)
    return _orig_md5(*args, **kwargs)


hashlib.md5 = _fips_safe_md5

from pydantic import TypeAdapter  # noqa: E402 — must follow md5 patch

# ── Module-level constants (rule 34) ──────────────────────────────

# Output path. The spec lives at the repo root so external tooling
# (Stoplight, openapi-generator, redocly, …) can pick it up via a
# stable URL relative to the project. Pinned absolute via
# ``Path(__file__).resolve().parent.parent`` so the script works
# regardless of CWD.
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_OPENAPI_OUT: Path = _REPO_ROOT / "openapi.json"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# OpenAPI 3.2.0 release: 2025-09-23. We pin the patch version so
# the post-processed doc declares its target spec unambiguously.
_OPENAPI_VERSION: str = "3.2.0"

# Routes that emit Server-Sent Events. Each tuple is
# ``(path, method)`` keyed exactly as the OpenAPI ``paths`` map
# stores them. If the route inventory grows (e.g. a new SSE
# endpoint), add the entry here so post-processing rewrites it.
_SSE_ROUTES: list[tuple[str, str]] = [
    ("/v1/responses", "post"),
    ("/v1/sessions/{session_id}/stream", "get"),
]


def _build_app_with_stub_stores() -> Any:
    """
    Build a FastAPI app with stub stores sufficient for OpenAPI generation.

    ``app.openapi()`` walks the route table and Pydantic models — it
    does not call any store methods. We use the SQLite-backed
    implementations against an on-disk temporary database. The temp
    file is best-effort cleaned up by the caller's filesystem.

    :returns: A configured :class:`fastapi.FastAPI` app.
    """
    import tempfile

    from omnigent.runtime.agent_cache import AgentCache
    from omnigent.server.app import create_app
    from omnigent.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
    from omnigent.stores.artifact_store.local import LocalArtifactStore
    from omnigent.stores.comment_store.sqlalchemy_store import SqlAlchemyCommentStore
    from omnigent.stores.conversation_store.sqlalchemy_store import (
        SqlAlchemyConversationStore,
    )
    from omnigent.stores.file_store.sqlalchemy_store import SqlAlchemyFileStore
    from omnigent.stores.host_store import HostStore
    from omnigent.stores.policy_store.sqlalchemy_store import SqlAlchemyPolicyStore

    # On-disk SQLite (mkdtemp ensures uniqueness so concurrent
    # invocations don't collide).
    workdir = Path(tempfile.mkdtemp(prefix="oa-openapi-"))
    db_path = workdir / "spec.sqlite"
    db_uri = f"sqlite:///{db_path}"
    artifact_store = LocalArtifactStore(str(workdir / "artifacts"))
    return create_app(
        agent_store=SqlAlchemyAgentStore(db_uri),
        file_store=SqlAlchemyFileStore(db_uri),
        conversation_store=SqlAlchemyConversationStore(db_uri),
        comment_store=SqlAlchemyCommentStore(db_uri),
        artifact_store=artifact_store,
        agent_cache=AgentCache(
            artifact_store=artifact_store,
            cache_dir=workdir / "cache",
        ),
        # Pass stores so conditionally-mounted routes stay in the spec.
        host_store=HostStore(db_uri),
        policy_store=SqlAlchemyPolicyStore(db_uri),
    )


def _server_stream_event_schema() -> dict[str, Any]:
    """
    Return the JSON-Schema dict for the ``ServerStreamEvent`` union.

    Pydantic's ``TypeAdapter.json_schema(ref_template=...)`` emits a
    schema with internal ``$ref`` pointers in OpenAPI's expected
    ``#/components/schemas/<name>`` form. We then split out the
    union-root schema and inline the variant definitions into the
    components map so each per-event class appears as a top-level
    component schema.

    :returns: A dict with two keys:

        * ``"root"`` — the discriminated-union schema (the value
          assigned to ``components.schemas.ServerStreamEvent``).
        * ``"definitions"`` — the per-variant component schemas
          (merged into ``components.schemas``).
    """
    from omnigent.server.schemas import ServerStreamEvent

    adapter: TypeAdapter[ServerStreamEvent] = TypeAdapter(ServerStreamEvent)
    schema = adapter.json_schema(ref_template="#/components/schemas/{model}")
    # Pydantic returns ``{"oneOf": [...], "discriminator": {...},
    # "$defs": {...}}``. We hoist ``$defs`` to top-level component
    # schemas and keep the rest as the union root.
    definitions = schema.pop("$defs", {})
    return {"root": schema, "definitions": definitions}


def _rewrite_sse_route(
    paths: dict[str, Any],
    path: str,
    method: str,
) -> None:
    """
    Rewrite one SSE route's ``text/event-stream`` content for OAS 3.2.

    FastAPI emits ``content: {text/event-stream: {schema: <ref>}}``;
    OAS 3.2 uses ``itemSchema`` for sequential media types so each
    event in the stream is described as one item. We rename the key.

    No-op if the route doesn't exist (e.g. a renamed endpoint that
    fell off the inventory) — the caller's job is to keep
    :data:`_SSE_ROUTES` accurate.

    :param paths: The OpenAPI ``paths`` map; mutated in place.
    :param path: Route path, e.g. ``"/v1/responses"``.
    :param method: HTTP method (lowercase), e.g. ``"post"``.
    """
    op = paths.get(path, {}).get(method)
    if op is None:
        return
    ok_response = op.get("responses", {}).get("200", {})
    content = ok_response.get("content", {})
    sse_entry = content.get("text/event-stream")
    if sse_entry is None:
        return
    # Rename ``schema`` → ``itemSchema``. The value (a ``$ref``) is
    # untouched because the union schema applies to each event
    # equally — itemSchema is "validate this against every item
    # in the stream" per the 3.2 spec.
    if "schema" in sse_entry:
        sse_entry["itemSchema"] = sse_entry.pop("schema")


def generate_spec() -> dict[str, Any]:
    """
    Build, generate, and post-process the OpenAPI 3.2 spec.

    Encapsulates every step (app construction, generation, version
    bump, schema injection, SSE rewrite) so callers can compare the
    generated dict against ``openapi.json`` without writing to disk.

    :returns: The post-processed OpenAPI dict, ready to serialize.
    """
    app = _build_app_with_stub_stores()
    spec = app.openapi()
    # Bump the OpenAPI version literal — we don't change any
    # 3.1-only constructs because FastAPI's emitted shape is also
    # valid 3.2.x (3.2 is JSON-Schema-aligned and largely additive
    # over 3.1).
    spec["openapi"] = _OPENAPI_VERSION

    # Inject the ServerStreamEvent union + per-variant defs into
    # ``components.schemas`` so the SSE routes' $ref points resolve.
    components = spec.setdefault("components", {})
    schemas = components.setdefault("schemas", {})
    union = _server_stream_event_schema()
    schemas["ServerStreamEvent"] = union["root"]
    for name, definition in union["definitions"].items():
        # Don't clobber a same-named schema FastAPI already
        # synthesized — the union's per-variant defs include
        # ``ResponseObject`` (referenced from terminal events), and
        # FastAPI also emits one. Keep FastAPI's version; the
        # serialized shape is identical for our models.
        schemas.setdefault(name, definition)

    # Rewrite SSE routes' content entries to use ``itemSchema``.
    paths = spec.get("paths", {})
    for path, method in _SSE_ROUTES:
        _rewrite_sse_route(paths, path, method)

    return spec  # type: ignore[no-any-return]


def main() -> int:
    """
    CLI entry point.

    With no arguments, regenerates ``openapi.json``. With
    ``--check``, compares the generated spec to the on-disk file
    and exits 1 if they differ.

    :returns: 0 on success / no drift, 1 on drift in ``--check``
        mode.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "CI mode — exit 1 if the on-disk openapi.json differs from "
            "the generated spec. Use to fail PRs that change the spec "
            "without regenerating."
        ),
    )
    args = parser.parse_args()

    spec = generate_spec()
    serialized = json.dumps(spec, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not _OPENAPI_OUT.exists():
            sys.stderr.write(
                f"openapi.json not found at {_OPENAPI_OUT}; "
                "run `python scripts/dump_openapi.py` to generate it.\n",
            )
            return 1
        existing = _OPENAPI_OUT.read_text()
        if existing != serialized:
            sys.stderr.write(
                "openapi.json is out of sync with the generated spec.\n"
                "Run `python scripts/dump_openapi.py` to regenerate.\n",
            )
            return 1
        sys.stdout.write("openapi.json is up to date.\n")
        return 0

    _OPENAPI_OUT.write_text(serialized)
    sys.stdout.write(f"Wrote {_OPENAPI_OUT}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
