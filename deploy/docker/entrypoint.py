"""OSS Docker entrypoint for the Omnigent server.

Mirrors ``deploy/databricks/src/app.py`` (the Databricks Apps entrypoint) but
configured for a plain Postgres database and a local-filesystem
artifact store. Intended to run inside the image built by
``deploy/docker/Dockerfile``.

Execution mode: external runners only. The server accepts runner
WebSocket connections at ``/v1/runner/tunnel`` and never spawns
harness subprocesses on its own. Users run ``omnigent run … --server
<url>`` on their own machine; that runner dials in.

Configuration is via environment variables:

  DATABASE_URL          Required. SQLAlchemy URL. Both PaaS-style URLs
                        (``postgresql://user:pw@host:5432/db``,
                        ``postgres://...``) and the explicit psycopg3
                        form (``postgresql+psycopg://...``) are accepted;
                        the prefix is normalized automatically.
  ARTIFACT_DIR          Directory for the local artifact store.
                        Defaults to ``/data/artifacts`` (the volume
                        mount point used by docker-compose).
  HOST, PORT            Bind address. Default ``0.0.0.0:8000``.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stderr, force=True)
logger = logging.getLogger("omnigent-docker")

# Defaults live as module-level constants — the Dockerfile and
# docker-compose.yaml both also set these, so the values here just
# document the contract for anyone running entrypoint.py outside the
# image. Source: deploy/docker/Dockerfile (ENV block) and
# deploy/docker/docker-compose.yaml.
_DEFAULT_HOST = "0.0.0.0"
# Pinned to 8000 by design (container/platform convention) — deliberately
# decoupled from the CLI's local-server default (6767 in host/local_server.py).
_DEFAULT_PORT = "8000"
_DEFAULT_ARTIFACT_DIR = "/data/artifacts"

try:
    import uvicorn

    from omnigent.db.utils import normalize_database_url
    from omnigent.runner.transports.ws_tunnel.limits import (
        RUNNER_TUNNEL_MAX_MESSAGE_BYTES,
    )
    from omnigent.server.app import create_app
    from omnigent.server.paas_env import detect_base_url, resolve_bind_host
    from omnigent.server.server_config import config_str_list, load_server_config

    # ── Configuration ────────────────────────────────────────
    # Non-secret settings come from a YAML config file (default
    # <data_dir>/config.yaml, e.g. /data/config.yaml on the volume, or
    # OMNIGENT_CONFIG) — the same experience a laptop gets from
    # `omnigent server -c`. Secrets stay in the environment:
    # DATABASE_URL (carries the password) and the cookie / OIDC secrets.
    cfg = load_server_config()

    # DATABASE_URL is env-first (compose/PaaS inject it; it's a secret),
    # with `database_uri:` in the config as a fallback for self-managed DBs.
    DATABASE_URL = os.environ.get("DATABASE_URL") or cfg.get("database_uri")
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is required (env), or set `database_uri:` in the server config. "
            "Accepted forms: "
            "'postgresql+psycopg://user:pw@host:5432/omnigent' (explicit psycopg3), "
            "or the 'postgres://' / 'postgresql://' URLs emitted by Railway, Render, etc."
        )
    # Normalize PaaS-style URLs (postgres:// or postgresql://) to the
    # psycopg3 dialect specifier that SQLAlchemy requires.
    DATABASE_URL = normalize_database_url(DATABASE_URL)

    # App settings are config-first, env fallback, then the built-in default.
    ARTIFACT_DIR = Path(
        cfg.get("artifact_location") or os.environ.get("ARTIFACT_DIR") or _DEFAULT_ARTIFACT_DIR
    )
    # resolve_bind_host strips the bracketed IPv6 form some platforms inject
    # ("[::]") and coerces Railway's IPv6 wildcard to IPv4 (its edge is v4-only).
    HOST = resolve_bind_host(
        cfg.get("host") or os.environ.get("HOST"), os.environ, default=_DEFAULT_HOST
    )
    PORT = int(cfg.get("port") or os.environ.get("PORT") or _DEFAULT_PORT)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Config: HOST=%s PORT=%d DB=%s ARTIFACTS=%s",
        HOST,
        PORT,
        DATABASE_URL.split("@", 1)[-1] if "@" in DATABASE_URL else DATABASE_URL,
        ARTIFACT_DIR,
    )

    # Containerized / remote deploys default to authenticated auth.
    # The framework-wide default (a bare local `omnigent server`) is
    # single-user header mode with no login, but a Docker / HF / PaaS
    # instance is typically network-exposed, so we opt it into the
    # multi-user login flow here — accounts by default, or OIDC if the
    # operator supplied OMNIGENT_OIDC_* config. An operator can still
    # force header/oidc/accounts via OMNIGENT_AUTH_PROVIDER, or turn
    # auth off with OMNIGENT_AUTH_ENABLED=0.
    os.environ.setdefault("OMNIGENT_AUTH_ENABLED", "1")

    # Kill-switch ergonomics: OMNIGENT_AUTH_ENABLED=0 means "no login,
    # single-user local container" (the documented local-dev posture).
    # Header mode now fails closed on a missing X-Forwarded-Email,
    # so without this marker a no-auth container would
    # 401 every request — nothing injects the header. Only the implicit
    # kill-switch path gets the marker: an EXPLICIT
    # OMNIGENT_AUTH_PROVIDER=header deploy declared a header-injecting
    # proxy and must stay strict.
    from omnigent.server.auth import env_var_is_truthy

    # Compose passes OMNIGENT_AUTH_PROVIDER as "" when unset
    # ("${VAR:-}"): empty and missing both mean "not explicitly pinned".
    _raw_auth_provider = os.environ.get("OMNIGENT_AUTH_PROVIDER")
    _auth_provider_explicit = bool(_raw_auth_provider and _raw_auth_provider.strip())
    if not _auth_provider_explicit and not env_var_is_truthy("OMNIGENT_AUTH_ENABLED"):
        os.environ.setdefault("OMNIGENT_LOCAL_SINGLE_USER", "1")

    # Accounts mode ergonomics: when the operator hasn't set them, supply the
    # two required vars (cookie secret + base URL) so a 1-click / `docker
    # compose up` deploy works with zero config. Gate on the *resolved*
    # selection so an explicit header/oidc deploy (or AUTH_ENABLED=0)
    # doesn't mint accounts secrets it never reads.
    from omnigent.server.auth import resolve_auth_source

    if resolve_auth_source() == "accounts":
        from omnigent.server.accounts_secret import load_or_generate_cookie_secret

        # Empty-check, not setdefault: compose passes these as empty strings
        # ("${VAR:-}"), which setdefault would leave in place — defeating the default.
        if not os.environ.get("OMNIGENT_ACCOUNTS_COOKIE_SECRET"):
            os.environ["OMNIGENT_ACCOUNTS_COOKIE_SECRET"] = load_or_generate_cookie_secret(
                ARTIFACT_DIR
            )
        if not os.environ.get("OMNIGENT_ACCOUNTS_BASE_URL"):
            # Auto-detect the public URL from the PaaS env (Render / Railway /
            # Fly / HF Spaces) so a 1-click deploy needs zero manual config;
            # falls back to the bind address for local / Docker / EC2.
            os.environ["OMNIGENT_ACCOUNTS_BASE_URL"] = detect_base_url(
                os.environ, host=HOST, port=PORT
            )

    # ── Migrations ───────────────────────────────────────────
    # Alembic upgrade runs before the stores boot — the SQLAlchemy
    # stores refuse to start on a stale schema.

    import sqlalchemy

    from omnigent.db.utils import _run_migrations as _run_alembic_upgrade

    _migration_engine = sqlalchemy.create_engine(DATABASE_URL)
    try:
        _run_alembic_upgrade(_migration_engine, DATABASE_URL)
    finally:
        _migration_engine.dispose()

    # ── Stores ───────────────────────────────────────────────

    from omnigent.runtime import init as init_runtime
    from omnigent.runtime import telemetry
    from omnigent.runtime.agent_cache import AgentCache
    from omnigent.runtime.caps import RuntimeCaps
    from omnigent.server.managed_hosts import parse_sandbox_config
    from omnigent.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
    from omnigent.stores.artifact_store.local import LocalArtifactStore
    from omnigent.stores.comment_store.sqlalchemy_store import (
        SqlAlchemyCommentStore,
    )
    from omnigent.stores.conversation_store.sqlalchemy_store import (
        SqlAlchemyConversationStore,
    )
    from omnigent.stores.file_store.sqlalchemy_store import SqlAlchemyFileStore
    from omnigent.stores.host_store import HostStore
    from omnigent.stores.permission_store.sqlalchemy_store import (
        SqlAlchemyPermissionStore,
    )

    telemetry.init()

    agent_store = SqlAlchemyAgentStore(DATABASE_URL)
    file_store = SqlAlchemyFileStore(DATABASE_URL)
    conversation_store = SqlAlchemyConversationStore(DATABASE_URL)
    comment_store = SqlAlchemyCommentStore(DATABASE_URL)
    permission_store = SqlAlchemyPermissionStore(DATABASE_URL)
    host_store = HostStore(DATABASE_URL)
    # Fail startup loud on a malformed `sandbox:` section (an operator
    # typo should not surface as a runtime 502 on the first managed
    # session); the startup catch-all below logs it.
    sandbox_config = parse_sandbox_config(cfg.get("sandbox"))
    artifact_store = LocalArtifactStore(str(ARTIFACT_DIR))

    agent_cache = AgentCache(
        artifact_store=artifact_store,
        cache_dir=ARTIFACT_DIR / ".cache",
    )

    init_runtime(
        agent_cache=agent_cache,
        caps=RuntimeCaps(),
        agent_store=agent_store,
        file_store=file_store,
        conversation_store=conversation_store,
        artifact_store=artifact_store,
        comment_store=comment_store,
    )

    # Build the auth provider from the live env (header/oidc/accounts).
    # Accounts mode also needs an AccountStore explicitly wired — the
    # entrypoint constructs it here rather than letting create_app do
    # so internally, so this same code path can opt out by passing
    # None for non-accounts deploys (matching the structural
    # contract used on the hosted product).
    from omnigent.server.auth import UnifiedAuthProvider as _UAP
    from omnigent.server.auth import create_auth_provider

    auth_provider = create_auth_provider()
    account_store = None
    if isinstance(auth_provider, _UAP) and auth_provider._source == "accounts":
        from omnigent.server.accounts_store import SqlAlchemyAccountStore

        account_store = SqlAlchemyAccountStore(DATABASE_URL)

    app = create_app(
        agent_store=agent_store,
        file_store=file_store,
        conversation_store=conversation_store,
        artifact_store=artifact_store,
        agent_cache=agent_cache,
        comment_store=comment_store,
        permission_store=permission_store,
        host_store=host_store,
        auth_provider=auth_provider,
        account_store=account_store,
        # Non-secret auth settings from the config file (admins are the
        # canonical, declarative roster; allowed_domains gates OIDC). Both
        # union with their runtime-editable files under <data_dir>.
        admins=config_str_list(cfg.get("admins")),
        allowed_domains=config_str_list(cfg.get("allowed_domains")),
        sandbox_config=sandbox_config,
    )

    if __name__ == "__main__":
        logger.info("Starting omnigent server on %s:%d", HOST, PORT)
        uvicorn.run(
            app,
            host=HOST,
            port=PORT,
            ws_max_size=RUNNER_TUNNEL_MAX_MESSAGE_BYTES,
        )

except Exception:  # noqa: BLE001 — startup catch-all so failures land in logs
    logger.error("FATAL: omnigent server failed to start:\n%s", traceback.format_exc())
    # Keep the process alive briefly so the container log capture has time
    # to flush before the orchestrator restarts us.
    import time

    time.sleep(30)
    sys.exit(1)
