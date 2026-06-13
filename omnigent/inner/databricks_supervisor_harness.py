"""
``executor.type: supervisor`` harness wrap.

Thin module exposing :func:`create_app` — the entrypoint the shared
:mod:`omnigent.runtime.harnesses._runner` invokes after the parent
process resolves ``"supervisor"`` to this module via
:data:`omnigent.runtime.harnesses._HARNESS_MODULES`.

Internally, instantiates
:class:`omnigent.runtime.harnesses._executor_adapter.ExecutorAdapter`
around an :class:`omnigent.inner.databricks_supervisor_executor.SupervisorExecutor`
configured from env vars the parent process sets before spawning.

Mirrors the codex / claude-sdk wraps; the supervisor harness is the
first to wrap a wholly-AP-internal executor (no third-party SDK), so
the env-var contract is shorter (no CLI path, no permission mode, no
OS-tool config) — see the ``HARNESS_SUPERVISOR_*`` keys in
:mod:`omnigent.inner.databricks_supervisor_executor`.

Env vars read at startup:

- ``HARNESS_SUPERVISOR_MODEL``: gateway model identifier, e.g.
  ``"databricks-claude-sonnet-4-6"``. Required.
- ``HARNESS_SUPERVISOR_DATABRICKS_PROFILE``: optional profile name
  from ``~/.databrickscfg``. ``None`` lets the SDK use its own
  resolution chain.
- ``HARNESS_SUPERVISOR_TOOLS_JSON``: optional JSON-encoded list of
  the verbatim nested-shape tool entries from the spec.
- ``HARNESS_SUPERVISOR_CONNECTION_JSON``: optional JSON-encoded dict
  with explicit ``base_url`` + ``api_key`` overrides.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from omnigent.inner.databricks_supervisor_executor import SupervisorExecutor
from omnigent.inner.executor import Executor
from omnigent.runtime.harnesses._executor_adapter import ExecutorAdapter

_logger = logging.getLogger(__name__)


def _build_supervisor_executor() -> Executor:
    """
    Construct an inner :class:`SupervisorExecutor`.

    The wrapper itself is cheap to construct (no network); the
    underlying runtime executor is built lazily on the first
    :meth:`run_turn`, which is when credential resolution actually
    fires. So a misconfigured workspace surfaces on the first
    request rather than at FastAPI app boot.

    :returns: A configured :class:`SupervisorExecutor` instance
        ready to be wrapped by :class:`ExecutorAdapter`.
    """
    return SupervisorExecutor()


def create_app() -> FastAPI:
    """
    Build the supervisor harness's FastAPI app.

    Required entry point per the harness contract — the runner
    imports this module (resolved from
    :data:`omnigent.runtime.harnesses._HARNESS_MODULES`) and
    invokes ``create_app()`` to get the app it serves.

    :returns: The FastAPI app from :class:`ExecutorAdapter`'s
        :meth:`build` method, with all routes from the harness
        API subset wired up. The wrapped
        :class:`SupervisorExecutor` is constructed at adapter-
        build time; its underlying runtime executor is built
        lazily on the first turn.
    """
    adapter = ExecutorAdapter(executor_factory=_build_supervisor_executor)
    return adapter.build()
