# Databricks supervisor — example bundle

A minimal agent that delegates the entire turn (model + tools) to the
[Databricks Agent Bricks Supervisor API](https://docs.databricks.com/aws/en/generative-ai/agent-bricks/supervisor-api).

For the integration overview — why the integration is partial today,
where the code lives, how to wire up a new tool, and which tool
types have portable MCP equivalents — see
[`designs/DATABRICKS_SUPERVISOR_API_INTEGRATION.md`](../../designs/DATABRICKS_SUPERVISOR_API_INTEGRATION.md).

## Prerequisites

- A workspace where the Supervisor API preview is enabled.
- A Databricks profile in `~/.databrickscfg` matching
  `executor.profile` in `config.yaml` (currently `oss`).
- A fresh OAuth token: `databricks auth login`.

## Run it — one terminal

```bash
uv run omnigent examples/databricks_supervisor
```

Spawns a local omnigent server, registers the bundle, streams
the response. Server logs land in `~/.omnigent/logs/server/server-*.log`
and are retained after the REPL exits for debugging.

## Run it — two terminals (live server logs)

**Terminal 1 — server:**

```bash
uv run omnigent server --agent examples/databricks_supervisor
```

**Terminal 2 — REPL:**

```bash
uv run omnigent run --server http://localhost:6767
```

Server stdout/stderr stream to Terminal 1 in real time. Re-run the
REPL command as many times as you want against the same server.
State persists in `./omnigent.db`; delete it for a clean slate.
