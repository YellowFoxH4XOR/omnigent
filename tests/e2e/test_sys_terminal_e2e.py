"""E2E tests for the AP-side ``sys_terminal_*`` tool family.

Per ``designs/OMNIGENT_TERMINAL_BRIDGE.md`` §8.3 — these tests
exercise the full Omnigent integration path: omnigent YAML
declares ``terminals:``, the compat translator threads it onto
``AgentSpec.terminals``, the AP-side ``ToolManager`` registers
the ``sys_terminal_*`` family, the LLM invokes them, the
:class:`omnigent.terminals.TerminalRegistry` spawns real
tmux sessions, and (per §4.4 corrected) cleanup fires only at
conversation deletion / Omnigent shutdown — NOT at workflow exit.

Skipped if tmux isn't installed on the host running the test.

Usage::

    pytest tests/e2e/test_sys_terminal_e2e.py \\
        --llm-api-key $LLM_API_KEY -v
"""

from __future__ import annotations

import json
import shutil

import httpx
import pytest

from tests.e2e.conftest import poll_until_terminal

pytestmark = pytest.mark.skipif(
    shutil.which("tmux") is None,
    reason="tmux not installed; sys_terminal_* e2e tests need tmux on PATH",
)


def _get_function_call_outputs(
    client: httpx.Client,
    conversation_id: str,
    tool_name: str,
) -> list[str]:
    """
    Return raw outputs of every ``tool_name`` call in conversation order.

    Walks ``function_call`` and ``function_call_output`` items in the
    conversation. Used so assertions land on deterministic tool
    output strings, not on flaky LLM prose summaries.

    :param client: HTTP client.
    :param conversation_id: Conversation to inspect.
    :param tool_name: Only outputs of calls to this tool are returned.
    :returns: Ordered list of raw output strings.
    """
    resp = client.get(f"/v1/sessions/{conversation_id}/items?limit=200")
    resp.raise_for_status()
    items = resp.json()["data"]
    calls_by_id: dict[str, dict] = {}
    for item in items:
        if item.get("type") == "function_call" and item.get("name") == tool_name:
            calls_by_id[item["call_id"]] = item
    outputs: list[str] = []
    for item in items:
        if item.get("type") == "function_call_output":
            cid = item.get("call_id")
            if cid in calls_by_id:
                outputs.append(str(item.get("output", "")))
    return outputs


def test_sys_terminal_basic_round_trip_e2e(
    live_server: str,
    sys_terminal_test_agent: str,
    http_client: httpx.Client,
) -> None:
    """
    Real LLM drives the full ``sys_terminal_*`` round trip
    against a real tmux. Asserts on the raw tool outputs (not
    prose) so flaky LLM wording can't fail the test.

    What this verifies:
      1. The compat translator threaded ``terminals:`` from
         omnigent YAML through to ``AgentSpec.terminals``.
      2. The AP-side ToolManager registered the
         ``sys_terminal_*`` family from
         ``ToolManager._register_terminal_tools``.
      3. The LLM successfully invoked launch + send + read.
      4. The :class:`TerminalRegistry` spawned a real tmux
         session; ``send`` reached it; ``read`` saw the marker.

    What breaks if this fails (top suspects):
      - ``AgentSpec.terminals=None`` after translation → tools
        not registered → "tool not available" mid-conversation.
      - Workflow path differs from test ToolManager registration
        path → tools register in tests but not at runtime.
      - tmux subprocess spawn fails silently → empty pane reads.
    """
    marker = "TERMINAL_E2E_MARKER_AAAA"
    prompt = (
        f"Use sys_terminal_launch to start the 'bash' terminal with "
        f"session 's1'. Then use sys_terminal_send to type "
        f"'echo {marker}' followed by Enter. Wait briefly for the "
        f"output, then call sys_terminal_read on session 's1'. "
        f"Report what you saw. Do this in one go, then reply 'done'."
    )
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": sys_terminal_test_agent,
            "input": prompt,
            "stream": False,
        },
        timeout=180.0,
    )
    resp.raise_for_status()
    body = resp.json()
    response_id = body["id"]
    body = poll_until_terminal(http_client, response_id, timeout=180)
    assert body["status"] == "completed", (
        f"Workflow failed before completion. Status={body['status']!r}; "
        f"error={body.get('error')!r}. If 'failed' with an exception "
        f"about ``sys_terminal_launch``, the tools didn't register on "
        f"the AP-side ToolManager."
    )

    conv_id = body["conversation"]["id"]

    # The LLM must have called launch — otherwise the rest of the
    # test would be testing nothing.
    launches = _get_function_call_outputs(http_client, conv_id, "sys_terminal_launch")
    assert len(launches) >= 1, (
        f"sys_terminal_launch was never called; conv_id={conv_id}. "
        f"If 0 calls, either the LLM ignored the prompt or the tool "
        f"wasn't on the schema (registration regression)."
    )
    launch_result = json.loads(launches[0])
    assert launch_result.get("status") == "launched", (
        f"First launch should report status='launched'; got "
        f"{launch_result!r}. If the value is 'already_running', the "
        f"registry reused stale state from a prior test run."
    )

    # The marker must appear in at least one read output. We
    # don't constrain the LLM's call ordering (it might read
    # twice, retry, etc.), only that the data flowed.
    reads = _get_function_call_outputs(http_client, conv_id, "sys_terminal_read")
    assert len(reads) >= 1, f"sys_terminal_read was never called; conv_id={conv_id}."
    combined_screens = " ".join(reads)
    assert marker in combined_screens, (
        f"Echo marker {marker!r} not seen in any sys_terminal_read "
        f"output. Reads: {reads!r}. If empty, the send didn't reach "
        f"tmux. If reads have a prompt but not the echo, the bash "
        f"command failed in tmux (e.g. shell-init error)."
    )


def test_sys_terminal_omnigent_yaml_threaded_through_e2e(
    live_server: str,
    sys_terminal_test_agent: str,
    http_client: httpx.Client,
) -> None:
    """
    The omnigent-flavored YAML's ``terminals:`` block reaches
    the LLM as registered tools — this is the load-bearing
    compat-translator integration test from §8.3 (test 8).

    Distinct from the round-trip test: that one asserts the
    tools *work*, this one asserts the LLM *sees them on the
    schema*. We probe by asking the agent to introspect its
    tool list (no actual terminal launch needed).

    What breaks if this fails:
      - Compat translator doesn't propagate
        ``AgentDef.terminals`` → ``AgentSpec.terminals``.
      - ``ToolManager._register_terminal_tools`` doesn't fire
        when ``spec.terminals`` is non-None.
      - The schema produced by ``SysTerminal*Tool.get_schema``
        is malformed and the LLM rejects the function entry
        silently.
    """
    prompt = (
        "List the names of every tool you have available. Reply with "
        "the bare comma-separated list, no prose, no extra wording."
    )
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": sys_terminal_test_agent,
            "input": prompt,
            "stream": False,
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    body = resp.json()
    body = poll_until_terminal(http_client, body["id"], timeout=120)
    assert body["status"] == "completed"

    parts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                txt = block.get("text")
                if txt:
                    parts.append(txt)
    full_text = "\n".join(parts).lower()

    # Every sys_terminal_* tool must be in the LLM's listed tools.
    # We don't pin exact spelling — the LLM may use different
    # quote / format conventions. Tool *names* are well-defined,
    # though, so substring match is reliable.
    expected = [
        "sys_terminal_launch",
        "sys_terminal_send",
        "sys_terminal_read",
        "sys_terminal_list",
        "sys_terminal_close",
    ]
    missing = [name for name in expected if name not in full_text]
    assert not missing, (
        f"LLM didn't report these sys_terminal_* tools as available: "
        f"{missing!r}. Full text: {full_text!r}. If the list is "
        f"complete except one or two, the schema may be malformed "
        f"for the missing entries; if all are missing, the compat "
        f"translator didn't propagate ``terminals:`` to "
        f"``AgentSpec.terminals``."
    )


def test_sys_terminal_persists_across_turns_e2e(
    live_server: str,
    sys_terminal_test_agent: str,
    http_client: httpx.Client,
) -> None:
    """
    A terminal launched in turn 1 is still alive on turn 2 of the
    same conversation. The load-bearing test that OMNIGENT_TERMINAL_BRIDGE
    §4.4's "cleanup at workflow exit" was the wrong rule: each turn
    is a workflow run, so per-workflow cleanup would kill the
    session between turns. Real cleanup fires at conversation
    deletion / Omnigent shutdown only.

    Turn 1: launch ``bash:work``, run ``cd /tmp`` inside it.
    Turn 2 (same conversation, separate POST): run ``pwd`` in the
    same ``bash:work``. Asserts the cd state survived.

    What breaks if this fails (top suspect, the bug this test was
    written to lock in):
      - ``cleanup_conversation`` got re-added to the workflow's
        ``finally:`` block. Every turn boundary kills tmux. Turn 2
        sees ``sys_terminal_list -> []`` and the agent has to
        relaunch.

    Other failure modes:
      - Registry's per-conversation slot is being garbage-collected
        between turns (would surface as the same symptom).
      - Sub-agent boundary issues affecting the LLM's call routing.
    """
    # ── Turn 1: launch + cd /tmp ───────────────────────────
    turn1_prompt = (
        "Use sys_terminal_launch to start the 'bash' terminal with "
        "session 'work'. Then use sys_terminal_send to type 'cd /tmp' "
        "followed by Enter. Then reply 'turn 1 done'. Do not call any "
        "other terminal tools."
    )
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": sys_terminal_test_agent,
            "input": turn1_prompt,
            "stream": False,
        },
        timeout=180.0,
    )
    resp.raise_for_status()
    turn1_body = poll_until_terminal(http_client, resp.json()["id"], timeout=180)
    assert turn1_body["status"] == "completed", (
        f"Turn 1 failed: {turn1_body.get('status')!r}, error={turn1_body.get('error')!r}."
    )
    conv_id = turn1_body["conversation"]["id"]
    response_id = turn1_body["id"]

    # Sanity: turn 1 actually launched a terminal.
    launches = _get_function_call_outputs(http_client, conv_id, "sys_terminal_launch")
    assert len(launches) >= 1, (
        f"Turn 1 didn't call sys_terminal_launch; conv_id={conv_id}. "
        f"Without a launch, the persistence assertion below is vacuous."
    )
    launch_result = json.loads(launches[0])
    assert launch_result.get("status") == "launched"

    # ── Turn 2 (continued conversation): pwd in same session ──
    turn2_prompt = (
        "Use sys_terminal_send on terminal 'bash' session 'work' to "
        "type 'pwd' followed by Enter. Wait briefly, then "
        "sys_terminal_read on the same session. Reply with the read "
        "output verbatim, no prose."
    )
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": sys_terminal_test_agent,
            "input": turn2_prompt,
            "previous_response_id": response_id,
            "stream": False,
        },
        timeout=180.0,
    )
    resp.raise_for_status()
    turn2_body = poll_until_terminal(http_client, resp.json()["id"], timeout=180)
    assert turn2_body["status"] == "completed", (
        f"Turn 2 failed: {turn2_body.get('status')!r}, error={turn2_body.get('error')!r}."
    )

    # The load-bearing assertion: turn 2's reads must show /tmp,
    # proving the cd state from turn 1 survived. If cleanup ran at
    # the turn 1 → turn 2 boundary, the session would be gone and
    # either (a) sys_terminal_send would error "not running", or
    # (b) the agent would relaunch and start fresh in $HOME.
    reads = _get_function_call_outputs(http_client, conv_id, "sys_terminal_read")
    # Only count reads from turn 2 (ones added after turn 1's items).
    # Easy proxy: just check ANY read shows /tmp; the only writers
    # were turn 1 (cd /tmp) and turn 2 (pwd).
    combined = " ".join(reads)
    assert "/tmp" in combined, (
        f"Expected '/tmp' in sys_terminal_read output across the "
        f"two turns, got reads={reads!r}. If '/tmp' is missing, "
        f"the bash:work session was torn down between turns and "
        f"the agent ended up in $HOME on turn 2's pwd. Most likely "
        f"cause: ``cleanup_conversation`` regressed back into the "
        f"workflow's finally block."
    )
    # Defensive sanity: the agent didn't have to relaunch. If it
    # did, it means the session disappeared and the LLM compensated
    # by spawning fresh — a bug we want flagged loudly.
    all_launches = _get_function_call_outputs(http_client, conv_id, "sys_terminal_launch")
    assert len(all_launches) == 1, (
        f"Expected exactly one sys_terminal_launch across both "
        f"turns (turn 1's). Got {len(all_launches)}. If >1, the "
        f"agent had to relaunch on turn 2 because the session was "
        f"gone — confirms the per-workflow cleanup regression."
    )


def test_sys_terminal_full_workflow_e2e(
    live_server: str,
    sys_terminal_test_agent: str,
    http_client: httpx.Client,
) -> None:
    """
    A coherent task that exercises ALL FIVE ``sys_terminal_*`` tools
    in one conversation. The LLM is asked to perform a small shell
    investigation, then verify the cleanup succeeded.

    Flow:
      1. ``sys_terminal_launch`` — start ``bash:investigate``.
      2. ``sys_terminal_send`` — write a marker to a tmp file.
      3. ``sys_terminal_read`` — capture the pane confirming the
         echo + the file-write completed.
      4. ``sys_terminal_list`` — confirm the registry shows
         ``bash:investigate`` as running.
      5. ``sys_terminal_close`` — kill the session.
      6. ``sys_terminal_list`` again — confirm the registry no
         longer reports ``bash:investigate``.

    What this catches that the focused tests don't:
      - ``sys_terminal_list`` schema/dispatch never gets exercised
        through the LLM in the focused tests; a malformed list
        schema or wrong return shape would only fail here.
      - ``sys_terminal_close`` likewise — the focused tests
        verify the registry-level close, but not the LLM-driven
        path through the AP-side ToolManager.
      - The post-close list is the only e2e check that close
        actually removed the registry entry (not just killed
        the process); without it, a leak that only surfaces
        across multiple turns / closes would be invisible.

    The prompt tells the LLM the sequence explicitly; the
    assertions check tool names appear in conversation items
    rather than trusting the LLM's prose summary. LLM ordering
    flexibility within reason: as long as all 5 tools fire and
    the markers / list states show up in the right order, the
    test passes.
    """
    marker = "FULL_WORKFLOW_MARKER_BBBB"
    prompt = (
        "Perform this exact sequence using sys_terminal_* tools. "
        "Do NOT skip steps. Reply only after step 6 completes.\n\n"
        f"  1. sys_terminal_launch terminal='bash' session='investigate'.\n"
        f"  2. sys_terminal_send terminal='bash' session='investigate' "
        f"text='echo {marker}' keys='Enter'.\n"
        "  3. sys_terminal_read terminal='bash' session='investigate'.\n"
        "  4. sys_terminal_list (no args) — capture the result.\n"
        "  5. sys_terminal_close terminal='bash' session='investigate'.\n"
        "  6. sys_terminal_list again (no args) — capture the result.\n\n"
        "Reply with 'done' once step 6 completes. No prose, no extra "
        "wording."
    )
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": sys_terminal_test_agent,
            "input": prompt,
            "stream": False,
        },
        timeout=240.0,
    )
    resp.raise_for_status()
    body = poll_until_terminal(http_client, resp.json()["id"], timeout=240)
    assert body["status"] == "completed", (
        f"Workflow failed: status={body.get('status')!r}, error={body.get('error')!r}."
    )
    conv_id = body["conversation"]["id"]

    # All 5 tool names must appear in the conversation. We pull
    # the raw call lists so a missing tool can be named in the
    # failure message.
    launches = _get_function_call_outputs(http_client, conv_id, "sys_terminal_launch")
    sends = _get_function_call_outputs(http_client, conv_id, "sys_terminal_send")
    reads = _get_function_call_outputs(http_client, conv_id, "sys_terminal_read")
    lists = _get_function_call_outputs(http_client, conv_id, "sys_terminal_list")
    closes = _get_function_call_outputs(http_client, conv_id, "sys_terminal_close")

    missing = [
        name
        for name, calls in [
            ("sys_terminal_launch", launches),
            ("sys_terminal_send", sends),
            ("sys_terminal_read", reads),
            ("sys_terminal_list", lists),
            ("sys_terminal_close", closes),
        ]
        if not calls
    ]
    assert not missing, (
        f"LLM didn't invoke these tools: {missing!r}. The full-workflow "
        f"test requires all 5. If sys_terminal_list or sys_terminal_close "
        f"is missing, those paths have no e2e coverage anywhere else."
    )

    # The marker must appear in at least one read — proves
    # send actually reached tmux and read captured the output.
    combined_reads = " ".join(reads)
    assert marker in combined_reads, (
        f"Marker {marker!r} not seen in sys_terminal_read output: "
        f"{reads!r}. send/read flow broken."
    )

    # At least one list call must have returned a non-empty list
    # (the pre-close one in step 4) and at least one must have
    # returned an empty list (the post-close one in step 6).
    # We don't pin which is which — LLM may make extra exploratory
    # list calls — but both states must exist among the calls.
    saw_running = False
    saw_empty = False
    for raw in lists:
        try:
            entries = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(entries, list):
            # ``sys_terminal_list`` entries expose ``session`` (the
            # LLM-facing key), not ``session_key`` (the registry's
            # internal field). Confirmed via real JSON output.
            if any(isinstance(e, dict) and e.get("session") == "investigate" for e in entries):
                saw_running = True
            if entries == []:
                saw_empty = True
    assert saw_running, (
        f"No sys_terminal_list call ever showed bash:investigate as "
        f"a registered terminal. Lists: {lists!r}. Either the launch "
        f"never registered (impossible — launches above non-empty), "
        f"or list returns the wrong shape."
    )
    assert saw_empty, (
        f"No sys_terminal_list call ever returned an empty list. "
        f"Lists: {lists!r}. Either close didn't remove the entry "
        f"(registry leak), or the LLM didn't call list after close."
    )

    # Close response must have status='closed' (not 'not_found').
    # This catches the case where the LLM closed a different
    # session than it launched.
    close_results = [json.loads(r) for r in closes if r]
    assert any(c.get("status") == "closed" for c in close_results), (
        f"No sys_terminal_close returned status='closed'. Got: "
        f"{close_results!r}. Either the LLM passed the wrong "
        f"session_key, or close didn't find the registered entry "
        f"(would be a registry-tooling bug)."
    )


def test_sys_terminal_cwd_default_is_workspace_e2e(
    live_server: str,
    sys_terminal_test_agent: str,
    http_client: httpx.Client,
) -> None:
    """
    §4.6 cwd-resolution precedence: when the spec sets
    ``os_env.cwd: "."`` (the default-placeholder case), a launched
    terminal lands in the per-task workspace — NOT AP's process cwd.

    The test agent's YAML
    (``tests/resources/agents/sys-terminal-test/sys-terminal-test.yaml``)
    has ``os_env.cwd: "."`` exactly to exercise this path. The
    agent runs ``pwd`` and reports back; we assert the path is
    NOT the test-runner's cwd (the omnigent repo root) —
    if §4.6 had regressed to "use AP's process cwd", that's where
    the terminal would land.

    What breaks if this fails:
      - ``_synthesize_parent_os_env`` regresses (e.g. stops
        substituting workspace when spec.cwd is ``"."``).
      - ``ToolContext.workspace`` stops being populated by the
        workflow before tool dispatch.
      - The legacy "process cwd" fallback gets reintroduced.
    """
    import os

    runner_cwd = os.getcwd()
    prompt = (
        "Use sys_terminal_launch to start the 'bash' terminal with "
        "session 'cwdtest'. Then sys_terminal_send 'pwd' followed by "
        "Enter. Wait briefly, then sys_terminal_read on session "
        "'cwdtest'. Reply 'done' once you've read the pane."
    )
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": sys_terminal_test_agent,
            "input": prompt,
            "stream": False,
        },
        timeout=180.0,
    )
    resp.raise_for_status()
    body = poll_until_terminal(http_client, resp.json()["id"], timeout=180)
    assert body["status"] == "completed", (
        f"Workflow failed: {body.get('status')!r}, error={body.get('error')!r}"
    )
    conv_id = body["conversation"]["id"]

    reads = _get_function_call_outputs(http_client, conv_id, "sys_terminal_read")
    assert len(reads) >= 1, f"sys_terminal_read never called; conv_id={conv_id}"
    combined = " ".join(reads)

    # The pwd output must show a path that is NOT the test runner's
    # cwd. If runner_cwd shows up in any read, §4.6 regressed and
    # the terminal landed in AP's process cwd instead of the
    # per-task workspace.
    assert runner_cwd not in combined, (
        f"Found runner cwd {runner_cwd!r} in pwd output. §4.6 "
        f"workspace-fallback regressed; the terminal landed in AP's "
        f"process cwd. Reads:\n{reads!r}"
    )
    # Sanity: pwd produced *some* path — empty reads would indicate
    # the send/read flow itself broke, not §4.6 specifically.
    assert "/" in combined, (
        f"sys_terminal_read returned no path-shaped content. "
        f"send/read flow likely broken. Reads: {reads!r}"
    )


def test_sys_terminal_send_keys_drives_interactive_e2e(
    live_server: str,
    sys_terminal_test_agent: str,
    http_client: httpx.Client,
) -> None:
    """
    Interactive driving — the load-bearing capability that
    ``sys_terminal_*`` adds over ``sys_os_shell``. Start a Python
    REPL inside the bash terminal, send ``print(2+2)``, assert
    ``4`` appears in the pane.

    A naive ``sys_os_shell`` ("python3 -c 'print(2+2)'") would
    work too, but proves nothing about *interactive* state. The
    test below requires the REPL to stay running across two
    separate ``send`` calls, with the second ``send`` interpreted
    by the live python process from the first.

    What breaks if this fails:
      - ``send_keys`` parsing regresses (Enter etc. mis-routed).
      - The 50ms ``asyncio.sleep`` between text and keys collapses
        and Enter fires before the text lands.
      - The per-instance lock over-serializes such
        that the python REPL never gets to read its own input.
    """
    prompt = (
        "Use sys_terminal_launch to start the 'bash' terminal with "
        "session 'pyrepl'. Then sys_terminal_send 'python3' followed "
        "by Enter. Wait briefly. Then sys_terminal_send "
        "'print(2+2)' followed by Enter. Wait briefly. Then "
        "sys_terminal_read on session 'pyrepl'. Reply 'done' once "
        "the read completes."
    )
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": sys_terminal_test_agent,
            "input": prompt,
            "stream": False,
        },
        timeout=180.0,
    )
    resp.raise_for_status()
    body = poll_until_terminal(http_client, resp.json()["id"], timeout=180)
    assert body["status"] == "completed", (
        f"Workflow failed: {body.get('status')!r}, error={body.get('error')!r}"
    )
    conv_id = body["conversation"]["id"]

    # Two distinct sends required — one for the REPL start, one
    # for the print expression. If only one fired, the LLM
    # short-circuited to a single sys_os_shell or merged the
    # commands; the test no longer proves interactive driving.
    sends = _get_function_call_outputs(http_client, conv_id, "sys_terminal_send")
    assert len(sends) >= 2, (
        f"Expected >=2 sys_terminal_send calls (python3 start + "
        f"print(2+2)), got {len(sends)}. Sends: {sends!r}. The LLM "
        f"may have collapsed both into a single send — test no "
        f"longer exercises interactive driving."
    )

    reads = _get_function_call_outputs(http_client, conv_id, "sys_terminal_read")
    assert len(reads) >= 1, f"sys_terminal_read never called; conv_id={conv_id}"
    combined = " ".join(reads)

    # The result of print(2+2) must show in the pane. If a Python
    # REPL prompt (>>>) shows but no 4, the print was swallowed
    # by the REPL's input buffer and never executed — points at
    # the keys=Enter handling regressing.
    assert "4" in combined, (
        f"Python REPL output '4' missing from pane after "
        f"print(2+2). Combined reads:\n{combined!r}\n"
        f"If the pane shows '>>>' but no 4, the second send's "
        f"Enter didn't reach python's stdin. If the pane shows "
        f"nothing useful at all, python3 may not be on PATH in "
        f"the tmux env."
    )


def _drain_sse_to_events(response: httpx.Response) -> list[dict]:
    """
    Drain an SSE streaming response into decoded JSON event dicts.

    Reads the byte stream, splits frames on the SSE
    ``\\n\\n`` boundary, and JSON-decodes each ``data:`` line.
    Used by the streaming REPL-shape e2e test so assertions can
    inspect the exact events the REPL would render from.

    :param response: An open ``httpx.Response`` already in
        streaming mode (caller is responsible for the
        ``with client.stream(...) as resp:`` envelope).
    :returns: Decoded event dicts in arrival order.
    """
    events: list[dict] = []
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
            try:
                events.append(json.loads(data_line[len("data:") :].strip()))
            except json.JSONDecodeError:
                continue
    return events


def test_sys_terminal_repl_tool_call_render_no_mcp_prefix_no_duplicates_e2e(
    live_server: str,
    sys_terminal_test_agent: str,
    http_client: httpx.Client,
) -> None:
    """
    REPL-shape regression: claude-sdk MCP tool calls must render
    bare names ONCE, not as ``mcp__omnigent__sys_terminal_launch``
    on the live SSE stream and again as ``sys_terminal_launch`` from
    a follow-up emission.

    The user-visible bug (2026-04-28): in the Omnigent REPL with
    ``harness: claude-sdk``, every terminal tool call appeared
    twice — first with the SDK's MCP prefix
    (``mcp__omnigent__sys_terminal_launch``) during streaming,
    then again with the prefix stripped after the turn completed.
    The fix strips the prefix in :func:`_observed_tool_call_sse_dicts`
    and :func:`_build_observed_tool_items` so the live SSE stream
    and the persisted conversation item agree on the bare name.

    Asserts on the same SSE event stream that the REPL's BlockStream
    (``sdks/python-client/omnigent_client/_stream.py``) consumes
    to render ``⏵ tool_name(args)`` lines, so the test fails at
    exactly the layer the user sees:

    1. Every ``response.output_item.done`` with
       ``item.type == "function_call"`` carries a NON-prefixed
       ``name`` (no leading ``mcp__``). A regression that
       reintroduced the prefix on the wire would fail here even
       though the persisted conversation item might still be bare
       (or vice versa).
    2. Per ``call_id``, exactly ONE such event is emitted on the
       live stream. Two emissions for the same call_id would surface
       in the REPL as duplicate ``⏵ sys_terminal_launch(...)``
       lines (the SDK yields one ``ToolGroup`` per ``ToolCall``
       parse — no dedup by call_id), which is the exact symptom
       the user reported.
    3. The persisted conversation item at
       ``GET /v1/sessions/{id}/items`` agrees with the SSE
       stream's name. Asymmetry would mean ``/history`` and
       ``/switch`` rebuild the same call as a different name from
       what the live stream shows — a more subtle form of the same
       bug.

    Uses the same ``sys-terminal-test`` agent as the rest of this
    file (claude-sdk via Databricks gateway) because the
    MCP-prefix surface ONLY exists on the claude-sdk path; codex
    / openai-agents-sdk harnesses don't use the
    ``mcp__{server}__{tool}`` naming.
    """
    prompt = (
        "Use sys_terminal_launch to start the 'bash' terminal "
        "with session 'render_test'. Just launch it and reply "
        "'done' — don't send anything into it."
    )
    body = {
        "model": sys_terminal_test_agent,
        "input": prompt,
        "stream": True,
    }

    streaming_events: list[dict] = []
    with http_client.stream(
        "POST",
        "/v1/responses",
        json=body,
        timeout=180.0,
    ) as resp:
        assert resp.status_code == 200, (
            f"Stream POST failed: {resp.status_code} — "
            f"{resp.read().decode(errors='replace')[:500]!r}"
        )
        streaming_events = _drain_sse_to_events(resp)

    # The stream must close cleanly; otherwise the assertions
    # below run against a truncated event list and produce a
    # confusing failure (e.g. "no function_call seen" when the
    # real cause was a server crash mid-stream).
    event_types = [e.get("type") for e in streaming_events]
    assert "response.completed" in event_types, (
        f"Stream did not reach response.completed; saw: {event_types}. "
        f"If the stream truncated mid-turn, the count assertions below "
        f"are unreliable."
    )

    # Filter to the events the REPL's BlockStream renders as
    # ``⏵ name(args)`` — the exact wire shape the bug surfaced
    # on. ``function_call_output`` items carry only ``call_id`` +
    # ``output``, no ``name``, so they're irrelevant here.
    function_call_items = [
        e["item"]
        for e in streaming_events
        if e.get("type") == "response.output_item.done"
        and isinstance(e.get("item"), dict)
        and e["item"].get("type") == "function_call"
    ]
    assert function_call_items, (
        f"No function_call output_item.done events on the SSE stream — "
        f"the LLM didn't call sys_terminal_launch (or the harness "
        f"emitted it via a different event shape). Event types seen: "
        f"{event_types}. Without a tool call, the regression assertions "
        f"below are vacuous."
    )

    # ── Assertion 1: bare names on the live stream ──
    # The fix strips the SDK's ``mcp__{server}__`` prefix in
    # ``_observed_tool_call_sse_dicts``. A regression that
    # restored the prefix would be caught here even when the
    # persisted item is bare (the user's exact bug ordering: live
    # stream prefixed, persistence stripped).
    prefixed = [
        item for item in function_call_items if str(item.get("name", "")).startswith("mcp__")
    ]
    assert not prefixed, (
        f"Found {len(prefixed)} function_call event(s) on the SSE stream "
        f"with the MCP prefix still attached. Names: "
        f"{[item.get('name') for item in prefixed]!r}. The REPL "
        f"renders these as ``⏵ mcp__omnigent__sys_terminal_launch(...)`` "
        f"instead of the bare ``⏵ sys_terminal_launch(...)`` — the "
        f"exact regression of the 2026-04-28 user-reported bug."
    )

    # ── Assertion 2: at most one event per (name, arguments) ──
    # ``BlockStream`` yields one ``ToolGroup`` per ``ToolCall``
    # parse with no dedup. Two function_call events for the same
    # logical call render as two ``⏵ sys_terminal_launch(...)``
    # lines — the double-render symptom from the user's report.
    #
    # Why dedup by (name, arguments) instead of call_id:
    # the legacy bug emitted the SAME tool with TWO DIFFERENT
    # call_ids — once as an "observed" event from the
    # ExecutorAdapter's :class:`ToolCallRequest` translator
    # (call_id = the SDK's tool_use_id, e.g. ``toolu_bdrk_*``)
    # and once as an "action_required" event from
    # :func:`_bridge_one_dispatch` -> ``ctx.dispatch_tool``
    # (call_id = a freshly allocated ``call_*``). Both events
    # carry the SAME tool name and arguments — the visual
    # duplicate the user sees. A call_id-keyed assertion would
    # PASS the buggy stream because each event has a unique
    # call_id; an (name, arguments) key catches the real
    # symptom.
    # Diagnostic: dump every function_call event we observed so a
    # failure here surfaces the actual emission sequence rather
    # than just a count.
    print(f"\n[TEST DIAG] function_call events ({len(function_call_items)}):")
    for idx, item in enumerate(function_call_items):
        print(
            f"  [{idx}] call_id={item.get('call_id')!r} "
            f"status={item.get('status')!r} "
            f"name={item.get('name')!r} args={item.get('arguments')!r}"
        )

    # Two function_call events per call_id are EXPECTED by design
    # post-2026-04-29: one ``status=in_progress`` (inline render
    # of the call line as the dispatch starts) plus one
    # ``status=completed`` (the post-dispatch flush from
    # ``response.completed`` for durable persistence). The SDK
    # ``BlockStream`` dedupes by call_id so the user-visible REPL
    # shows the call line ONCE. So the regression to guard against
    # is two events with the SAME (call_id, status) — that means
    # either the in_progress was emitted twice (the harness's
    # two-source action_required SSE didn't dedupe) OR the
    # observed-flush ran twice. Group by ``(call_id, status)``
    # instead of the older ``(name, arguments)`` key, which
    # false-positives on the intentional in_progress + completed
    # pair (same name + args, different status).
    by_call_status: dict[tuple[str, str], int] = {}
    for item in function_call_items:
        key = (str(item.get("call_id", "")), str(item.get("status", "")))
        by_call_status[key] = by_call_status.get(key, 0) + 1
    payload_dupes = {k: n for k, n in by_call_status.items() if n != 1}
    assert not payload_dupes, (
        f"Same (call_id, status) emitted multiple times: "
        f"{payload_dupes!r}. Two events with the SAME call_id but "
        f"DIFFERENT status are OK (in_progress + completed) — the "
        f"SDK BlockStream dedupes by call_id. Same (call_id, "
        f"status) means the upstream emitter fired twice. "
        f"Suspect sites: (a) ``ExecutorAdapter._translate_event`` "
        f"for ``ToolCallRequest`` re-emitting an MCP tool that "
        f"already round-trips via ``ctx.dispatch_tool``, or "
        f"(b) ``_emit_executor_live_only`` running twice for one "
        f"``ToolCallObserved`` (shouldn't happen but easy regression)."
    )

    # ── Assertion 3: persistence agrees with the wire ──
    # The persisted conversation item is what ``/history`` and
    # ``/switch`` rebuild. If only the SSE side were stripped,
    # those flows would still render prefixed names — a subtler
    # form of the same inconsistency. ``_build_observed_tool_items``
    # was changed alongside ``_observed_tool_call_sse_dicts``
    # specifically to keep these in sync.
    response_id = next(
        (
            e.get("response", {}).get("id")
            for e in streaming_events
            if e.get("type") == "response.created"
        ),
        None,
    )
    assert response_id is not None, (
        f"No response.created event seen — cannot fetch persisted state. "
        f"Event types: {event_types}"
    )
    body_after = poll_until_terminal(http_client, response_id, timeout=60)
    assert body_after["status"] == "completed", (
        f"Workflow status not completed: {body_after.get('status')!r}, "
        f"error={body_after.get('error')!r}"
    )
    conv_id = body_after["conversation"]["id"]
    items_resp = http_client.get(f"/v1/sessions/{conv_id}/items?limit=200")
    items_resp.raise_for_status()
    persisted_function_calls = [
        item for item in items_resp.json()["data"] if item.get("type") == "function_call"
    ]
    persisted_prefixed = [
        item for item in persisted_function_calls if str(item.get("name", "")).startswith("mcp__")
    ]
    assert not persisted_prefixed, (
        f"Persisted function_call items still carry the MCP prefix: "
        f"{[item.get('name') for item in persisted_prefixed]!r}. "
        f"``_build_observed_tool_items`` must strip the prefix so "
        f"``/history`` and ``/switch`` render the same bare name "
        f"the live stream did."
    )


def test_sys_terminal_ten_parallel_dispatches_complete_e2e(
    live_server: str,
    sys_terminal_test_agent: str,
    http_client: httpx.Client,
) -> None:
    """
    Ten parallel ``sys_terminal_*`` dispatches in a single turn must
    all succeed. Direct repro of the parallel-dispatch race: pre-fix, concurrent
    action_required dispatches raced on the parent agent workflow's
    ``function_id`` counter and produced
    ``DBOSUnexpectedStepError``, which surfaced in the REPL as a
    ``failed`` response.

    Per ``designs/TOOL_DISPATCH_CHILD_WORKFLOWS.md``: each dispatch
    now spawns its own DBOS workflow with an independent
    ``function_id`` namespace, so the race is gone by construction.

    The test:

    1. Asks the LLM to launch ten sandboxed/unsandboxed terminals
       in a single turn. The exact tool count varies because the LLM
       might split the work, but ten launches produces enough
       concurrent action_required events to expose the race.
    2. Asserts response status = ``"completed"``. Any
       ``DBOSUnexpectedStepError`` would surface as ``"failed"``.
    3. Asserts at least N child ``kind="tool"`` task rows under
       the parent — proving each dispatch DID spawn a child
       workflow (the architecture from the design doc, not a
       silent fallback).
    4. Asserts every persisted ``function_call_output`` has a
       non-empty ``output`` field — proving the PATCH back ran
       through the child workflow's ``_patch_to_harness`` step
       and the parent's ``response.completed`` flush stamped the
       result on the conversation history.

    Skipped automatically when ``tmux`` is missing — the
    ``pytestmark`` at module level handles that. Requires the
    ``--llm-api-key`` option (Databricks test-profile PAT for the
    claude-sdk + databricks gateway path).
    """
    prompt = (
        "Launch 10 separate bash terminals using sys_terminal_launch. Use "
        "session keys 't0', 't1', 't2', ..., 't9'. Just call sys_terminal_launch "
        "once per terminal — do not send anything into them and do not read "
        "from them. Reply 'done' once all 10 launches return."
    )
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": sys_terminal_test_agent,
            "input": prompt,
            "stream": False,
        },
        timeout=300.0,
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]
    body = poll_until_terminal(http_client, response_id, timeout=300)

    assert body["status"] == "completed", (
        f"Expected status='completed' but got status={body['status']!r}, "
        f"error={body.get('error')!r}. A 'failed' status here is the "
        f"exact regression that was fixed: concurrent action_required "
        f"dispatches racing on the parent agent workflow's "
        f"function_id counter. Re-check whether each dispatch is "
        f"spawning its own child workflow per "
        f"designs/TOOL_DISPATCH_CHILD_WORKFLOWS.md."
    )

    conv_id = body["conversation"]["id"]

    # Count actual launch tool calls + non-empty outputs. The LLM
    # may launch slightly more than 10 (retries on transient
    # errors) but at least 10 should land. Ten is the threshold
    # that historically reproduces the race; fewer wouldn't prove
    # the parallel-dispatch path was exercised.
    launches = _get_function_call_outputs(http_client, conv_id, "sys_terminal_launch")
    assert len(launches) >= 10, (
        f"Expected at least 10 sys_terminal_launch calls; got "
        f"{len(launches)}. The LLM may have collapsed the request — "
        f"if so the test no longer exercises the parallel-dispatch "
        f"path and needs a stronger prompt. Outputs seen: "
        f"{launches[:3]!r}{'...' if len(launches) > 3 else ''}"
    )
    # Every recorded output must be a real result envelope, not an
    # empty string. The pre-fix bug surfaced as response='failed'
    # with empty function_call_outputs because the workflow died
    # before the harness's response.completed flush ran.
    succeeded = 0
    for idx, out in enumerate(launches):
        if not out:
            # An empty outputs slipped through: not the race we're
            # guarding (the workflow completed) but worth surfacing
            # so the test author can investigate. Don't fail the
            # test on a single bad launch — the race regression
            # would produce N>>1 empties, which the
            # ``succeeded >= 10`` floor below catches.
            continue
        try:
            parsed_out = json.loads(out)
        except json.JSONDecodeError:
            continue
        if parsed_out.get("status") in {"launched", "already_running"}:
            succeeded += 1
    assert succeeded >= 10, (
        f"Expected at least 10 launches to report a successful "
        f"status, got {succeeded} of {len(launches)}. The pre-fix "
        f"race was that several dispatches died with empty outputs; "
        f"if this assertion fails, look at server.log for "
        f"DBOSUnexpectedStepError or other async-dispatch errors."
    )
