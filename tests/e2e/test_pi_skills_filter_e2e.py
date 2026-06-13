"""E2E test: pi executor's ``skills:`` field actually filters
which skills the agent sees.

The Pi executor translates ``skills_filter`` into Pi CLI args at
construction time (``_resolve_pi_skill_args``):

- ``"all"``  → ``--skill <bundle_path>`` for each bundle skill,
                no ``--no-skills`` (host auto-discovery on).
- ``"none"`` → ``["--no-skills"]`` (suppresses everything).
- list[name] → ``["--no-skills"] + ["--skill", <bundle_path>]`` for
                each named bundle skill (silent skip for missing).

This test parametrizes the three filter modes against three fixture
agent bundles whose ``skills/`` subdir ships two distinctively-named
SKILL.md files:

- ``pi-e2e-xyz-greet-c4a8d5``
- ``pi-e2e-xyz-count-d2f6e1``

(Hyphens, not underscores — Pi's skill spec requires names to be
``^[a-z0-9-]+$`` with no other characters.)

The unique suffixes (``c4a8d5`` / ``d2f6e1``) are unforgable — the
LLM cannot hallucinate them, so a string match in the agent's
enumerated output is unambiguous proof Pi actually loaded that skill.

Usage::

    pytest tests/e2e/test_pi_skills_filter_e2e.py \
        --llm-api-key $LLM_API_KEY --profile test-profile -v
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from tests.e2e._harness_probes import cli_unavailable_reason
from tests.e2e.conftest import poll_until_terminal, upload_agent

_FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "resources" / "agents"

# The two bundled skill names. Suffixes are intentionally
# distinctive so the assertions are unambiguous — if these strings
# show up in the model's response, Pi genuinely surfaced them.
_GREET_NAME = "pi-e2e-xyz-greet-c4a8d5"
_COUNT_NAME = "pi-e2e-xyz-count-d2f6e1"

_pytest_pi_unavailable = cli_unavailable_reason("pi")
pytestmark = pytest.mark.skipif(
    _pytest_pi_unavailable is not None,
    reason=(
        "pi skills e2e requires a runnable 'pi' CLI; "
        f"{_pytest_pi_unavailable}. Install/fix Pi to run this test."
    ),
)


def _extract_all_text(body: dict[str, Any]) -> str:
    """
    Concatenate all ``output_text`` blocks from a response body.

    :param body: The terminal response body returned by
        :func:`tests.e2e.conftest.poll_until_terminal`.
    :returns: All assistant text joined by newlines.
    """
    parts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                text = block.get("text")
                if text:
                    parts.append(text)
    return "\n".join(parts)


def _materialize_with_profile(
    src_dir: Path,
    dst_dir: Path,
    profile: str,
) -> Path:
    """
    Copy a fixture agent bundle and inject the Databricks profile.

    Mirror of the codex e2e variant. The fixture YAMLs intentionally
    omit ``executor.profile`` so the same fixtures work across
    developers with different ``~/.databrickscfg`` profile names. At
    test time we materialize a per-test copy with the actual
    ``--profile`` baked in. Without a profile the Pi harness wrap
    can't authenticate with the Databricks gateway and the agent run
    fails before skills are even consulted.

    :param src_dir: Path to the fixture under
        ``tests/resources/agents/pi_skills_*/``.
    :param dst_dir: Tmp directory to copy into.
    :param profile: Databricks profile name from ``--profile``.
    :returns: The materialized bundle directory ready for
        :func:`upload_agent`.
    """
    bundle = dst_dir / src_dir.name
    shutil.copytree(src_dir, bundle)
    yaml_path = bundle / f"{src_dir.name}.yaml"
    raw = yaml.safe_load(yaml_path.read_text())
    raw["executor"]["profile"] = profile
    yaml_path.write_text(yaml.safe_dump(raw, default_flow_style=False))
    return bundle


@pytest.fixture
def pi_profile(request: pytest.FixtureRequest) -> str:
    """
    Return the ``--profile`` CLI arg, or skip if not provided.

    This test is not parametrized over a ``harness`` argument, so the autouse
    harness gate in ``tests.e2e.conftest`` cannot infer that the Pi executor is
    required. The module-level ``pytestmark`` handles the Pi CLI prerequisite.

    :param request: Pytest request object.
    :returns: The Databricks profile name.
    :raises pytest.skip.Exception: If ``--profile`` was not passed.
    """
    profile: str = request.config.getoption("--profile")
    if not profile:
        pytest.skip(
            "pi skills e2e requires --profile <name> "
            "(e.g. --profile test-profile) so the harness wrap can "
            "authenticate the Databricks gateway"
        )
    return profile


@pytest.mark.parametrize(
    "fixture, expected_visible, expected_hidden",
    [
        # ``skills: all`` → both bundled skills exposed via
        # ``--skill <path>`` flags. Failure mode: resolver drops
        # bundle source, env-var bridge drops BUNDLE_DIR, or the
        # ``"all"`` branch is broken. Any of these would leave the
        # agent with zero bundle skills and the ``in text``
        # assertion would fail.
        (
            "pi_skills_all",
            [_GREET_NAME, _COUNT_NAME],
            [],
        ),
        # ``skills: none`` → ``--no-skills`` flag suppresses both
        # auto-discovery and explicit skills. Failure mode: env-var
        # bridge drops SKILLS_FILTER, harness wrap defaults to
        # ``"all"``, resolver's ``"none"`` branch emits stray
        # ``--skill`` flags. Any of these would leak bundle skills
        # and the ``not in text`` assertion would fail.
        (
            "pi_skills_none",
            [],
            [_GREET_NAME, _COUNT_NAME],
        ),
        # ``skills: [greet]`` → ``--no-skills`` plus exactly one
        # ``--skill`` for the named bundle skill. Failure mode:
        # per-name filter doesn't apply (counter leaks), or filter
        # applies but emits the wrong path (greet missing). Either
        # is caught by one of the two assertions.
        (
            "pi_skills_list",
            [_GREET_NAME],
            [_COUNT_NAME],
        ),
    ],
)
def test_pi_skills_filter_e2e(
    pi_profile: str,
    http_client: httpx.Client,
    fixture: str,
    expected_visible: list[str],
    expected_hidden: list[str],
    tmp_path: Path,
) -> None:
    """
    Pi's ``skills:`` filter actually controls what the model sees.

    Live e2e regression-pin for the pi skills bridge. Loaded with
    deterministic-name fixtures (suffixes unforgable by the LLM) so
    the assertions can string-match without an LLM judge: the
    presence of ``pi_e2e_xyz_greet_c4a8d5`` in the model's output is
    unambiguous proof Pi loaded that SKILL.md.

    **What breaks if the feature is wrong:**

    - If ``_resolve_pi_skill_args`` doesn't emit ``--skill`` flags
      for bundle skills (e.g. the bundle source is dropped), the
      ``"all"`` and ``"list"`` cases find no bundle-skill names →
      ``expected_visible`` assertion fires with a clear "not
      visible" message.
    - If ``_resolve_skills_filter`` defaults to ``"all"`` when the
      AP-side env-var bridge breaks (the original pre-fix
      regression), the ``"none"`` case leaks bundle skills →
      ``expected_hidden`` assertion fires with the leaked name.
    - If the per-name list filter is broken (matches everything or
      matches nothing), the ``"list"`` case fires either branch.

    Each of those breakages produces a specific failure message
    naming the offending skill so a regression triage can jump
    straight to the right layer.

    :param http_client: The session-scoped ``httpx.Client`` from
        ``tests.e2e.conftest``, pointed at a live Omnigent server.
    :param pi_profile: Databricks profile from ``--profile``.
    :param fixture: Fixture agent dir name; selects the spec's
        ``skills:`` value.
    :param expected_visible: Bundled skill names that MUST appear
        in the agent's output.
    :param expected_hidden: Bundled skill names that MUST NOT
        appear in the agent's output (scoped to our fixture's
        distinctive names so the user's host-installed Pi
        extensions don't pollute the assertion).
    :param tmp_path: Pytest-provided per-test tmpdir for the
        materialized bundle.
    """
    bundle = _materialize_with_profile(_FIXTURE_ROOT / fixture, tmp_path, pi_profile)
    agent = upload_agent(http_client, bundle)

    resp = http_client.post(
        "/v1/responses",
        json={
            "model": agent,
            "input": (
                "List every skill name available to you in this session. "
                "Output ONLY the names, one per line, exactly as they appear "
                "in your environment — do not paraphrase, do not abbreviate, "
                "do not invent skills you do not see. If you have no skills, "
                "output the literal string `NO_SKILLS_LOADED`."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]

    body = poll_until_terminal(http_client, response_id, timeout=120)
    assert body["status"] == "completed", (
        f"agent run failed: status={body.get('status')!r} error={body.get('error')!r}"
    )

    text = _extract_all_text(body)

    # Visibility assertions — the listed skill names MUST appear.
    # If a name is absent, Pi didn't see that skill in its session,
    # which means either the resolver didn't emit a ``--skill``
    # flag for it or the env-var bridge dropped
    # ``HARNESS_PI_BUNDLE_DIR`` / ``HARNESS_PI_SKILLS_FILTER``.
    for name in expected_visible:
        assert name in text, (
            f"fixture={fixture!r}: bundle skill {name!r} should be visible "
            f"to the agent but didn't appear in the enumerated output. "
            f"Likely the pi resolver didn't emit --skill flags for the "
            f"bundle, or the AP→harness env-var bridge dropped "
            f"HARNESS_PI_BUNDLE_DIR / HARNESS_PI_SKILLS_FILTER. "
            f"Agent output:\n{text[:1500]}"
        )

    # Suppression assertions — the listed names MUST NOT appear.
    # Scoped to the distinctive ``c4a8d5`` / ``d2f6e1`` suffixes;
    # any host-installed Pi extension with a different name won't
    # match so the assertion stays clean.
    for name in expected_hidden:
        assert name not in text, (
            f"fixture={fixture!r}: bundle skill {name!r} should be HIDDEN "
            f"from the agent but appeared in the output. The "
            f"``skills: {fixture.removeprefix('pi_skills_')!r}`` filter "
            f"didn't suppress this skill — likely the env-var bridge "
            f"dropped HARNESS_PI_SKILLS_FILTER, the harness wrap fell "
            f'back to ``"all"``, or the resolver emitted stray '
            f"``--skill`` flags. Agent output:\n{text[:1500]}"
        )
