#!/usr/bin/env bash
# Single source of truth for the Merge Ready outcome. Downstream steps
# just consume `state`, `short_desc`, and `long_desc`.
#
# Truth table (rows are mutually exclusive; first match wins):
#
#   force-merge | effective | CI eval  | state    | meaning
#   ------------+-----------+----------+----------+---------------------------
#   true        | true      | (skipped)| success  | maintainer bypass
#   *           | *         | success  | success  | CI green on its own merits
#   true        | false     | failure  | failure  | bypass attempted but rejected
#   false       | false     | failure  | failure  | CI red, no bypass attempted
#
# Row 2 (CI green with ineffective force-merge) is deliberately a
# success: applying the label without maintainer involvement should be
# a no-op, not a penalty.
#
# Env in: FORCE_MERGE, EFFECTIVE, REASON, EVAL, FAILED
# Out:    state, short_desc, long_desc on $GITHUB_OUTPUT

set -euo pipefail

if [[ "$FORCE_MERGE" == "true" && "$EFFECTIVE" == "true" ]]; then
  STATE=success
  SHORT="Bypassed via force-merge ($REASON)"
  LONG=":fast_forward: gate is green via \`force-merge\` ($REASON), merging now."
elif [[ "$EVAL" == "success" ]]; then
  STATE=success
  SHORT="All required checks green"
  LONG=":white_check_mark: gate is green, merging now."
elif [[ "$FORCE_MERGE" == "true" ]]; then
  STATE=failure
  SHORT="force-merge label is not effective: $REASON"
  LONG=":no_entry: \`force-merge\` is not effective: $REASON. The merge will not fire until a maintainer approves or one of them retriggers \`/merge\`."
else
  STATE=failure
  SHORT="Required checks not all green; force-merge requires maintainer approval"
  LONG=$':hourglass: gate not green yet. Required checks not satisfied:\n\n'"$FAILED"$'\nThe merge will fire once these turn green, or apply `force-merge` with maintainer approval to bypass.'
fi

# GitHub commit-status descriptions max out at 140 chars.
if [[ ${#SHORT} -gt 140 ]]; then
  SHORT="${SHORT:0:137}..."
fi

echo "state=$STATE" >> "$GITHUB_OUTPUT"
echo "short_desc=$SHORT" >> "$GITHUB_OUTPUT"
{
  echo "long_desc<<_LONG_EOF_"
  printf '%s' "$LONG"
  echo
  echo "_LONG_EOF_"
} >> "$GITHUB_OUTPUT"
echo "Computed gate: state=$STATE | $SHORT"
