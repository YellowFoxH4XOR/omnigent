#!/usr/bin/env bash
# Decides whether the `force-merge` label can bypass the CI gate.
#
# Effective only if a maintainer is on the hook for the change: the PR
# author is a maintainer, OR a maintainer's most recent decisive review
# (non-COMMENTED) on the PR is APPROVED.
#
# When the label is applied without either, we surface the reason in a
# red Merge Ready status rather than silently letting the bypass land.
#
# Env in: GH_TOKEN, REPO, PR, FORCE_MERGE, MAINTAINERS
# Out:    effective=true|false; reason=<human-readable>

set -euo pipefail

if [[ "$FORCE_MERGE" != "true" ]]; then
  echo "effective=false" >> "$GITHUB_OUTPUT"
  echo "reason=" >> "$GITHUB_OUTPUT"
  exit 0
fi

if [[ -z "${MAINTAINERS// /}" ]]; then
  echo "effective=false" >> "$GITHUB_OUTPUT"
  echo "reason=no maintainers configured in .github/MAINTAINER on main" >> "$GITHUB_OUTPUT"
  exit 0
fi

# GitHub usernames are case-insensitive (login is unique modulo case),
# so compare against a lowercase normalized list. Exact bash string
# compare on the lowercased pair -- not `grep -w`, which treats `-` as
# a word boundary and would let `alice` match `alice-admin`.
MAINTAINERS_LC=$(echo "$MAINTAINERS" | tr '[:upper:]' '[:lower:]')

AUTHOR=$(gh pr view "$PR" --repo "$REPO" --json author --jq '.author.login')
AUTHOR_LC=$(echo "$AUTHOR" | tr '[:upper:]' '[:lower:]')
for m in $MAINTAINERS_LC; do
  if [[ "$m" == "$AUTHOR_LC" ]]; then
    echo "effective=true" >> "$GITHUB_OUTPUT"
    echo "reason=author @$AUTHOR is a maintainer" >> "$GITHUB_OUTPUT"
    exit 0
  fi
done

# Latest decisive (non-COMMENTED) review per user; keep those whose
# latest state is APPROVED. Matches GitHub's UI: a later COMMENTED
# review doesn't supersede an approval, but CHANGES_REQUESTED or
# DISMISSED does.
APPROVERS=$(gh api "repos/$REPO/pulls/$PR/reviews" --paginate \
  --jq '[.[] | select(.state != "COMMENTED")] | group_by(.user.login) | map(max_by(.submitted_at)) | .[] | select(.state == "APPROVED") | .user.login')

for u in $APPROVERS; do
  u_lc=$(echo "$u" | tr '[:upper:]' '[:lower:]')
  for m in $MAINTAINERS_LC; do
    if [[ "$m" == "$u_lc" ]]; then
      echo "effective=true" >> "$GITHUB_OUTPUT"
      echo "reason=approved by maintainer @$u" >> "$GITHUB_OUTPUT"
      exit 0
    fi
  done
done

echo "effective=false" >> "$GITHUB_OUTPUT"
echo "reason=author @$AUTHOR is not a maintainer and no maintainer has approved this PR yet" >> "$GITHUB_OUTPUT"
