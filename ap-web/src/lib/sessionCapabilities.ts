/** UI-only session capability gates, derived from snapshot labels. */

const CLAUDE_NATIVE_WRAPPER = "claude-code-native-ui";

/**
 * Fail-closed gate for Web UI reasoning-effort controls.
 *
 * :param session: Session or sidebar row carrying labels. ``null`` or missing
 *     labels fail closed.
 * :returns: True only for explicit Claude-native sessions.
 */
export function supportsEffortControl(
  session: { labels?: Record<string, string | null> | null } | null | undefined,
): boolean {
  return session?.labels?.["omnigent.wrapper"] === CLAUDE_NATIVE_WRAPPER;
}
