import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RunnerOfflineError } from "@/hooks/useWorkspaceChangedFiles";
import { FolderTree } from "./FolderTree";

afterEach(cleanup);

/** Render FolderTree (the "All" files tab) with defaults, overriding per test. */
function renderTree(props: Partial<Parameters<typeof FolderTree>[0]> = {}) {
  return render(
    <FolderTree
      files={undefined}
      isLoading={false}
      isError={false}
      error={null}
      onFileSelect={vi.fn()}
      conversationId="conv_abc"
      showHidden={false}
      changedFiles={undefined}
      {...props}
    />,
  );
}

describe("FolderTree runner-offline state", () => {
  it("shows the reconnect hint when the runner went offline (session failed)", () => {
    // With runnerWentOffline the "All" tab shows the same reconnect hint as
    // the Changed tab, not the generic "Failed to load".
    renderTree({ isError: true, error: new RunnerOfflineError(), runnerWentOffline: true });

    expect(screen.getByText(/agent is asleep/i)).toBeInTheDocument();
    expect(screen.getByText(/send a message in the chat to reconnect/i)).toBeInTheDocument();
    expect(screen.queryByText(/failed to load/i)).not.toBeInTheDocument();
  });

  it("shows the empty state (not the asleep hint) for a new session that hasn't started", () => {
    // A new session 503s while connecting but never went "failed" — show
    // the normal empty state, not the asleep alarm.
    renderTree({ isError: true, error: new RunnerOfflineError(), runnerWentOffline: false });

    expect(screen.getByText(/no files in workspace/i)).toBeInTheDocument();
    expect(screen.queryByText(/agent is asleep/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/failed to load/i)).not.toBeInTheDocument();
  });

  it("still shows the raw error for a non-runner-offline failure", () => {
    renderTree({ isError: true, error: new Error("500 Internal Server Error") });

    expect(screen.getByText(/failed to load: 500 internal server error/i)).toBeInTheDocument();
    expect(screen.queryByText(/agent is asleep/i)).not.toBeInTheDocument();
  });
});
