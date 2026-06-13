import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { appendPromptHistoryEntry, usePromptHistory } from "./usePromptHistory";

const STORAGE_KEY = "omnigent:prompt-history";

function storedHistory(): string[] {
  const raw = localStorage.getItem(STORAGE_KEY);
  return raw ? (JSON.parse(raw) as string[]) : [];
}

describe("appendPromptHistoryEntry", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => localStorage.clear());

  it("persists a trimmed entry and returns the new history", () => {
    // The leading/trailing whitespace must be stripped before storage so
    // recall returns the prompt the way it was composed, not with stray
    // padding — and the return value is what the hook syncs its ref to.
    const result = appendPromptHistoryEntry("  read the README  ");
    expect(result).toEqual(["read the README"]);
    expect(storedHistory()).toEqual(["read the README"]);
  });

  it("skips empty / whitespace-only text without writing", () => {
    // A blank landing composer must never push an empty entry — pressing
    // ArrowUp afterwards would otherwise recall "" and clear the input.
    expect(appendPromptHistoryEntry("   ")).toBeNull();
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it("collapses a consecutive duplicate against the persisted tail", () => {
    // Sending the same prompt from the landing composer and then again in the
    // chat must not store it twice (shell HISTCONTROL=ignoredups). The second
    // call returns the unchanged history so the caller still syncs correctly.
    appendPromptHistoryEntry("hello");
    const result = appendPromptHistoryEntry("hello");
    expect(result).toEqual(["hello"]);
    expect(storedHistory()).toEqual(["hello"]);
  });

  it("keeps a non-consecutive repeat (only the immediate tail dedupes)", () => {
    appendPromptHistoryEntry("a");
    appendPromptHistoryEntry("b");
    appendPromptHistoryEntry("a");
    expect(storedHistory()).toEqual(["a", "b", "a"]);
  });

  it("caps at 100 entries with FIFO eviction of the oldest", () => {
    for (let i = 0; i < 105; i++) appendPromptHistoryEntry(`p${i}`);
    const history = storedHistory();
    expect(history).toHaveLength(100);
    // The five oldest (p0..p4) are evicted; the newest survives at the tail.
    expect(history[0]).toBe("p5");
    expect(history[history.length - 1]).toBe("p104");
  });
});

describe("usePromptHistory — landing → chat handoff", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => {
    cleanup();
    localStorage.clear();
  });

  it("recalls an entry written by appendPromptHistoryEntry before the hook mounted", () => {
    // This is the exact landing-composer flow: the home page writes the
    // prompt to localStorage, then navigates so a fresh chat composer mounts.
    // The hook hydrates from localStorage on mount, so the first ArrowUp must
    // return that prompt — proving the cross-surface handoff works.
    appendPromptHistoryEntry("the prompt I just sent");
    const { result } = renderHook(() => usePromptHistory());
    let recalled: string | null = null;
    act(() => {
      recalled = result.current.recallPrevious("");
    });
    expect(recalled).toBe("the prompt I just sent");
  });

  it("appendEntry from the chat composer is itself recallable", () => {
    const { result } = renderHook(() => usePromptHistory());
    act(() => result.current.appendEntry("typed in chat"));
    let recalled: string | null = null;
    act(() => {
      recalled = result.current.recallPrevious("");
    });
    expect(recalled).toBe("typed in chat");
    expect(storedHistory()).toEqual(["typed in chat"]);
  });
});
