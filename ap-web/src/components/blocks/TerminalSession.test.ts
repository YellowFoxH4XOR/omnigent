// Unit tests for the terminal session's pure helpers.
//
// The full TerminalSession constructor needs a real xterm + WebSocket
// + DOM container, so it's exercised via manual REPL verification (see
// TerminalView.test.ts). `openTerminalLink` is the one piece of our own
// logic the WebLinksAddon delegates to — the click handler that makes
// terminal URLs clickable — so we pin it here.

import { Terminal } from "@xterm/xterm";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  SHIFT_ENTER_CSI_U,
  SYNC_ECHO_MAX_BYTES,
  SYNC_ECHO_WINDOW_MS,
  applyTerminalCopy,
  loadWebglRenderer,
  openTerminalLink,
  shouldEchoSynchronously,
  terminalTheme,
  terminalKeyEventPayload,
} from "./TerminalSession";

describe("openTerminalLink", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("opens the uri in a new tab with noopener,noreferrer", () => {
    // Stub window.open so we observe the call without spawning a tab.
    const openSpy = vi.spyOn(window, "open").mockReturnValue(null);
    const event = new MouseEvent("click");

    openTerminalLink(event, "https://example.com/foo");

    // Proves the handler routes the detected URL to a new tab with the
    // hardening flags. If it regressed to navigating in-place, _blank
    // would be missing and the live terminal session would be torn down.
    expect(openSpy).toHaveBeenCalledWith(
      "https://example.com/foo",
      "_blank",
      "noopener,noreferrer",
    );
  });

  it("prevents the addon's default in-place navigation", () => {
    vi.spyOn(window, "open").mockReturnValue(null);
    const event = new MouseEvent("click");
    const preventSpy = vi.spyOn(event, "preventDefault");

    openTerminalLink(event, "https://example.com/foo");

    // The WebLinksAddon navigates the current document on click by
    // default; without preventDefault the click would unload the SPA
    // (and kill the WebSocket-attached terminal) before window.open's
    // tab is usable. A failure here means that suppression was dropped.
    expect(preventSpy).toHaveBeenCalledOnce();
  });
});

describe("applyTerminalCopy", () => {
  function copyEvent() {
    const setData = vi.fn();
    const preventDefault = vi.fn();
    const event: Pick<ClipboardEvent, "clipboardData" | "preventDefault"> = {
      clipboardData: { setData } as unknown as DataTransfer,
      preventDefault,
    };
    return { event, setData, preventDefault };
  }

  it("writes the selection to the clipboard and prevents default", () => {
    const { event, setData, preventDefault } = copyEvent();

    // A real selection must be placed on the clipboard as text/plain and
    // the browser's default (per-visual-row) copy suppressed, so a
    // soft-wrapped paragraph pastes as the single logical line that
    // getSelection() already reflowed.
    expect(applyTerminalCopy(event, "selected text")).toBe(true);
    expect(setData).toHaveBeenCalledWith("text/plain", "selected text");
    expect(preventDefault).toHaveBeenCalledOnce();
  });

  it("does nothing when there is no selection", () => {
    const { event, setData, preventDefault } = copyEvent();

    // With no selection the event must be left untouched so the browser's
    // default copy behavior still applies (and we never clobber the
    // clipboard with an empty string).
    expect(applyTerminalCopy(event, "")).toBe(false);
    expect(setData).not.toHaveBeenCalled();
    expect(preventDefault).not.toHaveBeenCalled();
  });
});

describe("shouldEchoSynchronously", () => {
  it("takes the sync path for a small chunk right after a keystroke", () => {
    // Echo/prompt-sized chunk arriving well within the window: paint it
    // synchronously so the keystroke echo lands without a queued-write
    // frame of latency.
    expect(shouldEchoSynchronously(64, 10)).toBe(true);
  });

  it("stays async when the user hasn't typed recently", () => {
    // Past the window, this is unsolicited output (an agent printing),
    // not an echo — the async write queue is correct.
    expect(shouldEchoSynchronously(64, SYNC_ECHO_WINDOW_MS)).toBe(false);
    expect(shouldEchoSynchronously(64, SYNC_ECHO_WINDOW_MS + 1)).toBe(false);
  });

  it("stays async for large chunks even right after a keystroke", () => {
    // A big chunk is a flood/redraw, not an echo; keeping it on the async
    // path stops one giant synchronous write from blocking the main
    // thread mid-type.
    expect(shouldEchoSynchronously(SYNC_ECHO_MAX_BYTES + 1, 10)).toBe(false);
    expect(shouldEchoSynchronously(SYNC_ECHO_MAX_BYTES, 10)).toBe(true);
  });
});

describe("loadWebglRenderer", () => {
  it("returns null without throwing when WebGL is unavailable", () => {
    // jsdom has no WebGL context (getContext() is unimplemented), so this
    // exercises the exact degraded environment the fallback exists for:
    // headless CI, a blocklisted GPU, or a browser with WebGL disabled.
    // The function must swallow the failure and return null so the caller
    // keeps the working DOM renderer — a throw here would crash terminal
    // construction and leave the user with no terminal at all.
    const term = new Terminal();
    const container = document.createElement("div");
    term.open(container);

    expect(loadWebglRenderer(term)).toBeNull();

    term.dispose();
  });
});

describe("terminalTheme", () => {
  it("uses a light ANSI bright-black in light mode", () => {
    const theme = terminalTheme(false);

    // Codex paints its prompt/input band with ANSI gray. In the web light
    // theme that gray must be a pale surface so dark prompt text remains
    // readable.
    expect(theme.background).toBe("#ffffff");
    expect(theme.foreground).toBe("#18181b");
    expect(theme.brightBlack).toBe("#e4e4e7");

    // CLIs that assume a dark terminal paint primary text with ANSI
    // white / bright-white. On the white card background those slots must
    // be dark, or the text renders white-on-white and disappears.
    expect(theme.white).toBe("#3f3f46");
    expect(theme.brightWhite).toBe("#18181b");
  });

  it("keeps dark mode terminal surfaces dark", () => {
    const theme = terminalTheme(true);

    // Dark mode should retain the terminal-like contrast the rest of the
    // app expects rather than inheriting the light prompt-band treatment.
    expect(theme.background).toBe("#131517");
    expect(theme.foreground).toBe("#e4e4e7");
    expect(theme.brightBlack).toBe("#71717a");
  });
});

describe("terminalKeyEventPayload", () => {
  function keyEvent(init: KeyboardEventInit): KeyboardEvent {
    return new KeyboardEvent("keydown", init);
  }

  it("encodes Shift+Enter as Kitty CSI-u", () => {
    const payload = terminalKeyEventPayload(keyEvent({ key: "Enter", shiftKey: true }));

    // This is the byte sequence prompt-toolkit maps to F20, which the
    // REPL binds to "insert newline". Returning "\x1b\r" here would be
    // the old Alt+Enter fallback, not Kitty/CSI-u support.
    expect(payload).toBe(SHIFT_ENTER_CSI_U);
    expect(payload).toBe("\x1b[13;2u");
  });

  it("leaves plain Enter on xterm's default path", () => {
    expect(terminalKeyEventPayload(keyEvent({ key: "Enter" }))).toBeNull();
  });

  it("does not override other modified Enter combinations", () => {
    expect(terminalKeyEventPayload(keyEvent({ key: "Enter", altKey: true }))).toBeNull();
    expect(terminalKeyEventPayload(keyEvent({ key: "Enter", ctrlKey: true }))).toBeNull();
    expect(terminalKeyEventPayload(keyEvent({ key: "Enter", metaKey: true }))).toBeNull();
    expect(
      terminalKeyEventPayload(keyEvent({ key: "Enter", shiftKey: true, altKey: true })),
    ).toBeNull();
  });
});
