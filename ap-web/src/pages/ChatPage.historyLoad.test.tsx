import { cleanup, fireEvent, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useChatStore } from "@/store/chatStore";
import { HistoryAutoLoader } from "./ChatPage";

const stickContext = vi.hoisted(() => ({
  scrollRef: { current: null as HTMLElement | null },
}));

vi.mock("use-stick-to-bottom", () => ({
  useStickToBottomContext: () => stickContext,
}));

const originalLoadMoreHistory = useChatStore.getState().loadMoreHistory;

/**
 * Installs mutable layout metrics on a jsdom element.
 *
 * @param el - Scroll container element used by the mocked StickToBottom context.
 * @param metrics - Mutable scroll state that the test can inspect and update.
 *     `clientHeight` defaults to 0 (jsdom default) so the viewport-fill guard
 *     stays dormant unless a test opts in.
 */
function setScrollMetrics(
  el: HTMLElement,
  metrics: { scrollTop: number; scrollHeight: number; clientHeight?: number },
) {
  Object.defineProperty(el, "scrollTop", {
    configurable: true,
    get: () => metrics.scrollTop,
    set: (value: number) => {
      metrics.scrollTop = value;
    },
  });
  Object.defineProperty(el, "scrollHeight", {
    configurable: true,
    get: () => metrics.scrollHeight,
  });
  Object.defineProperty(el, "clientHeight", {
    configurable: true,
    get: () => metrics.clientHeight ?? 0,
  });
}

describe("HistoryAutoLoader", () => {
  beforeEach(() => {
    stickContext.scrollRef.current = null;
  });

  afterEach(() => {
    cleanup();
    useChatStore.setState({ loadMoreHistory: originalLoadMoreHistory });
    vi.unstubAllGlobals();
  });

  it("renders no visible control", () => {
    const { container } = render(
      <HistoryAutoLoader hasMoreHistory={true} loadingMoreHistory={false} />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("preserves the visible scroll offset after a scroll-up prepend", () => {
    const loadMoreHistory = vi.fn(async () => {});
    useChatStore.setState({ loadMoreHistory });
    const scrollRoot = document.createElement("div");
    const metrics = { scrollTop: 24, scrollHeight: 100 };
    setScrollMetrics(scrollRoot, metrics);
    stickContext.scrollRef.current = scrollRoot;

    const { rerender } = render(
      <HistoryAutoLoader hasMoreHistory={true} loadingMoreHistory={false} />,
    );
    // Scroll near the top to trigger an older-history fetch.
    fireEvent.scroll(scrollRoot);
    rerender(<HistoryAutoLoader hasMoreHistory={true} loadingMoreHistory={true} />);
    metrics.scrollHeight = 180;
    rerender(<HistoryAutoLoader hasMoreHistory={false} loadingMoreHistory={false} />);

    expect(metrics.scrollTop).toBe(104);
  });

  it("loads older history when the user scrolls near the top", () => {
    const loadMoreHistory = vi.fn(async () => {});
    useChatStore.setState({ loadMoreHistory });
    const scrollRoot = document.createElement("div");
    setScrollMetrics(scrollRoot, { scrollTop: 299, scrollHeight: 100 });
    stickContext.scrollRef.current = scrollRoot;

    render(<HistoryAutoLoader hasMoreHistory={true} loadingMoreHistory={false} />);
    fireEvent.scroll(scrollRoot);

    expect(loadMoreHistory).toHaveBeenCalledTimes(1);
  });

  it("auto-loads when the window is too short to scroll", () => {
    const loadMoreHistory = vi.fn(async () => {});
    useChatStore.setState({ loadMoreHistory });
    const scrollRoot = document.createElement("div");
    // Content shorter than the viewport → no scrollbar, scroll trigger can't fire.
    setScrollMetrics(scrollRoot, { scrollTop: 0, scrollHeight: 100, clientHeight: 500 });
    stickContext.scrollRef.current = scrollRoot;

    render(<HistoryAutoLoader hasMoreHistory={true} loadingMoreHistory={false} />);

    expect(loadMoreHistory).toHaveBeenCalledTimes(1);
  });

  it("does not auto-load a short window once history is exhausted", () => {
    const loadMoreHistory = vi.fn(async () => {});
    useChatStore.setState({ loadMoreHistory });
    const scrollRoot = document.createElement("div");
    setScrollMetrics(scrollRoot, { scrollTop: 0, scrollHeight: 100, clientHeight: 500 });
    stickContext.scrollRef.current = scrollRoot;

    render(<HistoryAutoLoader hasMoreHistory={false} loadingMoreHistory={false} />);

    expect(loadMoreHistory).not.toHaveBeenCalled();
  });

  it("re-fills when the viewport grows so content no longer overflows", () => {
    const loadMoreHistory = vi.fn(async () => {});
    useChatStore.setState({ loadMoreHistory });

    const holder: { cb: (() => void) | null } = { cb: null };
    class StubResizeObserver {
      constructor(cb: () => void) {
        holder.cb = cb;
      }
      observe() {}
      unobserve() {}
      disconnect() {}
    }
    vi.stubGlobal("ResizeObserver", StubResizeObserver);

    const scrollRoot = document.createElement("div");
    // Content overflows the viewport at first — scrollbar present, fill dormant.
    const metrics = { scrollTop: 600, scrollHeight: 1000, clientHeight: 500 };
    setScrollMetrics(scrollRoot, metrics);
    stickContext.scrollRef.current = scrollRoot;

    render(<HistoryAutoLoader hasMoreHistory={true} loadingMoreHistory={false} />);
    expect(loadMoreHistory).not.toHaveBeenCalled();

    // Window grows: content no longer overflows, so the resize re-check pages.
    metrics.clientHeight = 1200;
    holder.cb?.();

    expect(loadMoreHistory).toHaveBeenCalledTimes(1);
  });
});
