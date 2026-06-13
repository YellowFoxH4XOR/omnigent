import { describe, expect, it } from "vitest";
import { formatToolDuration, getOutputPreview } from "./ToolCard";

describe("formatToolDuration", () => {
  it("formats subsecond, second, minute, and hour durations", () => {
    expect(formatToolDuration(0.042)).toBe("42ms");
    expect(formatToolDuration(3.25)).toBe("3.3s");
    expect(formatToolDuration(12.4)).toBe("12s");
    expect(formatToolDuration(61.2)).toBe("1m 1s");
    expect(formatToolDuration(3_599.9)).toBe("1h 0m");
    expect(formatToolDuration(3_901)).toBe("1h 5m");
  });

  it("handles invalid or negative durations as zero milliseconds", () => {
    expect(formatToolDuration(Number.NaN)).toBe("0ms");
    expect(formatToolDuration(-1)).toBe("0ms");
  });
});

describe("getOutputPreview", () => {
  it("keeps short output intact", () => {
    const preview = getOutputPreview("first\nsecond");

    expect(preview.text).toBe("first\nsecond");
    expect(preview.isTruncated).toBe(false);
    expect(preview.lineCount).toBe(2);
    expect(preview.hiddenLineCount).toBe(0);
    expect(preview.hiddenCharCount).toBe(0);
  });

  it("truncates long output by line count and reports hidden content", () => {
    const output = Array.from({ length: 85 }, (_, index) => `line ${index + 1}`).join("\n");
    const preview = getOutputPreview(output);

    expect(preview.isTruncated).toBe(true);
    expect(preview.shownLineCount).toBe(80);
    expect(preview.hiddenLineCount).toBe(5);
    expect(preview.text).toContain("line 80");
    expect(preview.text).not.toContain("line 81");
  });

  it("expands long output back to the full text", () => {
    const output = "x".repeat(12_050);
    const collapsed = getOutputPreview(output);
    const expanded = getOutputPreview(output, true);

    expect(collapsed.isTruncated).toBe(true);
    expect(collapsed.shownCharCount).toBe(12_000);
    expect(expanded.text).toBe(output);
    expect(expanded.isTruncated).toBe(false);
  });
});
