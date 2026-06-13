import { describe, expect, it } from "vitest";
import {
  isThemeMode,
  nextThemeMode,
  normalizeResolvedTheme,
  normalizeThemeMode,
  themeModes,
} from "./themeMode";

describe("theme mode helpers", () => {
  it("recognizes the supported selectable theme modes", () => {
    expect(themeModes).toEqual(["light", "dark", "system"]);
    expect(isThemeMode("light")).toBe(true);
    expect(isThemeMode("dark")).toBe(true);
    expect(isThemeMode("system")).toBe(true);
    expect(isThemeMode("sepia")).toBe(false);
    expect(isThemeMode(undefined)).toBe(false);
  });

  it("normalizes missing or unknown stored theme values to system", () => {
    expect(normalizeThemeMode("light")).toBe("light");
    expect(normalizeThemeMode("dark")).toBe("dark");
    expect(normalizeThemeMode("system")).toBe("system");
    expect(normalizeThemeMode("sepia")).toBe("system");
    expect(normalizeThemeMode(undefined)).toBe("system");
  });

  it("normalizes resolved theme values to the light/dark rendering modes", () => {
    expect(normalizeResolvedTheme("dark")).toBe("dark");
    expect(normalizeResolvedTheme("light")).toBe("light");
    expect(normalizeResolvedTheme("system")).toBe("light");
    expect(normalizeResolvedTheme(undefined)).toBe("light");
  });

  it("cycles system → dark → light → system on each click", () => {
    // The cycle must visit every mode exactly once before wrapping;
    // a wrong value here would skip a mode or trap the user in two states.
    expect(nextThemeMode("system")).toBe("dark");
    expect(nextThemeMode("dark")).toBe("light");
    expect(nextThemeMode("light")).toBe("system");
  });
});
