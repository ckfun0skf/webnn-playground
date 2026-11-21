import { describe, expect, it } from "vitest";
import { createSeededRandom } from "./random";

describe("createSeededRandom", () => {
  it("produces repeatable sequences for the same seed", () => {
    const first = createSeededRandom(42);
    const second = createSeededRandom(42);

    expect(Array.from({ length: 5 }, first)).toEqual(
      Array.from({ length: 5 }, second),
    );
  });

  it("produces values in the range [0, 1)", () => {
    const random = createSeededRandom(123);
    for (let index = 0; index < 100; index += 1) {
      const value = random();
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });
});
