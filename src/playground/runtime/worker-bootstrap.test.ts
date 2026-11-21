import { describe, expect, it } from "vitest";
import { PLAYGROUND_WORKER_SOURCE } from "./worker-bootstrap";

describe("Playground Worker bootstrap", () => {
  it("contains syntactically valid JavaScript", () => {
    expect(() => new Function(PLAYGROUND_WORKER_SOURCE)).not.toThrow();
  });
});
