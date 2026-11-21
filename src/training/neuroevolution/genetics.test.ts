import { describe, expect, it } from "vitest";
import { createMLP } from "../../core/model";
import { createSeededRandom } from "../../playground/runtime/random";
import { crossoverNetworks, createRandomNetwork, mutateNetwork } from "./genetics";

function createTemplate() {
  return createMLP({
    id: "template",
    inputSize: 2,
    layers: [{ units: 3, activation: "tanh" }, { units: 1, activation: "linear" }],
    random: () => 0.5,
  });
}

describe("neuroevolution genetics", () => {
  it("creates random candidates with the same architecture", () => {
    const template = createTemplate();
    const candidate = createRandomNetwork(template, "candidate", createSeededRandom(7));

    expect(candidate.inputSize).toBe(template.inputSize);
    expect(candidate.layers.map((layer) => layer.units)).toEqual([3, 1]);
    expect(candidate.layers.map((layer) => layer.activation)).toEqual(["tanh", "linear"]);
  });

  it("creates repeatable parameter sets and layer identifiers from the same seed", () => {
    const template = createTemplate();
    const first = createRandomNetwork(template, "repeatable", createSeededRandom(123));
    const second = createRandomNetwork(template, "repeatable", createSeededRandom(123));

    expect(first).toEqual(second);
  });

  it("performs deterministic crossover for a fixed seed", () => {
    const template = createTemplate();
    const parentA = createRandomNetwork(template, "a", createSeededRandom(1));
    const parentB = createRandomNetwork(template, "b", createSeededRandom(2));
    const childA = crossoverNetworks(parentA, parentB, "child", createSeededRandom(99));
    const childB = crossoverNetworks(parentA, parentB, "child", createSeededRandom(99));

    expect(childA).toEqual(childB);
  });

  it("mutates parameters without changing the source network", () => {
    const template = createTemplate();
    const original = createRandomNetwork(template, "original", createSeededRandom(5));
    const mutated = mutateNetwork(original, createSeededRandom(12), 1, 0.2, "mutated");

    expect(mutated.id).toBe("mutated");
    expect(mutated.layers[0].weights).not.toEqual(original.layers[0].weights);
    expect(original.id).toBe("original");
  });
});
