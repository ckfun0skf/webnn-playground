import { describe, expect, it } from "vitest";
import { appendDenseLayer, createMLP, removeLastDenseLayer, resizeLayerUnits, updateLayerParameters } from "./create";
import { validateNetwork } from "./validate";
import { compileMLP } from "../engine/forward";

describe("MLP model", () => {
  it("creates a valid network with correctly shaped parameters", () => {
    const network = createMLP({
      id: "test-network",
      inputSize: 2,
      layers: [{ units: 3 }, { units: 1, activation: "sigmoid" }],
      random: () => 0.5,
    });

    expect(validateNetwork(network)).toEqual([]);
    expect(network.layers[0].weights).toHaveLength(3);
    expect(network.layers[0].weights[0]).toHaveLength(2);
    expect(network.layers[1].weights[0]).toHaveLength(3);
  });

  it("runs forward propagation with known weights and biases", () => {
    const network = createMLP({
      id: "known-network",
      inputSize: 2,
      layers: [{ units: 1, activation: "linear" }],
      random: () => 0.5,
    });
    const configured = updateLayerParameters(network, network.layers[0].id, {
      weights: [[2, -1]],
      biases: [0.5],
    });

    expect(compileMLP(configured)([3, 2])).toEqual([4.5]);
  });

  it("applies sigmoid and softmax activations", () => {
    const sigmoid = createMLP({
      inputSize: 1,
      layers: [{ units: 1, activation: "sigmoid" }],
      random: () => 0.5,
    });
    const softmax = createMLP({
      inputSize: 1,
      layers: [{ units: 2, activation: "softmax" }],
      random: () => 0.5,
    });

    expect(compileMLP(sigmoid)([0])).toEqual([0.5]);
    expect(compileMLP(softmax)([0])).toEqual([0.5, 0.5]);
  });

  it("rejects input shape mismatches and invalid parameter shapes", () => {
    const network = createMLP({
      inputSize: 2,
      layers: [{ units: 1, activation: "linear" }],
      random: () => 0.5,
    });

    expect(() => compileMLP(network)([1])).toThrow(/Expected 2 input values/);
    expect(() => updateLayerParameters(network, network.layers[0].id, {
      weights: [[1, 2, 3]],
    })).toThrow(/Invalid neural network/);
  });

  it("resizes a hidden layer and keeps the following weight matrix aligned", () => {
    const network = createMLP({
      id: "resize-network",
      inputSize: 2,
      layers: [{ units: 3 }, { units: 2, activation: "softmax" }],
      random: () => 0.5,
    });

    const expanded = resizeLayerUnits(network, network.layers[0].id, 5, () => 0.5);
    expect(expanded.layers[0].weights).toHaveLength(5);
    expect(expanded.layers[1].weights[0]).toHaveLength(5);
    expect(validateNetwork(expanded)).toEqual([]);

    const reduced = resizeLayerUnits(expanded, expanded.layers[0].id, 2, () => 0.5);
    expect(reduced.layers[0].weights).toHaveLength(2);
    expect(reduced.layers[1].weights[0]).toHaveLength(2);
    expect(validateNetwork(reduced)).toEqual([]);
  });

  it("appends and removes dense layers without invalidating the network", () => {
    const network = createMLP({
      id: "layer-network",
      inputSize: 2,
      layers: [{ units: 3 }],
      random: () => 0.5,
    });
    const appended = appendDenseLayer(network, 4, "tanh", () => 0.5);

    expect(appended.layers).toHaveLength(2);
    expect(appended.layers[1].weights[0]).toHaveLength(3);
    expect(validateNetwork(appended)).toEqual([]);
    expect(removeLastDenseLayer(appended).layers).toHaveLength(1);
    expect(() => removeLastDenseLayer(network)).toThrow(/at least one dense layer/);
  });

  it("does not mutate the source network when updating parameters", () => {
    const network = createMLP({
      inputSize: 1,
      layers: [{ units: 1, activation: "linear" }],
      random: () => 0.5,
    });
    const updated = updateLayerParameters(network, network.layers[0].id, {
      biases: [7],
    });

    expect(network.layers[0].biases).toEqual([0]);
    expect(updated.layers[0].biases).toEqual([7]);
  });
});
