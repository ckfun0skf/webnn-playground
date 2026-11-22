import { describe, expect, it } from "vitest";
import type { MLPNetwork } from "./types";
import { exportOnnxNetwork, importOnnxNetwork, uniqueImportedNetwork } from "./io";
import { assertValidNetwork } from "./validate";

const sampleNetwork: MLPNetwork = {
  schemaVersion: 1,
  id: "sample-network",
  name: "Sample network",
  kind: "mlp",
  inputSize: 2,
  layers: [
    {
      id: "sample-layer-1",
      type: "dense",
      units: 3,
      activation: "relu",
      useBias: true,
      weights: [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
      biases: [0.01, 0.02, 0.03],
    },
    {
      id: "sample-layer-2",
      type: "dense",
      units: 1,
      activation: "sigmoid",
      useBias: true,
      weights: [[0.7, 0.8, 0.9]],
      biases: [0.04],
    },
  ],
};

describe("network interchange", () => {
  it("round-trips an MLP through ONNX without changing its parameters", () => {
    const bytes = exportOnnxNetwork(sampleNetwork);
    const imported = importOnnxNetwork(bytes, "sample.onnx");
    assertValidNetwork(imported);
    expect(imported.inputSize).toBe(sampleNetwork.inputSize);
    expect(imported.layers).toHaveLength(sampleNetwork.layers.length);
    expect(imported.layers.map((layer) => layer.units)).toEqual([3, 1]);
    expect(imported.layers.map((layer) => layer.activation)).toEqual(["relu", "sigmoid"]);
    expect(imported.layers[0].weights).toEqual(sampleNetwork.layers[0].weights);
    expect(imported.layers[0].biases).toEqual(sampleNetwork.layers[0].biases);
    expect(imported.layers[1].weights).toEqual(sampleNetwork.layers[1].weights);
    expect(imported.layers[1].biases).toEqual(sampleNetwork.layers[1].biases);
  });

  it("assigns a new network and layer identity when importing", () => {
    const imported = importOnnxNetwork(exportOnnxNetwork(sampleNetwork));
    const duplicated = uniqueImportedNetwork(imported, new Set([imported.id]));
    expect(duplicated.id).not.toBe(imported.id);
    expect(duplicated.layers[0].id).not.toBe(imported.layers[0].id);
    expect(duplicated.layers[0].weights).toEqual(imported.layers[0].weights);
  });

  it("rejects invalid ONNX bytes with a useful error", () => {
    expect(() => importOnnxNetwork(new Uint8Array([0, 1, 2, 3]))).toThrow(/valid binary ONNX/i);
  });
});
