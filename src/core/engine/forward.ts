import { applyActivation } from "./activations";
import type { MLPNetwork } from "../model/types";
import { assertValidNetwork } from "../model/validate";

export type Predictor = (input: readonly number[]) => number[];

export function compileMLP(network: MLPNetwork): Predictor {
  assertValidNetwork(network);

  // Snapshot parameters so a compiled predictor is stable if the editor later
  // replaces the project's immutable network object.
  const layers = network.layers.map((layer) => ({
    units: layer.units,
    activation: layer.activation,
    useBias: layer.useBias !== false,
    weights: layer.weights.map((row) => [...row]),
    biases: [...layer.biases],
  }));
  const inputSize = network.inputSize;

  return (input: readonly number[]): number[] => {
    if (input.length !== inputSize) {
      throw new Error(`Expected ${inputSize} input values, received ${input.length}.`);
    }
    let values = Array.from(input);
    if (values.some((value) => !Number.isFinite(value))) {
      throw new Error("Input values must all be finite numbers.");
    }

    for (const layer of layers) {
      const sums = Array.from({ length: layer.units }, (_, neuronIndex) => {
        const weightedSum = layer.weights[neuronIndex].reduce(
          (sum, weight, inputIndex) => sum + weight * values[inputIndex],
          layer.useBias ? layer.biases[neuronIndex] : 0,
        );
        return weightedSum;
      });
      values = applyActivation(sums, layer.activation);
    }

    return values;
  };
}

export function predictMLP(network: MLPNetwork, input: readonly number[]): number[] {
  return compileMLP(network)(input);
}
