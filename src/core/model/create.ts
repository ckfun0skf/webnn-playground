import {
  MODEL_SCHEMA_VERSION,
  type CreateMLPOptions,
  type DenseLayer,
  type MLPNetwork,
  type LayerParameterUpdate,
} from "./types";
import { assertValidNetwork } from "./validate";

function createId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

function weightLimit(activation: DenseLayer["activation"], fanIn: number, fanOut: number): number {
  return activation === "relu" ? Math.sqrt(6 / fanIn) : Math.sqrt(6 / (fanIn + fanOut));
}

function createWeightRow(fanIn: number, fanOut: number, activation: DenseLayer["activation"], random: () => number): number[] {
  const limit = weightLimit(activation, fanIn, fanOut);
  return Array.from({ length: fanIn }, () => (random() * 2 - 1) * limit);
}

export function createMLP(options: CreateMLPOptions): MLPNetwork {
  const random = options.random ?? Math.random;

  if (!Number.isInteger(options.inputSize) || options.inputSize < 1) {
    throw new Error("inputSize must be a positive integer.");
  }
  if (options.layers.length === 0) {
    throw new Error("An MLP must contain at least one dense layer.");
  }

  const networkId = options.id ?? createId("network");
  let fanIn = options.inputSize;
  const layers: DenseLayer[] = options.layers.map((definition, index) => {
    if (!Number.isInteger(definition.units) || definition.units < 1) {
      throw new Error(`layers[${index}].units must be a positive integer.`);
    }

    const activation = definition.activation ?? "relu";
    const fanOut = definition.units;
    const weights = Array.from({ length: fanOut }, () =>
      createWeightRow(fanIn, fanOut, activation, random),
    );

    const layer: DenseLayer = {
      id: `${networkId}-layer-${index + 1}`,
      type: "dense",
      units: fanOut,
      activation,
      weights,
      biases: Array.from({ length: fanOut }, () => 0),
    };

    fanIn = fanOut;
    return layer;
  });

  const network: MLPNetwork = {
    schemaVersion: MODEL_SCHEMA_VERSION,
    id: networkId,
    name: options.name ?? "Untitled MLP",
    kind: "mlp",
    inputSize: options.inputSize,
    layers,
  };

  assertValidNetwork(network);
  return network;
}

export function updateLayerParameters(
  network: MLPNetwork,
  layerId: string,
  update: LayerParameterUpdate,
): MLPNetwork {
  const layerIndex = network.layers.findIndex((layer) => layer.id === layerId);
  if (layerIndex < 0) {
    throw new Error(`Layer not found: ${layerId}`);
  }

  const layers = network.layers.map((layer, index) =>
    index === layerIndex ? { ...layer, ...update } : layer,
  );

  const updated: MLPNetwork = { ...network, layers };
  assertValidNetwork(updated);
  return updated;
}

export function appendDenseLayer(
  network: MLPNetwork,
  units: number,
  activation: DenseLayer["activation"] = "relu",
  random: () => number = Math.random,
): MLPNetwork {
  if (!Number.isInteger(units) || units < 1) {
    throw new Error("units must be a positive integer.");
  }

  const fanIn = network.layers[network.layers.length - 1].units;
  const layer: DenseLayer = {
    id: createId("layer"),
    type: "dense",
    units,
    activation,
    weights: Array.from({ length: units }, () => createWeightRow(fanIn, units, activation, random)),
    biases: Array.from({ length: units }, () => 0),
  };
  const updated: MLPNetwork = { ...network, layers: [...network.layers, layer] };
  assertValidNetwork(updated);
  return updated;
}

export function removeLastDenseLayer(network: MLPNetwork): MLPNetwork {
  if (network.layers.length <= 1) {
    throw new Error("An MLP must contain at least one dense layer.");
  }
  const updated: MLPNetwork = { ...network, layers: network.layers.slice(0, -1) };
  assertValidNetwork(updated);
  return updated;
}

export function resizeLayerUnits(
  network: MLPNetwork,
  layerId: string,
  units: number,
  random: () => number = Math.random,
): MLPNetwork {
  if (!Number.isInteger(units) || units < 1) {
    throw new Error("units must be a positive integer.");
  }

  const layerIndex = network.layers.findIndex((layer) => layer.id === layerId);
  if (layerIndex < 0) throw new Error("Layer not found: " + layerId);

  const current = network.layers[layerIndex];
  const fanIn = layerIndex === 0 ? network.inputSize : network.layers[layerIndex - 1].units;
  const nextLayers = network.layers.map((layer) => ({
    ...layer,
    weights: layer.weights.map((row) => [...row]),
    biases: [...layer.biases],
  }));

  const currentLimit = weightLimit(current.activation, fanIn, units);
  const nextWeights = Array.from({ length: units }, (_, rowIndex) =>
    rowIndex < current.units
      ? [...current.weights[rowIndex]]
      : Array.from({ length: fanIn }, () => (random() * 2 - 1) * currentLimit),
  );
  const nextBiases = Array.from({ length: units }, (_, index) => current.biases[index] ?? 0);

  nextLayers[layerIndex] = { ...current, units, weights: nextWeights, biases: nextBiases };

  const following = nextLayers[layerIndex + 1];
  if (following) {
    const followingLimit = weightLimit(following.activation, units, following.units);
    following.weights = following.weights.map((row) => {
      if (units <= current.units) return row.slice(0, units);
      return [
        ...row,
        ...Array.from({ length: units - current.units }, () => (random() * 2 - 1) * followingLimit),
      ];
    });
  }

  const updated: MLPNetwork = { ...network, layers: nextLayers };
  assertValidNetwork(updated);
  return updated;
}
