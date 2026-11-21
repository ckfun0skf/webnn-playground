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
    const limit = activation === "relu"
      ? Math.sqrt(6 / fanIn)
      : Math.sqrt(6 / (fanIn + fanOut));

    const weights = Array.from({ length: fanOut }, () =>
      Array.from({ length: fanIn }, () => (random() * 2 - 1) * limit),
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
