import {
  assertValidNetwork,
  createMLP,
  type MLPNetwork,
} from "../../core/model";

export type RandomSource = () => number;

export function cloneNetwork(network: MLPNetwork, id: string): MLPNetwork {
  return {
    ...network,
    id,
    layers: network.layers.map((layer) => ({
      ...layer,
      useBias: layer.useBias !== false,
      weights: layer.weights.map((row) => [...row]),
      biases: [...layer.biases],
    })),
    metadata: network.metadata
      ? { ...network.metadata, tags: network.metadata.tags ? [...network.metadata.tags] : undefined }
      : undefined,
  };
}

export function createRandomNetwork(
  template: MLPNetwork,
  id: string,
  random: RandomSource,
): MLPNetwork {
  return createMLP({
    id,
    name: template.name,
    inputSize: template.inputSize,
    layers: template.layers.map((layer) => ({
      units: layer.units,
      activation: layer.activation,
      useBias: layer.useBias !== false,
    })),
    random,
  });
}

export function crossoverNetworks(
  parentA: MLPNetwork,
  parentB: MLPNetwork,
  childId: string,
  random: RandomSource,
): MLPNetwork {
  if (
    parentA.inputSize !== parentB.inputSize ||
    parentA.layers.length !== parentB.layers.length ||
    parentA.layers.some((layer, index) =>
      layer.units !== parentB.layers[index].units ||
      layer.activation !== parentB.layers[index].activation ||
      (layer.useBias !== false) !== (parentB.layers[index].useBias !== false)
    )
  ) {
    throw new Error("Crossover requires networks with identical architectures.");
  }

  const child: MLPNetwork = {
    ...parentA,
    id: childId,
    name: parentA.name,
    layers: parentA.layers.map((layer, layerIndex) => {
      const other = parentB.layers[layerIndex];
      return {
        ...layer,
        weights: layer.weights.map((row, rowIndex) =>
          row.map((weight, columnIndex) =>
            random() < 0.5 ? weight : other.weights[rowIndex][columnIndex],
          ),
        ),
        biases: layer.useBias === false ? [...layer.biases] : layer.biases.map((bias, index) =>
          random() < 0.5 ? bias : other.biases[index],
        ),
      };
    }),
  };

  assertValidNetwork(child);
  return child;
}

function gaussian(random: RandomSource): number {
  const first = Math.max(random(), Number.EPSILON);
  const second = random();
  return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
}

export function mutateNetwork(
  network: MLPNetwork,
  random: RandomSource,
  mutationRate: number,
  mutationScale: number,
  id = network.id,
): MLPNetwork {
  const mutated: MLPNetwork = {
    ...network,
    id,
    layers: network.layers.map((layer) => ({
      ...layer,
      weights: layer.weights.map((row) =>
        row.map((weight) =>
          random() < mutationRate ? weight + gaussian(random) * mutationScale : weight,
        ),
      ),
      biases: layer.useBias === false ? [...layer.biases] : layer.biases.map((bias) =>
        random() < mutationRate ? bias + gaussian(random) * mutationScale : bias,
      ),
    })),
  };

  assertValidNetwork(mutated);
  return mutated;
}
