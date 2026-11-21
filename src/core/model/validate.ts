import { MODEL_SCHEMA_VERSION, type MLPNetwork } from "./types";

const ACTIVATIONS = new Set([
  "linear",
  "relu",
  "tanh",
  "sigmoid",
  "softmax",
]);

export function validateNetwork(network: unknown): string[] {
  const errors: string[] = [];

  if (!network || typeof network !== "object") {
    return ["Network must be an object."];
  }

  const candidate = network as Partial<MLPNetwork>;

  if (candidate.schemaVersion !== MODEL_SCHEMA_VERSION) {
    errors.push(`schemaVersion must be ${MODEL_SCHEMA_VERSION}.`);
  }
  if (candidate.kind !== "mlp") {
    errors.push('kind must be "mlp".');
  }
  if (typeof candidate.id !== "string" || candidate.id.length === 0) {
    errors.push("id must be a non-empty string.");
  }
  if (typeof candidate.name !== "string") {
    errors.push("name must be a string.");
  }
  if (!Number.isInteger(candidate.inputSize) || (candidate.inputSize ?? 0) < 1) {
    errors.push("inputSize must be a positive integer.");
  }
  if (!Array.isArray(candidate.layers) || candidate.layers.length === 0) {
    errors.push("layers must contain at least one dense layer.");
    return errors;
  }

  let expectedInputs = Number.isInteger(candidate.inputSize) ? candidate.inputSize! : 0;
  const layerIds = new Set<string>();

  candidate.layers.forEach((layer, index) => {
    const prefix = `layers[${index}]`;

    if (!layer || typeof layer !== "object") {
      errors.push(`${prefix} must be an object.`);
      return;
    }
    if (layer.type !== "dense") {
      errors.push(`${prefix}.type must be "dense".`);
    }
    if (typeof layer.id !== "string" || layer.id.length === 0) {
      errors.push(`${prefix}.id must be a non-empty string.`);
    } else if (layerIds.has(layer.id)) {
      errors.push(`${prefix}.id must be unique.`);
    } else {
      layerIds.add(layer.id);
    }
    if (!Number.isInteger(layer.units) || layer.units < 1) {
      errors.push(`${prefix}.units must be a positive integer.`);
    }
    if (!ACTIVATIONS.has(layer.activation)) {
      errors.push(`${prefix}.activation is not supported.`);
    }

    if (!Array.isArray(layer.weights) || layer.weights.length !== layer.units) {
      errors.push(`${prefix}.weights must have one row per unit.`);
    } else {
      layer.weights.forEach((row, rowIndex) => {
        if (!Array.isArray(row) || row.length !== expectedInputs) {
          errors.push(`${prefix}.weights[${rowIndex}] must contain ${expectedInputs} values.`);
          return;
        }
        row.forEach((value, columnIndex) => {
          if (!Number.isFinite(value)) {
            errors.push(`${prefix}.weights[${rowIndex}][${columnIndex}] must be finite.`);
          }
        });
      });
    }

    if (!Array.isArray(layer.biases) || layer.biases.length !== layer.units) {
      errors.push(`${prefix}.biases must contain one value per unit.`);
    } else {
      layer.biases.forEach((value, biasIndex) => {
        if (!Number.isFinite(value)) {
          errors.push(`${prefix}.biases[${biasIndex}] must be finite.`);
        }
      });
    }

    if (Number.isInteger(layer.units) && layer.units > 0) {
      expectedInputs = layer.units;
    }
  });

  return errors;
}

export function assertValidNetwork(network: unknown): asserts network is MLPNetwork {
  const errors = validateNetwork(network);
  if (errors.length > 0) {
    throw new Error(`Invalid neural network:\n- ${errors.join("\n- ")}`);
  }
}
