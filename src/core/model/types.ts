export const MODEL_SCHEMA_VERSION = 1 as const;

export type ActivationFunction =
  | "linear"
  | "relu"
  | "tanh"
  | "sigmoid"
  | "softmax";

export interface DenseLayer {
  id: string;
  type: "dense";
  units: number;
  activation: ActivationFunction;
  useBias?: boolean;
  /** One row per output neuron; each row contains one weight per input. */
  weights: number[][];
  /** One bias per output neuron. */
  biases: number[];
}

export interface MLPNetwork {
  schemaVersion: typeof MODEL_SCHEMA_VERSION;
  id: string;
  name: string;
  kind: "mlp";
  inputSize: number;
  layers: DenseLayer[];
  metadata?: {
    description?: string;
    tags?: string[];
  };
}

export interface DenseLayerDefinition {
  units: number;
  activation?: ActivationFunction;
  useBias?: boolean;
}

export interface CreateMLPOptions {
  id?: string;
  name?: string;
  inputSize: number;
  layers: DenseLayerDefinition[];
  random?: () => number;
}

export interface LayerParameterUpdate {
  weights?: number[][];
  biases?: number[];
  activation?: ActivationFunction;
}
