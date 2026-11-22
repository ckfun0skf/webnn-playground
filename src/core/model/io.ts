import { onnx } from "onnx-proto";
import { assertValidNetwork, MODEL_SCHEMA_VERSION, type ActivationFunction, type DenseLayer, type MLPNetwork } from "./index";

const MAX_IMPORT_BYTES = 100 * 1024 * 1024;
const ACTIVATION_BY_OPERATOR: Record<string, ActivationFunction> = {
  Relu: "relu",
  Tanh: "tanh",
  Sigmoid: "sigmoid",
  Softmax: "softmax",
  Identity: "linear",
};

type ProtoApi = {
  ModelProto: {
    decode(bytes: Uint8Array): any;
    encode(value: unknown): { finish(): Uint8Array };
  };
};

const proto = onnx as unknown as ProtoApi;

function createId(): string {
  const token = typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2) + Date.now().toString(36);
  return "network-import-" + token;
}

function safeName(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

function tensorValues(tensor: any): number[] {
  if (tensor?.dataType !== 1) {
    throw new Error(`ONNX initializer "${tensor?.name ?? "unnamed"}" must use FLOAT32 data.`);
  }
  if (tensor.externalData?.length || tensor.dataLocation === 1) {
    throw new Error("ONNX external tensor data is not supported. Embed the weights in the .onnx file and try again.");
  }
  const expected = (tensor.dims ?? []).reduce((total: number, dimension: number) => total * Number(dimension), 1);
  let values: number[];
  if (tensor.floatData?.length) {
    values = Array.from(tensor.floatData, Number);
  } else if (tensor.rawData?.length) {
    const bytes = tensor.rawData instanceof Uint8Array ? tensor.rawData : new Uint8Array(tensor.rawData);
    if (bytes.byteLength % 4 !== 0) throw new Error("ONNX tensor raw_data is not aligned to FLOAT32 values.");
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    values = Array.from({ length: bytes.byteLength / 4 }, (_, index) => view.getFloat32(index * 4, true));
  } else {
    throw new Error(`ONNX initializer "${tensor.name}" has no embedded FLOAT32 values.`);
  }
  if (values.length !== expected || values.some((value) => !Number.isFinite(value))) {
    throw new Error(`ONNX initializer "${tensor.name}" has an invalid shape or non-finite values.`);
  }
  return values;
}

function tensorMap(graph: any): Map<string, { dims: number[]; values: number[] }> {
  const result = new Map<string, { dims: number[]; values: number[] }>();
  for (const tensor of graph.initializer ?? []) {
    if (!tensor.name || result.has(tensor.name)) throw new Error("ONNX graph contains an unnamed or duplicate initializer.");
    result.set(tensor.name, { dims: Array.from(tensor.dims ?? [], Number), values: tensorValues(tensor) });
  }
  return result;
}

function attributeInteger(node: any, name: string, fallback: number): number {
  const attribute = (node.attribute ?? []).find((item: any) => item.name === name);
  return attribute ? Number(attribute.i) : fallback;
}

function graphInputSize(graph: any, initializerNames: Set<string>, fallback: number): number {
  const inputs = (graph.input ?? []).filter((input: any) => !initializerNames.has(input.name));
  if (inputs.length !== 1) throw new Error("Only ONNX models with exactly one non-initializer input are supported.");
  const dimensions = inputs[0].type?.tensorType?.shape?.dim ?? [];
  const known = dimensions.map((dimension: any) => Number(dimension.dimValue ?? 0)).filter((value: number) => value > 0);
  return known.length ? known[known.length - 1] : fallback;
}

/** Imports a sequential, fully-connected ONNX graph made from Gemm and common activations. */
export function importOnnxNetwork(bytes: Uint8Array, filename = "Imported ONNX network"): MLPNetwork {
  if (bytes.byteLength === 0 || bytes.byteLength > MAX_IMPORT_BYTES) {
    throw new Error("ONNX file is empty or exceeds the 100 MB import limit.");
  }
  let model: any;
  try {
    model = proto.ModelProto.decode(bytes);
  } catch {
    throw new Error("The file is not a valid binary ONNX ModelProto.");
  }
  const graph = model?.graph;
  if (!graph || !Array.isArray(graph.node) || graph.node.length === 0) {
    throw new Error("ONNX model does not contain a supported graph.");
  }

  const tensors = tensorMap(graph);
  const layers: DenseLayer[] = [];
  let currentValue: string | null = null;
  let pendingActivation: ActivationFunction | null = null;

  for (let index = 0; index < graph.node.length; index += 1) {
    const node = graph.node[index];
    if (node.domain && node.domain !== "") {
      throw new Error(`Unsupported ONNX operator domain "${node.domain}".`);
    }
    if (node.opType === "Gemm") {
      if (pendingActivation !== null && layers.length > 0) {
        layers[layers.length - 1].activation = pendingActivation;
        pendingActivation = null;
      }
      if (node.input?.[0] !== currentValue && currentValue !== null) {
        throw new Error("ONNX graph is not a simple sequential dense network.");
      }
      const weight = tensors.get(node.input?.[1]);
      if (!weight || weight.dims.length !== 2) throw new Error("Each ONNX Gemm layer must reference a 2D FLOAT32 weight initializer.");
      const transA = attributeInteger(node, "transA", 0);
      const transB = attributeInteger(node, "transB", 0);
      if (transA !== 0 || (transB !== 0 && transB !== 1)) {
        throw new Error("ONNX Gemm layers with transA or unsupported transB settings cannot be imported.");
      }
      const [dim0, dim1] = weight.dims;
      const inputSize = transB === 1 ? dim1 : dim0;
      const units = transB === 1 ? dim0 : dim1;
      const weights = Array.from({ length: units }, (_, outputIndex) =>
        Array.from({ length: inputSize }, (_, inputIndex) =>
          transB === 1
            ? weight.values[outputIndex * inputSize + inputIndex]
            : weight.values[inputIndex * units + outputIndex],
        ),
      );
      const biasTensor = node.input?.[2] ? tensors.get(node.input[2]) : undefined;
      const biases = biasTensor ? biasTensor.values : Array.from({ length: units }, () => 0);
      if (biasTensor && (biasTensor.dims.reduce((a, b) => a * b, 1) !== units || biases.length !== units)) {
        throw new Error("ONNX Gemm bias dimensions do not match the layer output.");
      }
      const layer: DenseLayer = {
        id: `imported-layer-${layers.length + 1}`,
        type: "dense",
        units,
        activation: "linear",
        useBias: Boolean(biasTensor),
        weights,
        biases,
      };
      layers.push(layer);
      currentValue = node.output?.[0] ?? null;
      if (!currentValue) throw new Error("ONNX Gemm node has no output.");
      continue;
    }

    const activation = ACTIVATION_BY_OPERATOR[node.opType];
    if (!activation || layers.length === 0 || node.input?.[0] !== currentValue) {
      throw new Error(`Unsupported ONNX graph operator or topology: ${node.opType ?? "unknown"}.`);
    }
    if (pendingActivation !== null) throw new Error("Multiple activation operators between dense layers are not supported.");
    pendingActivation = activation;
    currentValue = node.output?.[0] ?? null;
    if (!currentValue) throw new Error("ONNX activation node has no output.");
    layers[layers.length - 1].activation = activation;
    pendingActivation = null;
  }

  if (layers.length === 0) throw new Error("ONNX graph does not contain any Gemm dense layers.");
  const firstWeight = tensors.get(graph.node.find((node: any) => node.opType === "Gemm")?.input?.[1]);
  const inferredInputSize = firstWeight ? (attributeInteger(graph.node.find((node: any) => node.opType === "Gemm"), "transB", 0) === 1 ? firstWeight.dims[1] : firstWeight.dims[0]) : 0;
  const inputSize = graphInputSize(graph, new Set(tensors.keys()), inferredInputSize);
  if (inputSize !== inferredInputSize) throw new Error("ONNX graph input dimensions do not match the first dense layer.");
  const network: MLPNetwork = {
    schemaVersion: MODEL_SCHEMA_VERSION,
    id: createId(),
    name: safeName(graph.name, filename.replace(/\.onnx$/i, "")),
    kind: "mlp",
    inputSize,
    layers,
    metadata: { description: "Imported from ONNX. Only sequential Gemm + activation graphs are supported." },
  };
  assertValidNetwork(network);
  return network;
}

/** Exports an MLP as an ONNX graph using Gemm and standard activation operators. */
export function exportOnnxNetwork(network: MLPNetwork): Uint8Array {
  assertValidNetwork(network);
  const initializers: any[] = [];
  const nodes: any[] = [];
  let inputName = "input";
  let previousSize = network.inputSize;

  network.layers.forEach((layer, index) => {
    const weightName = `layer_${index + 1}_weight`;
    const biasName = `layer_${index + 1}_bias`;
    const linearOutput = `layer_${index + 1}_linear`;
    const layerOutput = `layer_${index + 1}_output`;
    const transposedWeights = Array.from({ length: previousSize }, (_, inputIndex) =>
      Array.from({ length: layer.units }, (_, outputIndex) => layer.weights[outputIndex][inputIndex]),
    ).flat();
    initializers.push({
      dims: [previousSize, layer.units],
      dataType: 1,
      name: weightName,
      floatData: transposedWeights,
    });
    const gemmInputs = [inputName, weightName];
    if (layer.useBias !== false) {
      initializers.push({ dims: [layer.units], dataType: 1, name: biasName, floatData: [...layer.biases] });
      gemmInputs.push(biasName);
    }
    const hasActivation = layer.activation !== "linear";
    nodes.push({
      input: gemmInputs,
      output: [hasActivation ? linearOutput : layerOutput],
      name: `Dense_${index + 1}`,
      opType: "Gemm",
    });
    if (hasActivation) {
      const operator = ({ relu: "Relu", tanh: "Tanh", sigmoid: "Sigmoid", softmax: "Softmax" } as const)[layer.activation as Exclude<ActivationFunction, "linear">];
      nodes.push({
        input: [linearOutput],
        output: [layerOutput],
        name: `Activation_${index + 1}`,
        opType: operator,
      });
    }
    inputName = layerOutput;
    previousSize = layer.units;
  });

  const valueInfo = (name: string, size: number) => ({
    name,
    type: { tensorType: { elemType: 1, shape: { dim: [{ dimValue: 1 }, { dimValue: size }] } } },
  });
  const model = {
    irVersion: 8,
    producerName: "WebNN Playground",
    producerVersion: "1.0",
    opsetImport: [{ domain: "", version: 13 }],
    graph: {
      name: network.name,
      node: nodes,
      initializer: initializers,
      input: [valueInfo("input", network.inputSize)],
      output: [valueInfo(inputName, previousSize)],
    },
  };
  return proto.ModelProto.encode(model).finish();
}

export function downloadNetworkFile(filename: string, content: BlobPart, type: string): void {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function uniqueImportedNetwork(network: MLPNetwork, existingIds: Set<string>): MLPNetwork {
  let id = createId();
  while (existingIds.has(id)) id = createId();
  const imported = { ...network, id, layers: network.layers.map((layer, index) => ({ ...layer, id: `${id}-layer-${index + 1}`, weights: layer.weights.map((row) => [...row]), biases: [...layer.biases] })) };
  assertValidNetwork(imported);
  return imported;
}
