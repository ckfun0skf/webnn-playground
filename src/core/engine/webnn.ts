import { assertValidNetwork } from "../model/validate";
import type { MLPNetwork } from "../model/types";
interface MLContextOptions { powerPreference?: "default" | "high-performance" | "low-power"; accelerated?: boolean; /** Chromium-specific legacy device selector; not part of the current WebNN standard. */ deviceType?: "npu" }
interface MLOperandDescriptor { dataType: "float32"; shape: number[] }
interface MLOperand {}
interface MLTensor { destroy(): void }
interface MLGraph { destroy(): void }
interface MLGraphBuilderLike {
 input(name: string, descriptor: MLOperandDescriptor): MLOperand;
 constant(descriptor: MLOperandDescriptor, data: Float32Array): MLOperand;
 gemm(a: MLOperand, b: MLOperand, options?: { c?: MLOperand }): MLOperand;
 relu(input: MLOperand): MLOperand; tanh(input: MLOperand): MLOperand;
 sigmoid(input: MLOperand): MLOperand; softmax(input: MLOperand): MLOperand;
 build(outputs: Record<string, MLOperand>): Promise<MLGraph>;
}
interface MLGraphBuilderConstructor { new(context: MLContextLike): MLGraphBuilderLike }
interface MLContextLike {
 createTensor(descriptor: MLOperandDescriptor & { writable?: boolean; readable?: boolean }): Promise<MLTensor>;
 writeTensor(tensor: MLTensor, data: Float32Array): void;
 dispatch(graph: MLGraph, inputs: Record<string, MLTensor>, outputs: Record<string, MLTensor>): void;
 readTensor(tensor: MLTensor): Promise<ArrayBuffer>; destroy(): void;
 readonly accelerated: boolean;
}
interface NavigatorWithML extends Navigator { ml?: { createContext(options?: MLContextOptions): Promise<MLContextLike> } }
export interface WebNNPredictor { predict(input: readonly number[]): Promise<number[]>; dispose(): void; accelerated: boolean }

export async function createWebNNPredictor(
 network: MLPNetwork,
 options: MLContextOptions = { powerPreference: "default", accelerated: true },
): Promise<WebNNPredictor> {
 assertValidNetwork(network);
 const ml = (navigator as NavigatorWithML).ml;
 const Builder = (globalThis as unknown as { MLGraphBuilder?: MLGraphBuilderConstructor }).MLGraphBuilder;
 if (!ml || typeof ml.createContext !== "function") throw new Error("WebNN is not exposed in this Worker. Enable WebNN in a supported browser and use HTTPS.");
 if (!Builder) throw new Error("MLGraphBuilder is unavailable in this Worker.");
 // The NPU deviceType hint is Chromium-specific and is not part of the current W3C WebNN API.
 // Other backends may ignore it; WebNN does not expose a portable way to verify the selected device.
 const context = await ml.createContext(options);
 let graph: MLGraph | undefined;
 let inputTensor: MLTensor | undefined;
 let outputTensor: MLTensor | undefined;
 try {
  const builder = new Builder(context);
  const input = builder.input("input", { dataType: "float32", shape: [1, network.inputSize] });
  let activationInput = input;
  let previousSize = network.inputSize;
  for (const layer of network.layers) {
   const transposedWeights = new Float32Array(previousSize * layer.units);
   for (let out = 0; out < layer.units; out += 1) for (let inp = 0; inp < previousSize; inp += 1) {
    transposedWeights[inp * layer.units + out] = layer.weights[out][inp];
   }
   const weights = builder.constant({ dataType: "float32", shape: [previousSize, layer.units] }, transposedWeights);
   const options: { c?: MLOperand } = {};
   if (layer.useBias !== false) options.c = builder.constant(
    { dataType: "float32", shape: [1, layer.units] }, new Float32Array(layer.biases),
   );
   let output = builder.gemm(activationInput, weights, options);
   switch (layer.activation) {
    case "relu": output = builder.relu(output); break;
    case "tanh": output = builder.tanh(output); break;
    case "sigmoid": output = builder.sigmoid(output); break;
    case "softmax": output = builder.softmax(output); break;
    case "linear": break;
   }
   activationInput = output;
   previousSize = layer.units;
  }
  graph = await builder.build({ output: activationInput });
  inputTensor = await context.createTensor({ dataType: "float32", shape: [1, network.inputSize], writable: true });
  outputTensor = await context.createTensor({ dataType: "float32", shape: [1, previousSize], readable: true });
  return {
   accelerated: context.accelerated,
   async predict(values) {
    if (values.length !== network.inputSize) throw new Error(`Expected ${network.inputSize} input values, received ${values.length}.`);
    if (values.some(value => !Number.isFinite(value))) throw new Error("Input values must all be finite numbers.");
    context.writeTensor(inputTensor!, new Float32Array(values));
    context.dispatch(graph!, { input: inputTensor! }, { output: outputTensor! });
    return Array.from(new Float32Array(await context.readTensor(outputTensor!)));
   },
   dispose() { inputTensor?.destroy(); outputTensor?.destroy(); graph?.destroy(); context.destroy(); },
  };
 } catch (error) {
  inputTensor?.destroy(); outputTensor?.destroy(); graph?.destroy(); context.destroy();
  throw error;
 }
}
