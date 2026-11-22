# Network Import and Export

The Network Library supports two interchange formats.

## Native JSON

Native JSON is the full-fidelity WebNN Playground format. It serializes the complete MLPNetwork representation, including schema version, network and layer IDs, activation functions, bias settings, weights, biases, and metadata.

Import validates the model against the current native schema. Imported networks receive new network and layer IDs so an import never overwrites an existing library entry; all model architecture, parameters, and metadata are retained.

## ONNX

ONNX import and export run locally in the browser. The current adapter intentionally supports a narrow, explicit subset that maps to this application's sequential dense MLP model:

- Gemm dense layers with FLOAT32 embedded initializers
- Optional Gemm bias
- Relu, Tanh, Sigmoid, Softmax (last dimension), and Identity activations
- Sequential single-input graphs with no branches

The importer accepts Gemm weights with either supported transB layout, and rejects unsupported Gemm scaling, transposition, operator domains, graph topology, tensor data types, and external tensor data. ONNX graphs using MatMul/Add patterns, convolution, normalization, recurrent layers, multiple inputs/outputs, or other operators are not currently convertible to the native MLP representation.

ONNX export produces a standard opset 13 graph with Gemm layers and standard activation operators. Native-only metadata and training history are not represented in ONNX.

A successful ONNX conversion means the graph was structurally converted into the supported native MLP subset; it does not imply that every arbitrary ONNX model can be imported.
