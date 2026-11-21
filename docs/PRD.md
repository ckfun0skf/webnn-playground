# Product Requirements

## 1. Vision

WebNN Playground is a browser-based laboratory for building, executing, training, and experimenting with neural networks. It should make it possible to move from a network design to a working agent in a user-defined simulation without requiring a backend service.

The product is intended for practical experimentation. Visual explanations and animation are useful, but the primary goal is to create and run real networks and observe their behavior.

## 2. Product principles

1. **Browser-native:** the complete application runs on GitHub Pages; no application server is required.
2. **JavaScript-first, WebNN-optional:** a JavaScript backend guarantees the baseline execution path. WebNN is an optional backend when available and suitable.
3. **One network model:** visual editing, code editing, inference, training, and saved networks operate on a shared canonical network representation.
4. **Generic environments:** a Playground is a JavaScript-defined simulation, not a game-specific builder.
5. **Engine-owned training lifecycle:** training algorithms manage populations, candidate networks, evaluation, fitness collection, and generations. Environment code defines the task.
6. **Extensible architecture:** start with MLP and expand to additional architectures and operators without making the core engine MLP-specific.
7. **Local ownership:** projects, environments, datasets, and networks can be saved locally and exported/imported.

## 3. Primary user workflows

### 3.1 Create and run a network

- Create a project.
- Add and configure layers/nodes in a visual editor.
- Edit the same network through a code representation/editor.
- Configure weights, biases, activation functions, and supported operators.
- Provide input values and run inference.
- Inspect outputs and, optionally, visualize signal propagation.
- Save and reload the network.

### 3.2 Train with supervised learning

- Import or create a dataset.
- Configure loss function, optimizer, learning rate, batch size, and training limits.
- Start, pause, resume, and stop training.
- Inspect loss and other metrics over time.
- Save checkpoints and export the trained network.

### 3.3 Train with neuroevolution

- Select a network architecture.
- Select a JavaScript Playground environment.
- Configure population size, mutation settings, crossover, selection strategy, generation limit, and fitness-related options.
- Run candidate evaluations through the generic session lifecycle.
- Inspect generation progress, candidate fitness, best-so-far network, and session metrics.
- Select any retained candidate and save it to the Network Library.
- Load a saved network into the environment for standalone play or evaluation.

### 3.4 Create a custom Playground

- Write JavaScript environment code in an editor.
- Define the required session lifecycle hooks.
- Request inference from the current candidate network whenever needed.
- End a session and report its fitness and optional metrics.
- Run the environment in a non-training/test mode where appropriate.
- Save, import, and export the environment as part of a project.

## 4. Core product areas

### Network Editor
- Visual network graph.
- Code editor/view of the canonical network definition.
- Layer, node, connection, weight, bias, and activation configuration.
- Input/output inspection and inference controls.
- Optional signal-flow animation, controllable in settings.

### Execution
- JavaScript backend as the baseline.
- Optional WebNN backend with capability detection and clear unsupported-operation reporting.
- Backend selection and result/performance comparison when both paths can execute the same model.
- No claim of hardware acceleration unless the selected browser/backend confirms it.

### Training Studio
- Supervised learning.
- Neuroevolution.
- Training configuration, controls, metrics, history, and checkpoint selection.

### Playground Studio
- Built-in example environments, beginning with Snake.
- Generic JavaScript environment editor for arbitrary simulations.
- Runtime lifecycle and network-inference API.
- Environment logs, session state, metrics, and visualization hooks.
- No requirement that custom environments use a gameboard or visual builder.

### Network Library
- Save and name networks/checkpoints.
- Store architecture and parameter data.
- Load a network into inference or a Playground.
- Import/export portable files.
- Preserve generation/session metadata where relevant.

### Project workspace
- Multiple projects/tabs.
- Local persistence in browser storage.
- Import/export project bundles.
- Clear unsaved-change and storage status.

## 5. Generic Playground requirements

The framework must not assume a particular environment domain.

At minimum, the neuroevolution integration must support:

- A session-start hook called for each candidate evaluation.
- Access to the current candidate's network through a stable inference API.
- Repeated inference calls during a session, with input/output shape validation.
- A session-end signal carrying a numeric fitness score and optional structured metrics.
- A timeout/cancellation path for environments that fail to finish.
- Deterministic seed information where possible, so evaluations can be reproduced.
- Session isolation so one candidate's state cannot leak into another candidate's evaluation.
- A way to run a saved network without an evolution population.

The exact public API is specified in the Architecture document and should be validated with a small end-to-end prototype before it is treated as stable.

## 6. Initial scope

### First complete vertical slice
1. Static GitHub Pages application shell.
2. Canonical MLP network representation.
3. JavaScript inference backend.
4. Generic Playground runtime contract.
5. Snake as the first built-in environment using that same contract.
6. Minimal neuroevolution loop with configurable population and mutation.
7. Candidate fitness history and best-network saving/loading.
8. Local project/network persistence and JSON import/export.

This vertical slice should prove that the generic framework works before expanding into more architectures or training algorithms.

## 7. Non-goals for the initial release

- Backend services, accounts, cloud synchronization, or server-side training.
- A visual custom-game or visual simulation builder.
- Arbitrary native code execution.
- Guaranteeing WebNN support in every browser or for every operation.
- Supporting every neural-network architecture in the first release.
- Treating the Snake demo as a separate training implementation.

## 8. Quality and safety requirements

- User-authored environment code must never run on the main UI thread.
- Run custom code in an isolated Worker-based runtime with a narrow message/API boundary.
- Provide execution time limits, cancellation, and error reporting.
- Do not expose DOM, browser storage, network access, or application internals to custom code by default.
- Treat Worker isolation as a reliability boundary, not a complete security sandbox against hostile code; document the trust model and evaluate stronger isolation before enabling untrusted public sharing.
- Validate all network inputs, outputs, parameter shapes, and fitness values.
- Keep the UI responsive during long simulations and training.
- Feature-detect WebNN and gracefully fall back to JavaScript.
- Include automated tests for engine, lifecycle, serialization, and example environments.

## 9. Success criteria

A user can define a non-game simulation in JavaScript, connect it to a neural network through the documented API, run repeated neuroevolution sessions, inspect fitness across generations, save a selected network, and run that saved network independently—all in a browser tab served from GitHub Pages.
