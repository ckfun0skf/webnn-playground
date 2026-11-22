# Architecture

## 1. Architectural overview

The application is a static, client-side web application deployed to GitHub Pages.

```text
UI / Project Workspace
  |-- Network Editor (visual + code)
  |-- Training Studio
  |-- Playground Studio
  |-- Network Library
  |
Application Services
  |-- Project & persistence service
  |-- Network/model service
  |-- Training orchestration
  |-- Playground orchestration
  |
Execution Layer
  |-- JavaScript backend (baseline)
  |-- WebNN backend (optional)
  |
Worker Runtime
  |-- Environment Worker(s)
  |-- Training/evaluation scheduler
  |-- Message protocol
```

The UI must remain separate from long-running computation. Inference, training, and user-authored environment code must not block the main thread.

## 2. Canonical network model

Visual editing, code editing, training, serialization, and execution must share one canonical network representation.

Initial implementation target: a feedforward MLP with explicit input/output dimensions, ordered layers, layer configuration, and parameter tensors (weights and biases). The representation should be versioned and designed for future graph-based architectures.

Do not make the UI graph itself the source of truth. The graph is a view/editor for the canonical model. Likewise, the code editor should edit or generate a validated representation rather than maintaining an unrelated runtime model.

The model layer must separate:
- Architecture/configuration.
- Trainable parameters.
- Runtime/backend-specific compiled artifacts.
- Training metadata and checkpoints.

## 2.1 Initial MLP schema and JavaScript execution

The first canonical model version is a sequential MLP:

- `schemaVersion`: versioned model format (currently `1`).
- `kind`: `mlp`.
- `inputSize`: required input vector length.
- `layers`: ordered dense layers. Each layer stores its stable ID, unit count, activation, a weight matrix (one row per output unit), and one bias per output unit.

The initial activation set is `linear`, `relu`, `tanh`, `sigmoid`, and `softmax`. Parameter editing produces a new immutable network object and validates matrix dimensions and finite values.

The JavaScript backend exposes `compileMLP(network)`, which validates and snapshots the model once, then returns a predictor function for repeated inference. This is the baseline semantic reference for future training algorithms and the optional WebNN adapter. A predictor rejects wrong-sized or non-finite input vectors.

## 3. Execution backends

### JavaScript backend
- Required baseline backend.
- Runs in browser-managed computation workers where practical.
- Defines reference semantics for supported operations.
- Used for broad compatibility and correctness comparisons.

### WebNN backend
- Optional backend, detected at runtime.
- Used only for supported operations and compatible model graphs.
- Must expose capability/compilation errors clearly.
- Must not be assumed available merely because the page is hosted over HTTPS.
- Backend comparison must use equivalent inputs and model parameters and report timing methodology.
- Do not promise deterministic CPU/GPU/NPU selection. The current WebNN specification abstracts the underlying device and does not expose a standardized device-enumeration or exact-device selection API. Where supported, expose the standardized power/performance preference and report only facts the browser actually exposes (for example, whether the context is accelerated), not an inferred GPU or NPU identity.

The network API must not depend directly on WebNN objects. Backend adapters translate the canonical model/operation representation into backend execution.

## 4. Generic Playground contract

A custom Playground is JavaScript source code that describes a simulation. It is not a game schema and does not require a visual environment builder.

The engine owns candidate networks, session identity, lifecycle, cancellation, timeouts, fitness collection, and population scheduling. Environment code owns domain-specific simulation state and rules.

### 4.1 Environment module

The initial contract is implemented in `src/playground/api` and `src/playground/runtime`. A custom environment is an ES module that exports `sessionStart(ctx)`; it may also export `sessionEnd(ctx, result)` for successful-session cleanup.

```js
export async function sessionStart(ctx) {
  let state = createInitialState(ctx.random);

  while (!isTerminal(state) && !ctx.signal.aborted) {
    const observation = observe(state);
    const action = await ctx.network.predict(observation);
    state = transition(state, action);

    ctx.report({ score: state.score, steps: state.steps });

    if (isTerminal(state)) {
      await ctx.finish({
        fitness: calculateFitness(state),
        metrics: { score: state.score, steps: state.steps }
      });
      return;
    }

    await ctx.yield();
  }
}

export async function sessionEnd(ctx, result) {
  // Optional cleanup after a successfully completed session.
}
```

The environment owns its simulation and fitness calculation. The host owns the candidate network, session lifecycle, inference execution, and result collection. The environment must return after calling `finish`.

### 4.2 Context API

The environment context should expose only narrow, documented capabilities:

| API | Purpose |
|---|---|
| `ctx.network.predict(input)` | Run inference using the current candidate network |
| `ctx.finish({ fitness, metrics })` | Signal that the session is complete and submit its result |
| `ctx.report(metrics)` | Publish optional progress metrics |
| `ctx.render(view)` | Publish a structured-clone-compatible visualization snapshot |
| `ctx.random()` | Seeded pseudo-random value for reproducible runs |
| `ctx.yield()` | Yield control so the scheduler/UI can process messages |
| `ctx.signal` | AbortSignal-like cancellation indicator |
| `ctx.session` | Read-only session metadata (generation, candidate index, seed, session ID) |
| `ctx.config` | User-defined environment configuration |

Fitness must be a finite number. Metrics should be structured clone-compatible data with size limits. The engine, not the environment, decides selection and reproduction.

### 4.3 Lifecycle semantics

For each candidate evaluation:

1. The scheduler creates an isolated session and binds the candidate network.
2. The runtime calls `sessionStart(ctx)`.
3. Environment code initializes state and runs its simulation.
4. The environment calls `ctx.network.predict(input)` whenever it needs an action/output.
5. The environment awaits `ctx.finish({ fitness, metrics })` exactly once when terminal.
6. The engine records the evaluation result and invokes optional `sessionEnd(ctx, result)`.
7. The scheduler disposes the session and proceeds according to the configured concurrency policy.

Failure paths:
- If `sessionStart` throws, mark the evaluation failed and record the error.
- If the environment returns without calling `finish`, mark it as incomplete unless the API explicitly defines a return-value completion mode.
- If a session exceeds its time/step budget, cancel it and record a timeout.
- If `finish` is called more than once, reject subsequent calls and report a contract error.
- Cancellation sends an abort signal to the Worker. Environments should observe `ctx.signal` and release their own resources in `finally` blocks; the host terminates the Worker if it does not cooperate.

### 4.4 Runtime isolation and trust

User code must execute outside the main UI thread. The initial runtime should use dedicated Web Workers and a message protocol with explicit request IDs for reports, finish acknowledgement, and cancellation. Network inference runs inside the environment Worker so repeated predictions do not block the UI thread.

Worker isolation protects UI responsiveness and limits accidental access to application state. It is not a complete security sandbox for hostile JavaScript. Do not market arbitrary user code as safely sandboxed. Before enabling execution of code from untrusted users or shared public environments, evaluate stronger isolation (for example, a separately-originated sandboxed frame with a restrictive policy) and document remaining risks.

No direct DOM access, application storage access, or unrestricted application object references should be exposed through the context API. Network access should be disabled by policy where a stronger isolation boundary is used; a Worker alone does not reliably prevent all network requests.

## 5. Neuroevolution engine

The neuroevolution engine is independent of any particular environment.

A Neural Network, a Playground, and a Training Run are separate persisted resources:
- A Neural Network stores architecture and parameters and can be reused by multiple Playgrounds.
- A Playground stores its source, versioned definition, input/output dimensions, and typed user parameters.
- A Training Run stores immutable snapshots of its Playground source, initial/best networks, evolution settings, environment parameters, seeds, and results.

Training selects a Playground first, then only offers networks whose input and output dimensions match the Playground definition. A completed best candidate can be saved as a new Neural Network without overwriting the source network.

Responsibilities:
- Initialize a population of candidate parameter sets compatible with one architecture.
- Evaluate each candidate by invoking the Playground session contract.
- Collect fitness and optional metrics.
- Apply selection, elitism, crossover, and mutation according to configuration.
- Track generation statistics and best-so-far candidates.
- Support pause, resume, stop, and cancellation.
- Retain selected checkpoints/candidates for saving to the Network Library.

The environment must not implement population management, mutation, selection, or generation progression.

Evaluation concurrency must be configurable and bounded by the device's available resources. Each candidate runs in its own Worker; population visualization may show a large grid, while the UI should batch/coalesce render updates to avoid excessive main-thread work. The evolution seed controls selection and mutation; a separate shared environment seed ensures candidates in the same comparison cohort see identical random conditions. Playground code should use the step-budget API to guarantee termination.

## 6. Supervised learning

Supervised training is a separate training algorithm that consumes a dataset and the canonical network model. It should share parameter storage, backend abstractions, metrics, checkpointing, and Network Library integration with neuroevolution.

Initial supervised scope is MLP backpropagation with a small set of loss functions and optimizers. The exact operator/optimizer matrix will be decided during implementation planning.

## 7. Persistence and project format

All project data must be locally persistable without a server.

- IndexedDB for projects, versioned Playgrounds, training runs, datasets, and saved networks.
- Versioned JSON for portable network/environment definitions.
- Project export/import should use a versioned bundle format.
- Runtime-only resources (Worker instances, WebNN contexts, compiled artifacts) must never be serialized.
- Saved network records should include architecture, parameters, model schema version, and optional training metadata.

## 8. Web application and GitHub Pages

- Build output must be static and deployable under a repository subpath.
- All asset paths and router behavior must support the GitHub Pages base path.
- No server-side routes or runtime API endpoints may be required.
- GitHub Actions should build and deploy the static site.
- The app should show browser capability information, including WebNN availability and selected backend.

## 9. Proposed source boundaries

These are conceptual boundaries; final folders may change as implementation begins.

- `core/model`: canonical model schema, validation, serialization.
- `core/engine`: graph/model execution contracts and tensor utilities.
- `backends/javascript`: baseline implementation.
- `backends/webnn`: optional WebNN adapter.
- `training/supervised`: supervised algorithms.
- `training/neuroevolution`: population/evolution algorithms.
- `playground/api`: environment contract and context types.
- `playground/runtime`: Worker protocol, scheduler, lifecycle.
- `playground/examples/snake`: first built-in environment, implemented through the public API.
- `storage`: IndexedDB and import/export.
- `ui`: editor, workspace, training, playground, and library screens.

## 10. Architectural proof

Before building the full UI, prove the generic contract with a minimal non-visual environment and Snake:

1. A tiny environment calls network inference multiple times.
2. It calls finish with fitness and metrics.
3. The engine evaluates more than one candidate and records results.
4. Snake uses exactly the same public contract.
5. A selected candidate is serialized, reloaded, and run in standalone inference mode.

If Snake needs a special engine path, the abstraction is wrong and should be revised.
