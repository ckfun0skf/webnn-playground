# WebNN Playground

A browser-based environment for building, running, training, and experimenting with neural networks.

The project is designed to run as a static site on GitHub Pages. All computation happens in the browser: a JavaScript execution backend provides broad compatibility, while WebNN is an optional backend for supported operations and browsers.

## Product direction

WebNN Playground is an experimentation platform, not just a neural-network visualizer or a collection of demos.

- Build neural networks with a visual editor and a code editor.
- Run inference and inspect network behavior.
- Train networks with supervised learning and neuroevolution.
- Create generic simulation environments in user-authored JavaScript.
- Use trained networks inside those environments, save selected networks, and run them again.
- Compare JavaScript and WebNN execution where supported.

## Generic Playground

A Playground is a user-authored JavaScript environment. It is not restricted to games or visual board editors.

For neuroevolution, the framework owns the population and evaluation lifecycle. Environment code defines how a session starts, how the simulation runs, how it invokes the current network, and when/how it reports the session's fitness.

The intended lifecycle is:

1. The engine starts an evaluation session for one candidate network.
2. The environment initializes its session state.
3. The environment requests network inference as often as needed.
4. The environment ends the session and reports fitness plus optional metrics.
5. The engine records the result and continues with the population/generation.

See [Playground API](docs/PLAYGROUND_API.md) for the environment contract and [Architecture](docs/ARCHITECTURE.md) for runtime boundaries.

## Current core APIs

- `createMLP(options)`: create a validated MLP with initialized parameters.
- `updateLayerParameters(network, layerId, update)`: immutably update layer weights, biases, or activation.
- `validateNetwork(network)`: inspect schema and parameter-shape errors.
- `compileMLP(network)`: compile a reusable JavaScript forward-pass predictor.
- `runPlaygroundSession(options)`: execute a user-authored environment in a Worker with network inference, fitness reporting, cancellation, and timeout handling.

The Network workspace includes a first working inference path with editable layer weights, biases, and activation functions. The Playground workspace includes a JavaScript environment editor wired to the Worker session runtime, progress reports, cancellation, and fitness output. Automated unit tests cover MLP construction, forward propagation, activation behavior, validation, seeded randomness, and Worker bootstrap syntax.

## Documentation

- [Product Requirements](docs/PRD.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Playground Environment API](docs/PLAYGROUND_API.md)
- [Roadmap](docs/ROADMAP.md)

## Deployment

The application must be deployable to GitHub Pages without a backend server. User code and simulations execute client-side. Browser storage is used for local projects and saved networks, with import/export for portability.

## Current implementation

The repository now contains the React + TypeScript + Vite workspace shell, an initial MLP model and JavaScript inference engine, editable network parameters, a basic interactive inference view, and a Worker-based generic Playground session runner with an environment code editor. Neuroevolution orchestration, the full visual network editor, environment code editor, persistence, and WebNN execution are still pending.
