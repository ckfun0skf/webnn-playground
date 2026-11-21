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

See [Architecture](docs/ARCHITECTURE.md) for the proposed environment contract and runtime boundaries.

## Documentation

- [Product Requirements](docs/PRD.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)

## Deployment

The application must be deployable to GitHub Pages without a backend server. User code and simulations execute client-side. Browser storage is used for local projects and saved networks, with import/export for portability.

## Status

Project foundation and architecture definition. Implementation has not started.
