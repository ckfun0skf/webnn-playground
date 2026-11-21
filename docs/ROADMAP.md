# Roadmap

The roadmap is organized around working vertical slices. The generic Playground contract is the highest-risk architectural item and must be proven early.

## Phase 0 — Foundation
- [x] Create repository README and product requirements.
- [x] Document the client-side architecture and generic Playground lifecycle.
- [x] Choose frontend/build tooling and establish project conventions (React, TypeScript, Vite).
- [x] Add GitHub Actions build, test, and GitHub Pages deployment workflow.
- [x] Add a minimal responsive application shell and capability report.

## Phase 1 — Core network and inference
- [x] Define and validate the versioned canonical network schema (MLP schema v1).
- [x] Implement MLP model construction and parameter initialization.
- [x] Implement JavaScript inference backend and activation functions.
- [x] Add unit tests for forward propagation and parameter shape validation.
- [x] Build basic inference UI for creating an MLP, entering inputs, and inspecting outputs.
- [ ] Add an editable visual network graph with layer/node/weight/bias controls.
- [ ] Add a code representation/editor backed by the same canonical model.
- [ ] Verify the build and test workflow on GitHub Actions.

## Phase 2 — Generic Playground runtime proof
- [x] Define and document the initial environment module API.
- [x] Implement Worker message protocol and session lifecycle.
- [x] Implement inference requests from environment code to the active candidate network.
- [x] Implement finish, fitness, metrics, cancellation, timeout, and error handling.
- [ ] Add a tiny non-game end-to-end environment test.
- [x] Implement deterministic seeded random values and unit tests.
- [ ] Verify session isolation and repeatable seeded runs in a browser.
- [x] Add a basic environment code editor and runtime logs.

## Phase 3 — Neuroevolution
- [ ] Implement population initialization.
- [ ] Implement fitness collection and generation statistics.
- [ ] Implement selection, elitism, mutation, and optional crossover.
- [ ] Add configurable population/evaluation settings.
- [ ] Add pause, resume, stop, and bounded evaluation concurrency.
- [ ] Add candidate history and best-network checkpointing.
- [ ] Validate the engine with the non-game environment.

## Phase 4 — Snake as a built-in Playground
- [ ] Implement Snake as a normal environment module using the public contract.
- [ ] Add a visual board renderer driven by environment state/events.
- [ ] Add Snake-specific configuration and fitness metrics.
- [ ] Train Snake agents through the generic neuroevolution engine.
- [ ] Select, save, reload, and run a trained Snake network independently.
- [ ] Confirm there is no Snake-specific path inside the evolution engine.

## Phase 5 — Persistence and project workspace
- [ ] Add IndexedDB persistence for projects, environments, and networks.
- [ ] Add versioned JSON import/export.
- [ ] Add Network Library and saved candidate management.
- [ ] Add multiple projects/tabs and unsaved-change handling.
- [ ] Add project bundle export/import.

## Phase 6 — Supervised learning
- [ ] Add dataset creation/import.
- [ ] Implement MLP backpropagation.
- [ ] Add loss functions and initial optimizers.
- [ ] Add training controls, metrics charts, and checkpoints.
- [ ] Reuse canonical model and Network Library.

## Phase 7 — WebNN backend
- [ ] Detect WebNN capabilities at runtime.
- [ ] Define supported operation mapping and explicit fallback behavior.
- [ ] Implement optional WebNN inference adapter.
- [ ] Add backend selection and correctness comparison.
- [ ] Add performance comparison with documented measurement methodology.

## Phase 8 — Extensibility
- [ ] Add additional layer/operator abstractions.
- [ ] Evaluate CNN and other architecture support.
- [ ] Expand built-in environment examples (for example, CartPole or pathfinding).
- [ ] Revisit stronger isolation for importing/running untrusted community environments.

## Release gates

A phase is not complete merely because its UI exists. Each phase requires automated tests for its core behavior. The first meaningful release requires the complete Phase 0–5 flow: create a network, train it in a generic environment, save a candidate, reload it, and run it again from GitHub Pages.
