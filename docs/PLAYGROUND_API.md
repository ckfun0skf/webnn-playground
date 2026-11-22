# Generic Playground Environment API

This document describes the initial JavaScript environment contract implemented by the browser runtime.

## Environment module

A Playground is an ES module supplied as JavaScript source. It must export `sessionStart(context)`; it may export `sessionEnd(context, result)`. It should also export a `playgroundDefinition` describing its stable ID, version, input/output dimensions, and user-configurable parameters. The Training workspace uses those dimensions to filter compatible Neural Networks and generates parameter controls from the definition.

```js
export const playgroundDefinition = {
  id: "snake",
  name: "Snake",
  version: 1,
  inputSize: 11,
  outputSize: 3,
  parameters: [
    { key: "maxStepWithoutApple", label: "Maximum steps without an apple", type: "number", default: 144, min: 1, max: 10000 },
    { key: "randomSeed", label: "Random seed", type: "randomSeed", default: 42 }
  ]
};
```

```js
export async function sessionStart(ctx) {
  const target = [0.25, 0.75];

  // A session may request inference as often as its simulation needs.
  const output = await ctx.network.predict([0, 1]);

  const error = output.reduce(
    (sum, value, index) => sum + (value - target[index]) ** 2,
    0,
  );

  ctx.report({ error });

  // Fitness is owned by the environment; the evolution engine only consumes it.
  await ctx.finish({
    fitness: 1 / (1 + error),
    metrics: { error },
  });
}

export async function sessionEnd(ctx, result) {
  // Optional per-session cleanup.
}
```

A session must call `ctx.finish({ fitness, metrics? })` exactly once. The fitness value must be a finite number. The environment should return from `sessionStart` after finishing; the runtime waits for the finish acknowledgement and then calls `sessionEnd`.

## Visualization contract

Environment JavaScript sends serializable view data through `ctx.render(view)`. It does not create DOM or React components. The host renders the data in the Playground panel; this keeps simulation code isolated in a Worker.

### Canvas view

Canvas is the flexible general-purpose renderer. It supports drawing commands plus optional header and footer labels:

```js
ctx.render({
  type: "canvas",
  width: 800,
  height: 500,
  background: "#101216",
  header: ["Generation 12", { label: "Agent", value: 4 }],
  commands: [
    { op: "fillRect", x: 20, y: 20, width: 40, height: 40, color: "#68d6a3" },
    { op: "line", x1: 20, y1: 80, x2: 200, y2: 80, color: "#9b8cff", lineWidth: 2 },
    { op: "circle", x: 100, y: 140, radius: 12, color: "#e7b66d" },
    { op: "text", text: "Agent 4", x: 20, y: 190, color: "#ffffff" }
  ],
  footer: ["Fitness: 24", "Steps: 180"]
});
```

Supported commands: `fillRect`, `strokeRect`, `line`, `circle`, and `text`. Coordinates are in canvas pixels. The command list is data, not executable code; the host interprets it on the Canvas 2D context.

### Built-in grid view

Use the grid renderer for cell-based environments without drawing each cell yourself:

```js
ctx.render({
  type: "grid", title: "Warehouse", width: 8, height: 6,
  cells: [
    { x: 1, y: 2, state: "active", label: "A" },
    { x: 5, y: 3, state: "target", label: "T" },
    { x: 3, y: 1, state: "wall" }
  ],
  header: ["Episode 3"], footer: ["Delivered: 4"]
});
```

Grid states: `active`, `target`, `wall`, `success`, `danger`, and `muted`. Grid views can also include optional `header` and `footer` label arrays.

### Metrics view

```js
ctx.render({
  type: "metrics", title: "Simulation status",
  metrics: [{ label: "Score", value: 12 }, { label: "Time", value: 34, unit: "s" }]
});
```

Unknown view types are shown as formatted JSON. Templates are ordinary Playground source files and metadata; the runtime does not contain template-specific or game-specific rendering logic.

## Context

| Member | Description |
|---|---|
| `ctx.network.predict(input)` | Asynchronously run the candidate network. Input must be a finite number array with the model's configured input length. |
| `ctx.finish({ fitness, metrics? })` | Complete this candidate evaluation. Returns a promise that resolves when the engine acknowledges the result. |
| `ctx.report(metrics)` | Send progress metrics to the host UI. |
| `ctx.render(view)` | Publish an environment-defined visualization snapshot to the host UI. The view can be any structured-clone-compatible data; a renderer can interpret its shape. |
| `ctx.random()` | Deterministic pseudo-random number based on the session seed. |
| `ctx.step()` | Advance one simulation step, apply the configured playback delay, and return `false` when the optional runtime safety step budget is reached. Environment-specific limits (such as Snake’s `maxStepWithoutApple`) belong in Playground parameters and are enforced by the environment code. |
| `ctx.yield()` | Yield to the Worker event loop without consuming a simulation step. |
| `ctx.signal` | AbortSignal for cooperative cancellation. |
| `ctx.session` | Read-only session metadata: ID, generation, candidate index, and seed. |
| `ctx.config` | Read-only Playground parameters and runtime settings. |

## Host API

`runPlaygroundSession(options)` runs one evaluation in a dedicated Worker and returns a promise for the session result.

Options:
- `source`: complete environment ES module source.
- `network`: validated canonical MLP network.
- `session`: session ID, generation, candidate index, and seed.
- `config`: optional Playground parameters, plus runtime settings such as `stepDelayMs` and an optional safety step budget.
- `session.seed`: deterministic environment seed. Neuroevolution resolves a `randomSeed` parameter once per generation and supplies the same seed to every candidate in that generation; the seed changes deterministically between generations.
- `timeoutMs`: optional session time limit (default 30 seconds).
- `signal`: optional AbortSignal for cancellation.
- `onReport`: optional callback for progress metrics.
- `onRender`: optional callback for visualization snapshots.

The environment and candidate network both execute inside the dedicated Worker. The Worker compiles the candidate model and exposes only the `ctx.network.predict(input)` function to the environment module; the main UI thread receives progress reports, visualization snapshots, and the final result. This keeps repeated inference off the UI thread.

## Lifecycle and failure behavior

1. The host creates a Worker and imports the environment module.
2. The runtime calls `sessionStart(ctx)`.
3. The Worker executes each `ctx.network.predict` call through its compiled JavaScript inference function.
4. The environment reports progress and eventually calls `ctx.finish`.
5. The host validates and acknowledges fitness.
6. The runtime calls optional `sessionEnd(ctx, result)`.
7. The host resolves the session promise and terminates the Worker.

A missing finish call, thrown error, invalid fitness, timeout, or cancellation rejects the host promise. Environments should use `await ctx.step()` in their simulation loop to enforce a finite step budget. Fitness is entirely environment-defined; simulation steps do not automatically contribute to fitness. Worker termination prevents a stuck simulation from blocking the main UI thread.

## Trust boundary

The Worker is a responsiveness and lifecycle boundary, not a complete security sandbox. Worker code can still use browser APIs such as network requests. Do not execute untrusted third-party environment code without a stronger isolation design and an explicit permission model.


## Random-seed parameters

A parameter with `type: "randomSeed"` is an integer base seed for environment randomness. During a standalone Playground run, the configured value is used unchanged. During Neuroevolution, the engine derives a deterministic seed for each generation:

```ts
generationSeed = (baseSeed + Math.imul(generation, 0x9e3779b9)) >>> 0;
```

All candidates in one generation receive the same `generationSeed` as both `ctx.session.seed` and `ctx.config.randomSeed`. Different generations receive different seeds. Re-running a training run with the same base seed reproduces the same sequence of generation seeds.
