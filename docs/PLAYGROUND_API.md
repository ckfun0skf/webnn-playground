# Generic Playground Environment API

This document describes the initial JavaScript environment contract implemented by the browser runtime.

## Environment module

An environment is an ES module supplied as JavaScript source. It must export `sessionStart(context)`. It may export `sessionEnd(context, result)` for cleanup.

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

## Context

| Member | Description |
|---|---|
| `ctx.network.predict(input)` | Asynchronously run the candidate network. Input must be a finite number array with the model's configured input length. |
| `ctx.finish({ fitness, metrics? })` | Complete this candidate evaluation. Returns a promise that resolves when the engine acknowledges the result. |
| `ctx.report(metrics)` | Send progress metrics to the host UI. |
| `ctx.random()` | Deterministic pseudo-random number based on the session seed. |
| `ctx.yield()` | Yield to the Worker event loop. |
| `ctx.signal` | AbortSignal for cooperative cancellation. |
| `ctx.session` | Read-only session metadata: ID, generation, candidate index, and seed. |
| `ctx.config` | Read-only top-level environment configuration. |

## Host API

`runPlaygroundSession(options)` runs one evaluation in a dedicated Worker and returns a promise for the session result.

Options:
- `source`: complete environment ES module source.
- `network`: validated canonical MLP network.
- `session`: session ID, generation, candidate index, and seed.
- `config`: optional environment configuration.
- `timeoutMs`: optional session time limit (default 30 seconds).
- `signal`: optional AbortSignal for cancellation.
- `onReport`: optional callback for progress metrics.

The runner compiles the candidate network on the host side and services inference requests from the Worker. The environment never receives the model parameters or a direct reference to the host predictor.

## Lifecycle and failure behavior

1. The host creates a Worker and imports the environment module.
2. The runtime calls `sessionStart(ctx)`.
3. Each `ctx.network.predict` request is serviced by the host's JavaScript inference backend.
4. The environment reports progress and eventually calls `ctx.finish`.
5. The host validates and acknowledges fitness.
6. The runtime calls optional `sessionEnd(ctx, result)`.
7. The host resolves the session promise and terminates the Worker.

A missing finish call, thrown error, invalid fitness, timeout, or cancellation rejects the host promise. Worker termination prevents a stuck simulation from blocking the main UI thread.

## Trust boundary

The Worker is a responsiveness and lifecycle boundary, not a complete security sandbox. Worker code can still use browser APIs such as network requests. Do not execute untrusted third-party environment code without a stronger isolation design and an explicit permission model.
