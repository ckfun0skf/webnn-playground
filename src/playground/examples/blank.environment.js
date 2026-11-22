export async function sessionStart(ctx) {
  const inputSize = Number(ctx.config.inputSize);
  const outputSize = Number(ctx.config.outputSize);
  let totalError = 0;

  // This example evaluates the network on three generated observations.
  for (let step = 0; step < 3; step += 1) {
    if (ctx.signal.aborted) return;

    const input = Array.from(
      { length: inputSize },
      (_, index) => index === step % inputSize ? 1 : 0
    );
    const target = Array.from(
      { length: outputSize },
      (_, index) => index === step % outputSize ? 1 : 0
    );

    const output = await ctx.network.predict(input);
    const error = output.reduce(
      (sum, value, index) => sum + (value - target[index]) ** 2,
      0
    );

    totalError += error;
    ctx.report({ step: step + 1, error, totalError });
    ctx.render({ type: "observation", step: step + 1, input, output, target, error });
    await ctx.yield();
  }

  await ctx.finish({
    fitness: 1 / (1 + totalError),
    metrics: { totalError, observations: 3 }
  });
}

export async function sessionEnd(ctx, result) {
  // Optional cleanup after a successful evaluation.
}