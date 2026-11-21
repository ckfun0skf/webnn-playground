export const PLAYGROUND_WORKER_SOURCE = `
let activeController;
let pendingFinishResolve;
let finished = false;

function serializeError(error) {
  return {
    message: error instanceof Error ? error.message : String(error),
    stack: error instanceof Error ? error.stack : undefined
  };
}

self.addEventListener("message", async (event) => {
  const message = event.data;
  if (message.type === "cancel") {
    activeController?.abort(message.reason);
    return;
  }
  if (message.type !== "start") return;

  const moduleUrl = URL.createObjectURL(
    new Blob([message.source], { type: "text/javascript" })
  );

  try {
    const environment = await import(moduleUrl);
    if (typeof environment.sessionStart !== "function") {
      throw new Error("Environment must export sessionStart(context).");
    }

    activeController = new AbortController();
    finished = false;
    let requestId = 0;
    const pendingPredictions = new Map();

    const context = {
      network: {
        predict(input) {
          if (!Array.isArray(input) || input.some((value) => !Number.isFinite(value))) {
            return Promise.reject(new Error("Network input must be an array of finite numbers."));
          }
          const id = ++requestId;
          self.postMessage({ type: "predict", requestId: id, input });
          return new Promise((resolve, reject) => pendingPredictions.set(id, { resolve, reject }));
        }
      },
      finish(result) {
        if (finished) return Promise.reject(new Error("finish() may only be called once."));
        if (!result || typeof result.fitness !== "number" || !Number.isFinite(result.fitness)) {
          return Promise.reject(new Error("finish() requires a finite numeric fitness value."));
        }
        finished = true;
        self.postMessage({ type: "finish", result });
        return new Promise((resolve) => { pendingFinishResolve = resolve; });
      },
      report(metrics) {
        if (!metrics || typeof metrics !== "object" || Array.isArray(metrics)) {
          throw new Error("report() expects a metrics object.");
        }
        self.postMessage({ type: "report", metrics });
      },
      random: (() => {
        let state = (message.session.seed >>> 0);
        return () => {
          state = (state + 0x6d2b79f5) >>> 0;
          let value = state;
          value = Math.imul(value ^ (value >>> 15), value | 1);
          value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
          return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
        };
      })(),
      yield: () => new Promise((resolve) => setTimeout(resolve, 0)),
      signal: activeController.signal,
      session: Object.freeze({ ...message.session }),
      config: Object.freeze({ ...message.config })
    };

    self.addEventListener("message", (responseEvent) => {
      const response = responseEvent.data;
      if (response.type === "infer-result") {
        const pending = pendingPredictions.get(response.requestId);
        if (!pending) return;
        pendingPredictions.delete(response.requestId);
        if (response.error) pending.reject(new Error(response.error));
        else pending.resolve(response.output);
      }
      if (response.type === "finish-ack") pendingFinishResolve?.();
      if (response.type === "cancel") activeController.abort(response.reason);
    });

    await environment.sessionStart(context);
    if (!finished) {
      self.postMessage({ type: "incomplete" });
      return;
    }

    await new Promise((resolve) => {
      if (!pendingFinishResolve) resolve();
      else {
        const previousResolve = pendingFinishResolve;
        pendingFinishResolve = () => { previousResolve(); resolve(); };
      }
    });

    if (typeof environment.sessionEnd === "function") {
      await environment.sessionEnd(context, message.__result);
    }
    self.postMessage({ type: "cleanup-complete" });
  } catch (error) {
    self.postMessage({ type: "error", ...serializeError(error) });
  } finally {
    URL.revokeObjectURL(moduleUrl);
  }
});
`;
