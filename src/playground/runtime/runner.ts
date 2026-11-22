import type {
  PlaygroundSessionResult,
  RunPlaygroundSessionOptions,
} from "../api/types";
import type { WorkerMessage, WorkerRequest } from "./protocol";

function createWorker(): Worker {
  return new Worker(
    new URL("./environment.worker.ts", import.meta.url),
    { type: "module", name: "playground-environment" },
  );
}

export function runPlaygroundSession(
  options: RunPlaygroundSessionOptions,
): Promise<PlaygroundSessionResult> {
  if (typeof Worker === "undefined") {
    return Promise.reject(new Error("Web Workers are not supported in this browser."));
  }
  if (!options.source.trim()) {
    return Promise.reject(new Error("Environment source cannot be empty."));
  }

  return new Promise((resolve, reject) => {
    const worker = createWorker();
    let settled = false;
    let finishReceived = false;
    let sessionResult: PlaygroundSessionResult | null = null;
    let cancellationTimer: ReturnType<typeof setTimeout> | undefined;
    const timeoutMs = options.timeoutMs ?? 30_000;

    const cleanup = () => {
      clearTimeout(timeout);
      if (cancellationTimer) clearTimeout(cancellationTimer);
      options.signal?.removeEventListener("abort", onAbort);
      worker.terminate();
    };

    const fail = (error: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(error);
    };

    const cancellationError = () =>
      new Error(options.signal?.reason?.toString() ?? "Session cancelled.");
    const onAbort = () => {
      worker.postMessage({
        type: "cancel",
        reason: cancellationError().message,
      } satisfies WorkerRequest);
      cancellationTimer = setTimeout(() => fail(cancellationError()), 500);
    };
    const timeout = setTimeout(
      () => fail(new Error(`Session timed out after ${timeoutMs}ms.`)),
      timeoutMs,
    );

    if (options.signal?.aborted) {
      fail(cancellationError());
      return;
    }
    options.signal?.addEventListener("abort", onAbort, { once: true });

    worker.onerror = (event) =>
      fail(new Error(event.message || "Environment worker failed."));
    worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
      const message = event.data;
      if (message.type === "backend") {
        options.onBackend?.({ backend: message.backend, detail: message.detail });
      } else if (message.type === "report") {
        try {
          options.onReport?.(message.metrics);
        } catch (error) {
          fail(error instanceof Error ? error : new Error(String(error)));
        }
      } else if (message.type === "render") {
        try {
          options.onRender?.(message.view);
        } catch (error) {
          fail(error instanceof Error ? error : new Error(String(error)));
        }
      } else if (message.type === "finish") {
        if (finishReceived) {
          fail(new Error("Environment called finish() more than once."));
          return;
        }
        if (!Number.isFinite(message.result.fitness)) {
          fail(new Error("Environment returned a non-finite fitness score."));
          return;
        }
        finishReceived = true;
        sessionResult = message.result;
        worker.postMessage({ type: "finish-ack" } satisfies WorkerRequest);
      } else if (message.type === "cancelled") {
        fail(cancellationError());
      } else if (message.type === "incomplete") {
        fail(new Error("Environment session ended without calling finish()."));
      } else if (message.type === "error") {
        fail(new Error(message.stack ? `${message.message}\n${message.stack}` : message.message));
      } else if (message.type === "cleanup-complete") {
        if (!sessionResult) {
          fail(new Error("Environment cleanup completed without a fitness result."));
          return;
        }
        settled = true;
        cleanup();
        resolve(sessionResult);
      }
    };

    try {
      worker.postMessage({
        type: "start",
        source: options.source,
        network: options.network,
        session: options.session,
        config: options.config ?? {},
        renderEnabled: Boolean(options.onRender),
        reportEnabled: Boolean(options.onReport),
        inferenceBackend: options.inferenceBackend ?? "auto",
      } satisfies WorkerRequest);
    } catch (error) {
      fail(error instanceof Error ? error : new Error(String(error)));
    }
  });
}
