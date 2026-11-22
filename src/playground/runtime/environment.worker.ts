import { compileMLP } from "../../core/engine";
import { createWebNNPredictor } from "../../core/engine/webnn";
import type { MLPNetwork } from "../../core/model";
import type { PlaygroundSessionMetadata, PlaygroundSessionResult } from "../api/types";

type StartMessage = {
  type: "start";
  source: string;
  network: MLPNetwork;
  session: PlaygroundSessionMetadata;
  config: Record<string, unknown>;
  renderEnabled: boolean;
  reportEnabled: boolean;
  inferenceBackend: "auto" | "npu" | "low-power" | "high-performance" | "webnn-cpu" | "javascript";
};
type IncomingMessage =
  | StartMessage
  | { type: "finish-ack" }
  | { type: "cancel"; reason: string };
type OutgoingMessage =
  | { type: "backend"; backend: "webnn" | "javascript"; detail: string }
  | { type: "report"; metrics: Record<string, unknown> }
  | { type: "render"; view: unknown }
  | { type: "finish"; result: PlaygroundSessionResult }
  | { type: "cleanup-complete" }
  | { type: "incomplete" }
  | { type: "cancelled" }
  | { type: "error"; message: string; stack?: string };

interface WorkerScope {
  addEventListener(type: "message", listener: (event: MessageEvent<IncomingMessage>) => void): void;
  postMessage(message: OutgoingMessage): void;
}

const scope = self as unknown as WorkerScope;
let activeController: AbortController | null = null;
let pendingFinishResolve: (() => void) | null = null;
let finishAckPromise: Promise<void> | null = null;
let sessionRunning = false;

function serializeError(error: unknown): { message: string; stack?: string } {
  return {
    message: error instanceof Error ? error.message : String(error),
    stack: error instanceof Error ? error.stack : undefined,
  };
}

function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

async function runSession(message: StartMessage): Promise<void> {
  const controller = new AbortController();
  activeController = controller;
  const moduleUrl = URL.createObjectURL(
    new Blob([message.source], { type: "text/javascript" }),
  );

  let disposePredictor: (() => void) | undefined;
  try {
    let predictor: (input: readonly number[]) => number[] | Promise<number[]>;
    if (message.inferenceBackend !== "javascript") {
      try {
        const powerPreference = message.inferenceBackend === "low-power"
          ? "low-power"
          : message.inferenceBackend === "high-performance"
            ? "high-performance"
            : "default";
        const webnnPredictor = await createWebNNPredictor(message.network, {
          powerPreference,
          accelerated: message.inferenceBackend !== "webnn-cpu",
          ...(message.inferenceBackend === "npu" ? { deviceType: "npu" as const } : {}),
        });
        disposePredictor = webnnPredictor.dispose;
        predictor = async (input) => {
          try {
            return await webnnPredictor.predict(input);
          } catch (inferenceError) {
            try { disposePredictor?.(); } catch { /* Continue with the CPU fallback. */ }
            disposePredictor = undefined;
            const cpuPredictor = compileMLP(message.network);
            predictor = cpuPredictor;
            scope.postMessage({
              type: "backend",
              backend: "javascript",
              detail: `WebNN inference failed; switched to JavaScript CPU inference. ${inferenceError instanceof Error ? inferenceError.message : String(inferenceError)}`,
            });
            return cpuPredictor(input);
          }
        };
        const preferenceLabel = message.inferenceBackend === "npu"
          ? "Chromium-specific NPU device request (fallback may still occur)"
          : message.inferenceBackend === "low-power"
          ? "low-power preference"
          : message.inferenceBackend === "high-performance"
            ? "high-performance preference"
            : message.inferenceBackend === "webnn-cpu"
              ? "CPU preference"
              : "automatic device selection";
        scope.postMessage({
          type: "backend",
          backend: "webnn",
          detail: message.inferenceBackend === "npu"
            ? `Requested NPU using Chromium-specific deviceType: npu. WebNN does not expose a portable confirmation of the physical device; the browser may fall back to GPU or CPU. Check Windows Task Manager or Chromium WebNN diagnostics to verify actual NPU use.`
            : webnnPredictor.accelerated
            ? `WebNN reports accelerated=true with ${preferenceLabel}. The browser selected an accelerated path but does not expose whether it is a GPU or NPU.`
            : `WebNN reports accelerated=false with ${preferenceLabel}; this context prefers CPU inference.`,
        });
      } catch (webnnError) {
        predictor = compileMLP(message.network);
        scope.postMessage({ type: "backend", backend: "javascript", detail: `WebNN unavailable; using JavaScript CPU inference. ${webnnError instanceof Error ? webnnError.message : String(webnnError)}` });
      }
    } else {
      predictor = compileMLP(message.network);
      scope.postMessage({ type: "backend", backend: "javascript", detail: "JavaScript CPU inference selected." });
    }
    const environment = await import(/* @vite-ignore */ moduleUrl);
    if (controller.signal.aborted) {
      scope.postMessage({ type: "cancelled" });
      return;
    }
    if (typeof environment.sessionStart !== "function") {
      throw new Error("Environment must export sessionStart(context).");
    }

    let sessionResult: PlaygroundSessionResult | null = null;
    finishAckPromise = new Promise((resolve) => {
      pendingFinishResolve = resolve;
    });

    let stepCount = 0;
    let lastRenderAt = Number.NEGATIVE_INFINITY;
    const configuredMaxSteps = Number(message.config.maxSteps);
    const maxSteps = Number.isInteger(configuredMaxSteps) && configuredMaxSteps > 0
      ? configuredMaxSteps
      : Number.POSITIVE_INFINITY;
    const stepDelay = () => new Promise<void>((resolve) => {
      const delay = Math.max(0, Math.min(2000, Number(message.config.stepDelayMs) || 0));
      setTimeout(resolve, delay);
    });

    const context = {
      network: {
        async predict(input: readonly number[]): Promise<number[]> {
          if (!Array.isArray(input) || input.some((value) => !Number.isFinite(value))) {
            throw new Error("Network input must be an array of finite numbers.");
          }
          return await predictor(input);
        },
      },
      finish(result: PlaygroundSessionResult): Promise<void> {
        if (sessionResult) throw new Error("finish() may only be called once.");
        if (!result || typeof result.fitness !== "number" || !Number.isFinite(result.fitness)) {
          throw new Error("finish() requires a finite numeric fitness value.");
        }
        sessionResult = result;
        scope.postMessage({ type: "finish", result });
        return finishAckPromise!;
      },
      report(metrics: Record<string, unknown>): void {
        if (!metrics || typeof metrics !== "object" || Array.isArray(metrics)) {
          throw new Error("report() expects a metrics object.");
        }
        if (message.reportEnabled) scope.postMessage({ type: "report", metrics });
      },
      render(view: unknown): void {
        if (!message.renderEnabled) return;
        const intervalMs = Math.max(0, Number(message.config.renderIntervalMs) || 0);
        const now = performance.now();
        if (now - lastRenderAt < intervalMs) return;
        lastRenderAt = now;
        scope.postMessage({ type: "render", view });
      },
      random: createSeededRandom(message.session.seed),
      step: async (): Promise<boolean> => {
        stepCount += 1;
        await stepDelay();
        return stepCount < maxSteps;
      },
      yield: stepDelay,
      signal: controller.signal,
      session: Object.freeze({ ...message.session }),
      config: Object.freeze({ ...message.config }),
    };

    await environment.sessionStart(context);

    const completedResult = sessionResult;
    if (!completedResult) {
      scope.postMessage({ type: controller.signal.aborted ? "cancelled" : "incomplete" });
      return;
    }

    await finishAckPromise;
    if (typeof environment.sessionEnd === "function") {
      await environment.sessionEnd(context, completedResult);
    }
    scope.postMessage({ type: "cleanup-complete" });
  } catch (error) {
    if (controller.signal.aborted) {
      scope.postMessage({ type: "cancelled" });
    } else {
      scope.postMessage({ type: "error", ...serializeError(error) });
    }
  } finally {
    try { disposePredictor?.(); } catch { /* Worker termination remains the final cleanup boundary. */ }
    URL.revokeObjectURL(moduleUrl);
    activeController = null;
    pendingFinishResolve = null;
    finishAckPromise = null;
    sessionRunning = false;
  }
}

scope.addEventListener("message", (event) => {
  const message = event.data;
  if (message.type === "cancel") {
    activeController?.abort(message.reason);
    return;
  }
  if (message.type === "finish-ack") {
    pendingFinishResolve?.();
    return;
  }
  if (message.type === "start") {
    if (sessionRunning) {
      scope.postMessage({ type: "error", message: "This Worker already has an active session." });
      return;
    }
    sessionRunning = true;
    void runSession(message);
  }
});
