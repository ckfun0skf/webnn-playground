import { compileMLP } from "../../core/engine";
import type { MLPNetwork } from "../../core/model";
import type { PlaygroundSessionMetadata, PlaygroundSessionResult } from "../api/types";

type StartMessage = {
  type: "start";
  source: string;
  network: MLPNetwork;
  session: PlaygroundSessionMetadata;
  config: Record<string, unknown>;
};
type IncomingMessage =
  | StartMessage
  | { type: "finish-ack" }
  | { type: "cancel"; reason: string };
type OutgoingMessage =
  | { type: "report"; metrics: Record<string, unknown> }
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
let finishResult: PlaygroundSessionResult | null = null;
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

  try {
    const predictor = compileMLP(message.network);
    const environment = await import(/* @vite-ignore */ moduleUrl);
    if (controller.signal.aborted) {
      scope.postMessage({ type: "cancelled" });
      return;
    }
    if (typeof environment.sessionStart !== "function") {
      throw new Error("Environment must export sessionStart(context).");
    }

    finishResult = null;
    finishAckPromise = new Promise((resolve) => {
      pendingFinishResolve = resolve;
    });

    const context = {
      network: {
        async predict(input: readonly number[]): Promise<number[]> {
          if (!Array.isArray(input) || input.some((value) => !Number.isFinite(value))) {
            throw new Error("Network input must be an array of finite numbers.");
          }
          return predictor(input);
        },
      },
      finish(result: PlaygroundSessionResult): Promise<void> {
        if (finishResult) throw new Error("finish() may only be called once.");
        if (!result || typeof result.fitness !== "number" || !Number.isFinite(result.fitness)) {
          throw new Error("finish() requires a finite numeric fitness value.");
        }
        finishResult = result;
        scope.postMessage({ type: "finish", result });
        return finishAckPromise!;
      },
      report(metrics: Record<string, unknown>): void {
        if (!metrics || typeof metrics !== "object" || Array.isArray(metrics)) {
          throw new Error("report() expects a metrics object.");
        }
        scope.postMessage({ type: "report", metrics });
      },
      random: createSeededRandom(message.session.seed),
      yield: () => new Promise<void>((resolve) => setTimeout(resolve, 0)),
      signal: controller.signal,
      session: Object.freeze({ ...message.session }),
      config: Object.freeze({ ...message.config }),
    };

    await environment.sessionStart(context);

    if (!finishResult) {
      scope.postMessage({ type: controller.signal.aborted ? "cancelled" : "incomplete" });
      return;
    }

    await finishAckPromise;
    if (typeof environment.sessionEnd === "function") {
      await environment.sessionEnd(context, finishResult);
    }
    scope.postMessage({ type: "cleanup-complete" });
  } catch (error) {
    if (controller.signal.aborted) {
      scope.postMessage({ type: "cancelled" });
    } else {
      scope.postMessage({ type: "error", ...serializeError(error) });
    }
  } finally {
    URL.revokeObjectURL(moduleUrl);
    activeController = null;
    pendingFinishResolve = null;
    finishAckPromise = null;
    finishResult = null;
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
