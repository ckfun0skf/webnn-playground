import { afterEach, describe, expect, it, vi } from "vitest";
import { createMLP } from "../../core/model";
import { runPlaygroundSession } from "./runner";

type FakeStartMessage = {
  type: "start";
  source: string;
  network: ReturnType<typeof createMLP>;
  session: { sessionId: string; generation: number; candidateIndex: number; seed: number };
  config: Record<string, unknown>;
};

class FakeWorker {
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  terminated = false;
  startMessage: FakeStartMessage | undefined;

  postMessage(message: { type: string; [key: string]: unknown }) {
    if (message.type === "start") {
      this.startMessage = message as FakeStartMessage;
      queueMicrotask(() => this.onmessage?.({
        data: { type: "finish", result: { fitness: 0.875, metrics: { score: 7 } } },
      } as MessageEvent));
    } else if (message.type === "finish-ack") {
      queueMicrotask(() => this.onmessage?.({
        data: { type: "cleanup-complete" },
      } as MessageEvent));
    } else if (message.type === "cancel") {
      queueMicrotask(() => this.onmessage?.({
        data: { type: "cancelled" },
      } as MessageEvent));
    }
  }

  terminate() {
    this.terminated = true;
  }
}

describe("runPlaygroundSession", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("passes the candidate network and resolves the environment fitness result", async () => {
    let worker: FakeWorker | undefined;
    vi.stubGlobal("Worker", class extends FakeWorker {
      constructor() {
        super();
        worker = this;
      }
    });

    const network = createMLP({
      id: "candidate-7",
      inputSize: 2,
      layers: [{ units: 1 }],
      random: () => 0.5,
    });
    const reports: Record<string, unknown>[] = [];

    const result = await runPlaygroundSession({
      source: "export async function sessionStart(ctx) {}",
      network,
      session: { sessionId: "test", generation: 2, candidateIndex: 7, seed: 42 },
      config: { scenario: "smoke-test" },
      onReport: (metrics) => reports.push(metrics),
    });

    expect(worker?.startMessage?.network.id).toBe("candidate-7");
    expect(worker?.startMessage?.session.candidateIndex).toBe(7);
    expect(result).toEqual({ fitness: 0.875, metrics: { score: 7 } });
    expect(worker?.terminated).toBe(true);
    expect(reports).toEqual([]);
  });

  it("rejects when the environment ends without submitting fitness", async () => {
    vi.stubGlobal("Worker", class extends FakeWorker {
      postMessage(message: { type: string; [key: string]: unknown }) {
        if (message.type === "start") {
          queueMicrotask(() => this.onmessage?.({
            data: { type: "incomplete" },
          } as MessageEvent));
        }
      }
    });

    const network = createMLP({
      inputSize: 2,
      layers: [{ units: 1 }],
      random: () => 0.5,
    });

    await expect(runPlaygroundSession({
      source: "export async function sessionStart(ctx) {}",
      network,
      session: { sessionId: "incomplete", generation: 0, candidateIndex: 0, seed: 1 },
    })).rejects.toThrow(/without calling finish/);
  });

  it("forwards cancellation to the environment Worker", async () => {
    let worker: FakeWorker | undefined;
    vi.stubGlobal("Worker", class extends FakeWorker {
      constructor() {
        super();
        worker = this;
      }
      postMessage(message: { type: string; [key: string]: unknown }) {
        if (message.type === "cancel") {
          queueMicrotask(() => this.onmessage?.({
            data: { type: "cancelled" },
          } as MessageEvent));
        }
      }
    });
    const controller = new AbortController();
    const network = createMLP({
      inputSize: 2,
      layers: [{ units: 1 }],
      random: () => 0.5,
    });

    const result = runPlaygroundSession({
      source: "export async function sessionStart(ctx) {}",
      network,
      session: { sessionId: "cancel", generation: 0, candidateIndex: 0, seed: 1 },
      signal: controller.signal,
    });
    controller.abort("test cancellation");

    await expect(result).rejects.toThrow("test cancellation");
    expect(worker?.terminated).toBe(true);
  });
});
