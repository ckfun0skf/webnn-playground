import { afterEach, describe, expect, it, vi } from "vitest";
import { createMLP, updateLayerParameters } from "../../core/model";
import { runPlaygroundSession } from "./runner";

class FakeWorker {
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  terminated = false;
  lastOutput: number[] | undefined;

  postMessage(message: { type: string; requestId?: number; input?: number[]; output?: number[] }) {
    if (message.type === "start") {
      queueMicrotask(() => this.onmessage?.({
        data: { type: "predict", requestId: 1, input: [3, 4] },
      } as MessageEvent));
    } else if (message.type === "infer-result") {
      this.lastOutput = message.output;
      queueMicrotask(() => this.onmessage?.({
        data: {
          type: "finish",
          result: { fitness: message.output?.[0] ?? -1, metrics: { output: message.output } },
        },
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
    vi.restoreAllMocks();
  });

  it("routes Worker inference to the candidate network and returns fitness", async () => {
    let worker: FakeWorker | undefined;
    vi.stubGlobal("Worker", class extends FakeWorker {
      constructor() {
        super();
        worker = this;
      }
    });
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:fake-worker");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined);

    const network = createMLP({
      inputSize: 2,
      layers: [{ units: 1, activation: "linear" }],
      random: () => 0.5,
    });
    const configured = updateLayerParameters(network, network.layers[0].id, {
      weights: [[1, 2]],
      biases: [0],
    });

    const result = await runPlaygroundSession({
      source: "export async function sessionStart(ctx) {}",
      network: configured,
      session: { sessionId: "test", generation: 0, candidateIndex: 0, seed: 42 },
    });

    expect(worker?.lastOutput).toEqual([11]);
    expect(result.fitness).toBe(11);
    expect(result.metrics).toEqual({ output: [11] });
    expect(worker?.terminated).toBe(true);
  });

  it("rejects when the environment ends without submitting fitness", async () => {
    vi.stubGlobal("Worker", class extends FakeWorker {
      postMessage(message: { type: string }) {
        if (message.type === "start") {
          queueMicrotask(() => this.onmessage?.({
            data: { type: "incomplete" },
          } as MessageEvent));
        }
      }
    });
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:fake-worker");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined);

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
});
