import { beforeEach, describe, expect, it, vi } from "vitest";
import { createMLP } from "../../core/model";
import { runPlaygroundSession } from "../../playground/runtime";
import { DEFAULT_NEUROEVOLUTION_CONFIG, runNeuroevolution } from "./engine";

vi.mock("../../playground/runtime", () => ({
  runPlaygroundSession: vi.fn(async ({ network }: { network: ReturnType<typeof createMLP> }) => ({
    fitness: network.layers[0].weights[0][0],
    metrics: { evaluated: true },
  })),
}));

describe("runNeuroevolution", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("evaluates every candidate across generations and returns the best network", async () => {
    const initialNetwork = createMLP({
      id: "seed",
      inputSize: 2,
      layers: [{ units: 2, activation: "tanh" }, { units: 1, activation: "linear" }],
      random: () => 0.5,
    });
    const generationEvents: number[] = [];
    const candidateEvents: number[] = [];

    const result = await runNeuroevolution({
      initialNetwork,
      environmentSource: "export async function sessionStart(ctx) {}",
      config: {
        populationSize: 4,
        generations: 3,
        eliteCount: 1,
        tournamentSize: 2,
        crossoverRate: 0.7,
        mutationRate: 0.2,
        mutationScale: 0.1,
        evaluationTimeoutMs: 1000,
        seed: 17,
        environmentSeed: 1234,
        includeInitialNetwork: true,
        maxConcurrentEvaluations: 2,
      },
      onCandidate: (candidate) => candidateEvents.push(candidate.candidateIndex),
      onGeneration: (summary) => generationEvents.push(summary.generation),
    });

    expect(vi.mocked(runPlaygroundSession)).toHaveBeenCalledTimes(12);
    for (const [callIndex, call] of vi.mocked(runPlaygroundSession).mock.calls.entries()) {
      const generation = Math.floor(callIndex / 4);
      const expectedSeed = (1234 + Math.imul(generation, 0x9e3779b9)) >>> 0;
      expect(call[0].session.seed).toBe(expectedSeed);
      expect(call[0].config?.randomSeed).toBe(expectedSeed);
      if (callIndex < 4) expect(call[0].session.generation).toBe(0);
    }
    expect(result.history).toHaveLength(3);
    expect(generationEvents).toEqual([0, 1, 2]);
    expect(candidateEvents).toHaveLength(12);
    expect(result.best.network).toBeDefined();
    expect(Number.isFinite(result.best.fitness)).toBe(true);
  });

  it("rejects invalid population and mutation settings", async () => {
    const initialNetwork = createMLP({
      inputSize: 1,
      layers: [{ units: 1 }],
      random: () => 0.5,
    });
    const config = {
      populationSize: 1,
      generations: 1,
      eliteCount: 1,
      tournamentSize: 1,
      crossoverRate: 0.7,
      mutationRate: 0.1,
      mutationScale: 0.1,
      evaluationTimeoutMs: 1000,
      seed: 1,
      environmentSeed: 1,
      includeInitialNetwork: true,
      maxConcurrentEvaluations: 2,
    };

    await expect(runNeuroevolution({
      initialNetwork,
      environmentSource: "export async function sessionStart(ctx) {}",
      config,
    })).rejects.toThrow(/populationSize/);

    expect(runPlaygroundSession).not.toHaveBeenCalled();
  });

  it("limits concurrent candidate sessions to the configured worker budget", async () => {
    const initialNetwork = createMLP({
      inputSize: 1,
      layers: [{ units: 1 }],
      random: () => 0.5,
    });
    let active = 0;
    let peak = 0;
    vi.mocked(runPlaygroundSession).mockImplementation(async ({ network }) => {
      active += 1;
      peak = Math.max(peak, active);
      await new Promise<void>((resolve) => setTimeout(resolve, 5));
      active -= 1;
      return { fitness: network.layers[0].weights[0][0] };
    });

    await runNeuroevolution({
      initialNetwork,
      environmentSource: "export async function sessionStart(ctx) {}",
      config: { ...DEFAULT_NEUROEVOLUTION_CONFIG, populationSize: 6, generations: 1, eliteCount: 1, maxConcurrentEvaluations: 2 },
    });

    expect(peak).toBe(2);
  });
});
