import { createSeededRandom } from "../../playground/runtime/random";
import { runPlaygroundSession } from "../../playground/runtime";
import type { MLPNetwork } from "../../core/model";
import {
  cloneNetwork,
  createRandomNetwork,
  crossoverNetworks,
  mutateNetwork,
} from "./genetics";
import type {
  CandidateEvaluation,
  GenerationSummary,
  NeuroevolutionResult,
  RunNeuroevolutionOptions,
} from "./types";

export const DEFAULT_NEUROEVOLUTION_CONFIG = {
  populationSize: 24,
  generations: 20,
  eliteCount: 2,
  tournamentSize: 3,
  crossoverRate: 0.7,
  mutationRate: 0.08,
  mutationScale: 0.15,
  evaluationTimeoutMs: 30_000,
  seed: 42,
  includeInitialNetwork: true,
} as const;

function validateOptions(options: RunNeuroevolutionOptions): void {
  const { config } = options;
  if (!options.environmentSource.trim()) {
    throw new Error("Environment source cannot be empty.");
  }
  if (!Number.isInteger(config.populationSize) || config.populationSize < 2) {
    throw new Error("populationSize must be an integer of at least 2.");
  }
  if (!Number.isInteger(config.generations) || config.generations < 1) {
    throw new Error("generations must be a positive integer.");
  }
  if (!Number.isInteger(config.eliteCount) || config.eliteCount < 1 || config.eliteCount >= config.populationSize) {
    throw new Error("eliteCount must be at least 1 and smaller than populationSize.");
  }
  if (!Number.isInteger(config.tournamentSize) || config.tournamentSize < 1) {
    throw new Error("tournamentSize must be a positive integer.");
  }
  if (!Number.isFinite(config.crossoverRate) || config.crossoverRate < 0 || config.crossoverRate > 1) {
    throw new Error("crossoverRate must be between 0 and 1.");
  }
  if (!Number.isFinite(config.mutationRate) || config.mutationRate < 0 || config.mutationRate > 1) {
    throw new Error("mutationRate must be between 0 and 1.");
  }
  if (!Number.isFinite(config.mutationScale) || config.mutationScale < 0) {
    throw new Error("mutationScale must be a finite non-negative number.");
  }
  if (!Number.isInteger(config.seed)) {
    throw new Error("seed must be an integer.");
  }
  if (!Number.isFinite(config.evaluationTimeoutMs) || config.evaluationTimeoutMs <= 0) {
    throw new Error("evaluationTimeoutMs must be a positive number.");
  }
}

function selectTournament(
  population: CandidateEvaluation[],
  tournamentSize: number,
  random: () => number,
): CandidateEvaluation {
  let selected: CandidateEvaluation | undefined;
  for (let index = 0; index < tournamentSize; index += 1) {
    const candidate = population[Math.floor(random() * population.length)];
    if (!selected || candidate.fitness > selected.fitness) selected = candidate;
  }
  return selected!;
}

function summarizeGeneration(
  generation: number,
  candidates: CandidateEvaluation[],
): GenerationSummary {
  const fitness = candidates.map((candidate) => candidate.fitness);
  return {
    generation,
    bestFitness: Math.max(...fitness),
    meanFitness: fitness.reduce((sum, value) => sum + value / fitness.length, 0),
    worstFitness: Math.min(...fitness),
    successfulEvaluations: candidates.filter((candidate) => !candidate.error).length,
    failedEvaluations: candidates.filter((candidate) => Boolean(candidate.error)).length,
  };
}

function createNextGeneration(
  ranked: CandidateEvaluation[],
  config: RunNeuroevolutionOptions["config"],
  random: () => number,
  generation: number,
): MLPNetwork[] {
  const next: MLPNetwork[] = ranked
    .slice(0, config.eliteCount)
    .map((candidate, index) =>
      cloneNetwork(candidate.network, `g${generation}-elite-${index}`),
    );

  while (next.length < config.populationSize) {
    const parentA = selectTournament(ranked, config.tournamentSize, random);
    const parentB = selectTournament(ranked, config.tournamentSize, random);
    const childId = `g${generation}-candidate-${next.length}`;
    const crossed = random() < config.crossoverRate
      ? crossoverNetworks(parentA.network, parentB.network, childId, random)
      : cloneNetwork(parentA.network, childId);
    next.push(mutateNetwork(
      crossed,
      random,
      config.mutationRate,
      config.mutationScale,
      childId,
    ));
  }

  return next;
}

export async function runNeuroevolution(
  options: RunNeuroevolutionOptions,
): Promise<NeuroevolutionResult> {
  validateOptions(options);
  const { config } = options;
  const random = createSeededRandom(config.seed);
  const evaluations: CandidateEvaluation[] = [];
  const history: GenerationSummary[] = [];
  let population: MLPNetwork[] = [];

  if (config.includeInitialNetwork) {
    population.push(cloneNetwork(options.initialNetwork, "g0-seed"));
  }
  while (population.length < config.populationSize) {
    population.push(createRandomNetwork(
      options.initialNetwork,
      `g0-candidate-${population.length}`,
      random,
    ));
  }

  let best: CandidateEvaluation | undefined;

  for (let generation = 0; generation < config.generations; generation += 1) {
    const generationResults: CandidateEvaluation[] = [];

    for (let candidateIndex = 0; candidateIndex < population.length; candidateIndex += 1) {
      if (options.signal?.aborted) {
        throw new Error(options.signal.reason?.toString() ?? "Neuroevolution cancelled.");
      }

      const network = population[candidateIndex];
      const candidateId = `g${generation}-c${candidateIndex}`;
      const seed = Math.floor(random() * 0xffffffff);

      let evaluation: CandidateEvaluation;
      try {
        const result = await runPlaygroundSession({
          source: options.environmentSource,
          network,
          session: {
            sessionId: candidateId,
            generation,
            candidateIndex,
            seed,
          },
          config: options.configForEnvironment,
          timeoutMs: config.evaluationTimeoutMs,
          signal: options.signal,
        });
        evaluation = {
          id: candidateId,
          generation,
          candidateIndex,
          network,
          fitness: result.fitness,
          metrics: result.metrics,
        };
      } catch (error) {
        if (options.signal?.aborted) {
          throw new Error(options.signal.reason?.toString() ?? "Neuroevolution cancelled.");
        }
        evaluation = {
          id: candidateId,
          generation,
          candidateIndex,
          network,
          fitness: -Number.MAX_VALUE,
          error: error instanceof Error ? error.message : String(error),
        };
      }

      generationResults.push(evaluation);
      evaluations.push(evaluation);
      if (!best || evaluation.fitness > best.fitness) best = evaluation;
      options.onCandidate?.(evaluation);
    }

    const ranked = [...generationResults].sort((a, b) => b.fitness - a.fitness);
    const summary = summarizeGeneration(generation, ranked);
    history.push(summary);
    options.onGeneration?.(summary);

    if (generation < config.generations - 1) {
      population = createNextGeneration(ranked, config, random, generation + 1);
    }
  }

  if (!best) throw new Error("Neuroevolution completed without evaluating any candidates.");
  return { best, history, evaluations };
}
