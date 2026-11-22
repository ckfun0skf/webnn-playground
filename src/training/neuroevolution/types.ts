import type { MLPNetwork } from "../../core/model";

export interface NeuroevolutionConfig {
  populationSize: number;
  generations: number;
  eliteCount: number;
  tournamentSize: number;
  crossoverRate: number;
  mutationRate: number;
  mutationScale: number;
  evaluationTimeoutMs: number;
  /** Controls selection, crossover, and mutation randomness. */
  seed: number;
  /** Shared by every candidate so they are evaluated under identical random conditions. */
  environmentSeed: number;
  includeInitialNetwork: boolean;
}

export interface CandidateEvaluation {
  id: string;
  generation: number;
  candidateIndex: number;
  network: MLPNetwork;
  fitness: number;
  metrics?: Record<string, unknown>;
  error?: string;
}

export interface GenerationSummary {
  generation: number;
  bestFitness: number;
  meanFitness: number;
  worstFitness: number;
  successfulEvaluations: number;
  failedEvaluations: number;
}

export interface RunNeuroevolutionOptions {
  initialNetwork: MLPNetwork;
  environmentSource: string;
  config: NeuroevolutionConfig;
  configForEnvironment?: Record<string, unknown>;
  signal?: AbortSignal;
  onCandidate?: (candidate: CandidateEvaluation) => void;
  onPopulationRender?: (generation: number, candidateIndex: number, view: unknown) => void;
  onGeneration?: (summary: GenerationSummary) => void;
}

export interface NeuroevolutionResult {
  best: CandidateEvaluation;
  history: GenerationSummary[];
  evaluations: CandidateEvaluation[];
}
