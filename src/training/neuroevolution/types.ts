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
  seed: number;
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
  onGeneration?: (summary: GenerationSummary) => void;
}

export interface NeuroevolutionResult {
  best: CandidateEvaluation;
  history: GenerationSummary[];
  evaluations: CandidateEvaluation[];
}
