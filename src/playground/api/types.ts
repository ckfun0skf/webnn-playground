import type { MLPNetwork } from "../../core/model";

export interface PlaygroundSessionMetadata {
  sessionId: string;
  generation: number;
  candidateIndex: number;
  seed: number;
}

export interface PlaygroundSessionResult {
  fitness: number;
  metrics?: Record<string, unknown>;
}

export interface PlaygroundEnvironmentContext {
  network: {
    predict(input: readonly number[]): Promise<number[]>;
  };
  finish(result: PlaygroundSessionResult): Promise<void>;
  report(metrics: Record<string, unknown>): void;
  render(view: unknown): void;
  random(): number;
  yield(): Promise<void>;
  signal: AbortSignal;
  session: Readonly<PlaygroundSessionMetadata>;
  config: Readonly<Record<string, unknown>>;
}

export interface PlaygroundEnvironmentModule {
  sessionStart(context: PlaygroundEnvironmentContext): void | Promise<void>;
  sessionEnd?(
    context: PlaygroundEnvironmentContext,
    result: PlaygroundSessionResult,
  ): void | Promise<void>;
}

export interface RunPlaygroundSessionOptions {
  source: string;
  network: MLPNetwork;
  session: PlaygroundSessionMetadata;
  config?: Record<string, unknown>;
  timeoutMs?: number;
  signal?: AbortSignal;
  onReport?: (metrics: Record<string, unknown>) => void;
  onRender?: (view: unknown) => void;
}

