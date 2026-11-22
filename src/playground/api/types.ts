import type { MLPNetwork } from "../../core/model";

export interface PlaygroundParameterDefinition {
  key: string;
  label: string;
  type: "number" | "boolean" | "string";
  default: number | boolean | string;
  min?: number;
  max?: number;
  step?: number;
  description?: string;
}

export interface PlaygroundDefinition {
  id: string;
  name: string;
  version: number;
  inputSize: number;
  outputSize: number;
  parameters: PlaygroundParameterDefinition[];
}

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
  /** Advances one simulation step, applies the configured playback delay, and returns false at maxSteps. */
  step(): Promise<boolean>;
  yield(): Promise<void>;
  signal: AbortSignal;
  session: Readonly<PlaygroundSessionMetadata>;
  config: Readonly<Record<string, unknown>>;
}

export interface PlaygroundEnvironmentModule {
  playgroundDefinition?: PlaygroundDefinition;
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
