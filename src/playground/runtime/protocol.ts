import type { PlaygroundSessionResult } from "../api/types";

export type WorkerRequest =
  | { type: "start"; source: string; session: unknown; config: unknown }
  | { type: "infer-result"; requestId: number; output?: number[]; error?: string }
  | { type: "finish-ack" }
  | { type: "cancel"; reason: string };

export type WorkerMessage =
  | { type: "predict"; requestId: number; input: number[] }
  | { type: "report"; metrics: Record<string, unknown> }
  | { type: "finish"; result: PlaygroundSessionResult }
  | { type: "cleanup-complete" }
  | { type: "incomplete" }
  | { type: "error"; message: string; stack?: string };
