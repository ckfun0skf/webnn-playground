import type { MLPNetwork } from "../../core/model";
import type { PlaygroundSessionMetadata, PlaygroundSessionResult } from "../api/types";

export type WorkerRequest =
  | {
      type: "start";
      source: string;
      network: MLPNetwork;
      session: PlaygroundSessionMetadata;
      config: Record<string, unknown>;
      renderEnabled: boolean;
    }
  | { type: "finish-ack" }
  | { type: "cancel"; reason: string };

export type WorkerMessage =
  | { type: "report"; metrics: Record<string, unknown> }
  | { type: "render"; view: unknown }
  | { type: "finish"; result: PlaygroundSessionResult }
  | { type: "cleanup-complete" }
  | { type: "incomplete" }
  | { type: "cancelled" }
  | { type: "error"; message: string; stack?: string };
