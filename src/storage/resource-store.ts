import type { MLPNetwork } from "../core/model";
import type { PlaygroundDefinition } from "../playground/api";

const DATABASE_NAME = "webnn-playground";
const DATABASE_VERSION = 2;
const PLAYGROUND_STORE = "playgrounds";
const TRAINING_RUN_STORE = "trainingRuns";

export interface SavedPlaygroundRecord {
  id: string;
  name: string;
  version: number;
  definition: PlaygroundDefinition;
  source: string;
  createdAt: number;
  updatedAt: number;
}

export interface SavedTrainingRunRecord {
  id: string;
  name: string;
  status: "completed" | "failed" | "cancelled";
  playgroundId: string;
  playgroundName: string;
  playgroundVersion: number;
  /** Immutable source snapshot makes a run reproducible even if the Playground is later edited. */
  playgroundSource: string;
  /** Immutable network snapshot used to seed the evolution run. */
  initialNetwork: MLPNetwork;
  evolutionConfig: Record<string, unknown>;
  environmentConfig: Record<string, unknown>;
  bestNetwork?: MLPNetwork;
  bestFitness?: number;
  generationHistory: Record<string, unknown>[];
  createdAt: number;
  completedAt?: number;
}

let databasePromise: Promise<IDBDatabase> | null = null;

function openDatabase(): Promise<IDBDatabase> {
  if (typeof indexedDB === "undefined") {
    return Promise.reject(new Error("IndexedDB is not available in this browser."));
  }
  if (databasePromise) return databasePromise;
  databasePromise = new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains("networks")) {
        database.createObjectStore("networks", { keyPath: "id" });
      }
      if (!database.objectStoreNames.contains(PLAYGROUND_STORE)) {
        database.createObjectStore(PLAYGROUND_STORE, { keyPath: "id" });
      }
      if (!database.objectStoreNames.contains(TRAINING_RUN_STORE)) {
        database.createObjectStore(TRAINING_RUN_STORE, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Unable to open local resource storage."));
    request.onblocked = () => reject(new Error("Resource storage upgrade is blocked by another open tab."));
  }).catch((error: unknown) => {
    databasePromise = null;
    throw error;
  });
  return databasePromise;
}

function transactionDone(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error ?? new Error("IndexedDB transaction failed."));
    transaction.onabort = () => reject(transaction.error ?? new Error("IndexedDB transaction was aborted."));
  });
}

async function listRecords<T>(storeName: string): Promise<T[]> {
  const database = await openDatabase();
  const transaction = database.transaction(storeName, "readonly");
  const request = transaction.objectStore(storeName).getAll() as IDBRequest<T[]>;
  const records = await new Promise<T[]>((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Unable to read local resources."));
  });
  await transactionDone(transaction);
  return records;
}

async function putRecord<T extends { id: string }>(storeName: string, record: T): Promise<T> {
  const database = await openDatabase();
  const transaction = database.transaction(storeName, "readwrite");
  transaction.objectStore(storeName).put(record);
  await transactionDone(transaction);
  return record;
}

async function deleteRecord(storeName: string, id: string): Promise<void> {
  const database = await openDatabase();
  const transaction = database.transaction(storeName, "readwrite");
  transaction.objectStore(storeName).delete(id);
  await transactionDone(transaction);
}

export async function listSavedPlaygrounds(): Promise<SavedPlaygroundRecord[]> {
  return (await listRecords<SavedPlaygroundRecord>(PLAYGROUND_STORE))
    .sort((a, b) => b.updatedAt - a.updatedAt);
}

export async function savePlayground(
  definition: PlaygroundDefinition,
  source: string,
  name = definition.name,
): Promise<SavedPlaygroundRecord> {
  const existing = (await listSavedPlaygrounds()).find((record) => record.id === definition.id);
  const now = Date.now();
  return putRecord(PLAYGROUND_STORE, {
    id: definition.id,
    name,
    version: definition.version,
    definition,
    source,
    createdAt: existing?.createdAt ?? now,
    updatedAt: now,
  });
}

export async function deleteSavedPlayground(id: string): Promise<void> {
  await deleteRecord(PLAYGROUND_STORE, id);
}

export async function saveTrainingRun(
  run: SavedTrainingRunRecord,
): Promise<SavedTrainingRunRecord> {
  return putRecord(TRAINING_RUN_STORE, run);
}

export async function listTrainingRuns(): Promise<SavedTrainingRunRecord[]> {
  return (await listRecords<SavedTrainingRunRecord>(TRAINING_RUN_STORE))
    .sort((a, b) => b.createdAt - a.createdAt);
}

export async function deleteTrainingRun(id: string): Promise<void> {
  await deleteRecord(TRAINING_RUN_STORE, id);
}
