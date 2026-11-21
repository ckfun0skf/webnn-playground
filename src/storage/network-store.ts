import type { MLPNetwork } from "../core/model";

const DATABASE_NAME = "webnn-playground";
const DATABASE_VERSION = 1;
const NETWORK_STORE = "networks";

export interface SavedNetworkRecord {
  id: string;
  name: string;
  network: MLPNetwork;
  createdAt: number;
  updatedAt: number;
}

let databasePromise: Promise<IDBDatabase> | null = null;

function openDatabase(): Promise<IDBDatabase> {
  if (typeof indexedDB === "undefined") {
    return Promise.reject(new Error("IndexedDB is not available in this browser."));
  }
  if (databasePromise) return databasePromise;

  databasePromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(NETWORK_STORE)) {
        database.createObjectStore(NETWORK_STORE, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Unable to open local network storage."));
    request.onblocked = () => reject(new Error("Network storage upgrade is blocked by another open tab."));
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

export async function listSavedNetworks(): Promise<SavedNetworkRecord[]> {
  const database = await openDatabase();
  const transaction = database.transaction(NETWORK_STORE, "readonly");
  const request = transaction.objectStore(NETWORK_STORE).getAll() as IDBRequest<SavedNetworkRecord[]>;
  const records = await new Promise<SavedNetworkRecord[]>((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Unable to read saved networks."));
  });
  await transactionDone(transaction);
  return records.sort((a, b) => b.updatedAt - a.updatedAt);
}

export async function saveNetwork(
  network: MLPNetwork,
  name = network.name,
): Promise<SavedNetworkRecord> {
  const database = await openDatabase();
  const transaction = database.transaction(NETWORK_STORE, "readwrite");
  const store = transaction.objectStore(NETWORK_STORE);
  const existingRequest = store.get(network.id) as IDBRequest<SavedNetworkRecord | undefined>;
  const now = Date.now();

  const record = await new Promise<SavedNetworkRecord>((resolve, reject) => {
    existingRequest.onsuccess = () => {
      const existing = existingRequest.result;
      const next: SavedNetworkRecord = {
        id: network.id,
        name,
        network,
        createdAt: existing?.createdAt ?? now,
        updatedAt: now,
      };
      try {
        store.put(next);
        resolve(next);
      } catch (error) {
        reject(error);
      }
    };
    existingRequest.onerror = () => reject(existingRequest.error ?? new Error("Unable to save network."));
  });

  await transactionDone(transaction);
  return record;
}

export async function deleteSavedNetwork(id: string): Promise<void> {
  const database = await openDatabase();
  const transaction = database.transaction(NETWORK_STORE, "readwrite");
  transaction.objectStore(NETWORK_STORE).delete(id);
  await transactionDone(transaction);
}
