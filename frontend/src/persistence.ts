import {
  sameLocation,
  type LocationState,
  type NavigationSnapshot,
} from "./navigation.js";

export const STORAGE_SCHEMA_VERSION = 1;
export const WORKSPACE_COMPATIBILITY_EPOCH = 1;
export const MAX_PERSISTED_HISTORY = 100;

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface WorkspaceState {
  graphIdentity: string;
  graphFormat: string;
  graphVersion: number;
  backendStateVersion: string;
  savedAt: number;
  location: LocationState;
  history: NavigationSnapshot;
  expandedNodeIds: string[];
  treeFilter: string;
  treeScrollTop: number;
  hexTopOffset: number;
  inspectorScrollTop: number;
}

export interface WorkspaceContext {
  graphIdentity: string;
  graphFormat: string;
  graphVersion: number;
  backendStateVersion: string;
  validateLocation(location: LocationState): boolean;
  validateNodeId(nodeId: string): boolean;
  validateHexTopOffset(artifactId: string, offset: number): boolean;
}

interface StorageEnvelope<T> {
  schemaVersion: number;
  compatibilityEpoch: number;
  value: T;
}

export function workspaceStorageKey(graphIdentity: string): string {
  return `npunlock.viz.workspace:${encodeURIComponent(graphIdentity)}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNonnegative(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function isLocation(value: unknown): value is LocationState {
  if (!isRecord(value)
    || typeof value.artifactId !== "string"
    || !Number.isSafeInteger(value.offset)
    || !Number.isSafeInteger(value.end)) return false;
  if (value.nodeId !== undefined && typeof value.nodeId !== "string") return false;
  if (value.fieldName !== undefined && typeof value.fieldName !== "string") return false;
  const spanIndex = value.spanIndex;
  if (spanIndex !== undefined
    && (typeof spanIndex !== "number" || !Number.isSafeInteger(spanIndex) || spanIndex < 0)) return false;
  if (value.arrival !== undefined) {
    if (!isRecord(value.arrival)
      || typeof value.arrival.sourceNodeId !== "string"
      || typeof value.arrival.targetNodeId !== "string"
      || typeof value.arrival.kind !== "string") return false;
  }
  return true;
}

function remove(storage: StorageLike, key: string): void {
  try {
    storage.removeItem(key);
  } catch {
    // Storage is optional; a denied cleanup must not make the viewer unusable.
  }
}

export function loadCompatibleValue<T>(
  storage: StorageLike,
  key: string,
  validate: (value: unknown) => value is T,
): T | undefined {
  try {
    const raw = storage.getItem(key);
    if (raw === null) return undefined;
    const envelope = JSON.parse(raw) as unknown;
    if (!isRecord(envelope)
      || envelope.schemaVersion !== STORAGE_SCHEMA_VERSION
      || envelope.compatibilityEpoch !== WORKSPACE_COMPATIBILITY_EPOCH
      || !validate(envelope.value)) {
      remove(storage, key);
      return undefined;
    }
    return envelope.value;
  } catch {
    remove(storage, key);
    return undefined;
  }
}

export function saveCompatibleValue<T>(storage: StorageLike, key: string, value: T): void {
  const envelope: StorageEnvelope<T> = {
    schemaVersion: STORAGE_SCHEMA_VERSION,
    compatibilityEpoch: WORKSPACE_COMPATIBILITY_EPOCH,
    value,
  };
  try {
    storage.setItem(key, JSON.stringify(envelope));
  } catch {
    // Storage denial and quota exhaustion are nonfatal.
  }
}

function validateWorkspace(value: unknown, context: WorkspaceContext): value is WorkspaceState {
  if (!isRecord(value)
    || value.graphIdentity !== context.graphIdentity
    || value.graphFormat !== context.graphFormat
    || value.graphVersion !== context.graphVersion
    || value.backendStateVersion !== context.backendStateVersion
    || !isFiniteNonnegative(value.savedAt)
    || !isLocation(value.location)
    || !context.validateLocation(value.location)
    || !isRecord(value.history)) return false;
  const entries = value.history.entries;
  const index = value.history.index;
  if (!Array.isArray(entries)
    || entries.length < 1
    || entries.length > MAX_PERSISTED_HISTORY
    || typeof index !== "number"
    || !Number.isInteger(index)
    || index < 0
    || index >= entries.length
    || !entries.every(entry => isLocation(entry) && context.validateLocation(entry))) return false;
  const current = entries[index];
  if (!isLocation(current) || !sameLocation(value.location, current)) return false;
  if (!Array.isArray(value.expandedNodeIds)
    || !value.expandedNodeIds.every(nodeId => typeof nodeId === "string" && context.validateNodeId(nodeId))
    || new Set(value.expandedNodeIds).size !== value.expandedNodeIds.length
    || typeof value.treeFilter !== "string"
    || value.treeFilter.length > 2048
    || !isFiniteNonnegative(value.treeScrollTop)
    || typeof value.hexTopOffset !== "number"
    || !Number.isSafeInteger(value.hexTopOffset)
    || !context.validateHexTopOffset(value.location.artifactId, value.hexTopOffset)
    || !isFiniteNonnegative(value.inspectorScrollTop)) return false;
  return true;
}

export function loadWorkspace(
  storage: StorageLike,
  key: string,
  context: WorkspaceContext,
): WorkspaceState | undefined {
  return loadCompatibleValue(storage, key, (value): value is WorkspaceState => validateWorkspace(value, context));
}

export function saveWorkspace(storage: StorageLike, key: string, value: WorkspaceState): void {
  saveCompatibleValue(storage, key, value);
}
