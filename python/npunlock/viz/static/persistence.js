import { sameLocation, } from "./navigation.js";
export const STORAGE_SCHEMA_VERSION = 1;
export const WORKSPACE_COMPATIBILITY_EPOCH = 1;
export const MAX_PERSISTED_HISTORY = 100;
export function workspaceStorageKey(graphIdentity) {
    return `npunlock.viz.workspace:${encodeURIComponent(graphIdentity)}`;
}
function isRecord(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}
function isFiniteNonnegative(value) {
    return typeof value === "number" && Number.isFinite(value) && value >= 0;
}
function isLocation(value) {
    if (!isRecord(value)
        || typeof value.artifactId !== "string"
        || !Number.isSafeInteger(value.offset)
        || !Number.isSafeInteger(value.end))
        return false;
    if (value.nodeId !== undefined && typeof value.nodeId !== "string")
        return false;
    if (value.fieldName !== undefined && typeof value.fieldName !== "string")
        return false;
    const spanIndex = value.spanIndex;
    if (spanIndex !== undefined
        && (typeof spanIndex !== "number" || !Number.isSafeInteger(spanIndex) || spanIndex < 0))
        return false;
    if (value.arrival !== undefined) {
        if (!isRecord(value.arrival)
            || typeof value.arrival.sourceNodeId !== "string"
            || typeof value.arrival.targetNodeId !== "string"
            || typeof value.arrival.kind !== "string")
            return false;
    }
    return true;
}
function remove(storage, key) {
    try {
        storage.removeItem(key);
    }
    catch {
    }
}
export function loadCompatibleValue(storage, key, validate) {
    try {
        const raw = storage.getItem(key);
        if (raw === null)
            return undefined;
        const envelope = JSON.parse(raw);
        if (!isRecord(envelope)
            || envelope.schemaVersion !== STORAGE_SCHEMA_VERSION
            || envelope.compatibilityEpoch !== WORKSPACE_COMPATIBILITY_EPOCH
            || !validate(envelope.value)) {
            remove(storage, key);
            return undefined;
        }
        return envelope.value;
    }
    catch {
        remove(storage, key);
        return undefined;
    }
}
export function saveCompatibleValue(storage, key, value) {
    const envelope = {
        schemaVersion: STORAGE_SCHEMA_VERSION,
        compatibilityEpoch: WORKSPACE_COMPATIBILITY_EPOCH,
        value,
    };
    try {
        storage.setItem(key, JSON.stringify(envelope));
    }
    catch {
    }
}
function validateWorkspace(value, context) {
    if (!isRecord(value)
        || value.graphIdentity !== context.graphIdentity
        || value.graphFormat !== context.graphFormat
        || value.graphVersion !== context.graphVersion
        || value.backendStateVersion !== context.backendStateVersion
        || !isFiniteNonnegative(value.savedAt)
        || !isLocation(value.location)
        || !context.validateLocation(value.location)
        || !isRecord(value.history))
        return false;
    const entries = value.history.entries;
    const index = value.history.index;
    if (!Array.isArray(entries)
        || entries.length < 1
        || entries.length > MAX_PERSISTED_HISTORY
        || typeof index !== "number"
        || !Number.isInteger(index)
        || index < 0
        || index >= entries.length
        || !entries.every(entry => isLocation(entry) && context.validateLocation(entry)))
        return false;
    const current = entries[index];
    if (!isLocation(current) || !sameLocation(value.location, current))
        return false;
    if (!Array.isArray(value.expandedNodeIds)
        || !value.expandedNodeIds.every(nodeId => typeof nodeId === "string" && context.validateNodeId(nodeId))
        || new Set(value.expandedNodeIds).size !== value.expandedNodeIds.length
        || typeof value.treeFilter !== "string"
        || value.treeFilter.length > 2048
        || !isFiniteNonnegative(value.treeScrollTop)
        || typeof value.hexTopOffset !== "number"
        || !Number.isSafeInteger(value.hexTopOffset)
        || !context.validateHexTopOffset(value.location.artifactId, value.hexTopOffset)
        || !isFiniteNonnegative(value.inspectorScrollTop))
        return false;
    return true;
}
export function loadWorkspace(storage, key, context) {
    return loadCompatibleValue(storage, key, (value) => validateWorkspace(value, context));
}
export function saveWorkspace(storage, key, value) {
    saveCompatibleValue(storage, key, value);
}
