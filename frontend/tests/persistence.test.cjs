const assert = require("node:assert/strict");
const test = require("node:test");

const {
  STORAGE_SCHEMA_VERSION,
  WORKSPACE_COMPATIBILITY_EPOCH,
  loadCompatibleValue,
  loadWorkspace,
  saveCompatibleValue,
  saveWorkspace,
  workspaceStorageKey,
} = require("../.test-build/persistence.js");

class MemoryStorage {
  values = new Map();
  removed = [];

  getItem(key) { return this.values.get(key) ?? null; }
  setItem(key, value) { this.values.set(key, value); }
  removeItem(key) { this.removed.push(key); this.values.delete(key); }
}

function location(offset = 8) {
  return { artifactId: "artifact", offset, end: offset + 4, nodeId: "node", fieldName: "field", spanIndex: 0 };
}

function workspace(overrides = {}) {
  const current = location();
  return {
    graphIdentity: '"graph-sha256"',
    graphFormat: "npunlock.artifact-graph",
    graphVersion: 2,
    backendStateVersion: "backend-1",
    savedAt: 42,
    location: current,
    history: { entries: [current], index: 0 },
    expandedNodeIds: ["node"],
    treeFilter: "kernel",
    treeScrollTop: 12,
    hexTopOffset: 0,
    inspectorScrollTop: 24,
    ...overrides,
  };
}

function context(overrides = {}) {
  return {
    graphIdentity: '"graph-sha256"',
    graphFormat: "npunlock.artifact-graph",
    graphVersion: 2,
    backendStateVersion: "backend-1",
    validateLocation: value => value.artifactId === "artifact"
      && value.offset >= 0 && value.end > value.offset && value.end <= 64
      && value.nodeId === "node" && value.fieldName === "field" && value.spanIndex === 0,
    validateNodeId: value => value === "node",
    validateHexTopOffset: (artifactId, value) => artifactId === "artifact" && value >= 0 && value < 64,
    ...overrides,
  };
}

test("compatible workspace state round trips exactly", () => {
  const storage = new MemoryStorage();
  saveWorkspace(storage, "workspace", workspace());
  assert.deepEqual(loadWorkspace(storage, "workspace", context()), workspace());
  assert.deepEqual(storage.removed, []);
});

test("workspace compatibility and graph drift discard the entire snapshot", () => {
  const incompatible = [
    envelope => { envelope.schemaVersion = STORAGE_SCHEMA_VERSION + 1; },
    envelope => { envelope.compatibilityEpoch = WORKSPACE_COMPATIBILITY_EPOCH + 1; },
    envelope => { envelope.value.graphIdentity = '"different"'; },
    envelope => { envelope.value.graphVersion = 1; },
    envelope => { envelope.value.backendStateVersion = "backend-2"; },
    envelope => { envelope.value.location.nodeId = "missing"; },
    envelope => { envelope.value.history.index = 3; },
    envelope => { envelope.value.hexTopOffset = 64; },
  ];
  for (const mutate of incompatible) {
    const storage = new MemoryStorage();
    saveWorkspace(storage, "workspace", workspace());
    const envelope = JSON.parse(storage.getItem("workspace"));
    mutate(envelope);
    storage.setItem("workspace", JSON.stringify(envelope));
    assert.equal(loadWorkspace(storage, "workspace", context()), undefined);
    assert.equal(storage.getItem("workspace"), null);
    assert.deepEqual(storage.removed, ["workspace"]);
  }
});

test("workspace keys are stable per exact graph identity", () => {
  assert.equal(workspaceStorageKey('"graph-sha256"'), workspaceStorageKey('"graph-sha256"'));
  assert.notEqual(workspaceStorageKey('"graph-sha256"'), workspaceStorageKey('"other"'));
});

test("malformed state and denied storage are nonfatal", () => {
  const storage = new MemoryStorage();
  storage.setItem("workspace", "not-json");
  assert.equal(loadWorkspace(storage, "workspace", context()), undefined);
  assert.equal(storage.getItem("workspace"), null);

  const denied = {
    getItem() { throw new Error("denied"); },
    setItem() { throw new Error("denied"); },
    removeItem() { throw new Error("denied"); },
  };
  assert.equal(loadWorkspace(denied, "workspace", context()), undefined);
  assert.doesNotThrow(() => saveWorkspace(denied, "workspace", workspace()));
});

test("global options use the same fail-closed envelope", () => {
  const storage = new MemoryStorage();
  const valid = value => typeof value === "object" && value !== null && value.enabled === true;
  saveCompatibleValue(storage, "options", { enabled: true });
  assert.deepEqual(loadCompatibleValue(storage, "options", valid), { enabled: true });

  const envelope = JSON.parse(storage.getItem("options"));
  envelope.compatibilityEpoch += 1;
  storage.setItem("options", JSON.stringify(envelope));
  assert.equal(loadCompatibleValue(storage, "options", valid), undefined);
  assert.equal(storage.getItem("options"), null);
});
