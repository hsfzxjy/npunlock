const assert = require("node:assert/strict");
const test = require("node:test");

const { NavigationHistory, sameLocation } = require("../.test-build/navigation.js");

function location(offset, extra = {}) {
  return { artifactId: "a", offset, end: offset + 1, ...extra };
}

test("location equality includes ranges and semantic context", () => {
  assert.equal(sameLocation(location(4), location(4)), true);
  assert.equal(sameLocation(location(4), { ...location(4), end: 8 }), false);
  assert.equal(sameLocation(location(4), location(4, { fieldName: "size" })), false);
  assert.equal(sameLocation(
    location(4, { arrival: { sourceNodeId: "x", targetNodeId: "y", kind: "uses" } }),
    location(4, { arrival: { sourceNodeId: "x", targetNodeId: "y", kind: "uses" } }),
  ), true);
});

test("byte stepping coalesces without losing the semantic origin", () => {
  const history = new NavigationHistory();
  history.record(location(0, { nodeId: "header" }), "replace");
  history.record(location(1, { nodeId: "header" }), "coalesce", "byte-step");
  history.record(location(2, { nodeId: "field" }), "coalesce", "byte-step");
  history.record(location(3, { nodeId: "field" }), "coalesce", "byte-step");
  assert.equal(history.length, 2);
  assert.deepEqual(history.move(-1), location(0, { nodeId: "header" }));
  assert.deepEqual(history.move(1), location(3, { nodeId: "field" }));
});

test("a semantic jump after back discards the forward branch", () => {
  const history = new NavigationHistory();
  history.record(location(0), "replace");
  history.record(location(8, { nodeId: "first" }), "push");
  history.record(location(16, { nodeId: "second" }), "push");
  assert.equal(history.canBack, true);
  assert.deepEqual(history.move(-1), location(8, { nodeId: "first" }));
  history.record(location(24, { nodeId: "replacement" }), "push");
  assert.equal(history.canForward, false);
  assert.equal(history.length, 3);
  assert.deepEqual(history.move(-1), location(8, { nodeId: "first" }));
});
