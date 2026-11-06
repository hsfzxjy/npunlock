const assert = require("node:assert/strict");
const test = require("node:test");

const { flattenTree, visibleWindow } = require("../.test-build/tree.js");

test("tree flattening preserves hierarchy, expansion, and filtered ancestors", () => {
  const root = { id: "root" };
  const first = { id: "first" };
  const second = { id: "second" };
  const leaf = { id: "leaf" };
  const children = new Map([
    ["root", [first, second]],
    ["first", [leaf]],
  ]);
  assert.deepEqual(
    flattenTree([root], children, new Set(["root"])).map(value => [value.node.id, value.depth]),
    [["root", 0], ["first", 1], ["second", 1]],
  );
  assert.deepEqual(
    flattenTree([root], children, new Set(), new Set(["root", "first", "leaf"]), true)
      .map(value => [value.node.id, value.depth]),
    [["root", 0], ["first", 1], ["leaf", 2]],
  );
});

test("virtual tree window remains bounded for ten thousand rows", () => {
  const root = { id: "node-0" };
  const children = Array.from({ length: 9_999 }, (_, index) => ({ id: `node-${index + 1}` }));
  const flattened = flattenTree([root], new Map([[root.id, children]]), new Set([root.id]));
  assert.equal(flattened.length, 10_000);
  const first = visibleWindow(10_000, 0, 630, 21, 6);
  const middle = visibleWindow(10_000, 105_000, 630, 21, 6);
  const last = visibleWindow(10_000, 210_000, 630, 21, 6);
  assert.deepEqual(first, { start: 0, end: 43 });
  assert.ok(middle.start > 4_900 && middle.end - middle.start <= 43);
  assert.equal(last.end, 10_000);
  assert.ok(last.end - last.start <= 43);
});
