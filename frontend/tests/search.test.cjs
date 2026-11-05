const assert = require("node:assert/strict");
const test = require("node:test");

const { parseLocationExpression, searchEntries } = require("../.test-build/search.js");

test("location expressions preserve artifact and half-open range semantics", () => {
  const sizes = new Map([["native", 0x100], ["Kernel", 0x80]]);
  assert.deepEqual(parseLocationExpression("20", "native", sizes), {
    artifactId: "native", start: 0x20, end: 0x21,
  });
  assert.deepEqual(parseLocationExpression("kernel:0x10-0x20", "native", sizes), {
    artifactId: "Kernel", start: 0x10, end: 0x20,
  });
  assert.equal(parseLocationExpression("kernel:0x70-0x90", "native", sizes), undefined);
  assert.equal(parseLocationExpression("virtual:0x20", "native", sizes), undefined);
});

test("search favors exact and prefix labels while retaining stable ties", () => {
  const entries = [
    { id: "1", label: "KernelText", detail: "section", terms: ["code"], value: 1 },
    { id: "2", label: "KernelParams", detail: "section", terms: ["parameters"], value: 2 },
    { id: "3", label: "range", detail: "uses KernelText", terms: ["reference"], value: 3 },
  ];
  assert.deepEqual(searchEntries(entries, "KernelText").map(entry => entry.id), ["1", "3"]);
  assert.deepEqual(searchEntries(entries, "kernel").map(entry => entry.id), ["1", "2", "3"]);
  assert.deepEqual(searchEntries(entries, "parameters").map(entry => entry.id), ["2"]);
});
