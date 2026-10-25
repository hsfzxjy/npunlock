interface SpanData { artifact: string; start: number; end: number }
interface FieldData {
  value: unknown;
  display?: string;
  description?: string;
  properties?: Record<string, unknown>;
  spans?: SpanData[];
}
interface AddressData { space: string; value: number }
interface NodeData {
  id: string;
  label: string;
  description?: string;
  kind: string;
  category: string;
  spans: SpanData[];
  properties: Record<string, unknown>;
  fields: Record<string, FieldData>;
  addresses: Record<string, AddressData>;
  parent?: string;
}
interface LinkData { source: string; target: string; kind: string; properties: Record<string, unknown> }
interface ArtifactData { id: string; name: string; size: number; sha256: string; media_type: string }
interface CategoryData { id: string; label: string; color: string }
interface GraphData {
  format: string;
  version: number;
  title: string;
  artifacts: ArtifactData[];
  categories: CategoryData[];
  nodes: NodeData[];
  links: LinkData[];
}
interface IndexedSpan extends SpanData { node: NodeData; length: number; fieldName?: string }
interface LocationState { artifactId: string; offset: number; end: number; nodeId?: string }
interface ArtifactKnowledge {
  entityCount: number;
  relationshipCount: number;
  fieldCount: number;
  explainedFieldCount: number;
  mappedBytes: number;
  decodedFieldBytes: number;
}
interface ViewerOptions {
  dimIrrelevant: boolean;
  treeVisible: boolean;
  inspectorVisible: boolean;
  treeWidth: number | null;
  inspectorWidth: number | null;
}

const ROW_BYTES = 16;
const ROW_HEIGHT = 21;
const PAGE_BYTES = 16 * 1024;
const MAX_SCROLL_HEIGHT = 16_000_000;
const OVERSCAN = 8;
const OPTIONS_KEY = "npunlock.viz.options.v1";
const DEFAULT_OPTIONS: ViewerOptions = {
  dimIrrelevant: true,
  treeVisible: true,
  inspectorVisible: true,
  treeWidth: null,
  inspectorWidth: null,
};

function required<T extends HTMLElement>(id: string): T {
  const value = document.getElementById(id);
  if (!(value instanceof HTMLElement)) throw new Error(`missing interface element: ${id}`);
  return value as T;
}

function hex(value: number, width = 0): string {
  return `0x${Math.max(0, value).toString(16).padStart(width, "0")}`;
}

function printable(value: number): string {
  return value >= 0x20 && value <= 0x7e ? String.fromCharCode(value) : ".";
}

function displayValue(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number") return Number.isInteger(value) ? `${value} (${hex(value)})` : String(value);
  if (value === null) return "null";
  if (typeof value === "boolean") return String(value);
  return JSON.stringify(value, null, 2) ?? String(value);
}

function byteCount(value: number): string {
  if (value < 1024) return `${value} B`;
  const units = ["KiB", "MiB", "GiB"];
  let scaled = value / 1024;
  let index = 0;
  while (scaled >= 1024 && index < units.length - 1) {
    scaled /= 1024;
    index += 1;
  }
  return `${scaled >= 10 ? scaled.toFixed(0) : scaled.toFixed(1)} ${units[index]}`;
}

function coverage(value: number, total: number): string {
  const percent = total > 0 ? value * 100 / total : 0;
  return `${byteCount(value)} (${percent < 0.1 && value > 0 ? "<0.1" : percent.toFixed(1)}%)`;
}

function unionLength(spans: SpanData[], size: number): number {
  const ordered = spans
    .map(span => ({ start: Math.max(0, span.start), end: Math.min(size, span.end) }))
    .filter(span => span.end > span.start)
    .sort((a, b) => a.start - b.start || a.end - b.end);
  let total = 0;
  let start = -1;
  let end = -1;
  for (const span of ordered) {
    if (span.start > end) {
      if (end > start) total += end - start;
      start = span.start;
      end = span.end;
    } else {
      end = Math.max(end, span.end);
    }
  }
  if (end > start) total += end - start;
  return total;
}

function evidenceText(field: FieldData): string | undefined {
  const confidence = field.properties?.confidence;
  const scope = field.properties?.scope;
  const relevance = field.properties?.relevance;
  const labels: Record<string, string> = {
    confirmed: "confirmed",
    "strong-inference": "strong inference",
    "execution-tested-inference": "execution-tested inference",
    raw: "raw observation",
  };
  const parts: string[] = [];
  if (typeof confidence === "string") parts.push(labels[confidence] ?? confidence);
  if (relevance === "irrelevant") parts.push("irrelevant to supported interpretation");
  else if (relevance === "context") parts.push("context only");
  if (typeof scope === "string") parts.push(scope);
  return parts.length ? parts.join(" · ") : undefined;
}

function loadOptions(): ViewerOptions {
  try {
    const value = JSON.parse(localStorage.getItem(OPTIONS_KEY) ?? "null") as Partial<ViewerOptions> | null;
    if (!value || typeof value !== "object") return { ...DEFAULT_OPTIONS };
    return {
      dimIrrelevant: typeof value.dimIrrelevant === "boolean" ? value.dimIrrelevant : true,
      treeVisible: typeof value.treeVisible === "boolean" ? value.treeVisible : true,
      inspectorVisible: typeof value.inspectorVisible === "boolean" ? value.inspectorVisible : true,
      treeWidth: typeof value.treeWidth === "number" && Number.isFinite(value.treeWidth) ? value.treeWidth : null,
      inspectorWidth: typeof value.inspectorWidth === "number" && Number.isFinite(value.inspectorWidth) ? value.inspectorWidth : null,
    };
  } catch {
    return { ...DEFAULT_OPTIONS };
  }
}

function saveOptions(value: ViewerOptions): void {
  try {
    localStorage.setItem(OPTIONS_KEY, JSON.stringify(value));
  } catch {
    // The viewer remains usable when browser policy disables local storage.
  }
}

class ByteCache {
  private pages = new Map<number, Promise<Uint8Array>>();
  constructor(private artifact: ArtifactData) {}

  private page(index: number): Promise<Uint8Array> {
    let value = this.pages.get(index);
    if (!value) {
      const offset = index * PAGE_BYTES;
      const length = Math.min(PAGE_BYTES, this.artifact.size - offset);
      const url = new URL(`api/artifacts/${encodeURIComponent(this.artifact.id)}/bytes`, window.location.href);
      url.searchParams.set("offset", String(offset));
      url.searchParams.set("length", String(length));
      value = fetch(url).then(async response => {
        if (!response.ok) throw new Error(`byte request failed: ${response.status}`);
        return new Uint8Array(await response.arrayBuffer());
      });
      this.pages.set(index, value);
    }
    return value;
  }

  async range(start: number, end: number): Promise<Uint8Array> {
    if (end <= start) return new Uint8Array();
    const first = Math.floor(start / PAGE_BYTES);
    const last = Math.floor((end - 1) / PAGE_BYTES);
    const values = await Promise.all(Array.from({ length: last - first + 1 }, (_, index) => this.page(first + index)));
    const output = new Uint8Array(end - start);
    let outputOffset = 0;
    for (let pageIndex = first; pageIndex <= last; pageIndex += 1) {
      const page = values[pageIndex - first];
      if (!page) continue;
      const pageStart = pageIndex * PAGE_BYTES;
      const copyStart = Math.max(start, pageStart);
      const copyEnd = Math.min(end, pageStart + page.length);
      output.set(page.subarray(copyStart - pageStart, copyEnd - pageStart), outputOffset);
      outputOffset += copyEnd - copyStart;
    }
    return output;
  }
}

class Viewer {
  private artifactTabs = required<HTMLElement>("artifact-tabs");
  private tree = required<HTMLElement>("tree");
  private inspector = required<HTMLElement>("inspector");
  private hexHead = required<HTMLElement>("hex-head");
  private hexScroll = required<HTMLElement>("hex-scroll");
  private hexSpace = required<HTMLElement>("hex-space");
  private hexRows = required<HTMLElement>("hex-rows");
  private status = required<HTMLElement>("status");
  private workspace = required<HTMLElement>("workspace");
  private offsetInput = required<HTMLInputElement>("offset-input");
  private nodes = new Map<string, NodeData>();
  private artifacts = new Map<string, ArtifactData>();
  private categories = new Map<string, CategoryData>();
  private children = new Map<string, NodeData[]>();
  private roots: NodeData[] = [];
  private depths = new Map<string, number>();
  private spans = new Map<string, IndexedSpan[]>();
  private cache = new Map<string, ByteCache>();
  private knowledge = new Map<string, ArtifactKnowledge>();
  private expanded = new Set<string>();
  private current!: LocationState;
  private history: LocationState[] = [];
  private historyIndex = -1;
  private hoveredNode?: string;
  private treeFocusId?: string;
  private renderToken = 0;
  private options = loadOptions();

  constructor(private graph: GraphData) {
    for (const artifact of graph.artifacts) {
      this.artifacts.set(artifact.id, artifact);
      this.cache.set(artifact.id, new ByteCache(artifact));
      this.spans.set(artifact.id, []);
    }
    for (const category of graph.categories) this.categories.set(category.id, category);
    for (const node of graph.nodes) {
      this.nodes.set(node.id, node);
      if (node.parent) {
        const values = this.children.get(node.parent) ?? [];
        values.push(node);
        this.children.set(node.parent, values);
      } else {
        this.roots.push(node);
      }
      for (const span of node.spans) {
        this.spans.get(span.artifact)?.push({ ...span, node, length: span.end - span.start });
      }
      for (const [fieldName, field] of Object.entries(node.fields)) {
        for (const span of field.spans ?? []) {
          this.spans.get(span.artifact)?.push({ ...span, node, length: span.end - span.start, fieldName });
        }
      }
    }
    for (const values of this.spans.values()) values.sort((a, b) => a.start - b.start || a.end - b.end);
    for (const node of graph.nodes) {
      if ((this.children.get(node.id)?.length ?? 0) > 0) this.expanded.add(node.id);
    }
    for (const root of this.roots) {
      this.assignDepth(root, 0);
    }
    for (const artifact of graph.artifacts) this.knowledge.set(artifact.id, this.buildKnowledge(artifact));
    const first = graph.artifacts[0];
    if (!first) throw new Error("artifact graph contains no artifacts");
    this.current = { artifactId: first.id, offset: 0, end: Math.min(1, first.size) };
    this.applyOptions();
    this.bind();
    this.renderTabs();
    this.renderTree();
    this.navigate(this.current, true, false);
    new ResizeObserver(() => this.renderHex()).observe(this.hexScroll);
  }

  private assignDepth(node: NodeData, depth: number): void {
    this.depths.set(node.id, depth);
    for (const child of this.children.get(node.id) ?? []) this.assignDepth(child, depth + 1);
  }

  private buildKnowledge(artifact: ArtifactData): ArtifactKnowledge {
    const touchedNodes = this.graph.nodes.filter(node =>
      node.spans.some(span => span.artifact === artifact.id)
      || Object.values(node.fields).some(field => (field.spans ?? []).some(span => span.artifact === artifact.id))
    );
    const touchedIds = new Set(touchedNodes.map(node => node.id));
    const fields = touchedNodes.flatMap(node => Object.values(node.fields).filter(field => {
      const spans = field.spans ?? [];
      return spans.length === 0 || spans.some(span => span.artifact === artifact.id);
    }));
    const mappedSpans = touchedNodes.flatMap(node => node.spans.filter(span =>
      span.artifact === artifact.id
      && !(node.parent === undefined && span.start === 0 && span.end === artifact.size)
    ));
    const fieldSpans = fields.flatMap(field => (field.spans ?? []).filter(span => span.artifact === artifact.id));
    return {
      entityCount: touchedNodes.length,
      relationshipCount: this.graph.links.filter(link => touchedIds.has(link.source) || touchedIds.has(link.target)).length,
      fieldCount: fields.length,
      explainedFieldCount: fields.filter(field => Boolean(field.description)).length,
      mappedBytes: unionLength(mappedSpans, artifact.size),
      decodedFieldBytes: unionLength(fieldSpans, artifact.size),
    };
  }

  private bind(): void {
    this.hexScroll.addEventListener("scroll", () => {
      this.hexHead.scrollLeft = this.hexScroll.scrollLeft;
      this.renderHex();
    });
    this.hexScroll.addEventListener("keydown", event => this.hexKey(event));
    required<HTMLFormElement>("offset-form").addEventListener("submit", event => {
      event.preventDefault();
      const raw = this.offsetInput.value.trim().toLowerCase();
      const parsed = Number.parseInt(raw.startsWith("0x") ? raw.slice(2) : raw, 16);
      const artifact = this.artifacts.get(this.current.artifactId)!;
      if (!Number.isFinite(parsed) || parsed < 0 || parsed >= artifact.size) {
        this.setStatus(`offset must be between ${hex(0)} and ${hex(Math.max(0, artifact.size - 1))}`);
        return;
      }
      this.navigate({ artifactId: artifact.id, offset: parsed, end: parsed + 1 }, true, true);
    });
    required<HTMLButtonElement>("copy").addEventListener("click", () => this.copyLocation());
    required<HTMLButtonElement>("back").addEventListener("click", () => this.moveHistory(-1));
    required<HTMLButtonElement>("forward").addEventListener("click", () => this.moveHistory(1));
    required<HTMLButtonElement>("toggle-tree").addEventListener("click", event => this.togglePanel("tree", event));
    required<HTMLButtonElement>("toggle-inspector").addEventListener("click", event => this.togglePanel("inspector", event));
    required<HTMLButtonElement>("toggle-dim-irrelevant").addEventListener("click", () => this.toggleDimIrrelevant());
    this.bindSplitter(required<HTMLElement>("left-splitter"), "--tree-width", "treeWidth", true, 150, 520);
    this.bindSplitter(required<HTMLElement>("right-splitter"), "--inspector-width", "inspectorWidth", false, 210, 620);
    window.addEventListener("keydown", event => {
      if (event.altKey && event.key === "ArrowLeft") { event.preventDefault(); this.moveHistory(-1); }
      if (event.altKey && event.key === "ArrowRight") { event.preventDefault(); this.moveHistory(1); }
    });
  }

  private applyOptions(): void {
    this.workspace.classList.toggle("tree-hidden", !this.options.treeVisible);
    this.workspace.classList.toggle("inspector-hidden", !this.options.inspectorVisible);
    required<HTMLButtonElement>("toggle-tree").setAttribute("aria-pressed", String(this.options.treeVisible));
    required<HTMLButtonElement>("toggle-inspector").setAttribute("aria-pressed", String(this.options.inspectorVisible));
    required<HTMLButtonElement>("toggle-dim-irrelevant").setAttribute("aria-pressed", String(this.options.dimIrrelevant));
    if (this.options.treeWidth !== null) {
      document.documentElement.style.setProperty("--tree-width", `${Math.max(150, Math.min(520, this.options.treeWidth))}px`);
    }
    if (this.options.inspectorWidth !== null) {
      document.documentElement.style.setProperty("--inspector-width", `${Math.max(210, Math.min(620, this.options.inspectorWidth))}px`);
    }
  }

  private bindSplitter(
    splitter: HTMLElement,
    variable: string,
    option: "treeWidth" | "inspectorWidth",
    fromLeft: boolean,
    minimum: number,
    maximum: number,
  ): void {
    const setValue = (value: number, persist = true) => {
      const bounded = Math.max(minimum, Math.min(maximum, value));
      document.documentElement.style.setProperty(variable, `${bounded}px`);
      splitter.setAttribute("aria-valuenow", String(Math.round(bounded)));
      this.options[option] = bounded;
      if (persist) saveOptions(this.options);
      this.renderHex();
    };
    const currentValue = () => {
      const raw = getComputedStyle(document.documentElement).getPropertyValue(variable);
      const value = Number.parseFloat(raw);
      return Number.isFinite(value) ? value : minimum;
    };
    splitter.setAttribute("aria-valuemin", String(minimum));
    splitter.setAttribute("aria-valuemax", String(maximum));
    splitter.setAttribute("aria-valuenow", String(Math.round(currentValue())));
    splitter.addEventListener("pointerdown", event => {
      splitter.setPointerCapture(event.pointerId);
      splitter.classList.add("dragging");
    });
    splitter.addEventListener("pointermove", event => {
      if (!splitter.hasPointerCapture(event.pointerId)) return;
      const bounds = this.workspace.getBoundingClientRect();
      const value = fromLeft ? event.clientX - bounds.left : bounds.right - event.clientX;
      setValue(value, false);
    });
    const finish = (event: PointerEvent) => {
      if (splitter.hasPointerCapture(event.pointerId)) splitter.releasePointerCapture(event.pointerId);
      splitter.classList.remove("dragging");
      saveOptions(this.options);
    };
    splitter.addEventListener("pointerup", finish);
    splitter.addEventListener("pointercancel", finish);
    splitter.addEventListener("keydown", event => {
      let value: number | undefined;
      if (event.key === "Home") value = minimum;
      if (event.key === "End") value = maximum;
      if (event.key === "ArrowLeft") value = currentValue() + (fromLeft ? -12 : 12);
      if (event.key === "ArrowRight") value = currentValue() + (fromLeft ? 12 : -12);
      if (value === undefined) return;
      event.preventDefault();
      setValue(value);
    });
  }

  private togglePanel(panel: "tree" | "inspector", event: Event): void {
    const button = event.currentTarget as HTMLButtonElement;
    const className = `${panel}-hidden`;
    const hidden = this.workspace.classList.toggle(className);
    button.setAttribute("aria-pressed", String(!hidden));
    if (panel === "tree") this.options.treeVisible = !hidden;
    else this.options.inspectorVisible = !hidden;
    saveOptions(this.options);
    this.renderHex();
  }

  private toggleDimIrrelevant(): void {
    this.options.dimIrrelevant = !this.options.dimIrrelevant;
    required<HTMLButtonElement>("toggle-dim-irrelevant")
      .setAttribute("aria-pressed", String(this.options.dimIrrelevant));
    saveOptions(this.options);
    this.renderTree();
    this.renderInspector();
    this.renderHex();
  }

  private renderTabs(): void {
    this.artifactTabs.replaceChildren();
    for (const artifact of this.graph.artifacts) {
      const button = document.createElement("button");
      button.className = "artifact-tab";
      button.type = "button";
      button.role = "tab";
      button.textContent = artifact.name;
      button.title = `${artifact.name}\n${artifact.size.toLocaleString()} bytes\nsha256 ${artifact.sha256}`;
      const selected = artifact.id === this.current.artifactId;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
      button.dataset.artifactId = artifact.id;
      button.addEventListener("click", () => this.selectArtifact(artifact, false));
      button.addEventListener("keydown", event => {
        const tabs = this.graph.artifacts;
        const index = tabs.findIndex(value => value.id === artifact.id);
        let next: ArtifactData | undefined;
        if (event.key === "ArrowLeft") next = tabs[(index - 1 + tabs.length) % tabs.length];
        if (event.key === "ArrowRight") next = tabs[(index + 1) % tabs.length];
        if (event.key === "Home") next = tabs[0];
        if (event.key === "End") next = tabs[tabs.length - 1];
        if (!next) return;
        event.preventDefault();
        this.selectArtifact(next, true);
      });
      this.artifactTabs.append(button);
    }
  }

  private selectArtifact(artifact: ArtifactData, restoreFocus: boolean): void {
    const offset = artifact.id === this.current.artifactId ? this.current.offset : 0;
    this.navigate({ artifactId: artifact.id, offset, end: Math.min(artifact.size, offset + 1) }, true, true);
    if (restoreFocus) {
      requestAnimationFrame(() => this.artifactTabs.querySelector<HTMLElement>(`[data-artifact-id="${CSS.escape(artifact.id)}"]`)?.focus());
    }
  }

  private renderTree(): void {
    this.tree.replaceChildren();
    const fragment = document.createDocumentFragment();
    const appendNode = (node: NodeData, depth: number) => {
      const row = document.createElement("div");
      row.className = `tree-row${this.current.nodeId === node.id ? " selected" : ""}`;
      if (this.options.dimIrrelevant && this.isIrrelevant(node)) row.classList.add("irrelevant");
      row.role = "treeitem";
      row.dataset.nodeId = node.id;
      row.tabIndex = (this.treeFocusId ?? this.current.nodeId ?? this.roots[0]?.id) === node.id ? 0 : -1;
      row.style.paddingLeft = `${depth * 14 + 3}px`;
      row.setAttribute("aria-level", String(depth + 1));
      row.setAttribute("aria-selected", String(this.current.nodeId === node.id));
      const children = this.children.get(node.id) ?? [];
      if (children.length) row.setAttribute("aria-expanded", String(this.expanded.has(node.id)));
      const expander = document.createElement("button");
      expander.type = "button";
      expander.className = `tree-expander${children.length ? "" : " empty"}`;
      expander.tabIndex = -1;
      expander.textContent = this.expanded.has(node.id) ? "▾" : "▸";
      expander.setAttribute("aria-label", this.expanded.has(node.id) ? "Collapse" : "Expand");
      expander.addEventListener("click", event => {
        event.stopPropagation();
        if (this.expanded.has(node.id)) this.expanded.delete(node.id); else this.expanded.add(node.id);
        this.renderTree();
      });
      const mark = document.createElement("span");
      mark.className = "tree-mark";
      mark.style.background = this.displayColor(this.categoryColor(node.category), this.isIrrelevant(node));
      const label = document.createElement("span");
      label.className = "tree-label";
      label.textContent = node.label;
      label.title = node.description ? `${node.label}\n${node.description}` : node.label;
      const kind = document.createElement("span");
      kind.className = "tree-kind";
      kind.textContent = node.kind;
      row.append(expander, mark, label, kind);
      row.addEventListener("click", () => {
        this.treeFocusId = node.id;
        this.selectNode(node.id, true);
      });
      row.addEventListener("focus", () => { this.treeFocusId = node.id; });
      row.addEventListener("keydown", event => this.treeKey(event, node));
      row.addEventListener("mouseenter", () => this.setHovered(node.id));
      row.addEventListener("mouseleave", () => this.setHovered(undefined));
      fragment.append(row);
      if (this.expanded.has(node.id)) for (const child of children) appendNode(child, depth + 1);
    };
    for (const root of this.roots) appendNode(root, 0);
    this.tree.append(fragment);
  }

  private treeKey(event: KeyboardEvent, node: NodeData): void {
    const rows = Array.from(this.tree.querySelectorAll<HTMLElement>(".tree-row"));
    const index = rows.findIndex(row => row.dataset.nodeId === node.id);
    let target: HTMLElement | undefined;
    if (event.key === "ArrowDown") target = rows[index + 1];
    if (event.key === "ArrowUp") target = rows[index - 1];
    if (event.key === "Home") target = rows[0];
    if (event.key === "End") target = rows[rows.length - 1];
    if (event.key === "ArrowRight") {
      const children = this.children.get(node.id) ?? [];
      if (children.length && !this.expanded.has(node.id)) {
        this.expanded.add(node.id);
        this.renderTree();
        target = this.tree.querySelector<HTMLElement>(`[data-node-id="${CSS.escape(node.id)}"]`) ?? undefined;
      } else if (children.length) {
        target = this.tree.querySelector<HTMLElement>(`[data-node-id="${CSS.escape(children[0]!.id)}"]`) ?? undefined;
      }
    }
    if (event.key === "ArrowLeft") {
      if (this.expanded.has(node.id) && (this.children.get(node.id)?.length ?? 0) > 0) {
        this.expanded.delete(node.id);
        this.renderTree();
        target = this.tree.querySelector<HTMLElement>(`[data-node-id="${CSS.escape(node.id)}"]`) ?? undefined;
      } else if (node.parent) {
        target = this.tree.querySelector<HTMLElement>(`[data-node-id="${CSS.escape(node.parent)}"]`) ?? undefined;
      }
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      this.treeFocusId = node.id;
      this.selectNode(node.id, true);
      requestAnimationFrame(() => this.tree.querySelector<HTMLElement>(`[data-node-id="${CSS.escape(node.id)}"]`)?.focus());
      return;
    }
    if (!target) return;
    event.preventDefault();
    for (const row of rows) row.tabIndex = -1;
    target.tabIndex = 0;
    this.treeFocusId = target.dataset.nodeId;
    target.focus();
  }

  private selectNode(nodeId: string, push: boolean): void {
    const node = this.nodes.get(nodeId);
    if (!node) return;
    let span = node.spans.find(value => value.artifact === this.current.artifactId) ?? node.spans[0];
    if (!span) {
      this.navigate({ ...this.current, nodeId }, push, false);
      return;
    }
    this.navigate({ artifactId: span.artifact, offset: span.start, end: span.end, nodeId }, push, true);
  }

  private navigate(location: LocationState, push: boolean, scroll: boolean): void {
    const artifact = this.artifacts.get(location.artifactId);
    if (!artifact) return;
    const offset = Math.max(0, Math.min(Math.max(0, artifact.size - 1), location.offset));
    const end = Math.max(offset + (artifact.size ? 1 : 0), Math.min(artifact.size, location.end));
    this.current = { ...location, offset, end };
    if (push) {
      const previous = this.history[this.historyIndex];
      if (!previous || previous.artifactId !== this.current.artifactId || previous.offset !== this.current.offset || previous.nodeId !== this.current.nodeId) {
        this.history = this.history.slice(0, this.historyIndex + 1);
        this.history.push({ ...this.current });
        this.historyIndex = this.history.length - 1;
      }
    }
    if (this.current.nodeId) this.revealTree(this.current.nodeId);
    this.offsetInput.value = hex(this.current.offset);
    this.renderTabs();
    this.renderTree();
    this.renderInspector();
    this.updateHistoryButtons();
    if (scroll) this.scrollToOffset(this.current.offset);
    else this.renderHex();
    this.setStatus(`${artifact.name}  ${hex(this.current.offset)}  ${this.current.end - this.current.offset} byte${this.current.end - this.current.offset === 1 ? "" : "s"}`);
  }

  private revealTree(nodeId: string): void {
    let node = this.nodes.get(nodeId);
    while (node?.parent) {
      this.expanded.add(node.parent);
      node = this.nodes.get(node.parent);
    }
    requestAnimationFrame(() => this.tree.querySelector<HTMLElement>(`[data-node-id="${CSS.escape(nodeId)}"]`)?.scrollIntoView({ block: "nearest" }));
  }

  private moveHistory(delta: number): void {
    const next = this.historyIndex + delta;
    if (next < 0 || next >= this.history.length) return;
    this.historyIndex = next;
    const location = this.history[next];
    if (location) this.navigate(location, false, true);
  }

  private updateHistoryButtons(): void {
    required<HTMLButtonElement>("back").disabled = this.historyIndex <= 0;
    required<HTMLButtonElement>("forward").disabled = this.historyIndex >= this.history.length - 1;
  }

  private scrollToOffset(offset: number): void {
    const artifact = this.artifacts.get(this.current.artifactId)!;
    const rowCount = Math.ceil(artifact.size / ROW_BYTES);
    const total = rowCount * ROW_HEIGHT;
    const visual = Math.min(MAX_SCROLL_HEIGHT, Math.max(ROW_HEIGHT, total));
    this.hexSpace.style.height = `${visual}px`;
    if (total <= MAX_SCROLL_HEIGHT) {
      this.hexScroll.scrollTop = Math.floor(offset / ROW_BYTES) * ROW_HEIGHT;
    } else {
      const maxScroll = Math.max(1, visual - this.hexScroll.clientHeight);
      this.hexScroll.scrollTop = Math.floor((offset / Math.max(1, artifact.size - 1)) * maxScroll);
    }
    this.renderHex();
  }

  private async renderHex(): Promise<void> {
    const token = ++this.renderToken;
    const artifact = this.artifacts.get(this.current.artifactId)!;
    const rowCount = Math.ceil(artifact.size / ROW_BYTES);
    const totalHeight = Math.max(ROW_HEIGHT, rowCount * ROW_HEIGHT);
    const visualHeight = Math.min(MAX_SCROLL_HEIGHT, totalHeight);
    this.hexSpace.style.height = `${visualHeight}px`;
    const visible = Math.max(1, Math.ceil(this.hexScroll.clientHeight / ROW_HEIGHT));
    let first: number;
    let scaled = false;
    if (totalHeight <= MAX_SCROLL_HEIGHT) {
      first = Math.max(0, Math.floor(this.hexScroll.scrollTop / ROW_HEIGHT) - OVERSCAN);
    } else {
      scaled = true;
      const denominator = Math.max(1, visualHeight - this.hexScroll.clientHeight);
      const progress = this.hexScroll.scrollTop / denominator;
      first = Math.max(0, Math.min(Math.max(0, rowCount - visible), Math.floor(progress * Math.max(0, rowCount - visible))) - OVERSCAN);
    }
    const last = Math.min(rowCount, first + visible + OVERSCAN * 2);
    const start = first * ROW_BYTES;
    const end = Math.min(artifact.size, last * ROW_BYTES);
    let bytes: Uint8Array;
    try {
      bytes = await this.cache.get(artifact.id)!.range(start, end);
    } catch (error) {
      if (token === this.renderToken) this.hexRows.replaceChildren(this.errorElement(String(error)));
      return;
    }
    if (token !== this.renderToken) return;
    const cover: IndexedSpan[][] = Array.from({ length: end - start }, () => []);
    for (const span of this.spans.get(artifact.id) ?? []) {
      if (span.start >= end) break;
      if (span.end <= start) continue;
      for (let offset = Math.max(start, span.start); offset < Math.min(end, span.end); offset += 1) cover[offset - start]?.push(span);
    }
    for (const values of cover) {
      values.sort((a, b) => a.length - b.length
        || Number(Boolean(b.fieldName)) - Number(Boolean(a.fieldName))
        || (this.depths.get(b.node.id) ?? 0) - (this.depths.get(a.node.id) ?? 0));
    }
    const fragment = document.createDocumentFragment();
    for (let rowIndex = first; rowIndex < last; rowIndex += 1) {
      const row = document.createElement("div");
      row.className = "hex-row";
      row.style.top = `${scaled ? this.hexScroll.scrollTop + (rowIndex - first) * ROW_HEIGHT : rowIndex * ROW_HEIGHT}px`;
      const rowOffset = rowIndex * ROW_BYTES;
      const offsetElement = document.createElement("span");
      offsetElement.className = "row-offset";
      offsetElement.textContent = rowOffset.toString(16).padStart(Math.max(8, artifact.size.toString(16).length), "0");
      const hexCells = document.createElement("span");
      hexCells.className = "hex-cells";
      const asciiCells = document.createElement("span");
      asciiCells.className = "ascii-cells";
      for (let column = 0; column < ROW_BYTES; column += 1) {
        const offset = rowOffset + column;
        const value = offset < artifact.size ? bytes[offset - start] : undefined;
        const annotations = offset < end ? cover[offset - start] ?? [] : [];
        hexCells.append(this.byteElement(offset, value, annotations, false));
        asciiCells.append(this.byteElement(offset, value, annotations, true));
      }
      row.append(offsetElement, hexCells, asciiCells);
      fragment.append(row);
    }
    this.hexRows.replaceChildren(fragment);
  }

  private byteElement(offset: number, value: number | undefined, annotations: IndexedSpan[], ascii: boolean): HTMLElement {
    const cell = document.createElement("button");
    cell.type = "button";
    cell.tabIndex = -1;
    cell.className = "byte";
    if (value === undefined) {
      cell.classList.add("empty");
      cell.textContent = ascii ? " " : "00";
      return cell;
    }
    cell.textContent = ascii ? printable(value) : value.toString(16).padStart(2, "0");
    const primary = annotations[0];
    const preferredField = (nodeId: string | undefined) => {
      const fields = annotations.filter(span => span.node.id === nodeId && span.fieldName);
      return fields.find(span => !this.isIrrelevant(span.node, span.fieldName)) ?? fields[0];
    };
    const selectedField = this.current.nodeId ? preferredField(this.current.nodeId) : undefined;
    if (primary) {
      cell.classList.add("annotated");
      const semanticSpan = selectedField ?? preferredField(primary.node.id) ?? primary;
      const baseColor = semanticSpan.fieldName
        ? this.fieldColor(semanticSpan.node, semanticSpan.fieldName)
        : this.categoryColor(semanticSpan.node.category);
      const irrelevant = this.isIrrelevant(semanticSpan.node, semanticSpan.fieldName);
      const semanticColor = this.displayColor(baseColor, irrelevant);
      if (this.options.dimIrrelevant && irrelevant) cell.classList.add("irrelevant");
      cell.style.setProperty("--semantic-color", semanticColor);
      if (selectedField) cell.classList.add("field-highlight");
      if (this.hoveredNode && annotations.some(span => span.node.id === this.hoveredNode)) cell.classList.add("hovered");
      cell.title = annotations
        .map(span => {
          if (!span.fieldName) return span.node.description ? `${span.node.label} — ${span.node.description}` : span.node.label;
          const field = span.node.fields[span.fieldName];
          const label = `${span.node.label} · ${span.fieldName}`;
          const explanation = field?.description ? `${label} — ${field.description}` : label;
          const evidence = field ? evidenceText(field) : undefined;
          return evidence ? `${explanation} [${evidence}]` : explanation;
        })
        .filter((value, index, all) => all.indexOf(value) === index)
        .join("\n");
    }
    if (offset === this.current.offset) cell.classList.add("selected");
    else if (offset >= this.current.offset && offset < this.current.end) cell.classList.add("range-selected");
    cell.addEventListener("click", event => {
      const mostSpecific = annotations[0]?.node;
      const start = event.shiftKey ? Math.min(this.current.offset, offset) : offset;
      const end = event.shiftKey ? Math.max(this.current.offset, offset) + 1 : offset + 1;
      this.navigate({ artifactId: this.current.artifactId, offset: start, end, nodeId: mostSpecific?.id }, true, false);
      this.hexScroll.focus();
    });
    if (primary) {
      cell.addEventListener("mouseenter", () => this.setHovered(primary.node.id));
      cell.addEventListener("mouseleave", () => this.setHovered(undefined));
    }
    return cell;
  }

  private categoryColor(category: string): string {
    return this.categories.get(category)?.color ?? "#5e6670";
  }

  private fieldColor(node: NodeData, fieldName: string): string {
    const index = Math.max(0, Object.keys(node.fields).indexOf(fieldName));
    const hue = Math.round((index * 137.508 + 8) % 360);
    return `hsl(${hue} 48% 42%)`;
  }

  private isIrrelevant(node: NodeData, fieldName?: string): boolean {
    if (fieldName && node.fields[fieldName]?.properties?.relevance === "irrelevant") return true;
    return node.properties.relevance === "irrelevant";
  }

  private displayColor(color: string, irrelevant: boolean): string {
    if (!this.options.dimIrrelevant || !irrelevant) return color;
    return `color-mix(in srgb, ${color} 24%, var(--text-faint))`;
  }

  private covering(artifactId: string, offset: number): NodeData[] {
    const values = (this.spans.get(artifactId) ?? []).filter(span => span.start <= offset && offset < span.end);
    values.sort((a, b) => a.length - b.length || (this.depths.get(b.node.id) ?? 0) - (this.depths.get(a.node.id) ?? 0));
    return values.map(value => value.node).filter((node, index, all) => all.findIndex(value => value.id === node.id) === index);
  }

  private renderInspector(): void {
    this.inspector.replaceChildren();
    const node = this.current.nodeId ? this.nodes.get(this.current.nodeId) : undefined;
    const artifact = this.artifacts.get(this.current.artifactId)!;
    const title = document.createElement("h1");
    title.className = "inspector-title";
    title.textContent = node?.label ?? `${artifact.name} ${hex(this.current.offset)}`;
    const subtitle = document.createElement("div");
    subtitle.className = "inspector-subtitle";
    subtitle.textContent = node ? `${node.kind} · ${node.id}` : `${this.current.end - this.current.offset} selected byte${this.current.end - this.current.offset === 1 ? "" : "s"}`;
    this.inspector.append(title, subtitle, this.knowledgeElement(artifact));
    if (node) {
      if (node.description) {
        const description = document.createElement("p");
        description.className = "inspector-description";
        description.textContent = node.description;
        this.inspector.append(description);
      }
      if (node.spans.length) {
        const section = this.section("byte spans");
        for (const span of node.spans) section.append(this.spanButton(span));
        this.inspector.append(section);
      }
      if (Object.keys(node.addresses).length) {
        const section = this.section("addresses");
        for (const [name, address] of Object.entries(node.addresses)) section.append(this.kv(name, `${address.space}:${hex(address.value)}`));
        this.inspector.append(section);
      }
      if (Object.keys(node.fields).length) {
        const section = this.section("fields");
        for (const [name, field] of Object.entries(node.fields)) {
          const spans = field.spans ?? [];
          const element = document.createElement(spans.length ? "button" : "div");
          if (element instanceof HTMLButtonElement) {
            element.type = "button";
            element.title = "Navigate to field bytes";
            element.addEventListener("click", () => this.navigateSpan(spans[0]!, node.id));
          }
          const irrelevant = this.isIrrelevant(node, name);
          element.className = `field-row${spans.length ? " jump" : ""}`;
          if (this.options.dimIrrelevant && irrelevant) element.classList.add("irrelevant");
          const mark = document.createElement("span");
          mark.className = "field-mark";
          mark.style.background = this.displayColor(this.fieldColor(node, name), irrelevant);
          const label = document.createElement("span");
          label.className = "field-name";
          label.textContent = name;
          const value = document.createElement("span");
          value.className = "field-value";
          value.textContent = field.display ?? displayValue(field.value);
          const size = document.createElement("span");
          size.className = "field-size";
          const byteSize = spans.reduce((total, span) => total + span.end - span.start, 0);
          size.textContent = spans.length ? `${byteSize} byte${byteSize === 1 ? "" : "s"}` : "size unknown";
          element.append(mark, label, value, size);
          if (field.description) {
            const description = document.createElement("span");
            description.className = "field-description";
            description.textContent = field.description;
            element.append(description);
          }
          const evidence = evidenceText(field);
          if (evidence) {
            const evidenceElement = document.createElement("span");
            evidenceElement.className = "field-evidence";
            evidenceElement.textContent = evidence;
            evidenceElement.title = "Confirmed meanings come from a standard or validated contract. Inferences are qualified by their evidence. Irrelevant means the bytes are understood but carry no meaning for the supported interpretation.";
            element.append(evidenceElement);
          }
          section.append(element);
        }
        this.inspector.append(section);
      }
      if (Object.keys(node.properties).length) {
        const section = this.section();
        for (const [name, value] of Object.entries(node.properties)) section.append(this.kv(name, displayValue(value)));
        this.inspector.append(section);
      }
      const outgoing = this.graph.links.filter(link => link.source === node.id);
      const incoming = this.graph.links.filter(link => link.target === node.id);
      if (outgoing.length || incoming.length) {
        const section = this.section("references");
        for (const link of outgoing) section.append(this.referenceButton(link, true));
        for (const link of incoming) section.append(this.referenceButton(link, false));
        this.inspector.append(section);
      }
    }
    const covering = this.covering(this.current.artifactId, this.current.offset).filter(value => value.id !== node?.id);
    if (covering.length) {
      const section = this.section("also covers this byte");
      for (const value of covering) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "jump";
        button.textContent = value.label;
        button.addEventListener("click", () => this.selectNode(value.id, true));
        section.append(button);
      }
      this.inspector.append(section);
    }
  }

  private knowledgeElement(artifact: ArtifactData): HTMLElement {
    const knowledge = this.knowledge.get(artifact.id)!;
    const element = document.createElement("div");
    element.className = "artifact-knowledge";
    element.title = "Mapped coverage excludes a whole-file root annotation. Decoded coverage is the union of exact field spans; neither percentage claims that opaque bytes are understood.";
    const coverageLine = document.createElement("div");
    coverageLine.textContent = `regions mapped ${coverage(knowledge.mappedBytes, artifact.size)} · fields decoded ${coverage(knowledge.decodedFieldBytes, artifact.size)}`;
    const counts = document.createElement("div");
    counts.className = "artifact-knowledge-counts";
    const fieldSummary = knowledge.fieldCount
      ? `${knowledge.explainedFieldCount}/${knowledge.fieldCount} fields explained`
      : "no decoded fields";
    counts.textContent = `${knowledge.entityCount} entities · ${knowledge.relationshipCount} relationships · ${fieldSummary}`;
    element.append(coverageLine, counts);
    return element;
  }

  private section(label?: string): HTMLElement {
    const section = document.createElement("section");
    section.className = "inspector-section";
    if (label) {
      const heading = document.createElement("h2");
      heading.textContent = label;
      section.append(heading);
    }
    return section;
  }

  private kv(key: string, value: string): HTMLElement {
    const row = document.createElement("div");
    row.className = "kv";
    const keyElement = document.createElement("span");
    keyElement.className = "kv-key";
    keyElement.textContent = key;
    const valueElement = document.createElement("span");
    valueElement.className = "kv-value";
    valueElement.textContent = value;
    row.append(keyElement, valueElement);
    return row;
  }

  private spanButton(span: SpanData): HTMLElement {
    const artifact = this.artifacts.get(span.artifact)!;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "jump";
    button.textContent = `${artifact.name}  ${hex(span.start)}–${hex(span.end)}  (${span.end - span.start})`;
    button.addEventListener("click", () => this.navigateSpan(span, this.current.nodeId));
    return button;
  }

  private navigateSpan(span: SpanData, nodeId?: string): void {
    this.navigate({ artifactId: span.artifact, offset: span.start, end: span.end, nodeId }, true, true);
  }

  private referenceButton(link: LinkData, outgoing: boolean): HTMLElement {
    const targetId = outgoing ? link.target : link.source;
    const target = this.nodes.get(targetId);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "jump";
    const prefix = document.createElement("span");
    prefix.className = "ref-kind";
    prefix.textContent = `${outgoing ? "→" : "←"} ${link.kind}`;
    const label = document.createTextNode(target ? `  ${target.label}` : `  ${targetId}`);
    button.append(prefix, label);
    button.title = displayValue(link.properties);
    button.addEventListener("click", () => this.selectNode(targetId, true));
    return button;
  }

  private hexKey(event: KeyboardEvent): void {
    const artifact = this.artifacts.get(this.current.artifactId)!;
    let offset = this.current.offset;
    if (event.key === "ArrowLeft") offset -= 1;
    else if (event.key === "ArrowRight") offset += 1;
    else if (event.key === "ArrowUp") offset -= ROW_BYTES;
    else if (event.key === "ArrowDown") offset += ROW_BYTES;
    else if (event.key === "PageUp") offset -= ROW_BYTES * Math.max(1, Math.floor(this.hexScroll.clientHeight / ROW_HEIGHT));
    else if (event.key === "PageDown") offset += ROW_BYTES * Math.max(1, Math.floor(this.hexScroll.clientHeight / ROW_HEIGHT));
    else if (event.key === "Home") offset = event.ctrlKey ? 0 : Math.floor(offset / ROW_BYTES) * ROW_BYTES;
    else if (event.key === "End") offset = event.ctrlKey ? artifact.size - 1 : Math.min(artifact.size - 1, Math.floor(offset / ROW_BYTES) * ROW_BYTES + 15);
    else return;
    event.preventDefault();
    offset = Math.max(0, Math.min(Math.max(0, artifact.size - 1), offset));
    const primary = this.covering(artifact.id, offset)[0];
    this.navigate({ artifactId: artifact.id, offset, end: offset + 1, nodeId: primary?.id }, true, true);
  }

  private async copyLocation(): Promise<void> {
    const artifact = this.artifacts.get(this.current.artifactId)!;
    const range = this.current.end > this.current.offset + 1 ? `${hex(this.current.offset)}-${hex(this.current.end)}` : hex(this.current.offset);
    const value = `${artifact.id}:${range}`;
    try {
      await navigator.clipboard.writeText(value);
      this.setStatus(`copied ${value}`);
    } catch {
      this.setStatus(`copy unavailable · ${value}`);
    }
  }

  private setHovered(nodeId?: string): void {
    if (this.hoveredNode === nodeId) return;
    this.hoveredNode = nodeId;
    if (nodeId) {
      const node = this.nodes.get(nodeId);
      if (node) this.setStatus(`${node.label} · ${node.kind}`);
    } else {
      const artifact = this.artifacts.get(this.current.artifactId)!;
      this.setStatus(`${artifact.name}  ${hex(this.current.offset)}  ${this.current.end - this.current.offset} byte${this.current.end - this.current.offset === 1 ? "" : "s"}`);
    }
    this.renderHex();
  }

  private setStatus(value: string): void { this.status.textContent = value; }

  private errorElement(value: string): HTMLElement {
    const element = document.createElement("div");
    element.className = "error";
    element.textContent = value;
    return element;
  }
}

async function main(): Promise<void> {
  const response = await fetch(new URL("api/graph", window.location.href));
  if (!response.ok) throw new Error(`graph request failed: ${response.status}`);
  const graph = await response.json() as GraphData;
  if (graph.format !== "npunlock.artifact-graph" || graph.version !== 1) throw new Error("unsupported artifact graph format");
  new Viewer(graph);
}

main().catch(error => {
  const app = required<HTMLElement>("app");
  const element = document.createElement("pre");
  element.className = "error";
  element.textContent = error instanceof Error ? error.message : String(error);
  app.replaceChildren(element);
});
