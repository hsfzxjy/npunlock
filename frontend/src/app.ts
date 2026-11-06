import {
  NavigationHistory,
  sameLocation,
  sameShareableLocation,
  type ArrivalState,
  type HistoryMode,
  type LocationState,
} from "./navigation.js";
import {
  loadCompatibleValue,
  loadWorkspace,
  MAX_PERSISTED_HISTORY,
  saveCompatibleValue,
  saveWorkspace,
  workspaceStorageKey,
  type StorageLike,
  type WorkspaceState,
} from "./persistence.js";
import {
  formatLocationExpression,
  parseLocationExpression,
  searchEntries,
  type ParsedLocation,
  type SearchEntry,
} from "./search.js";
import { flattenTree, visibleWindow, type VisibleTreeItem } from "./tree.js";

interface SpanData { artifact: string; start: number; end: number; id?: string; label?: string; role?: string }
interface FieldData {
  value: unknown;
  display?: string;
  description?: string;
  properties?: Record<string, unknown>;
  spans?: SpanData[];
  confidence?: string;
  scope?: string;
  relevance?: string;
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
  confidence?: string;
  scope?: string;
  relevance?: string;
  contributes_to_coverage?: boolean;
}
interface LinkData {
  source: string;
  target: string;
  kind: string;
  properties: Record<string, unknown>;
  description?: string;
  confidence?: string;
  scope?: string;
}
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
interface OverviewSpan { start: number; end: number; color: string; irrelevant: boolean }
interface OverviewBase { canvas: HTMLCanvasElement; width: number; height: number; dimIrrelevant: boolean }
interface SelectionCandidate { node: NodeData; fieldName?: string; span: IndexedSpan }
interface NavigateOptions { history: HistoryMode; scroll: boolean; coalesceKey?: string }
type TreeMode = "all" | "relevant" | "artifact" | "covers" | "referenced";
type SearchTarget =
  | { kind: "node"; nodeId: string }
  | { kind: "field"; nodeId: string; fieldName: string }
  | { kind: "link"; link: LinkData; targetId: string }
  | { kind: "address"; nodeId: string; space: string; value: number }
  | {
    kind: "location";
    artifactId: string;
    start: number;
    end: number;
    nodeId?: string;
    fieldName?: string;
    spanIndex?: number;
  };
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
  collapseIrrelevantFields: boolean;
  treeVisible: boolean;
  inspectorVisible: boolean;
  treeWidth: number | null;
  inspectorWidth: number | null;
  treeMode: TreeMode;
}

const ROW_BYTES = 16;
const ROW_HEIGHT = 21;
const PAGE_BYTES = 16 * 1024;
const MAX_SCROLL_HEIGHT = 16_000_000;
const OVERSCAN = 8;
const TREE_OVERSCAN = 6;
const OPTIONS_KEY = "npunlock.viz.options.v1";
const DEFAULT_OPTIONS: ViewerOptions = {
  dimIrrelevant: true,
  collapseIrrelevantFields: false,
  treeVisible: true,
  inspectorVisible: true,
  treeWidth: null,
  inspectorWidth: null,
  treeMode: "all",
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

function evidenceText(value: {
  confidence?: string;
  scope?: string;
  relevance?: string;
  properties?: Record<string, unknown>;
}): string | undefined {
  const confidence = value.confidence ?? value.properties?.confidence;
  const scope = value.scope ?? value.properties?.scope;
  const relevance = value.relevance ?? value.properties?.relevance;
  const labels: Record<string, string> = {
    confirmed: "confirmed",
    "strong-inference": "strong inference",
    "execution-tested-inference": "execution-tested inference",
    raw: "raw observation",
    unresolved: "unresolved",
  };
  const parts: string[] = [];
  if (typeof confidence === "string") parts.push(labels[confidence] ?? confidence);
  if (relevance === "irrelevant") parts.push("irrelevant to supported interpretation");
  else if (relevance === "context") parts.push("context only");
  if (typeof scope === "string") parts.push(scope);
  return parts.length ? parts.join(" · ") : undefined;
}

function browserStorage(kind: "local" | "session"): StorageLike | undefined {
  try {
    return kind === "local" ? window.localStorage : window.sessionStorage;
  } catch {
    return undefined;
  }
}

function isViewerOptions(value: unknown): value is ViewerOptions {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const options = value as Record<string, unknown>;
  const width = (candidate: unknown, minimum: number, maximum: number) => candidate === null
    || (typeof candidate === "number"
      && Number.isFinite(candidate)
      && candidate >= minimum
      && candidate <= maximum);
  return typeof options.dimIrrelevant === "boolean"
    && typeof options.collapseIrrelevantFields === "boolean"
    && typeof options.treeVisible === "boolean"
    && typeof options.inspectorVisible === "boolean"
    && width(options.treeWidth, 150, 520)
    && width(options.inspectorWidth, 210, 620)
    && typeof options.treeMode === "string"
    && ["all", "relevant", "artifact", "covers", "referenced"].includes(options.treeMode);
}

function loadOptions(): ViewerOptions {
  const storage = browserStorage("local");
  if (!storage) return { ...DEFAULT_OPTIONS };
  const value = loadCompatibleValue(storage, OPTIONS_KEY, isViewerOptions);
  return value ? { ...value } : { ...DEFAULT_OPTIONS };
}

function saveOptions(value: ViewerOptions): void {
  const storage = browserStorage("local");
  if (storage) saveCompatibleValue(storage, OPTIONS_KEY, value);
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
  private overview = required<HTMLCanvasElement>("overview");
  private status = required<HTMLElement>("status");
  private workspace = required<HTMLElement>("workspace");
  private offsetInput = required<HTMLInputElement>("offset-input");
  private treeFilter = required<HTMLInputElement>("tree-filter");
  private treeMode = required<HTMLSelectElement>("tree-mode");
  private goToDialog = required<HTMLDialogElement>("go-to-dialog");
  private goToInput = required<HTMLInputElement>("go-to-input");
  private goToResults = required<HTMLElement>("go-to-results");
  private nodes = new Map<string, NodeData>();
  private artifacts = new Map<string, ArtifactData>();
  private categories = new Map<string, CategoryData>();
  private children = new Map<string, NodeData[]>();
  private roots: NodeData[] = [];
  private depths = new Map<string, number>();
  private spans = new Map<string, IndexedSpan[]>();
  private cache = new Map<string, ByteCache>();
  private knowledge = new Map<string, ArtifactKnowledge>();
  private overviewSpans = new Map<string, OverviewSpan[]>();
  private overviewBases = new Map<string, OverviewBase>();
  private expanded = new Set<string>();
  private current!: LocationState;
  private history = new NavigationHistory();
  private searchIndex: SearchEntry<SearchTarget>[] = [];
  private visibleSearchResults: SearchEntry<SearchTarget>[] = [];
  private searchSelection = 0;
  private hoveredNode?: string;
  private hoveredField?: string;
  private treeFocusId?: string;
  private treeRows: VisibleTreeItem<NodeData>[] = [];
  private treeMatches = new Set<string>();
  private treeFilterActive = false;
  private treeStructureDirty = true;
  private treeRenderFrame = 0;
  private treeSpace = document.createElement("div");
  private treeRowsLayer = document.createElement("div");
  private narrowMedia = window.matchMedia("(max-width: 820px)");
  private narrowDrawer?: "tree" | "inspector";
  private renderToken = 0;
  private options = loadOptions();
  private workspaceStorage = browserStorage("local");
  private workspaceKey?: string;
  private workspaceSaveTimer = 0;
  private restoringWorkspace = false;

  constructor(
    private graph: GraphData,
    private graphIdentity?: string,
    private backendStateVersion?: string,
  ) {
    if (graphIdentity && backendStateVersion) this.workspaceKey = workspaceStorageKey(graphIdentity);
    this.treeSpace.className = "tree-space";
    this.treeRowsLayer.className = "tree-rows";
    this.treeSpace.append(this.treeRowsLayer);
    for (const artifact of graph.artifacts) {
      this.artifacts.set(artifact.id, artifact);
      this.cache.set(artifact.id, new ByteCache(artifact));
      this.spans.set(artifact.id, []);
      this.overviewSpans.set(artifact.id, []);
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
        if (node.contributes_to_coverage !== false) {
          this.overviewSpans.get(span.artifact)?.push({
            start: span.start,
            end: span.end,
            color: this.categoryColor(node.category),
            irrelevant: this.isIrrelevant(node),
          });
        }
      }
      for (const [fieldName, field] of Object.entries(node.fields)) {
        for (const span of field.spans ?? []) {
          this.spans.get(span.artifact)?.push({ ...span, node, length: span.end - span.start, fieldName });
        }
      }
    }
    for (const values of this.spans.values()) values.sort((a, b) => a.start - b.start || a.end - b.end);
    for (const values of this.overviewSpans.values()) {
      values.sort((left, right) => (right.end - right.start) - (left.end - left.start));
    }
    for (const node of graph.nodes) {
      if ((this.children.get(node.id)?.length ?? 0) > 0) this.expanded.add(node.id);
    }
    for (const root of this.roots) {
      this.assignDepth(root, 0);
    }
    this.searchIndex = this.buildSearchIndex();
    for (const artifact of graph.artifacts) this.knowledge.set(artifact.id, this.buildKnowledge(artifact));
    const first = graph.artifacts[0];
    if (!first) throw new Error("artifact graph contains no artifacts");
    this.current = { artifactId: first.id, offset: 0, end: Math.min(1, first.size) };
    const initial = this.selectionCandidates(first.id, 0)[0];
    if (initial) {
      this.current.nodeId = initial.node.id;
      this.current.fieldName = initial.fieldName;
      this.current.spanIndex = this.candidateSpanIndex(initial);
    }
    const workspace = this.loadWorkspaceState();
    let restoredWorkspace: WorkspaceState | undefined;
    const fragment = window.location.hash.slice(1);
    let fragmentLocation: LocationState | undefined;
    if (fragment) {
      const sizes = new Map(this.graph.artifacts.map(artifact => [artifact.id, artifact.size]));
      const parsed = parseLocationExpression(fragment, this.current.artifactId, sizes);
      const restored = parsed ? this.resolveLocation(parsed) : undefined;
      if (restored) {
        fragmentLocation = {
          artifactId: restored.artifactId,
          offset: restored.start,
          end: restored.end,
          nodeId: restored.nodeId,
          fieldName: restored.fieldName,
          spanIndex: restored.spanIndex,
        };
      }
    }
    if (fragmentLocation) {
      this.current = fragmentLocation;
      if (workspace && sameShareableLocation(workspace.location, fragmentLocation)) {
        this.current = workspace.location;
        restoredWorkspace = workspace;
      }
      else if (workspace) this.discardWorkspaceState();
    } else if (workspace) {
      this.current = workspace.location;
      restoredWorkspace = workspace;
    }
    if (restoredWorkspace) {
      this.expanded = new Set(restoredWorkspace.expandedNodeIds);
      this.treeFilter.value = restoredWorkspace.treeFilter;
      if (!this.history.restore(restoredWorkspace.history)) {
        this.discardWorkspaceState();
        restoredWorkspace = undefined;
      }
    }
    this.restoringWorkspace = restoredWorkspace !== undefined;
    this.applyOptions();
    this.bind();
    this.renderTabs();
    this.renderTree();
    this.navigate(this.current, {
      history: restoredWorkspace ? "none" : "replace",
      scroll: Boolean(fragmentLocation) && !restoredWorkspace,
    });
    new ResizeObserver(() => this.renderHex()).observe(this.hexScroll);
    new ResizeObserver(() => this.renderTreeViewport()).observe(this.tree);
    new ResizeObserver(() => this.renderOverview()).observe(this.overview);
    if (restoredWorkspace) {
      requestAnimationFrame(() => this.restoreWorkspaceViewport(restoredWorkspace));
    } else {
      this.scheduleWorkspaceSave(0);
    }
  }

  private validateStoredLocation(location: LocationState): boolean {
    const artifact = this.artifacts.get(location.artifactId);
    if (!artifact
      || !Number.isSafeInteger(location.offset)
      || !Number.isSafeInteger(location.end)
      || location.offset < 0
      || (artifact.size === 0 ? location.offset !== 0 || location.end !== 0
        : location.offset >= artifact.size || location.end <= location.offset || location.end > artifact.size)) return false;
    if (!location.nodeId) {
      return location.fieldName === undefined
        && location.spanIndex === undefined
        && location.arrival === undefined;
    }
    const node = this.nodes.get(location.nodeId);
    if (!node) return false;
    const spans = location.fieldName === undefined
      ? node.spans
      : Object.hasOwn(node.fields, location.fieldName) ? node.fields[location.fieldName]!.spans ?? [] : undefined;
    if (!spans) return false;
    if (location.spanIndex !== undefined
      && (!Number.isSafeInteger(location.spanIndex)
        || location.spanIndex < 0
        || location.spanIndex >= spans.length)) return false;
    if (location.arrival) {
      if (location.arrival.targetNodeId !== location.nodeId
        || !this.nodes.has(location.arrival.sourceNodeId)
        || !this.nodes.has(location.arrival.targetNodeId)
        || !this.graph.links.some(link => link.source === location.arrival!.sourceNodeId
          && link.target === location.arrival!.targetNodeId
          && link.kind === location.arrival!.kind)) return false;
    }
    return true;
  }

  private loadWorkspaceState(): WorkspaceState | undefined {
    if (!this.workspaceStorage || !this.workspaceKey || !this.graphIdentity || !this.backendStateVersion) {
      return undefined;
    }
    return loadWorkspace(this.workspaceStorage, this.workspaceKey, {
      graphIdentity: this.graphIdentity,
      graphFormat: this.graph.format,
      graphVersion: this.graph.version,
      backendStateVersion: this.backendStateVersion,
      validateLocation: location => this.validateStoredLocation(location),
      validateNodeId: nodeId => this.nodes.has(nodeId),
      validateHexTopOffset: (artifactId, offset) => {
        const artifact = this.artifacts.get(artifactId);
        return Boolean(artifact) && offset >= 0 && (artifact!.size === 0 ? offset === 0 : offset < artifact!.size);
      },
    });
  }

  private discardWorkspaceState(): void {
    if (!this.workspaceStorage || !this.workspaceKey) return;
    try {
      this.workspaceStorage.removeItem(this.workspaceKey);
    } catch {
      // Storage is optional.
    }
  }

  private saveWorkspaceState(): void {
    this.workspaceSaveTimer = 0;
    if (this.restoringWorkspace
      || !this.workspaceStorage
      || !this.workspaceKey
      || !this.graphIdentity
      || !this.backendStateVersion) return;
    const history = this.history.snapshot(MAX_PERSISTED_HISTORY);
    const historyLocation = history.entries[history.index];
    if (!historyLocation || !sameLocation(historyLocation, this.current)) return;
    saveWorkspace(this.workspaceStorage, this.workspaceKey, {
      graphIdentity: this.graphIdentity,
      graphFormat: this.graph.format,
      graphVersion: this.graph.version,
      backendStateVersion: this.backendStateVersion,
      savedAt: Date.now(),
      location: this.current,
      history,
      expandedNodeIds: [...this.expanded],
      treeFilter: this.treeFilter.value,
      treeScrollTop: this.tree.scrollTop,
      hexTopOffset: this.hexTopOffset(),
      inspectorScrollTop: this.inspector.scrollTop,
    });
  }

  private scheduleWorkspaceSave(delay = 160): void {
    if (this.restoringWorkspace
      || !this.workspaceStorage
      || !this.workspaceKey
      || !this.graphIdentity
      || !this.backendStateVersion) return;
    if (this.workspaceSaveTimer) window.clearTimeout(this.workspaceSaveTimer);
    this.workspaceSaveTimer = window.setTimeout(() => this.saveWorkspaceState(), delay);
  }

  private flushWorkspaceState(): void {
    if (this.workspaceSaveTimer) window.clearTimeout(this.workspaceSaveTimer);
    this.workspaceSaveTimer = 0;
    this.saveWorkspaceState();
  }

  private restoreWorkspaceViewport(workspace: WorkspaceState): void {
    this.tree.scrollTop = workspace.treeScrollTop;
    this.renderTreeViewport();
    this.setHexTopOffset(workspace.hexTopOffset);
    this.inspector.scrollTop = workspace.inspectorScrollTop;
    this.restoringWorkspace = false;
    this.scheduleWorkspaceSave(0);
  }

  private assignDepth(node: NodeData, depth: number): void {
    this.depths.set(node.id, depth);
    for (const child of this.children.get(node.id) ?? []) this.assignDepth(child, depth + 1);
  }

  private nodePath(node: NodeData): string {
    const labels = [node.label];
    let current = node;
    while (current.parent) {
      const parent = this.nodes.get(current.parent);
      if (!parent) break;
      labels.push(parent.label);
      current = parent;
    }
    return labels.reverse().join(" › ");
  }

  private spanDetail(spans: SpanData[]): string {
    const span = spans[0];
    if (!span) return "logical entity";
    const artifact = this.artifacts.get(span.artifact);
    const label = span.label ? `${span.label} · ` : "";
    return `${label}${artifact?.name ?? span.artifact} ${hex(span.start)}–${hex(span.end)} · ${span.end - span.start} B`;
  }

  private buildSearchIndex(): SearchEntry<SearchTarget>[] {
    const entries: SearchEntry<SearchTarget>[] = [];
    for (const node of this.graph.nodes) {
      const path = this.nodePath(node);
      entries.push({
        id: `node:${node.id}`,
        label: node.label,
        detail: `${path} · ${this.spanDetail(node.spans)} · ${node.kind}`,
        terms: [node.id, node.kind, node.description ?? "", path],
        value: { kind: "node", nodeId: node.id },
      });
      for (const [fieldName, field] of Object.entries(node.fields)) {
        entries.push({
          id: `field:${node.id}:${fieldName}`,
          label: `${node.label} › ${fieldName}`,
          detail: `${path} · ${this.spanDetail(field.spans ?? [])}`,
          terms: [node.id, node.kind, fieldName, field.description ?? "", evidenceText(field) ?? ""],
          value: { kind: "field", nodeId: node.id, fieldName },
        });
      }
      for (const [name, address] of Object.entries(node.addresses)) {
        entries.push({
          id: `address:${node.id}:${name}`,
          label: `${address.space}:${hex(address.value)}`,
          detail: `${path} · ${name}`,
          terms: [node.id, node.label, name, address.space],
          value: { kind: "address", nodeId: node.id, space: address.space, value: address.value },
        });
      }
    }
    for (const [index, link] of this.graph.links.entries()) {
      const source = this.nodes.get(link.source);
      const target = this.nodes.get(link.target);
      entries.push({
        id: `link:${index}`,
        label: `${source?.label ?? link.source} → ${target?.label ?? link.target}`,
        detail: `${link.kind} · ${target ? this.nodePath(target) : link.target}`,
        terms: [link.kind, link.source, link.target, displayValue(link.properties)],
        value: { kind: "link", link, targetId: link.target },
      });
    }
    return entries;
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
      && node.contributes_to_coverage !== false
      && !(this.graph.version === 1 && node.parent === undefined && span.start === 0 && span.end === artifact.size)
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
      this.scheduleWorkspaceSave();
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
      this.navigate(
        { artifactId: artifact.id, offset: parsed, end: parsed + 1 },
        { history: "push", scroll: true },
      );
    });
    required<HTMLButtonElement>("copy").addEventListener("click", () => this.copyLocation());
    required<HTMLButtonElement>("open-go-to").addEventListener("click", () => this.openGoTo());
    required<HTMLButtonElement>("back").addEventListener("click", () => this.moveHistory(-1));
    required<HTMLButtonElement>("forward").addEventListener("click", () => this.moveHistory(1));
    required<HTMLButtonElement>("toggle-tree").addEventListener("click", event => this.togglePanel("tree", event));
    required<HTMLButtonElement>("toggle-inspector").addEventListener("click", event => this.togglePanel("inspector", event));
    required<HTMLButtonElement>("toggle-dim-irrelevant").addEventListener("click", () => this.toggleDimIrrelevant());
    this.overview.addEventListener("click", event => {
      const bounds = this.overview.getBoundingClientRect();
      this.navigateOverview((event.clientY - bounds.top) / Math.max(1, bounds.height), "push");
    });
    this.overview.addEventListener("keydown", event => this.overviewKey(event));
    this.treeFilter.addEventListener("input", () => {
      this.treeStructureDirty = true;
      this.renderTree();
      this.scheduleWorkspaceSave();
    });
    this.treeMode.addEventListener("change", () => {
      const mode = this.treeMode.value;
      if (!(["all", "relevant", "artifact", "covers", "referenced"] as string[]).includes(mode)) return;
      this.options.treeMode = mode as TreeMode;
      saveOptions(this.options);
      this.treeStructureDirty = true;
      this.renderTree();
    });
    this.tree.addEventListener("scroll", () => {
      this.scheduleWorkspaceSave();
      if (this.treeRenderFrame) return;
      this.treeRenderFrame = requestAnimationFrame(() => {
        this.treeRenderFrame = 0;
        this.renderTreeViewport();
      });
    });
    this.inspector.addEventListener("scroll", () => this.scheduleWorkspaceSave());
    this.narrowMedia.addEventListener("change", () => {
      this.narrowDrawer = undefined;
      this.applyOptions();
      this.renderHex();
    });
    required<HTMLElement>("hex-panel").addEventListener("pointerdown", () => this.closeNarrowDrawer());
    this.goToInput.addEventListener("input", () => this.renderGoTo());
    this.goToInput.addEventListener("keydown", event => this.goToKey(event));
    required<HTMLFormElement>("go-to-form").addEventListener("submit", event => {
      event.preventDefault();
      this.activateSearchResult(this.visibleSearchResults[this.searchSelection]);
    });
    this.goToDialog.addEventListener("click", event => {
      if (event.target === this.goToDialog) this.goToDialog.close();
    });
    this.bindSplitter(required<HTMLElement>("left-splitter"), "--tree-width", "treeWidth", true, 150, 520);
    this.bindSplitter(required<HTMLElement>("right-splitter"), "--inspector-width", "inspectorWidth", false, 210, 620);
    window.addEventListener("keydown", event => {
      if (event.altKey && event.key === "ArrowLeft") { event.preventDefault(); this.moveHistory(-1); }
      if (event.altKey && event.key === "ArrowRight") { event.preventDefault(); this.moveHistory(1); }
      if (event.ctrlKey && event.key.toLowerCase() === "k") { event.preventDefault(); this.openGoTo(); }
      if (event.key === "Escape" && this.narrowDrawer) {
        event.preventDefault();
        this.closeNarrowDrawer();
      }
      if (event.key === "/" && !event.ctrlKey && !event.altKey && !event.metaKey
        && !(event.target instanceof HTMLInputElement) && !(event.target instanceof HTMLSelectElement)) {
        event.preventDefault();
        this.treeFilter.focus();
        this.treeFilter.select();
      }
    });
    window.addEventListener("pagehide", () => this.flushWorkspaceState());
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden") this.flushWorkspaceState();
    });
  }

  private applyOptions(): void {
    this.workspace.classList.toggle("tree-hidden", !this.options.treeVisible);
    this.workspace.classList.toggle("inspector-hidden", !this.options.inspectorVisible);
    this.workspace.classList.toggle("narrow-tree-open", this.narrowDrawer === "tree");
    this.workspace.classList.toggle("narrow-inspector-open", this.narrowDrawer === "inspector");
    const narrow = this.narrowMedia.matches;
    required<HTMLButtonElement>("toggle-tree").setAttribute(
      "aria-pressed", String(narrow ? this.narrowDrawer === "tree" : this.options.treeVisible));
    required<HTMLButtonElement>("toggle-inspector").setAttribute(
      "aria-pressed", String(narrow ? this.narrowDrawer === "inspector" : this.options.inspectorVisible));
    required<HTMLButtonElement>("toggle-dim-irrelevant").setAttribute("aria-pressed", String(this.options.dimIrrelevant));
    this.treeMode.value = this.options.treeMode;
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
    if (this.narrowMedia.matches) {
      this.narrowDrawer = this.narrowDrawer === panel ? undefined : panel;
      this.applyOptions();
      if (this.narrowDrawer === "tree") this.renderTree();
      if (this.narrowDrawer === "inspector") this.renderInspector();
      this.renderHex();
      return;
    }
    const button = event.currentTarget as HTMLButtonElement;
    const className = `${panel}-hidden`;
    const hidden = this.workspace.classList.toggle(className);
    button.setAttribute("aria-pressed", String(!hidden));
    if (panel === "tree") this.options.treeVisible = !hidden;
    else this.options.inspectorVisible = !hidden;
    saveOptions(this.options);
    this.renderHex();
  }

  private closeNarrowDrawer(): void {
    if (!this.narrowDrawer) return;
    this.narrowDrawer = undefined;
    this.applyOptions();
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
    this.renderOverview();
  }

  private openGoTo(): void {
    if (!this.goToDialog.open) this.goToDialog.showModal();
    this.goToInput.value = "";
    this.searchSelection = 0;
    this.renderGoTo();
    requestAnimationFrame(() => this.goToInput.focus());
  }

  private renderGoTo(): void {
    const query = this.goToInput.value;
    const artifactSizes = new Map(this.graph.artifacts.map(artifact => [artifact.id, artifact.size]));
    const parsed = parseLocationExpression(query, this.current.artifactId, artifactSizes);
    const location = parsed ? this.resolveLocation(parsed) : undefined;
    const matches = searchEntries(this.searchIndex, query, 60);
    this.visibleSearchResults = location
      ? [{
        id: `location:${formatLocationExpression(location)}`,
        label: `${location.artifactId}:${hex(location.start)}${location.end > location.start + 1 ? `-${hex(location.end)}` : ""}`,
        detail: [
          `${location.end - location.start} byte${location.end - location.start === 1 ? "" : "s"}`,
          location.nodeId ? this.nodePath(this.nodes.get(location.nodeId)!) : "file offset",
          location.fieldName,
        ].filter(Boolean).join(" · "),
        terms: [],
        value: location,
      }, ...matches]
      : matches;
    this.searchSelection = Math.max(0, Math.min(this.searchSelection, this.visibleSearchResults.length - 1));
    this.goToResults.replaceChildren();
    if (!this.visibleSearchResults.length) {
      const empty = document.createElement("div");
      empty.className = "go-to-empty";
      empty.textContent = query.trim() ? "no matching semantic location" : "type a node, field, relation, or offset";
      this.goToResults.append(empty);
      this.goToInput.removeAttribute("aria-activedescendant");
      return;
    }
    const fragment = document.createDocumentFragment();
    for (const [index, entry] of this.visibleSearchResults.entries()) {
      const button = document.createElement("button");
      button.type = "button";
      button.id = `go-to-result-${index}`;
      button.className = "go-to-result";
      button.role = "option";
      button.tabIndex = -1;
      button.setAttribute("aria-selected", String(index === this.searchSelection));
      const label = document.createElement("span");
      label.className = "go-to-result-label";
      label.textContent = entry.label;
      const detail = document.createElement("span");
      detail.className = "go-to-result-detail";
      detail.textContent = entry.detail;
      button.append(label, detail);
      button.addEventListener("mousemove", () => {
        if (this.searchSelection === index) return;
        this.searchSelection = index;
        this.updateSearchSelection();
      });
      button.addEventListener("click", () => this.activateSearchResult(entry));
      fragment.append(button);
    }
    this.goToResults.append(fragment);
    this.updateSearchSelection();
  }

  private resolveLocation(parsed: ParsedLocation): Extract<SearchTarget, { kind: "location" }> | undefined {
    if (!parsed.nodeId) return { kind: "location", ...parsed };
    const node = this.nodes.get(parsed.nodeId);
    if (!node) return undefined;
    if (parsed.fieldName && !node.fields[parsed.fieldName]) return undefined;
    const spans = parsed.fieldName ? node.fields[parsed.fieldName]!.spans ?? [] : node.spans;
    if (parsed.spanIndex !== undefined && parsed.spanIndex >= spans.length) return undefined;
    return { kind: "location", ...parsed };
  }

  private updateSearchSelection(): void {
    for (const [index, element] of Array.from(this.goToResults.querySelectorAll<HTMLElement>(".go-to-result")).entries()) {
      element.setAttribute("aria-selected", String(index === this.searchSelection));
      if (index === this.searchSelection) element.scrollIntoView({ block: "nearest" });
    }
    if (this.visibleSearchResults.length) {
      this.goToInput.setAttribute("aria-activedescendant", `go-to-result-${this.searchSelection}`);
    }
  }

  private goToKey(event: KeyboardEvent): void {
    if (!this.visibleSearchResults.length) return;
    let next = this.searchSelection;
    if (event.key === "ArrowDown") next = Math.min(this.visibleSearchResults.length - 1, next + 1);
    else if (event.key === "ArrowUp") next = Math.max(0, next - 1);
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = this.visibleSearchResults.length - 1;
    else return;
    event.preventDefault();
    this.searchSelection = next;
    this.updateSearchSelection();
  }

  private activateSearchResult(entry: SearchEntry<SearchTarget> | undefined): void {
    if (!entry) return;
    const target = entry.value;
    this.goToDialog.close();
    if (target.kind === "location") {
      this.navigate(
        {
          artifactId: target.artifactId,
          offset: target.start,
          end: target.end,
          nodeId: target.nodeId,
          fieldName: target.fieldName,
          spanIndex: target.spanIndex,
        },
        { history: "push", scroll: true },
      );
      this.hexScroll.focus();
      return;
    }
    if (target.kind === "node" || target.kind === "address") {
      this.selectNode(target.nodeId, true);
      return;
    }
    if (target.kind === "link") {
      this.selectNode(target.targetId, true, {
        sourceNodeId: target.link.source,
        targetNodeId: target.link.target,
        kind: target.link.kind,
      });
      return;
    }
    const node = this.nodes.get(target.nodeId);
    const field = node?.fields[target.fieldName];
    const spans = field?.spans ?? [];
    let index = spans.findIndex(span => span.artifact === this.current.artifactId);
    if (index < 0) index = spans.length ? 0 : -1;
    const span = index >= 0 ? spans[index] : undefined;
    if (span) this.navigateSpan(span, target.nodeId, target.fieldName, index);
    else this.selectNode(target.nodeId, true);
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
    this.navigate(
      { artifactId: artifact.id, offset, end: Math.min(artifact.size, offset + 1) },
      { history: "push", scroll: true },
    );
    if (restoreFocus) {
      requestAnimationFrame(() => this.artifactTabs.querySelector<HTMLElement>(`[data-artifact-id="${CSS.escape(artifact.id)}"]`)?.focus());
    }
  }

  private treeFilterState(): { included?: Set<string>; matches: Set<string>; active: boolean } {
    const query = this.treeFilter.value.trim().toLowerCase();
    const mode = this.options.treeMode;
    const referenced = new Set<string>();
    if (mode === "referenced" && this.current.nodeId) {
      referenced.add(this.current.nodeId);
      for (const link of this.graph.links) {
        if (link.source === this.current.nodeId) referenced.add(link.target);
        if (link.target === this.current.nodeId) referenced.add(link.source);
      }
    }
    const covering = mode === "covers"
      ? new Set(this.covering(this.current.artifactId, this.current.offset).map(node => node.id))
      : new Set<string>();
    const matches = new Set<string>();
    for (const node of this.graph.nodes) {
      let modeMatch = true;
      if (mode === "relevant") modeMatch = !this.isIrrelevant(node);
      else if (mode === "artifact") {
        modeMatch = node.spans.some(span => span.artifact === this.current.artifactId)
          || Object.values(node.fields).some(field =>
            (field.spans ?? []).some(span => span.artifact === this.current.artifactId));
      } else if (mode === "covers") modeMatch = covering.has(node.id);
      else if (mode === "referenced") modeMatch = referenced.has(node.id);
      if (!modeMatch) continue;
      const text = [
        node.label,
        node.id,
        node.kind,
        node.description ?? "",
        ...Object.entries(node.fields).flatMap(([name, field]) => [name, field.description ?? ""]),
      ].join(" ").toLowerCase();
      if (!query || query.split(/\s+/).every(token => text.includes(token))) matches.add(node.id);
    }
    const active = Boolean(query) || mode !== "all";
    if (!active) return { matches, active };
    const included = new Set(matches);
    for (const nodeId of matches) {
      let node = this.nodes.get(nodeId);
      while (node?.parent) {
        included.add(node.parent);
        node = this.nodes.get(node.parent);
      }
    }
    return { included, matches, active };
  }

  private renderTree(): void {
    if (this.treeStructureDirty) {
      const filter = this.treeFilterState();
      this.treeMatches = filter.matches;
      this.treeFilterActive = filter.active;
      this.treeRows = flattenTree(this.roots, this.children, this.expanded, filter.included, filter.active);
      this.treeStructureDirty = false;
      this.treeSpace.style.height = `${Math.max(1, this.treeRows.length * ROW_HEIGHT + 6)}px`;
    }
    if (!this.treeRows.length) {
      const empty = document.createElement("div");
      empty.className = "go-to-empty";
      empty.textContent = "no matching entities";
      this.tree.replaceChildren(empty);
      return;
    }
    if (this.treeSpace.parentElement !== this.tree) this.tree.replaceChildren(this.treeSpace);
    this.renderTreeViewport();
  }

  private renderTreeViewport(): void {
    if (!this.treeRows.length || this.treeSpace.parentElement !== this.tree) return;
    const window = visibleWindow(
      this.treeRows.length,
      this.tree.scrollTop,
      this.tree.clientHeight,
      ROW_HEIGHT,
      TREE_OVERSCAN,
    );
    const fragment = document.createDocumentFragment();
    for (let index = window.start; index < window.end; index += 1) {
      const item = this.treeRows[index]!;
      fragment.append(this.treeRow(item.node, item.depth, index));
    }
    this.treeRowsLayer.replaceChildren(fragment);
  }

  private treeRow(node: NodeData, depth: number, index: number): HTMLElement {
    const row = document.createElement("div");
    row.className = `tree-row${this.current.nodeId === node.id ? " selected" : ""}`;
    if (this.options.dimIrrelevant && this.isIrrelevant(node)) row.classList.add("irrelevant");
    row.role = "treeitem";
    row.dataset.nodeId = node.id;
    row.dataset.rowIndex = String(index);
    row.tabIndex = (this.treeFocusId ?? this.current.nodeId ?? this.roots[0]?.id) === node.id ? 0 : -1;
    row.style.top = `${index * ROW_HEIGHT + 3}px`;
    row.style.paddingLeft = `${depth * 14 + 3}px`;
    row.setAttribute("aria-level", String(depth + 1));
    const siblings = node.parent ? this.children.get(node.parent) ?? [] : this.roots;
    row.setAttribute("aria-setsize", String(siblings.length));
    row.setAttribute("aria-posinset", String(siblings.findIndex(value => value.id === node.id) + 1));
    row.setAttribute("aria-selected", String(this.current.nodeId === node.id));
    const children = this.children.get(node.id) ?? [];
    const effectivelyExpanded = this.treeFilterActive || this.expanded.has(node.id);
    if (children.length) row.setAttribute("aria-expanded", String(effectivelyExpanded));
    const expander = document.createElement("button");
    expander.type = "button";
    expander.className = `tree-expander${children.length ? "" : " empty"}`;
    expander.tabIndex = -1;
    expander.textContent = effectivelyExpanded ? "▾" : "▸";
    expander.setAttribute("aria-label", effectivelyExpanded ? "Collapse" : "Expand");
    expander.disabled = this.treeFilterActive;
    expander.addEventListener("click", event => {
      event.stopPropagation();
      if (this.expanded.has(node.id)) this.expanded.delete(node.id); else this.expanded.add(node.id);
      this.treeStructureDirty = true;
      this.renderTree();
      this.focusTreeNode(node.id, false);
      this.scheduleWorkspaceSave(0);
    });
    const mark = document.createElement("span");
    mark.className = "tree-mark";
    mark.style.background = this.displayColor(this.categoryColor(node.category), this.isIrrelevant(node));
    const label = document.createElement("span");
    label.className = "tree-label";
    if (this.treeFilter.value.trim() && this.treeMatches.has(node.id)) label.classList.add("match");
    label.textContent = node.label;
    label.title = node.description ? `${node.label}\n${node.description}` : node.label;
    const kind = document.createElement("span");
    kind.className = "tree-kind";
    kind.textContent = node.kind;
    row.append(expander, mark, label, kind);
    row.addEventListener("click", () => {
      this.treeFocusId = node.id;
      this.closeNarrowDrawer();
      this.selectNode(node.id, true);
    });
    row.addEventListener("focus", () => { this.treeFocusId = node.id; });
    row.addEventListener("keydown", event => this.treeKey(event, node));
    row.addEventListener("mouseenter", () => this.setHovered(node.id));
    row.addEventListener("mouseleave", () => this.setHovered(undefined));
    return row;
  }

  private treeKey(event: KeyboardEvent, node: NodeData): void {
    const index = this.treeRows.findIndex(item => item.node.id === node.id);
    let targetId: string | undefined;
    if (event.key === "ArrowDown") targetId = this.treeRows[index + 1]?.node.id;
    if (event.key === "ArrowUp") targetId = this.treeRows[index - 1]?.node.id;
    if (event.key === "Home") targetId = this.treeRows[0]?.node.id;
    if (event.key === "End") targetId = this.treeRows[this.treeRows.length - 1]?.node.id;
    if (event.key === "ArrowRight") {
      const children = this.children.get(node.id) ?? [];
      if (children.length && !this.expanded.has(node.id)) {
        this.expanded.add(node.id);
        this.treeStructureDirty = true;
        this.renderTree();
        this.scheduleWorkspaceSave(0);
        targetId = node.id;
      } else if (children.length) {
        targetId = children[0]!.id;
      }
    }
    if (event.key === "ArrowLeft") {
      if (this.expanded.has(node.id) && (this.children.get(node.id)?.length ?? 0) > 0) {
        this.expanded.delete(node.id);
        this.treeStructureDirty = true;
        this.renderTree();
        this.scheduleWorkspaceSave(0);
        targetId = node.id;
      } else if (node.parent) {
        targetId = node.parent;
      }
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      this.treeFocusId = node.id;
      this.selectNode(node.id, true);
      this.focusTreeNode(node.id, true);
      return;
    }
    if (!targetId) return;
    event.preventDefault();
    this.focusTreeNode(targetId, true);
  }

  private focusTreeNode(nodeId: string, focus: boolean): void {
    const index = this.treeRows.findIndex(item => item.node.id === nodeId);
    if (index < 0) return;
    this.treeFocusId = nodeId;
    const top = index * ROW_HEIGHT + 3;
    const bottom = top + ROW_HEIGHT;
    if (this.tree.clientHeight <= 0) this.tree.scrollTop = top;
    else if (top < this.tree.scrollTop) this.tree.scrollTop = top;
    else if (bottom > this.tree.scrollTop + this.tree.clientHeight) {
      this.tree.scrollTop = Math.max(0, bottom - this.tree.clientHeight);
    }
    this.renderTreeViewport();
    if (focus) {
      requestAnimationFrame(() => this.tree.querySelector<HTMLElement>(
        `[data-node-id="${CSS.escape(nodeId)}"]`,
      )?.focus());
    }
  }

  private selectNode(nodeId: string, push: boolean, arrival?: ArrivalState): void {
    const node = this.nodes.get(nodeId);
    if (!node) return;
    this.closeNarrowDrawer();
    let spanIndex = node.spans.findIndex(value => value.artifact === this.current.artifactId);
    if (spanIndex < 0) spanIndex = node.spans.length ? 0 : -1;
    const span = spanIndex >= 0 ? node.spans[spanIndex] : undefined;
    if (!span) {
      this.navigate(
        { ...this.current, nodeId, fieldName: undefined, spanIndex: undefined, arrival },
        { history: push ? "push" : "none", scroll: false },
      );
      return;
    }
    this.navigate(
      {
        artifactId: span.artifact,
        offset: span.start,
        end: span.end,
        nodeId,
        spanIndex,
        arrival,
      },
      { history: push ? "push" : "none", scroll: true },
    );
  }

  private navigate(location: LocationState, options: NavigateOptions): void {
    const artifact = this.artifacts.get(location.artifactId);
    if (!artifact) return;
    const offset = Math.max(0, Math.min(Math.max(0, artifact.size - 1), location.offset));
    const end = Math.max(offset + (artifact.size ? 1 : 0), Math.min(artifact.size, location.end));
    this.current = { ...location, offset, end };
    this.history.record(this.current, options.history, options.coalesceKey);
    window.history.replaceState(null, "", `#${this.currentLocationExpression()}`);
    const expanded = this.current.nodeId ? this.revealTree(this.current.nodeId) : false;
    if (expanded || ["artifact", "covers", "referenced"].includes(this.options.treeMode)) {
      this.treeStructureDirty = true;
    }
    this.offsetInput.value = hex(this.current.offset);
    this.renderTabs();
    this.renderTree();
    if (this.current.nodeId) this.focusTreeNode(this.current.nodeId, false);
    this.renderInspector();
    this.updateHistoryButtons();
    if (options.scroll) this.scrollToOffset(this.current.offset);
    else this.renderHex();
    this.renderOverview();
    this.renderContextStatus();
    this.scheduleWorkspaceSave(0);
  }

  private revealTree(nodeId: string): boolean {
    let changed = false;
    let node = this.nodes.get(nodeId);
    while (node?.parent) {
      if (!this.expanded.has(node.parent)) changed = true;
      this.expanded.add(node.parent);
      node = this.nodes.get(node.parent);
    }
    return changed;
  }

  private moveHistory(delta: number): void {
    const location = this.history.move(delta);
    if (location) this.navigate(location, { history: "none", scroll: true });
  }

  private updateHistoryButtons(): void {
    required<HTMLButtonElement>("back").disabled = !this.history.canBack;
    required<HTMLButtonElement>("forward").disabled = !this.history.canForward;
  }

  private scrollToOffset(offset: number): void {
    this.setHexTopOffset(offset);
  }

  private setHexTopOffset(offset: number): void {
    const artifact = this.artifacts.get(this.current.artifactId)!;
    const rowCount = Math.ceil(artifact.size / ROW_BYTES);
    const total = Math.max(ROW_HEIGHT, rowCount * ROW_HEIGHT);
    const visual = Math.min(MAX_SCROLL_HEIGHT, total);
    this.hexSpace.style.height = `${visual}px`;
    if (total <= MAX_SCROLL_HEIGHT) {
      this.hexScroll.scrollTop = Math.floor(offset / ROW_BYTES) * ROW_HEIGHT;
    } else {
      const visible = Math.max(1, Math.ceil(this.hexScroll.clientHeight / ROW_HEIGHT));
      const maximumFirst = Math.max(0, rowCount - visible);
      const first = Math.max(0, Math.min(maximumFirst, Math.floor(offset / ROW_BYTES)));
      const maxScroll = Math.max(1, visual - this.hexScroll.clientHeight);
      this.hexScroll.scrollTop = maximumFirst ? (first / maximumFirst) * maxScroll : 0;
    }
    this.renderHex();
  }

  private hexTopOffset(): number {
    const artifact = this.artifacts.get(this.current.artifactId)!;
    if (!artifact.size) return 0;
    const rowCount = Math.ceil(artifact.size / ROW_BYTES);
    const total = Math.max(ROW_HEIGHT, rowCount * ROW_HEIGHT);
    let row: number;
    if (total <= MAX_SCROLL_HEIGHT) {
      row = Math.floor(this.hexScroll.scrollTop / ROW_HEIGHT);
    } else {
      const visible = Math.max(1, Math.ceil(this.hexScroll.clientHeight / ROW_HEIGHT));
      const maximumFirst = Math.max(0, rowCount - visible);
      const maximumScroll = Math.max(1, MAX_SCROLL_HEIGHT - this.hexScroll.clientHeight);
      row = Math.floor((this.hexScroll.scrollTop / maximumScroll) * maximumFirst);
    }
    return Math.min(artifact.size - 1, Math.max(0, row * ROW_BYTES));
  }

  private renderOverview(): void {
    const artifact = this.artifacts.get(this.current.artifactId);
    const context = this.overview.getContext("2d");
    const bounds = this.overview.getBoundingClientRect();
    if (!artifact || !context || bounds.width <= 0 || bounds.height <= 0) return;
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(bounds.width * ratio));
    const height = Math.max(1, Math.round(bounds.height * ratio));
    if (this.overview.width !== width || this.overview.height !== height) {
      this.overview.width = width;
      this.overview.height = height;
    }
    context.setTransform(1, 0, 0, 1, 0, 0);
    let base = this.overviewBases.get(artifact.id);
    if (!base || base.width !== width || base.height !== height
      || base.dimIrrelevant !== this.options.dimIrrelevant) {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const baseContext = canvas.getContext("2d")!;
      baseContext.fillStyle = getComputedStyle(this.overview).backgroundColor;
      baseContext.fillRect(0, 0, width, height);
      const baseScale = height / Math.max(1, artifact.size);
      for (const span of this.overviewSpans.get(artifact.id) ?? []) {
        const top = Math.floor(span.start * baseScale);
        const bottom = Math.max(top + 1, Math.ceil(span.end * baseScale));
        baseContext.globalAlpha = span.irrelevant ? (this.options.dimIrrelevant ? 0.12 : 0.3) : 0.22;
        baseContext.fillStyle = span.color;
        baseContext.fillRect(ratio, top, Math.max(1, width - ratio), bottom - top);
      }
      base = { canvas, width, height, dimIrrelevant: this.options.dimIrrelevant };
      this.overviewBases.set(artifact.id, base);
    }
    context.clearRect(0, 0, width, height);
    context.drawImage(base.canvas, 0, 0);
    const scale = height / Math.max(1, artifact.size);
    const referenced = new Set<string>();
    if (this.current.nodeId) {
      for (const link of this.graph.links) {
        if (link.source === this.current.nodeId) referenced.add(link.target);
        if (link.target === this.current.nodeId) referenced.add(link.source);
      }
    }
    context.globalAlpha = 0.85;
    context.fillStyle = this.categoryColor("reference");
    for (const nodeId of referenced) {
      for (const span of this.nodes.get(nodeId)?.spans ?? []) {
        if (span.artifact !== artifact.id) continue;
        const top = Math.floor(span.start * scale);
        context.fillRect(0, top, width, Math.max(1, ratio));
      }
    }
    const selectionTop = Math.floor(this.current.offset * scale);
    const selectionBottom = Math.max(selectionTop + 2, Math.ceil(this.current.end * scale));
    context.globalAlpha = 1;
    context.fillStyle = "#176da8";
    context.fillRect(0, selectionTop, width, Math.max(2 * ratio, selectionBottom - selectionTop));
    context.globalAlpha = 1;
    this.overview.title = `${artifact.name} · ${byteCount(artifact.size)}\nsemantic regions, references, and current selection`;
  }

  private navigateOverview(position: number, history: HistoryMode): void {
    const artifact = this.artifacts.get(this.current.artifactId)!;
    if (!artifact.size) return;
    const offset = Math.max(0, Math.min(artifact.size - 1, Math.floor(
      Math.max(0, Math.min(1, position)) * artifact.size,
    )));
    const primary = this.selectionCandidates(artifact.id, offset)[0];
    this.navigate(
      {
        artifactId: artifact.id,
        offset,
        end: offset + 1,
        nodeId: primary?.node.id,
        fieldName: primary?.fieldName,
        spanIndex: primary ? this.candidateSpanIndex(primary) : undefined,
      },
      { history, scroll: true },
    );
    this.overview.focus();
  }

  private overviewKey(event: KeyboardEvent): void {
    const artifact = this.artifacts.get(this.current.artifactId)!;
    if (!artifact.size) return;
    const small = Math.max(1, Math.floor(artifact.size / 100));
    const large = Math.max(1, Math.floor(artifact.size / 10));
    let offset = this.current.offset;
    if (event.key === "ArrowUp") offset -= small;
    else if (event.key === "ArrowDown") offset += small;
    else if (event.key === "PageUp") offset -= large;
    else if (event.key === "PageDown") offset += large;
    else if (event.key === "Home") offset = 0;
    else if (event.key === "End") offset = artifact.size - 1;
    else return;
    event.preventDefault();
    this.navigateOverview(offset / Math.max(1, artifact.size), "push");
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
      values.sort((a, b) => this.compareIndexedSpans(a, b));
    }
    const focusedNode = this.current.nodeId ? this.nodes.get(this.current.nodeId) : undefined;
    const focusedSource = this.current.fieldName
      ? focusedNode?.fields[this.current.fieldName]?.spans ?? []
      : focusedNode?.spans ?? [];
    const focusedSpans = focusedSource.filter(span => span.artifact === artifact.id);
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
      row.append(offsetElement, hexCells, asciiCells, this.focusedRangeGuide(rowOffset, focusedSpans));
      fragment.append(row);
    }
    this.hexRows.replaceChildren(fragment);
  }

  private focusedRangeGuide(rowOffset: number, spans: SpanData[]): HTMLElement {
    const guide = document.createElement("span");
    guide.className = "range-guide";
    guide.setAttribute("aria-hidden", "true");
    const rowEnd = rowOffset + ROW_BYTES;
    for (const span of spans) {
      if (span.start >= rowEnd || span.end <= rowOffset) continue;
      const segment = document.createElement("span");
      segment.className = "range-guide-segment";
      if (span.start >= rowOffset) segment.classList.add("range-start");
      if (span.end <= rowEnd) segment.classList.add("range-end");
      guide.append(segment);
    }
    return guide;
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
    const selectedField = this.current.nodeId && this.current.fieldName
      ? annotations.find(span => span.node.id === this.current.nodeId && span.fieldName === this.current.fieldName)
      : this.current.nodeId ? preferredField(this.current.nodeId) : undefined;
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
      if (this.hoveredNode && annotations.some(span =>
        span.node.id === this.hoveredNode && (!this.hoveredField || span.fieldName === this.hoveredField))) {
        cell.classList.add("hovered");
      }
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
      const candidates = this.selectionCandidates(this.current.artifactId, offset);
      let selected = candidates[0];
      if (!event.shiftKey && offset === this.current.offset && candidates.length > 1) {
        const currentIndex = candidates.findIndex(candidate =>
          candidate.node.id === this.current.nodeId && candidate.fieldName === this.current.fieldName);
        if (currentIndex >= 0) selected = candidates[(currentIndex + 1) % candidates.length];
      }
      const start = event.shiftKey ? Math.min(this.current.offset, offset) : offset;
      const end = event.shiftKey ? Math.max(this.current.offset, offset) + 1 : offset + 1;
      this.navigate(
        {
          artifactId: this.current.artifactId,
          offset: start,
          end,
          nodeId: selected?.node.id,
          fieldName: selected?.fieldName,
          spanIndex: selected ? this.candidateSpanIndex(selected) : undefined,
        },
        { history: "push", scroll: false },
      );
      this.hexScroll.focus();
    });
    if (primary) {
      cell.addEventListener("mouseenter", () => this.setHovered(primary.node.id, primary.fieldName, offset));
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
    if (fieldName) {
      const field = node.fields[fieldName];
      if (field?.relevance === "irrelevant" || field?.properties?.relevance === "irrelevant") return true;
    }
    return node.relevance === "irrelevant" || node.properties.relevance === "irrelevant";
  }

  private isConfirmedIrrelevantField(field: FieldData): boolean {
    const relevance = field.relevance ?? field.properties?.relevance;
    const confidence = field.confidence ?? field.properties?.confidence;
    return relevance === "irrelevant" && confidence === "confirmed";
  }

  private displayColor(color: string, irrelevant: boolean): string {
    if (!this.options.dimIrrelevant || !irrelevant) return color;
    return `color-mix(in srgb, ${color} 24%, var(--text-faint))`;
  }

  private compareIndexedSpans(left: IndexedSpan, right: IndexedSpan): number {
    return Number(Boolean(right.fieldName)) - Number(Boolean(left.fieldName))
      || left.length - right.length
      || (this.depths.get(right.node.id) ?? 0) - (this.depths.get(left.node.id) ?? 0)
      || left.start - right.start
      || left.node.id.localeCompare(right.node.id)
      || (left.fieldName ?? "").localeCompare(right.fieldName ?? "");
  }

  private coveringSpans(artifactId: string, offset: number): IndexedSpan[] {
    return (this.spans.get(artifactId) ?? [])
      .filter(span => span.start <= offset && offset < span.end)
      .sort((left, right) => this.compareIndexedSpans(left, right));
  }

  private selectionCandidates(artifactId: string, offset: number): SelectionCandidate[] {
    const result: SelectionCandidate[] = [];
    for (const span of this.coveringSpans(artifactId, offset)) {
      if (result.some(value => value.node.id === span.node.id && value.fieldName === span.fieldName)) continue;
      result.push({ node: span.node, fieldName: span.fieldName, span });
    }
    return result;
  }

  private candidateSpanIndex(candidate: SelectionCandidate): number {
    const spans = candidate.fieldName
      ? candidate.node.fields[candidate.fieldName]?.spans ?? []
      : candidate.node.spans;
    return Math.max(0, spans.findIndex(span =>
      span.id !== undefined && candidate.span.id !== undefined
        ? span.id === candidate.span.id
        : span.artifact === candidate.span.artifact
          && span.start === candidate.span.start
          && span.end === candidate.span.end));
  }

  private cycleOverlap(delta: number): void {
    const candidates = this.selectionCandidates(this.current.artifactId, this.current.offset);
    if (candidates.length < 2) return;
    const currentIndex = candidates.findIndex(candidate =>
      candidate.node.id === this.current.nodeId && candidate.fieldName === this.current.fieldName);
    const index = currentIndex < 0
      ? (delta > 0 ? 0 : candidates.length - 1)
      : (currentIndex + delta + candidates.length) % candidates.length;
    const selected = candidates[index]!;
    this.navigate(
      {
        ...this.current,
        nodeId: selected.node.id,
        fieldName: selected.fieldName,
        spanIndex: this.candidateSpanIndex(selected),
        arrival: undefined,
      },
      { history: "push", scroll: false },
    );
  }

  private selectedSpans(): SpanData[] {
    const node = this.current.nodeId ? this.nodes.get(this.current.nodeId) : undefined;
    return this.current.fieldName ? node?.fields[this.current.fieldName]?.spans ?? [] : node?.spans ?? [];
  }

  private moveSelectedSpan(delta: number): void {
    const spans = this.selectedSpans();
    if (spans.length < 2) return;
    const current = this.current.spanIndex ?? spans.findIndex(span =>
      span.artifact === this.current.artifactId && span.start === this.current.offset && span.end === this.current.end);
    const index = (Math.max(0, current) + delta + spans.length) % spans.length;
    this.navigateSpan(spans[index]!, this.current.nodeId, this.current.fieldName, index);
  }

  private covering(artifactId: string, offset: number): NodeData[] {
    return this.selectionCandidates(artifactId, offset)
      .map(value => value.node)
      .filter((node, index, all) => all.findIndex(value => value.id === node.id) === index);
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
    this.inspector.append(title, subtitle);
    if (node) {
      const meaning = this.meaningElement(node);
      if (meaning) this.inspector.append(meaning);
      this.inspector.append(this.locationElement(node));
      const fields = this.fieldsElement(node);
      if (fields) this.inspector.append(fields);
      const outgoing = this.graph.links.filter(link => link.source === node.id);
      const incoming = this.graph.links.filter(link => link.target === node.id);
      if (outgoing.length || incoming.length) {
        const section = this.section("references");
        for (const group of this.referenceGroups(outgoing, true)) section.append(group);
        for (const group of this.referenceGroups(incoming, false)) section.append(group);
        this.inspector.append(section);
      }
      if (Object.keys(node.addresses).length) {
        const section = this.section("addresses");
        for (const [name, address] of Object.entries(node.addresses)) section.append(this.kv(name, `${address.space}:${hex(address.value)}`));
        this.inspector.append(section);
      }
      if (Object.keys(node.properties).length) {
        const section = this.section("properties");
        for (const [name, value] of Object.entries(node.properties)) section.append(this.kv(name, displayValue(value)));
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
    this.inspector.append(this.semanticHelpElement(), this.knowledgeElement(artifact));
  }

  private meaningElement(node: NodeData): HTMLElement | undefined {
    const evidence = evidenceText(node);
    if (!node.description && !evidence && !this.current.arrival) return undefined;
    const element = document.createElement("div");
    element.className = "inspector-meaning";
    if (node.description) {
      const description = document.createElement("p");
      description.className = "inspector-description";
      description.textContent = node.description;
      element.append(description);
    }
    if (evidence) {
      const evidenceElement = document.createElement("div");
      evidenceElement.className = "semantic-evidence";
      evidenceElement.textContent = evidence;
      element.append(evidenceElement);
    }
    if (this.current.arrival) {
      const source = this.nodes.get(this.current.arrival.sourceNodeId);
      const arrival = document.createElement("div");
      arrival.className = "semantic-evidence";
      arrival.textContent = `arrived via ${this.current.arrival.kind}${source ? ` from ${source.label}` : ""}`;
      element.append(arrival);
    }
    return element;
  }

  private locationElement(node: NodeData): HTMLElement {
    const section = this.section("location");
    const lineage: NodeData[] = [];
    let cursor: NodeData | undefined = node;
    while (cursor) {
      lineage.push(cursor);
      cursor = cursor.parent ? this.nodes.get(cursor.parent) : undefined;
    }
    lineage.reverse();
    const path = document.createElement("div");
    path.className = "containment-path";
    for (const [index, pathNode] of lineage.entries()) {
      if (index) path.append(document.createTextNode(" › "));
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = pathNode.label;
      button.title = pathNode.description ?? pathNode.id;
      button.addEventListener("click", () => this.selectNode(pathNode.id, true));
      path.append(button);
    }
    section.append(path);
    if (node.spans.length) {
      for (const [index, span] of node.spans.entries()) section.append(this.spanButton(span, index));
    } else {
      const logical = document.createElement("div");
      logical.className = "muted";
      logical.textContent = "logical entity · no physical byte span";
      section.append(logical);
    }
    return section;
  }

  private fieldsElement(node: NodeData): HTMLElement | undefined {
    const indexed = Object.entries(node.fields).map(([name, field], order) => ({
      name,
      field,
      order,
      offset: Math.min(...(field.spans ?? []).map(span => span.start), Number.POSITIVE_INFINITY),
    }));
    if (!indexed.length) return undefined;
    indexed.sort((left, right) => left.offset - right.offset || left.order - right.order);
    const collapsible = indexed.filter(({ field }) => this.isConfirmedIrrelevantField(field)).length;
    let action: HTMLButtonElement | undefined;
    if (collapsible) {
      action = document.createElement("button");
      action.type = "button";
      action.className = "section-action";
      action.textContent = this.options.collapseIrrelevantFields
        ? `show ${collapsible} irrelevant`
        : `hide ${collapsible} irrelevant`;
      action.title = "Only fields explicitly marked both confirmed and irrelevant are collapsible";
      action.addEventListener("click", () => {
        this.options.collapseIrrelevantFields = !this.options.collapseIrrelevantFields;
        saveOptions(this.options);
        this.renderInspector();
      });
    }
    const section = this.section("fields", action);
    for (const { name, field } of indexed) {
      if (this.options.collapseIrrelevantFields && this.isConfirmedIrrelevantField(field)) continue;
      section.append(this.fieldElement(node, name, field));
    }
    return section;
  }

  private fieldElement(node: NodeData, name: string, field: FieldData): HTMLElement {
    const spans = field.spans ?? [];
    const element = document.createElement(spans.length ? "button" : "div");
    if (element instanceof HTMLButtonElement) {
      element.type = "button";
      element.title = "Navigate to field bytes";
      element.addEventListener("click", () => this.navigateSpan(spans[0]!, node.id, name, 0));
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
      element.append(evidenceElement);
    }
    return element;
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

  private semanticHelpElement(): HTMLElement {
    const details = document.createElement("details");
    details.className = "semantic-help";
    const summary = document.createElement("summary");
    summary.textContent = "semantic color and evidence";
    const content = document.createElement("p");
    content.textContent = "Color groups bytes by semantic category; field marks distinguish decoded fields. Confirmed is specified or validated, an inference is evidence-qualified, raw is observed without interpretation, unresolved is deliberately undecided, and irrelevant means understood but unused by the stated supported interpretation.";
    details.append(summary, content);
    return details;
  }

  private section(label?: string, action?: HTMLElement): HTMLElement {
    const section = document.createElement("section");
    section.className = "inspector-section";
    if (label) {
      const headingRow = document.createElement("div");
      headingRow.className = "section-heading";
      const heading = document.createElement("h2");
      heading.textContent = label;
      headingRow.append(heading);
      if (action) headingRow.append(action);
      section.append(headingRow);
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

  private spanButton(span: SpanData, spanIndex: number): HTMLElement {
    const artifact = this.artifacts.get(span.artifact)!;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "jump";
    const selectedNode = this.current.nodeId ? this.nodes.get(this.current.nodeId) : undefined;
    if (selectedNode?.spans[spanIndex] === span
      && span.artifact === this.current.artifactId
      && span.start === this.current.offset
      && span.end === this.current.end) button.classList.add("active-span");
    const label = span.label ? `${span.label} · ` : "";
    button.textContent = `${label}${artifact.name}  ${hex(span.start)}–${hex(span.end)}  (${span.end - span.start})`;
    button.addEventListener("click", () => this.navigateSpan(span, this.current.nodeId, undefined, spanIndex));
    return button;
  }

  private navigateSpan(span: SpanData, nodeId?: string, fieldName?: string, spanIndex?: number): void {
    this.closeNarrowDrawer();
    this.navigate(
      { artifactId: span.artifact, offset: span.start, end: span.end, nodeId, fieldName, spanIndex },
      { history: "push", scroll: true },
    );
  }

  private referenceGroups(links: LinkData[], outgoing: boolean): HTMLElement[] {
    const grouped = new Map<string, LinkData[]>();
    for (const link of links) {
      const values = grouped.get(link.kind) ?? [];
      values.push(link);
      grouped.set(link.kind, values);
    }
    return [...grouped.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([kind, values]) => {
        const group = document.createElement("div");
        group.className = "reference-group";
        const heading = document.createElement("div");
        heading.className = "reference-heading";
        const kindElement = document.createElement("span");
        kindElement.className = "ref-kind";
        kindElement.textContent = `${outgoing ? "→" : "←"} ${kind}`;
        const count = document.createElement("span");
        count.className = "reference-count";
        count.textContent = String(values.length);
        heading.append(kindElement, count);
        group.append(heading);
        const description = values.find(link => link.description)?.description;
        const evidence = evidenceText(values[0]!);
        if (description) {
          const element = document.createElement("div");
          element.className = "reference-description";
          element.textContent = description;
          group.append(element);
        }
        if (evidence) {
          const element = document.createElement("div");
          element.className = "semantic-evidence";
          element.textContent = evidence;
          group.append(element);
        }
        for (const link of values) group.append(this.referenceButton(link, outgoing));
        return group;
      });
  }

  private referenceButton(link: LinkData, outgoing: boolean): HTMLElement {
    const targetId = outgoing ? link.target : link.source;
    const target = this.nodes.get(targetId);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "reference-target";
    const label = document.createElement("span");
    label.className = "reference-target-label";
    label.textContent = target?.label ?? targetId;
    const location = document.createElement("span");
    location.className = "reference-target-location";
    location.textContent = target
      ? `${this.nodePath(target)} · ${this.spanDetail(target.spans)}`
      : targetId;
    button.append(label, location);
    if (target?.description) {
      const description = document.createElement("span");
      description.className = "reference-target-description";
      description.textContent = target.description;
      button.append(description);
    }
    const targetEvidence = target ? evidenceText(target) : undefined;
    if (targetEvidence) {
      const evidence = document.createElement("span");
      evidence.className = "semantic-evidence";
      evidence.textContent = targetEvidence;
      button.append(evidence);
    }
    button.title = Object.keys(link.properties).length ? displayValue(link.properties) : "Follow reference";
    button.addEventListener("click", () => this.selectNode(targetId, true, {
      sourceNodeId: link.source,
      targetNodeId: link.target,
      kind: link.kind,
    }));
    button.addEventListener("mouseenter", () => this.setHovered(targetId, undefined, target?.spans[0]?.start));
    button.addEventListener("mouseleave", () => this.setHovered(undefined));
    return button;
  }

  private hexKey(event: KeyboardEvent): void {
    if (event.key === "[" || event.key === "]") {
      event.preventDefault();
      if (event.altKey) this.moveSelectedSpan(event.key === "]" ? 1 : -1);
      else this.cycleOverlap(event.key === "]" ? 1 : -1);
      return;
    }
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
    const primary = this.selectionCandidates(artifact.id, offset)[0];
    this.navigate(
      {
        artifactId: artifact.id,
        offset,
        end: offset + 1,
        nodeId: primary?.node.id,
        fieldName: primary?.fieldName,
        spanIndex: primary ? this.candidateSpanIndex(primary) : undefined,
      },
      { history: "coalesce", coalesceKey: "byte-step", scroll: true },
    );
  }

  private async copyLocation(): Promise<void> {
    const value = this.currentLocationExpression();
    try {
      await navigator.clipboard.writeText(value);
      this.setStatus(`copied ${value}`);
    } catch {
      this.setStatus(`copy unavailable · ${value}`);
    }
  }

  private currentLocationExpression(): string {
    return formatLocationExpression({
      artifactId: this.current.artifactId,
      start: this.current.offset,
      end: this.current.end,
      nodeId: this.current.nodeId,
      fieldName: this.current.fieldName,
      spanIndex: this.current.spanIndex,
    });
  }

  private renderContextStatus(nodeId = this.current.nodeId, fieldName = this.current.fieldName, offset?: number): void {
    this.status.replaceChildren();
    const node = nodeId ? this.nodes.get(nodeId) : undefined;
    const contextOffset = offset ?? this.current.offset;
    const candidate = node
      ? this.coveringSpans(this.current.artifactId, contextOffset).find(span =>
        span.node.id === node.id && (fieldName === undefined || span.fieldName === fieldName))
      : undefined;
    const span = candidate ?? (fieldName ? node?.fields[fieldName]?.spans?.[0] : node?.spans[0]);
    const artifact = this.artifacts.get(span?.artifact ?? this.current.artifactId)!;
    const path: NodeData[] = [];
    let cursor = node;
    while (cursor) {
      path.push(cursor);
      cursor = cursor.parent ? this.nodes.get(cursor.parent) : undefined;
    }
    path.reverse();
    const artifactElement = document.createElement("span");
    artifactElement.className = "status-artifact";
    artifactElement.textContent = artifact.name;
    this.status.append(artifactElement);
    for (const pathNode of path) {
      this.status.append(this.statusSeparator());
      const button = document.createElement("button");
      button.type = "button";
      button.className = "status-crumb";
      button.textContent = pathNode.label;
      button.title = pathNode.description ?? pathNode.label;
      button.addEventListener("click", () => this.selectNode(pathNode.id, true));
      this.status.append(button);
    }
    const field = node && fieldName ? node.fields[fieldName] : undefined;
    if (fieldName) {
      this.status.append(this.statusSeparator());
      const fieldElement = document.createElement("span");
      fieldElement.className = "status-field";
      fieldElement.textContent = fieldName;
      fieldElement.title = field?.description ?? fieldName;
      this.status.append(fieldElement);
    }
    const rangeStart = span?.start ?? this.current.offset;
    const rangeEnd = span?.end ?? this.current.end;
    const range = document.createElement("span");
    range.className = "status-meta";
    const spanLabel = span?.label ? `${span.label} · ` : "";
    range.textContent = `${spanLabel}${hex(rangeStart)}–${hex(rangeEnd)} · ${rangeEnd - rangeStart} B`;
    this.status.append(range);
    if (field) {
      const value = document.createElement("span");
      value.className = "status-meta";
      value.textContent = field.display ?? displayValue(field.value);
      const evidence = evidenceText(field);
      value.title = [field.description, evidence].filter(Boolean).join("\n");
      this.status.append(value);
      if (evidence) {
        const evidenceElement = document.createElement("span");
        evidenceElement.className = "status-evidence";
        evidenceElement.textContent = evidence;
        this.status.append(evidenceElement);
      }
    }
    if (offset === undefined) {
      const candidates = this.selectionCandidates(this.current.artifactId, this.current.offset);
      if (candidates.length > 1) {
        const index = candidates.findIndex(value =>
          value.node.id === this.current.nodeId && value.fieldName === this.current.fieldName);
        const overlap = document.createElement("button");
        overlap.type = "button";
        overlap.className = "status-control";
        overlap.textContent = `${Math.max(0, index) + 1}/${candidates.length} covering`;
        overlap.title = "Cycle covering semantics ([ and ])";
        overlap.addEventListener("click", () => this.cycleOverlap(1));
        this.status.append(overlap);
      }
      const spans = this.selectedSpans();
      if (spans.length > 1) {
        const index = Math.max(0, Math.min(spans.length - 1, this.current.spanIndex ?? 0));
        const previous = document.createElement("button");
        previous.type = "button";
        previous.className = "status-control compact";
        previous.textContent = "‹";
        previous.title = "Previous span (Alt+[)";
        previous.addEventListener("click", () => this.moveSelectedSpan(-1));
        const count = document.createElement("span");
        count.className = "status-meta";
        count.textContent = `span ${index + 1}/${spans.length}`;
        const next = document.createElement("button");
        next.type = "button";
        next.className = "status-control compact";
        next.textContent = "›";
        next.title = "Next span (Alt+])";
        next.addEventListener("click", () => this.moveSelectedSpan(1));
        this.status.append(previous, count, next);
      }
      if (this.current.arrival) {
        const arrival = document.createElement("span");
        arrival.className = "status-evidence";
        arrival.textContent = `via ${this.current.arrival.kind}`;
        this.status.append(arrival);
      }
    }
  }

  private statusSeparator(): HTMLElement {
    const separator = document.createElement("span");
    separator.className = "status-separator";
    separator.textContent = "›";
    return separator;
  }

  private setHovered(nodeId?: string, fieldName?: string, offset?: number): void {
    if (this.hoveredNode === nodeId && this.hoveredField === fieldName) return;
    this.hoveredNode = nodeId;
    this.hoveredField = fieldName;
    if (nodeId) {
      this.renderContextStatus(nodeId, fieldName, offset);
    } else {
      this.renderContextStatus();
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
  if (graph.format !== "npunlock.artifact-graph" || ![1, 2].includes(graph.version)) {
    throw new Error("unsupported artifact graph format");
  }
  new Viewer(
    graph,
    response.headers.get("ETag") ?? undefined,
    response.headers.get("X-NPUnlock-Viz-State-Version") ?? undefined,
  );
}

main().catch(error => {
  const app = required<HTMLElement>("app");
  const element = document.createElement("pre");
  element.className = "error";
  element.textContent = error instanceof Error ? error.message : String(error);
  app.replaceChildren(element);
});
