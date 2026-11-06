import { NavigationHistory, sameLocation, sameShareableLocation, } from "./navigation.js";
import { loadCompatibleValue, loadWorkspace, MAX_PERSISTED_HISTORY, saveCompatibleValue, saveWorkspace, workspaceStorageKey, } from "./persistence.js";
import { formatLocationExpression, parseLocationExpression, searchEntries, } from "./search.js";
import { flattenTree, visibleWindow } from "./tree.js";
const ROW_BYTES = 16;
const ROW_HEIGHT = 21;
const PAGE_BYTES = 16 * 1024;
const MAX_SCROLL_HEIGHT = 16_000_000;
const OVERSCAN = 8;
const TREE_OVERSCAN = 6;
const OPTIONS_KEY = "npunlock.viz.options.v1";
const DEFAULT_OPTIONS = {
    dimIrrelevant: true,
    collapseIrrelevantFields: false,
    treeVisible: true,
    inspectorVisible: true,
    treeWidth: null,
    inspectorWidth: null,
    treeMode: "all",
};
function required(id) {
    const value = document.getElementById(id);
    if (!(value instanceof HTMLElement))
        throw new Error(`missing interface element: ${id}`);
    return value;
}
function hex(value, width = 0) {
    return `0x${Math.max(0, value).toString(16).padStart(width, "0")}`;
}
function printable(value) {
    return value >= 0x20 && value <= 0x7e ? String.fromCharCode(value) : ".";
}
function displayValue(value) {
    if (typeof value === "string")
        return value;
    if (typeof value === "number")
        return Number.isInteger(value) ? `${value} (${hex(value)})` : String(value);
    if (value === null)
        return "null";
    if (typeof value === "boolean")
        return String(value);
    return JSON.stringify(value, null, 2) ?? String(value);
}
function byteCount(value) {
    if (value < 1024)
        return `${value} B`;
    const units = ["KiB", "MiB", "GiB"];
    let scaled = value / 1024;
    let index = 0;
    while (scaled >= 1024 && index < units.length - 1) {
        scaled /= 1024;
        index += 1;
    }
    return `${scaled >= 10 ? scaled.toFixed(0) : scaled.toFixed(1)} ${units[index]}`;
}
function coverage(value, total) {
    const percent = total > 0 ? value * 100 / total : 0;
    return `${byteCount(value)} (${percent < 0.1 && value > 0 ? "<0.1" : percent.toFixed(1)}%)`;
}
function unionLength(spans, size) {
    const ordered = spans
        .map(span => ({ start: Math.max(0, span.start), end: Math.min(size, span.end) }))
        .filter(span => span.end > span.start)
        .sort((a, b) => a.start - b.start || a.end - b.end);
    let total = 0;
    let start = -1;
    let end = -1;
    for (const span of ordered) {
        if (span.start > end) {
            if (end > start)
                total += end - start;
            start = span.start;
            end = span.end;
        }
        else {
            end = Math.max(end, span.end);
        }
    }
    if (end > start)
        total += end - start;
    return total;
}
function evidenceText(value) {
    const confidence = value.confidence ?? value.properties?.confidence;
    const scope = value.scope ?? value.properties?.scope;
    const relevance = value.relevance ?? value.properties?.relevance;
    const labels = {
        confirmed: "confirmed",
        "strong-inference": "strong inference",
        "execution-tested-inference": "execution-tested inference",
        raw: "raw observation",
        unresolved: "unresolved",
    };
    const parts = [];
    if (typeof confidence === "string")
        parts.push(labels[confidence] ?? confidence);
    if (relevance === "irrelevant")
        parts.push("irrelevant to supported interpretation");
    else if (relevance === "context")
        parts.push("context only");
    if (typeof scope === "string")
        parts.push(scope);
    return parts.length ? parts.join(" · ") : undefined;
}
function browserStorage(kind) {
    try {
        return kind === "local" ? window.localStorage : window.sessionStorage;
    }
    catch {
        return undefined;
    }
}
function isViewerOptions(value) {
    if (typeof value !== "object" || value === null || Array.isArray(value))
        return false;
    const options = value;
    const width = (candidate, minimum, maximum) => candidate === null
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
function loadOptions() {
    const storage = browserStorage("local");
    if (!storage)
        return { ...DEFAULT_OPTIONS };
    const value = loadCompatibleValue(storage, OPTIONS_KEY, isViewerOptions);
    return value ? { ...value } : { ...DEFAULT_OPTIONS };
}
function saveOptions(value) {
    const storage = browserStorage("local");
    if (storage)
        saveCompatibleValue(storage, OPTIONS_KEY, value);
}
class ByteCache {
    artifact;
    pages = new Map();
    constructor(artifact) {
        this.artifact = artifact;
    }
    page(index) {
        let value = this.pages.get(index);
        if (!value) {
            const offset = index * PAGE_BYTES;
            const length = Math.min(PAGE_BYTES, this.artifact.size - offset);
            const url = new URL(`api/artifacts/${encodeURIComponent(this.artifact.id)}/bytes`, window.location.href);
            url.searchParams.set("offset", String(offset));
            url.searchParams.set("length", String(length));
            value = fetch(url).then(async (response) => {
                if (!response.ok)
                    throw new Error(`byte request failed: ${response.status}`);
                return new Uint8Array(await response.arrayBuffer());
            });
            this.pages.set(index, value);
        }
        return value;
    }
    async range(start, end) {
        if (end <= start)
            return new Uint8Array();
        const first = Math.floor(start / PAGE_BYTES);
        const last = Math.floor((end - 1) / PAGE_BYTES);
        const values = await Promise.all(Array.from({ length: last - first + 1 }, (_, index) => this.page(first + index)));
        const output = new Uint8Array(end - start);
        let outputOffset = 0;
        for (let pageIndex = first; pageIndex <= last; pageIndex += 1) {
            const page = values[pageIndex - first];
            if (!page)
                continue;
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
    graph;
    graphIdentity;
    backendStateVersion;
    artifactTabs = required("artifact-tabs");
    tree = required("tree");
    inspector = required("inspector");
    hexHead = required("hex-head");
    hexScroll = required("hex-scroll");
    hexSpace = required("hex-space");
    hexRows = required("hex-rows");
    overview = required("overview");
    status = required("status");
    workspace = required("workspace");
    offsetInput = required("offset-input");
    treeFilter = required("tree-filter");
    treeMode = required("tree-mode");
    goToDialog = required("go-to-dialog");
    goToInput = required("go-to-input");
    goToResults = required("go-to-results");
    nodes = new Map();
    artifacts = new Map();
    categories = new Map();
    children = new Map();
    roots = [];
    depths = new Map();
    spans = new Map();
    cache = new Map();
    knowledge = new Map();
    overviewSpans = new Map();
    overviewBases = new Map();
    expanded = new Set();
    current;
    history = new NavigationHistory();
    searchIndex = [];
    visibleSearchResults = [];
    searchSelection = 0;
    hoveredNode;
    hoveredField;
    treeFocusId;
    treeRows = [];
    treeMatches = new Set();
    treeFilterActive = false;
    treeStructureDirty = true;
    treeRenderFrame = 0;
    treeSpace = document.createElement("div");
    treeRowsLayer = document.createElement("div");
    narrowMedia = window.matchMedia("(max-width: 820px)");
    narrowDrawer;
    renderToken = 0;
    options = loadOptions();
    workspaceStorage = browserStorage("local");
    workspaceKey;
    workspaceSaveTimer = 0;
    restoringWorkspace = false;
    constructor(graph, graphIdentity, backendStateVersion) {
        this.graph = graph;
        this.graphIdentity = graphIdentity;
        this.backendStateVersion = backendStateVersion;
        if (graphIdentity && backendStateVersion)
            this.workspaceKey = workspaceStorageKey(graphIdentity);
        this.treeSpace.className = "tree-space";
        this.treeRowsLayer.className = "tree-rows";
        this.treeSpace.append(this.treeRowsLayer);
        for (const artifact of graph.artifacts) {
            this.artifacts.set(artifact.id, artifact);
            this.cache.set(artifact.id, new ByteCache(artifact));
            this.spans.set(artifact.id, []);
            this.overviewSpans.set(artifact.id, []);
        }
        for (const category of graph.categories)
            this.categories.set(category.id, category);
        for (const node of graph.nodes) {
            this.nodes.set(node.id, node);
            if (node.parent) {
                const values = this.children.get(node.parent) ?? [];
                values.push(node);
                this.children.set(node.parent, values);
            }
            else {
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
        for (const values of this.spans.values())
            values.sort((a, b) => a.start - b.start || a.end - b.end);
        for (const values of this.overviewSpans.values()) {
            values.sort((left, right) => (right.end - right.start) - (left.end - left.start));
        }
        for (const node of graph.nodes) {
            if ((this.children.get(node.id)?.length ?? 0) > 0)
                this.expanded.add(node.id);
        }
        for (const root of this.roots) {
            this.assignDepth(root, 0);
        }
        this.searchIndex = this.buildSearchIndex();
        for (const artifact of graph.artifacts)
            this.knowledge.set(artifact.id, this.buildKnowledge(artifact));
        const first = graph.artifacts[0];
        if (!first)
            throw new Error("artifact graph contains no artifacts");
        this.current = { artifactId: first.id, offset: 0, end: Math.min(1, first.size) };
        const initial = this.selectionCandidates(first.id, 0)[0];
        if (initial) {
            this.current.nodeId = initial.node.id;
            this.current.fieldName = initial.fieldName;
            this.current.spanIndex = this.candidateSpanIndex(initial);
        }
        const workspace = this.loadWorkspaceState();
        let restoredWorkspace;
        const fragment = window.location.hash.slice(1);
        let fragmentLocation;
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
            else if (workspace)
                this.discardWorkspaceState();
        }
        else if (workspace) {
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
        }
        else {
            this.scheduleWorkspaceSave(0);
        }
    }
    validateStoredLocation(location) {
        const artifact = this.artifacts.get(location.artifactId);
        if (!artifact
            || !Number.isSafeInteger(location.offset)
            || !Number.isSafeInteger(location.end)
            || location.offset < 0
            || (artifact.size === 0 ? location.offset !== 0 || location.end !== 0
                : location.offset >= artifact.size || location.end <= location.offset || location.end > artifact.size))
            return false;
        if (!location.nodeId) {
            return location.fieldName === undefined
                && location.spanIndex === undefined
                && location.arrival === undefined;
        }
        const node = this.nodes.get(location.nodeId);
        if (!node)
            return false;
        const spans = location.fieldName === undefined
            ? node.spans
            : Object.hasOwn(node.fields, location.fieldName) ? node.fields[location.fieldName].spans ?? [] : undefined;
        if (!spans)
            return false;
        if (location.spanIndex !== undefined
            && (!Number.isSafeInteger(location.spanIndex)
                || location.spanIndex < 0
                || location.spanIndex >= spans.length))
            return false;
        if (location.arrival) {
            if (location.arrival.targetNodeId !== location.nodeId
                || !this.nodes.has(location.arrival.sourceNodeId)
                || !this.nodes.has(location.arrival.targetNodeId)
                || !this.graph.links.some(link => link.source === location.arrival.sourceNodeId
                    && link.target === location.arrival.targetNodeId
                    && link.kind === location.arrival.kind))
                return false;
        }
        return true;
    }
    loadWorkspaceState() {
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
                return Boolean(artifact) && offset >= 0 && (artifact.size === 0 ? offset === 0 : offset < artifact.size);
            },
        });
    }
    discardWorkspaceState() {
        if (!this.workspaceStorage || !this.workspaceKey)
            return;
        try {
            this.workspaceStorage.removeItem(this.workspaceKey);
        }
        catch {
        }
    }
    saveWorkspaceState() {
        this.workspaceSaveTimer = 0;
        if (this.restoringWorkspace
            || !this.workspaceStorage
            || !this.workspaceKey
            || !this.graphIdentity
            || !this.backendStateVersion)
            return;
        const history = this.history.snapshot(MAX_PERSISTED_HISTORY);
        const historyLocation = history.entries[history.index];
        if (!historyLocation || !sameLocation(historyLocation, this.current))
            return;
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
    scheduleWorkspaceSave(delay = 160) {
        if (this.restoringWorkspace
            || !this.workspaceStorage
            || !this.workspaceKey
            || !this.graphIdentity
            || !this.backendStateVersion)
            return;
        if (this.workspaceSaveTimer)
            window.clearTimeout(this.workspaceSaveTimer);
        this.workspaceSaveTimer = window.setTimeout(() => this.saveWorkspaceState(), delay);
    }
    flushWorkspaceState() {
        if (this.workspaceSaveTimer)
            window.clearTimeout(this.workspaceSaveTimer);
        this.workspaceSaveTimer = 0;
        this.saveWorkspaceState();
    }
    restoreWorkspaceViewport(workspace) {
        this.tree.scrollTop = workspace.treeScrollTop;
        this.renderTreeViewport();
        this.setHexTopOffset(workspace.hexTopOffset);
        this.inspector.scrollTop = workspace.inspectorScrollTop;
        this.restoringWorkspace = false;
        this.scheduleWorkspaceSave(0);
    }
    assignDepth(node, depth) {
        this.depths.set(node.id, depth);
        for (const child of this.children.get(node.id) ?? [])
            this.assignDepth(child, depth + 1);
    }
    nodePath(node) {
        const labels = [node.label];
        let current = node;
        while (current.parent) {
            const parent = this.nodes.get(current.parent);
            if (!parent)
                break;
            labels.push(parent.label);
            current = parent;
        }
        return labels.reverse().join(" › ");
    }
    spanDetail(spans) {
        const span = spans[0];
        if (!span)
            return "logical entity";
        const artifact = this.artifacts.get(span.artifact);
        const label = span.label ? `${span.label} · ` : "";
        return `${label}${artifact?.name ?? span.artifact} ${hex(span.start)}–${hex(span.end)} · ${span.end - span.start} B`;
    }
    buildSearchIndex() {
        const entries = [];
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
    buildKnowledge(artifact) {
        const touchedNodes = this.graph.nodes.filter(node => node.spans.some(span => span.artifact === artifact.id)
            || Object.values(node.fields).some(field => (field.spans ?? []).some(span => span.artifact === artifact.id)));
        const touchedIds = new Set(touchedNodes.map(node => node.id));
        const fields = touchedNodes.flatMap(node => Object.values(node.fields).filter(field => {
            const spans = field.spans ?? [];
            return spans.length === 0 || spans.some(span => span.artifact === artifact.id);
        }));
        const mappedSpans = touchedNodes.flatMap(node => node.spans.filter(span => span.artifact === artifact.id
            && node.contributes_to_coverage !== false
            && !(this.graph.version === 1 && node.parent === undefined && span.start === 0 && span.end === artifact.size)));
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
    bind() {
        this.hexScroll.addEventListener("scroll", () => {
            this.hexHead.scrollLeft = this.hexScroll.scrollLeft;
            this.renderHex();
            this.scheduleWorkspaceSave();
        });
        this.hexScroll.addEventListener("keydown", event => this.hexKey(event));
        required("offset-form").addEventListener("submit", event => {
            event.preventDefault();
            const raw = this.offsetInput.value.trim().toLowerCase();
            const parsed = Number.parseInt(raw.startsWith("0x") ? raw.slice(2) : raw, 16);
            const artifact = this.artifacts.get(this.current.artifactId);
            if (!Number.isFinite(parsed) || parsed < 0 || parsed >= artifact.size) {
                this.setStatus(`offset must be between ${hex(0)} and ${hex(Math.max(0, artifact.size - 1))}`);
                return;
            }
            this.navigate({ artifactId: artifact.id, offset: parsed, end: parsed + 1 }, { history: "push", scroll: true });
        });
        required("copy").addEventListener("click", () => this.copyLocation());
        required("open-go-to").addEventListener("click", () => this.openGoTo());
        required("back").addEventListener("click", () => this.moveHistory(-1));
        required("forward").addEventListener("click", () => this.moveHistory(1));
        required("toggle-tree").addEventListener("click", event => this.togglePanel("tree", event));
        required("toggle-inspector").addEventListener("click", event => this.togglePanel("inspector", event));
        required("toggle-dim-irrelevant").addEventListener("click", () => this.toggleDimIrrelevant());
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
            if (!["all", "relevant", "artifact", "covers", "referenced"].includes(mode))
                return;
            this.options.treeMode = mode;
            saveOptions(this.options);
            this.treeStructureDirty = true;
            this.renderTree();
        });
        this.tree.addEventListener("scroll", () => {
            this.scheduleWorkspaceSave();
            if (this.treeRenderFrame)
                return;
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
        required("hex-panel").addEventListener("pointerdown", () => this.closeNarrowDrawer());
        this.goToInput.addEventListener("input", () => this.renderGoTo());
        this.goToInput.addEventListener("keydown", event => this.goToKey(event));
        required("go-to-form").addEventListener("submit", event => {
            event.preventDefault();
            this.activateSearchResult(this.visibleSearchResults[this.searchSelection]);
        });
        this.goToDialog.addEventListener("click", event => {
            if (event.target === this.goToDialog)
                this.goToDialog.close();
        });
        this.bindSplitter(required("left-splitter"), "--tree-width", "treeWidth", true, 150, 520);
        this.bindSplitter(required("right-splitter"), "--inspector-width", "inspectorWidth", false, 210, 620);
        window.addEventListener("keydown", event => {
            if (event.altKey && event.key === "ArrowLeft") {
                event.preventDefault();
                this.moveHistory(-1);
            }
            if (event.altKey && event.key === "ArrowRight") {
                event.preventDefault();
                this.moveHistory(1);
            }
            if (event.ctrlKey && event.key.toLowerCase() === "k") {
                event.preventDefault();
                this.openGoTo();
            }
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
            if (document.visibilityState === "hidden")
                this.flushWorkspaceState();
        });
    }
    applyOptions() {
        this.workspace.classList.toggle("tree-hidden", !this.options.treeVisible);
        this.workspace.classList.toggle("inspector-hidden", !this.options.inspectorVisible);
        this.workspace.classList.toggle("narrow-tree-open", this.narrowDrawer === "tree");
        this.workspace.classList.toggle("narrow-inspector-open", this.narrowDrawer === "inspector");
        const narrow = this.narrowMedia.matches;
        required("toggle-tree").setAttribute("aria-pressed", String(narrow ? this.narrowDrawer === "tree" : this.options.treeVisible));
        required("toggle-inspector").setAttribute("aria-pressed", String(narrow ? this.narrowDrawer === "inspector" : this.options.inspectorVisible));
        required("toggle-dim-irrelevant").setAttribute("aria-pressed", String(this.options.dimIrrelevant));
        this.treeMode.value = this.options.treeMode;
        if (this.options.treeWidth !== null) {
            document.documentElement.style.setProperty("--tree-width", `${Math.max(150, Math.min(520, this.options.treeWidth))}px`);
        }
        if (this.options.inspectorWidth !== null) {
            document.documentElement.style.setProperty("--inspector-width", `${Math.max(210, Math.min(620, this.options.inspectorWidth))}px`);
        }
    }
    bindSplitter(splitter, variable, option, fromLeft, minimum, maximum) {
        const setValue = (value, persist = true) => {
            const bounded = Math.max(minimum, Math.min(maximum, value));
            document.documentElement.style.setProperty(variable, `${bounded}px`);
            splitter.setAttribute("aria-valuenow", String(Math.round(bounded)));
            this.options[option] = bounded;
            if (persist)
                saveOptions(this.options);
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
            if (!splitter.hasPointerCapture(event.pointerId))
                return;
            const bounds = this.workspace.getBoundingClientRect();
            const value = fromLeft ? event.clientX - bounds.left : bounds.right - event.clientX;
            setValue(value, false);
        });
        const finish = (event) => {
            if (splitter.hasPointerCapture(event.pointerId))
                splitter.releasePointerCapture(event.pointerId);
            splitter.classList.remove("dragging");
            saveOptions(this.options);
        };
        splitter.addEventListener("pointerup", finish);
        splitter.addEventListener("pointercancel", finish);
        splitter.addEventListener("keydown", event => {
            let value;
            if (event.key === "Home")
                value = minimum;
            if (event.key === "End")
                value = maximum;
            if (event.key === "ArrowLeft")
                value = currentValue() + (fromLeft ? -12 : 12);
            if (event.key === "ArrowRight")
                value = currentValue() + (fromLeft ? 12 : -12);
            if (value === undefined)
                return;
            event.preventDefault();
            setValue(value);
        });
    }
    togglePanel(panel, event) {
        if (this.narrowMedia.matches) {
            this.narrowDrawer = this.narrowDrawer === panel ? undefined : panel;
            this.applyOptions();
            if (this.narrowDrawer === "tree")
                this.renderTree();
            if (this.narrowDrawer === "inspector")
                this.renderInspector();
            this.renderHex();
            return;
        }
        const button = event.currentTarget;
        const className = `${panel}-hidden`;
        const hidden = this.workspace.classList.toggle(className);
        button.setAttribute("aria-pressed", String(!hidden));
        if (panel === "tree")
            this.options.treeVisible = !hidden;
        else
            this.options.inspectorVisible = !hidden;
        saveOptions(this.options);
        this.renderHex();
    }
    closeNarrowDrawer() {
        if (!this.narrowDrawer)
            return;
        this.narrowDrawer = undefined;
        this.applyOptions();
        this.renderHex();
    }
    toggleDimIrrelevant() {
        this.options.dimIrrelevant = !this.options.dimIrrelevant;
        required("toggle-dim-irrelevant")
            .setAttribute("aria-pressed", String(this.options.dimIrrelevant));
        saveOptions(this.options);
        this.renderTree();
        this.renderInspector();
        this.renderHex();
        this.renderOverview();
    }
    openGoTo() {
        if (!this.goToDialog.open)
            this.goToDialog.showModal();
        this.goToInput.value = "";
        this.searchSelection = 0;
        this.renderGoTo();
        requestAnimationFrame(() => this.goToInput.focus());
    }
    renderGoTo() {
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
                        location.nodeId ? this.nodePath(this.nodes.get(location.nodeId)) : "file offset",
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
                if (this.searchSelection === index)
                    return;
                this.searchSelection = index;
                this.updateSearchSelection();
            });
            button.addEventListener("click", () => this.activateSearchResult(entry));
            fragment.append(button);
        }
        this.goToResults.append(fragment);
        this.updateSearchSelection();
    }
    resolveLocation(parsed) {
        if (!parsed.nodeId)
            return { kind: "location", ...parsed };
        const node = this.nodes.get(parsed.nodeId);
        if (!node)
            return undefined;
        if (parsed.fieldName && !node.fields[parsed.fieldName])
            return undefined;
        const spans = parsed.fieldName ? node.fields[parsed.fieldName].spans ?? [] : node.spans;
        if (parsed.spanIndex !== undefined && parsed.spanIndex >= spans.length)
            return undefined;
        return { kind: "location", ...parsed };
    }
    updateSearchSelection() {
        for (const [index, element] of Array.from(this.goToResults.querySelectorAll(".go-to-result")).entries()) {
            element.setAttribute("aria-selected", String(index === this.searchSelection));
            if (index === this.searchSelection)
                element.scrollIntoView({ block: "nearest" });
        }
        if (this.visibleSearchResults.length) {
            this.goToInput.setAttribute("aria-activedescendant", `go-to-result-${this.searchSelection}`);
        }
    }
    goToKey(event) {
        if (!this.visibleSearchResults.length)
            return;
        let next = this.searchSelection;
        if (event.key === "ArrowDown")
            next = Math.min(this.visibleSearchResults.length - 1, next + 1);
        else if (event.key === "ArrowUp")
            next = Math.max(0, next - 1);
        else if (event.key === "Home")
            next = 0;
        else if (event.key === "End")
            next = this.visibleSearchResults.length - 1;
        else
            return;
        event.preventDefault();
        this.searchSelection = next;
        this.updateSearchSelection();
    }
    activateSearchResult(entry) {
        if (!entry)
            return;
        const target = entry.value;
        this.goToDialog.close();
        if (target.kind === "location") {
            this.navigate({
                artifactId: target.artifactId,
                offset: target.start,
                end: target.end,
                nodeId: target.nodeId,
                fieldName: target.fieldName,
                spanIndex: target.spanIndex,
            }, { history: "push", scroll: true });
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
        if (index < 0)
            index = spans.length ? 0 : -1;
        const span = index >= 0 ? spans[index] : undefined;
        if (span)
            this.navigateSpan(span, target.nodeId, target.fieldName, index);
        else
            this.selectNode(target.nodeId, true);
    }
    renderTabs() {
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
                let next;
                if (event.key === "ArrowLeft")
                    next = tabs[(index - 1 + tabs.length) % tabs.length];
                if (event.key === "ArrowRight")
                    next = tabs[(index + 1) % tabs.length];
                if (event.key === "Home")
                    next = tabs[0];
                if (event.key === "End")
                    next = tabs[tabs.length - 1];
                if (!next)
                    return;
                event.preventDefault();
                this.selectArtifact(next, true);
            });
            this.artifactTabs.append(button);
        }
    }
    selectArtifact(artifact, restoreFocus) {
        const offset = artifact.id === this.current.artifactId ? this.current.offset : 0;
        this.navigate({ artifactId: artifact.id, offset, end: Math.min(artifact.size, offset + 1) }, { history: "push", scroll: true });
        if (restoreFocus) {
            requestAnimationFrame(() => this.artifactTabs.querySelector(`[data-artifact-id="${CSS.escape(artifact.id)}"]`)?.focus());
        }
    }
    treeFilterState() {
        const query = this.treeFilter.value.trim().toLowerCase();
        const mode = this.options.treeMode;
        const referenced = new Set();
        if (mode === "referenced" && this.current.nodeId) {
            referenced.add(this.current.nodeId);
            for (const link of this.graph.links) {
                if (link.source === this.current.nodeId)
                    referenced.add(link.target);
                if (link.target === this.current.nodeId)
                    referenced.add(link.source);
            }
        }
        const covering = mode === "covers"
            ? new Set(this.covering(this.current.artifactId, this.current.offset).map(node => node.id))
            : new Set();
        const matches = new Set();
        for (const node of this.graph.nodes) {
            let modeMatch = true;
            if (mode === "relevant")
                modeMatch = !this.isIrrelevant(node);
            else if (mode === "artifact") {
                modeMatch = node.spans.some(span => span.artifact === this.current.artifactId)
                    || Object.values(node.fields).some(field => (field.spans ?? []).some(span => span.artifact === this.current.artifactId));
            }
            else if (mode === "covers")
                modeMatch = covering.has(node.id);
            else if (mode === "referenced")
                modeMatch = referenced.has(node.id);
            if (!modeMatch)
                continue;
            const text = [
                node.label,
                node.id,
                node.kind,
                node.description ?? "",
                ...Object.entries(node.fields).flatMap(([name, field]) => [name, field.description ?? ""]),
            ].join(" ").toLowerCase();
            if (!query || query.split(/\s+/).every(token => text.includes(token)))
                matches.add(node.id);
        }
        const active = Boolean(query) || mode !== "all";
        if (!active)
            return { matches, active };
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
    renderTree() {
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
        if (this.treeSpace.parentElement !== this.tree)
            this.tree.replaceChildren(this.treeSpace);
        this.renderTreeViewport();
    }
    renderTreeViewport() {
        if (!this.treeRows.length || this.treeSpace.parentElement !== this.tree)
            return;
        const window = visibleWindow(this.treeRows.length, this.tree.scrollTop, this.tree.clientHeight, ROW_HEIGHT, TREE_OVERSCAN);
        const fragment = document.createDocumentFragment();
        for (let index = window.start; index < window.end; index += 1) {
            const item = this.treeRows[index];
            fragment.append(this.treeRow(item.node, item.depth, index));
        }
        this.treeRowsLayer.replaceChildren(fragment);
    }
    treeRow(node, depth, index) {
        const row = document.createElement("div");
        row.className = `tree-row${this.current.nodeId === node.id ? " selected" : ""}`;
        if (this.options.dimIrrelevant && this.isIrrelevant(node))
            row.classList.add("irrelevant");
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
        if (children.length)
            row.setAttribute("aria-expanded", String(effectivelyExpanded));
        const expander = document.createElement("button");
        expander.type = "button";
        expander.className = `tree-expander${children.length ? "" : " empty"}`;
        expander.tabIndex = -1;
        expander.textContent = effectivelyExpanded ? "▾" : "▸";
        expander.setAttribute("aria-label", effectivelyExpanded ? "Collapse" : "Expand");
        expander.disabled = this.treeFilterActive;
        expander.addEventListener("click", event => {
            event.stopPropagation();
            if (this.expanded.has(node.id))
                this.expanded.delete(node.id);
            else
                this.expanded.add(node.id);
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
        if (this.treeFilter.value.trim() && this.treeMatches.has(node.id))
            label.classList.add("match");
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
    treeKey(event, node) {
        const index = this.treeRows.findIndex(item => item.node.id === node.id);
        let targetId;
        if (event.key === "ArrowDown")
            targetId = this.treeRows[index + 1]?.node.id;
        if (event.key === "ArrowUp")
            targetId = this.treeRows[index - 1]?.node.id;
        if (event.key === "Home")
            targetId = this.treeRows[0]?.node.id;
        if (event.key === "End")
            targetId = this.treeRows[this.treeRows.length - 1]?.node.id;
        if (event.key === "ArrowRight") {
            const children = this.children.get(node.id) ?? [];
            if (children.length && !this.expanded.has(node.id)) {
                this.expanded.add(node.id);
                this.treeStructureDirty = true;
                this.renderTree();
                this.scheduleWorkspaceSave(0);
                targetId = node.id;
            }
            else if (children.length) {
                targetId = children[0].id;
            }
        }
        if (event.key === "ArrowLeft") {
            if (this.expanded.has(node.id) && (this.children.get(node.id)?.length ?? 0) > 0) {
                this.expanded.delete(node.id);
                this.treeStructureDirty = true;
                this.renderTree();
                this.scheduleWorkspaceSave(0);
                targetId = node.id;
            }
            else if (node.parent) {
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
        if (!targetId)
            return;
        event.preventDefault();
        this.focusTreeNode(targetId, true);
    }
    focusTreeNode(nodeId, focus) {
        const index = this.treeRows.findIndex(item => item.node.id === nodeId);
        if (index < 0)
            return;
        this.treeFocusId = nodeId;
        const top = index * ROW_HEIGHT + 3;
        const bottom = top + ROW_HEIGHT;
        if (this.tree.clientHeight <= 0)
            this.tree.scrollTop = top;
        else if (top < this.tree.scrollTop)
            this.tree.scrollTop = top;
        else if (bottom > this.tree.scrollTop + this.tree.clientHeight) {
            this.tree.scrollTop = Math.max(0, bottom - this.tree.clientHeight);
        }
        this.renderTreeViewport();
        if (focus) {
            requestAnimationFrame(() => this.tree.querySelector(`[data-node-id="${CSS.escape(nodeId)}"]`)?.focus());
        }
    }
    selectNode(nodeId, push, arrival) {
        const node = this.nodes.get(nodeId);
        if (!node)
            return;
        this.closeNarrowDrawer();
        let spanIndex = node.spans.findIndex(value => value.artifact === this.current.artifactId);
        if (spanIndex < 0)
            spanIndex = node.spans.length ? 0 : -1;
        const span = spanIndex >= 0 ? node.spans[spanIndex] : undefined;
        if (!span) {
            this.navigate({ ...this.current, nodeId, fieldName: undefined, spanIndex: undefined, arrival }, { history: push ? "push" : "none", scroll: false });
            return;
        }
        this.navigate({
            artifactId: span.artifact,
            offset: span.start,
            end: span.end,
            nodeId,
            spanIndex,
            arrival,
        }, { history: push ? "push" : "none", scroll: true });
    }
    navigate(location, options) {
        const artifact = this.artifacts.get(location.artifactId);
        if (!artifact)
            return;
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
        if (this.current.nodeId)
            this.focusTreeNode(this.current.nodeId, false);
        this.renderInspector();
        this.updateHistoryButtons();
        if (options.scroll)
            this.scrollToOffset(this.current.offset);
        else
            this.renderHex();
        this.renderOverview();
        this.renderContextStatus();
        this.scheduleWorkspaceSave(0);
    }
    revealTree(nodeId) {
        let changed = false;
        let node = this.nodes.get(nodeId);
        while (node?.parent) {
            if (!this.expanded.has(node.parent))
                changed = true;
            this.expanded.add(node.parent);
            node = this.nodes.get(node.parent);
        }
        return changed;
    }
    moveHistory(delta) {
        const location = this.history.move(delta);
        if (location)
            this.navigate(location, { history: "none", scroll: true });
    }
    updateHistoryButtons() {
        required("back").disabled = !this.history.canBack;
        required("forward").disabled = !this.history.canForward;
    }
    scrollToOffset(offset) {
        this.setHexTopOffset(offset);
    }
    setHexTopOffset(offset) {
        const artifact = this.artifacts.get(this.current.artifactId);
        const rowCount = Math.ceil(artifact.size / ROW_BYTES);
        const total = Math.max(ROW_HEIGHT, rowCount * ROW_HEIGHT);
        const visual = Math.min(MAX_SCROLL_HEIGHT, total);
        this.hexSpace.style.height = `${visual}px`;
        if (total <= MAX_SCROLL_HEIGHT) {
            this.hexScroll.scrollTop = Math.floor(offset / ROW_BYTES) * ROW_HEIGHT;
        }
        else {
            const visible = Math.max(1, Math.ceil(this.hexScroll.clientHeight / ROW_HEIGHT));
            const maximumFirst = Math.max(0, rowCount - visible);
            const first = Math.max(0, Math.min(maximumFirst, Math.floor(offset / ROW_BYTES)));
            const maxScroll = Math.max(1, visual - this.hexScroll.clientHeight);
            this.hexScroll.scrollTop = maximumFirst ? (first / maximumFirst) * maxScroll : 0;
        }
        this.renderHex();
    }
    hexTopOffset() {
        const artifact = this.artifacts.get(this.current.artifactId);
        if (!artifact.size)
            return 0;
        const rowCount = Math.ceil(artifact.size / ROW_BYTES);
        const total = Math.max(ROW_HEIGHT, rowCount * ROW_HEIGHT);
        let row;
        if (total <= MAX_SCROLL_HEIGHT) {
            row = Math.floor(this.hexScroll.scrollTop / ROW_HEIGHT);
        }
        else {
            const visible = Math.max(1, Math.ceil(this.hexScroll.clientHeight / ROW_HEIGHT));
            const maximumFirst = Math.max(0, rowCount - visible);
            const maximumScroll = Math.max(1, MAX_SCROLL_HEIGHT - this.hexScroll.clientHeight);
            row = Math.floor((this.hexScroll.scrollTop / maximumScroll) * maximumFirst);
        }
        return Math.min(artifact.size - 1, Math.max(0, row * ROW_BYTES));
    }
    renderOverview() {
        const artifact = this.artifacts.get(this.current.artifactId);
        const context = this.overview.getContext("2d");
        const bounds = this.overview.getBoundingClientRect();
        if (!artifact || !context || bounds.width <= 0 || bounds.height <= 0)
            return;
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
            const baseContext = canvas.getContext("2d");
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
        const referenced = new Set();
        if (this.current.nodeId) {
            for (const link of this.graph.links) {
                if (link.source === this.current.nodeId)
                    referenced.add(link.target);
                if (link.target === this.current.nodeId)
                    referenced.add(link.source);
            }
        }
        context.globalAlpha = 0.85;
        context.fillStyle = this.categoryColor("reference");
        for (const nodeId of referenced) {
            for (const span of this.nodes.get(nodeId)?.spans ?? []) {
                if (span.artifact !== artifact.id)
                    continue;
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
    navigateOverview(position, history) {
        const artifact = this.artifacts.get(this.current.artifactId);
        if (!artifact.size)
            return;
        const offset = Math.max(0, Math.min(artifact.size - 1, Math.floor(Math.max(0, Math.min(1, position)) * artifact.size)));
        const primary = this.selectionCandidates(artifact.id, offset)[0];
        this.navigate({
            artifactId: artifact.id,
            offset,
            end: offset + 1,
            nodeId: primary?.node.id,
            fieldName: primary?.fieldName,
            spanIndex: primary ? this.candidateSpanIndex(primary) : undefined,
        }, { history, scroll: true });
        this.overview.focus();
    }
    overviewKey(event) {
        const artifact = this.artifacts.get(this.current.artifactId);
        if (!artifact.size)
            return;
        const small = Math.max(1, Math.floor(artifact.size / 100));
        const large = Math.max(1, Math.floor(artifact.size / 10));
        let offset = this.current.offset;
        if (event.key === "ArrowUp")
            offset -= small;
        else if (event.key === "ArrowDown")
            offset += small;
        else if (event.key === "PageUp")
            offset -= large;
        else if (event.key === "PageDown")
            offset += large;
        else if (event.key === "Home")
            offset = 0;
        else if (event.key === "End")
            offset = artifact.size - 1;
        else
            return;
        event.preventDefault();
        this.navigateOverview(offset / Math.max(1, artifact.size), "push");
    }
    async renderHex() {
        const token = ++this.renderToken;
        const artifact = this.artifacts.get(this.current.artifactId);
        const rowCount = Math.ceil(artifact.size / ROW_BYTES);
        const totalHeight = Math.max(ROW_HEIGHT, rowCount * ROW_HEIGHT);
        const visualHeight = Math.min(MAX_SCROLL_HEIGHT, totalHeight);
        this.hexSpace.style.height = `${visualHeight}px`;
        const visible = Math.max(1, Math.ceil(this.hexScroll.clientHeight / ROW_HEIGHT));
        let first;
        let scaled = false;
        if (totalHeight <= MAX_SCROLL_HEIGHT) {
            first = Math.max(0, Math.floor(this.hexScroll.scrollTop / ROW_HEIGHT) - OVERSCAN);
        }
        else {
            scaled = true;
            const denominator = Math.max(1, visualHeight - this.hexScroll.clientHeight);
            const progress = this.hexScroll.scrollTop / denominator;
            first = Math.max(0, Math.min(Math.max(0, rowCount - visible), Math.floor(progress * Math.max(0, rowCount - visible))) - OVERSCAN);
        }
        const last = Math.min(rowCount, first + visible + OVERSCAN * 2);
        const start = first * ROW_BYTES;
        const end = Math.min(artifact.size, last * ROW_BYTES);
        let bytes;
        try {
            bytes = await this.cache.get(artifact.id).range(start, end);
        }
        catch (error) {
            if (token === this.renderToken)
                this.hexRows.replaceChildren(this.errorElement(String(error)));
            return;
        }
        if (token !== this.renderToken)
            return;
        const cover = Array.from({ length: end - start }, () => []);
        for (const span of this.spans.get(artifact.id) ?? []) {
            if (span.start >= end)
                break;
            if (span.end <= start)
                continue;
            for (let offset = Math.max(start, span.start); offset < Math.min(end, span.end); offset += 1)
                cover[offset - start]?.push(span);
        }
        for (const values of cover) {
            values.sort((a, b) => this.compareIndexedSpans(a, b));
        }
        const focusedNode = this.current.nodeId ? this.nodes.get(this.current.nodeId) : undefined;
        const focusedSource = this.current.fieldName
            ? focusedNode?.fields[this.current.fieldName]?.spans ?? []
            : focusedNode?.spans ?? [];
        const focusedSpans = focusedSource.filter(span => span.artifact === artifact.id);
        const annotationCapacity = Math.max(1, Math.min(6, Math.floor(Math.max(170, this.hexScroll.clientWidth - 550) / 170)));
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
            const rowEnd = Math.min(artifact.size, rowOffset + ROW_BYTES);
            const starts = new Set();
            for (let offset = Math.max(rowOffset, start); offset < Math.min(rowEnd, end); offset += 1) {
                for (const span of cover[offset - start] ?? []) {
                    if (span.start === offset)
                        starts.add(span);
                }
            }
            row.append(offsetElement, hexCells, asciiCells, this.focusedRangeGuide(rowOffset, focusedSpans), this.semanticRowAnnotations([...starts], annotationCapacity));
            fragment.append(row);
        }
        this.hexRows.replaceChildren(fragment);
    }
    focusedRangeGuide(rowOffset, spans) {
        const guide = document.createElement("span");
        guide.className = "range-guide";
        guide.setAttribute("aria-hidden", "true");
        const rowEnd = rowOffset + ROW_BYTES;
        for (const span of spans) {
            if (span.start >= rowEnd || span.end <= rowOffset)
                continue;
            const segment = document.createElement("span");
            segment.className = "range-guide-segment";
            if (span.start >= rowOffset)
                segment.classList.add("range-start");
            if (span.end <= rowEnd)
                segment.classList.add("range-end");
            guide.append(segment);
        }
        return guide;
    }
    semanticRowAnnotations(starts, capacity) {
        const lane = document.createElement("span");
        lane.className = "row-annotations";
        const unique = new Map();
        for (const span of starts.sort((left, right) => this.compareIndexedSpans(left, right))) {
            const key = `${span.node.id}\u0000${span.fieldName ?? ""}`;
            if (!unique.has(key))
                unique.set(key, span);
        }
        const candidates = [...unique.values()];
        const regions = candidates.filter(span => !span.fieldName);
        const fields = candidates.filter(span => span.fieldName)
            .sort((left, right) => left.start - right.start || this.compareIndexedSpans(left, right));
        const selected = candidates.find(span => span.node.id === this.current.nodeId
            && span.fieldName === this.current.fieldName);
        const prioritized = [selected, ...regions.slice(0, 2), ...fields, ...regions.slice(2)]
            .filter((span) => span !== undefined);
        const ordered = prioritized.filter((span, index) => prioritized.indexOf(span) === index);
        const visible = ordered.slice(0, capacity);
        for (const span of visible)
            lane.append(this.semanticRowAnnotation(span));
        if (ordered.length > visible.length) {
            const more = document.createElement("span");
            more.className = "row-annotation-more";
            more.textContent = `+${ordered.length - visible.length}`;
            more.title = ordered.slice(visible.length).map(span => span.fieldName ? `${span.node.label} › ${span.fieldName}` : span.node.label).join("\n");
            lane.append(more);
        }
        return lane;
    }
    semanticRowAnnotation(span) {
        const button = document.createElement("button");
        button.type = "button";
        button.tabIndex = -1;
        button.className = "row-annotation";
        button.style.setProperty("--semantic-color", this.categoryColor(span.node.category));
        const field = span.fieldName ? span.node.fields[span.fieldName] : undefined;
        const rawValue = field ? displayValue(field.value).replace(/\s+/g, " ") : undefined;
        const compactValue = rawValue && rawValue.length > 64 ? `${rawValue.slice(0, 63)}…` : rawValue;
        button.textContent = span.fieldName
            ? `${span.fieldName}${compactValue ? `: ${compactValue}` : ""}`
            : span.node.label;
        const detail = [
            span.fieldName ? `${this.nodePath(span.node)} › ${span.fieldName}` : this.nodePath(span.node),
            field?.description ?? span.node.description,
            `${hex(span.start)}–${hex(span.end)} · ${span.end - span.start} B`,
            evidenceText(field ?? span.node),
        ].filter(Boolean);
        button.title = detail.join("\n");
        const selected = this.current.nodeId === span.node.id
            && this.current.fieldName === span.fieldName;
        if (selected)
            button.classList.add("selected");
        if (this.options.dimIrrelevant && this.isIrrelevant(span.node, span.fieldName)) {
            button.classList.add("irrelevant");
        }
        button.addEventListener("click", () => {
            this.navigate({
                artifactId: span.artifact,
                offset: span.start,
                end: span.end,
                nodeId: span.node.id,
                fieldName: span.fieldName,
                spanIndex: this.candidateSpanIndex({ node: span.node, fieldName: span.fieldName, span }),
            }, { history: "push", scroll: false });
        });
        return button;
    }
    byteElement(offset, value, annotations, ascii) {
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
        const preferredField = (nodeId) => {
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
            if (this.options.dimIrrelevant && irrelevant)
                cell.classList.add("irrelevant");
            cell.style.setProperty("--semantic-color", semanticColor);
            if (selectedField)
                cell.classList.add("field-highlight");
            if (this.hoveredNode && annotations.some(span => span.node.id === this.hoveredNode && (!this.hoveredField || span.fieldName === this.hoveredField))) {
                cell.classList.add("hovered");
            }
            cell.title = annotations
                .map(span => {
                if (!span.fieldName)
                    return span.node.description ? `${span.node.label} — ${span.node.description}` : span.node.label;
                const field = span.node.fields[span.fieldName];
                const label = `${span.node.label} · ${span.fieldName}`;
                const explanation = field?.description ? `${label} — ${field.description}` : label;
                const evidence = field ? evidenceText(field) : undefined;
                return evidence ? `${explanation} [${evidence}]` : explanation;
            })
                .filter((value, index, all) => all.indexOf(value) === index)
                .join("\n");
        }
        if (offset === this.current.offset)
            cell.classList.add("selected");
        else if (offset >= this.current.offset && offset < this.current.end)
            cell.classList.add("range-selected");
        cell.addEventListener("click", event => {
            const candidates = this.selectionCandidates(this.current.artifactId, offset);
            let selected = candidates[0];
            if (!event.shiftKey && offset === this.current.offset && candidates.length > 1) {
                const currentIndex = candidates.findIndex(candidate => candidate.node.id === this.current.nodeId && candidate.fieldName === this.current.fieldName);
                if (currentIndex >= 0)
                    selected = candidates[(currentIndex + 1) % candidates.length];
            }
            const start = event.shiftKey ? Math.min(this.current.offset, offset) : offset;
            const end = event.shiftKey ? Math.max(this.current.offset, offset) + 1 : offset + 1;
            this.navigate({
                artifactId: this.current.artifactId,
                offset: start,
                end,
                nodeId: selected?.node.id,
                fieldName: selected?.fieldName,
                spanIndex: selected ? this.candidateSpanIndex(selected) : undefined,
            }, { history: "push", scroll: false });
            this.hexScroll.focus();
        });
        if (primary) {
            cell.addEventListener("mouseenter", () => this.setHovered(primary.node.id, primary.fieldName, offset));
            cell.addEventListener("mouseleave", () => this.setHovered(undefined));
        }
        return cell;
    }
    categoryColor(category) {
        return this.categories.get(category)?.color ?? "#5e6670";
    }
    fieldColor(node, fieldName) {
        const index = Math.max(0, Object.keys(node.fields).indexOf(fieldName));
        const hue = Math.round((index * 137.508 + 8) % 360);
        return `hsl(${hue} 48% 42%)`;
    }
    isIrrelevant(node, fieldName) {
        if (fieldName) {
            const field = node.fields[fieldName];
            if (field?.relevance === "irrelevant" || field?.properties?.relevance === "irrelevant")
                return true;
        }
        return node.relevance === "irrelevant" || node.properties.relevance === "irrelevant";
    }
    isConfirmedIrrelevantField(field) {
        const relevance = field.relevance ?? field.properties?.relevance;
        const confidence = field.confidence ?? field.properties?.confidence;
        return relevance === "irrelevant" && confidence === "confirmed";
    }
    displayColor(color, irrelevant) {
        if (!this.options.dimIrrelevant || !irrelevant)
            return color;
        return `color-mix(in srgb, ${color} 24%, var(--text-faint))`;
    }
    compareIndexedSpans(left, right) {
        return Number(Boolean(right.fieldName)) - Number(Boolean(left.fieldName))
            || left.length - right.length
            || (this.depths.get(right.node.id) ?? 0) - (this.depths.get(left.node.id) ?? 0)
            || left.start - right.start
            || left.node.id.localeCompare(right.node.id)
            || (left.fieldName ?? "").localeCompare(right.fieldName ?? "");
    }
    coveringSpans(artifactId, offset) {
        return (this.spans.get(artifactId) ?? [])
            .filter(span => span.start <= offset && offset < span.end)
            .sort((left, right) => this.compareIndexedSpans(left, right));
    }
    selectionCandidates(artifactId, offset) {
        const result = [];
        for (const span of this.coveringSpans(artifactId, offset)) {
            if (result.some(value => value.node.id === span.node.id && value.fieldName === span.fieldName))
                continue;
            result.push({ node: span.node, fieldName: span.fieldName, span });
        }
        return result;
    }
    candidateSpanIndex(candidate) {
        const spans = candidate.fieldName
            ? candidate.node.fields[candidate.fieldName]?.spans ?? []
            : candidate.node.spans;
        return Math.max(0, spans.findIndex(span => span.id !== undefined && candidate.span.id !== undefined
            ? span.id === candidate.span.id
            : span.artifact === candidate.span.artifact
                && span.start === candidate.span.start
                && span.end === candidate.span.end));
    }
    cycleOverlap(delta) {
        const candidates = this.selectionCandidates(this.current.artifactId, this.current.offset);
        if (candidates.length < 2)
            return;
        const currentIndex = candidates.findIndex(candidate => candidate.node.id === this.current.nodeId && candidate.fieldName === this.current.fieldName);
        const index = currentIndex < 0
            ? (delta > 0 ? 0 : candidates.length - 1)
            : (currentIndex + delta + candidates.length) % candidates.length;
        const selected = candidates[index];
        this.navigate({
            ...this.current,
            nodeId: selected.node.id,
            fieldName: selected.fieldName,
            spanIndex: this.candidateSpanIndex(selected),
            arrival: undefined,
        }, { history: "push", scroll: false });
    }
    selectedSpans() {
        const node = this.current.nodeId ? this.nodes.get(this.current.nodeId) : undefined;
        return this.current.fieldName ? node?.fields[this.current.fieldName]?.spans ?? [] : node?.spans ?? [];
    }
    moveSelectedSpan(delta) {
        const spans = this.selectedSpans();
        if (spans.length < 2)
            return;
        const current = this.current.spanIndex ?? spans.findIndex(span => span.artifact === this.current.artifactId && span.start === this.current.offset && span.end === this.current.end);
        const index = (Math.max(0, current) + delta + spans.length) % spans.length;
        this.navigateSpan(spans[index], this.current.nodeId, this.current.fieldName, index);
    }
    covering(artifactId, offset) {
        return this.selectionCandidates(artifactId, offset)
            .map(value => value.node)
            .filter((node, index, all) => all.findIndex(value => value.id === node.id) === index);
    }
    renderInspector() {
        this.inspector.replaceChildren();
        const node = this.current.nodeId ? this.nodes.get(this.current.nodeId) : undefined;
        const artifact = this.artifacts.get(this.current.artifactId);
        const title = document.createElement("h1");
        title.className = "inspector-title";
        title.textContent = node?.label ?? `${artifact.name} ${hex(this.current.offset)}`;
        const subtitle = document.createElement("div");
        subtitle.className = "inspector-subtitle";
        subtitle.textContent = node ? `${node.kind} · ${node.id}` : `${this.current.end - this.current.offset} selected byte${this.current.end - this.current.offset === 1 ? "" : "s"}`;
        this.inspector.append(title, subtitle);
        if (node) {
            const meaning = this.meaningElement(node);
            if (meaning)
                this.inspector.append(meaning);
            this.inspector.append(this.locationElement(node));
            const fields = this.fieldsElement(node);
            if (fields)
                this.inspector.append(fields);
            const outgoing = this.graph.links.filter(link => link.source === node.id);
            const incoming = this.graph.links.filter(link => link.target === node.id);
            if (outgoing.length || incoming.length) {
                const section = this.section("references");
                for (const group of this.referenceGroups(outgoing, true))
                    section.append(group);
                for (const group of this.referenceGroups(incoming, false))
                    section.append(group);
                this.inspector.append(section);
            }
            if (Object.keys(node.addresses).length) {
                const section = this.section("addresses");
                for (const [name, address] of Object.entries(node.addresses))
                    section.append(this.kv(name, `${address.space}:${hex(address.value)}`));
                this.inspector.append(section);
            }
            if (Object.keys(node.properties).length) {
                const section = this.section("properties");
                for (const [name, value] of Object.entries(node.properties))
                    section.append(this.kv(name, displayValue(value)));
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
    meaningElement(node) {
        const evidence = evidenceText(node);
        if (!node.description && !evidence && !this.current.arrival)
            return undefined;
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
    locationElement(node) {
        const section = this.section("location");
        const lineage = [];
        let cursor = node;
        while (cursor) {
            lineage.push(cursor);
            cursor = cursor.parent ? this.nodes.get(cursor.parent) : undefined;
        }
        lineage.reverse();
        const path = document.createElement("div");
        path.className = "containment-path";
        for (const [index, pathNode] of lineage.entries()) {
            if (index)
                path.append(document.createTextNode(" › "));
            const button = document.createElement("button");
            button.type = "button";
            button.textContent = pathNode.label;
            button.title = pathNode.description ?? pathNode.id;
            button.addEventListener("click", () => this.selectNode(pathNode.id, true));
            path.append(button);
        }
        section.append(path);
        if (node.spans.length) {
            for (const [index, span] of node.spans.entries())
                section.append(this.spanButton(span, index));
        }
        else {
            const logical = document.createElement("div");
            logical.className = "muted";
            logical.textContent = "logical entity · no physical byte span";
            section.append(logical);
        }
        return section;
    }
    fieldsElement(node) {
        const indexed = Object.entries(node.fields).map(([name, field], order) => ({
            name,
            field,
            order,
            offset: Math.min(...(field.spans ?? []).map(span => span.start), Number.POSITIVE_INFINITY),
        }));
        if (!indexed.length)
            return undefined;
        indexed.sort((left, right) => left.offset - right.offset || left.order - right.order);
        const collapsible = indexed.filter(({ field }) => this.isConfirmedIrrelevantField(field)).length;
        let action;
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
            if (this.options.collapseIrrelevantFields && this.isConfirmedIrrelevantField(field))
                continue;
            section.append(this.fieldElement(node, name, field));
        }
        return section;
    }
    fieldElement(node, name, field) {
        const spans = field.spans ?? [];
        const element = document.createElement(spans.length ? "button" : "div");
        if (element instanceof HTMLButtonElement) {
            element.type = "button";
            element.title = "Navigate to field bytes";
            element.addEventListener("click", () => this.navigateSpan(spans[0], node.id, name, 0));
        }
        const irrelevant = this.isIrrelevant(node, name);
        element.className = `field-row${spans.length ? " jump" : ""}`;
        if (this.options.dimIrrelevant && irrelevant)
            element.classList.add("irrelevant");
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
    knowledgeElement(artifact) {
        const knowledge = this.knowledge.get(artifact.id);
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
    semanticHelpElement() {
        const details = document.createElement("details");
        details.className = "semantic-help";
        const summary = document.createElement("summary");
        summary.textContent = "semantic color and evidence";
        const content = document.createElement("p");
        content.textContent = "Color groups bytes by semantic category; field marks distinguish decoded fields. Confirmed is specified or validated, an inference is evidence-qualified, raw is observed without interpretation, unresolved is deliberately undecided, and irrelevant means understood but unused by the stated supported interpretation.";
        details.append(summary, content);
        return details;
    }
    section(label, action) {
        const section = document.createElement("section");
        section.className = "inspector-section";
        if (label) {
            const headingRow = document.createElement("div");
            headingRow.className = "section-heading";
            const heading = document.createElement("h2");
            heading.textContent = label;
            headingRow.append(heading);
            if (action)
                headingRow.append(action);
            section.append(headingRow);
        }
        return section;
    }
    kv(key, value) {
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
    spanButton(span, spanIndex) {
        const artifact = this.artifacts.get(span.artifact);
        const button = document.createElement("button");
        button.type = "button";
        button.className = "jump";
        const selectedNode = this.current.nodeId ? this.nodes.get(this.current.nodeId) : undefined;
        if (selectedNode?.spans[spanIndex] === span
            && span.artifact === this.current.artifactId
            && span.start === this.current.offset
            && span.end === this.current.end)
            button.classList.add("active-span");
        const label = span.label ? `${span.label} · ` : "";
        button.textContent = `${label}${artifact.name}  ${hex(span.start)}–${hex(span.end)}  (${span.end - span.start})`;
        button.addEventListener("click", () => this.navigateSpan(span, this.current.nodeId, undefined, spanIndex));
        return button;
    }
    navigateSpan(span, nodeId, fieldName, spanIndex) {
        this.closeNarrowDrawer();
        this.navigate({ artifactId: span.artifact, offset: span.start, end: span.end, nodeId, fieldName, spanIndex }, { history: "push", scroll: true });
    }
    referenceGroups(links, outgoing) {
        const grouped = new Map();
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
            const evidence = evidenceText(values[0]);
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
            for (const link of values)
                group.append(this.referenceButton(link, outgoing));
            return group;
        });
    }
    referenceButton(link, outgoing) {
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
    hexKey(event) {
        if (event.key === "[" || event.key === "]") {
            event.preventDefault();
            if (event.altKey)
                this.moveSelectedSpan(event.key === "]" ? 1 : -1);
            else
                this.cycleOverlap(event.key === "]" ? 1 : -1);
            return;
        }
        const artifact = this.artifacts.get(this.current.artifactId);
        let offset = this.current.offset;
        if (event.key === "ArrowLeft")
            offset -= 1;
        else if (event.key === "ArrowRight")
            offset += 1;
        else if (event.key === "ArrowUp")
            offset -= ROW_BYTES;
        else if (event.key === "ArrowDown")
            offset += ROW_BYTES;
        else if (event.key === "PageUp")
            offset -= ROW_BYTES * Math.max(1, Math.floor(this.hexScroll.clientHeight / ROW_HEIGHT));
        else if (event.key === "PageDown")
            offset += ROW_BYTES * Math.max(1, Math.floor(this.hexScroll.clientHeight / ROW_HEIGHT));
        else if (event.key === "Home")
            offset = event.ctrlKey ? 0 : Math.floor(offset / ROW_BYTES) * ROW_BYTES;
        else if (event.key === "End")
            offset = event.ctrlKey ? artifact.size - 1 : Math.min(artifact.size - 1, Math.floor(offset / ROW_BYTES) * ROW_BYTES + 15);
        else
            return;
        event.preventDefault();
        offset = Math.max(0, Math.min(Math.max(0, artifact.size - 1), offset));
        const primary = this.selectionCandidates(artifact.id, offset)[0];
        this.navigate({
            artifactId: artifact.id,
            offset,
            end: offset + 1,
            nodeId: primary?.node.id,
            fieldName: primary?.fieldName,
            spanIndex: primary ? this.candidateSpanIndex(primary) : undefined,
        }, { history: "coalesce", coalesceKey: "byte-step", scroll: true });
    }
    async copyLocation() {
        const value = this.currentLocationExpression();
        try {
            await navigator.clipboard.writeText(value);
            this.setStatus(`copied ${value}`);
        }
        catch {
            this.setStatus(`copy unavailable · ${value}`);
        }
    }
    currentLocationExpression() {
        return formatLocationExpression({
            artifactId: this.current.artifactId,
            start: this.current.offset,
            end: this.current.end,
            nodeId: this.current.nodeId,
            fieldName: this.current.fieldName,
            spanIndex: this.current.spanIndex,
        });
    }
    renderContextStatus(nodeId = this.current.nodeId, fieldName = this.current.fieldName, offset) {
        this.status.replaceChildren();
        const node = nodeId ? this.nodes.get(nodeId) : undefined;
        const contextOffset = offset ?? this.current.offset;
        const candidate = node
            ? this.coveringSpans(this.current.artifactId, contextOffset).find(span => span.node.id === node.id && (fieldName === undefined || span.fieldName === fieldName))
            : undefined;
        const span = candidate ?? (fieldName ? node?.fields[fieldName]?.spans?.[0] : node?.spans[0]);
        const artifact = this.artifacts.get(span?.artifact ?? this.current.artifactId);
        const path = [];
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
                const index = candidates.findIndex(value => value.node.id === this.current.nodeId && value.fieldName === this.current.fieldName);
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
    statusSeparator() {
        const separator = document.createElement("span");
        separator.className = "status-separator";
        separator.textContent = "›";
        return separator;
    }
    setHovered(nodeId, fieldName, offset) {
        if (this.hoveredNode === nodeId && this.hoveredField === fieldName)
            return;
        this.hoveredNode = nodeId;
        this.hoveredField = fieldName;
        if (nodeId) {
            this.renderContextStatus(nodeId, fieldName, offset);
        }
        else {
            this.renderContextStatus();
        }
        this.renderHex();
    }
    setStatus(value) { this.status.textContent = value; }
    errorElement(value) {
        const element = document.createElement("div");
        element.className = "error";
        element.textContent = value;
        return element;
    }
}
async function main() {
    const response = await fetch(new URL("api/graph", window.location.href));
    if (!response.ok)
        throw new Error(`graph request failed: ${response.status}`);
    const graph = await response.json();
    if (graph.format !== "npunlock.artifact-graph" || ![1, 2].includes(graph.version)) {
        throw new Error("unsupported artifact graph format");
    }
    new Viewer(graph, response.headers.get("ETag") ?? undefined, response.headers.get("X-NPUnlock-Viz-State-Version") ?? undefined);
}
main().catch(error => {
    const app = required("app");
    const element = document.createElement("pre");
    element.className = "error";
    element.textContent = error instanceof Error ? error.message : String(error);
    app.replaceChildren(element);
});
