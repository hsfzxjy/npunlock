"use strict";
const ROW_BYTES = 16;
const ROW_HEIGHT = 21;
const PAGE_BYTES = 16 * 1024;
const MAX_SCROLL_HEIGHT = 16_000_000;
const OVERSCAN = 8;
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
    artifactTabs = required("artifact-tabs");
    tree = required("tree");
    inspector = required("inspector");
    hexScroll = required("hex-scroll");
    hexSpace = required("hex-space");
    hexRows = required("hex-rows");
    status = required("status");
    workspace = required("workspace");
    offsetInput = required("offset-input");
    nodes = new Map();
    artifacts = new Map();
    categories = new Map();
    children = new Map();
    roots = [];
    depths = new Map();
    spans = new Map();
    cache = new Map();
    expanded = new Set();
    current;
    history = [];
    historyIndex = -1;
    hoveredNode;
    renderToken = 0;
    constructor(graph) {
        this.graph = graph;
        for (const artifact of graph.artifacts) {
            this.artifacts.set(artifact.id, artifact);
            this.cache.set(artifact.id, new ByteCache(artifact));
            this.spans.set(artifact.id, []);
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
            }
            for (const field of Object.values(node.fields)) {
                for (const span of field.spans ?? []) {
                    this.spans.get(span.artifact)?.push({ ...span, node, length: span.end - span.start });
                }
            }
        }
        for (const values of this.spans.values())
            values.sort((a, b) => a.start - b.start || a.end - b.end);
        for (const root of this.roots) {
            this.expanded.add(root.id);
            this.assignDepth(root, 0);
        }
        const first = graph.artifacts[0];
        if (!first)
            throw new Error("artifact graph contains no artifacts");
        this.current = { artifactId: first.id, offset: 0, end: Math.min(1, first.size) };
        this.bind();
        this.renderTabs();
        this.renderTree();
        this.navigate(this.current, true, false);
        new ResizeObserver(() => this.renderHex()).observe(this.hexScroll);
    }
    assignDepth(node, depth) {
        this.depths.set(node.id, depth);
        for (const child of this.children.get(node.id) ?? [])
            this.assignDepth(child, depth + 1);
    }
    bind() {
        this.hexScroll.addEventListener("scroll", () => this.renderHex());
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
            this.navigate({ artifactId: artifact.id, offset: parsed, end: parsed + 1 }, true, true);
        });
        required("copy").addEventListener("click", () => this.copyLocation());
        required("back").addEventListener("click", () => this.moveHistory(-1));
        required("forward").addEventListener("click", () => this.moveHistory(1));
        required("toggle-tree").addEventListener("click", event => this.togglePanel("tree", event));
        required("toggle-inspector").addEventListener("click", event => this.togglePanel("inspector", event));
        this.bindSplitter(required("left-splitter"), "--tree-width", true, 150, 520);
        this.bindSplitter(required("right-splitter"), "--inspector-width", false, 210, 620);
        window.addEventListener("keydown", event => {
            if (event.altKey && event.key === "ArrowLeft") {
                event.preventDefault();
                this.moveHistory(-1);
            }
            if (event.altKey && event.key === "ArrowRight") {
                event.preventDefault();
                this.moveHistory(1);
            }
        });
    }
    bindSplitter(splitter, variable, fromLeft, minimum, maximum) {
        splitter.addEventListener("pointerdown", event => {
            splitter.setPointerCapture(event.pointerId);
            splitter.classList.add("dragging");
        });
        splitter.addEventListener("pointermove", event => {
            if (!splitter.hasPointerCapture(event.pointerId))
                return;
            const bounds = this.workspace.getBoundingClientRect();
            const value = fromLeft ? event.clientX - bounds.left : bounds.right - event.clientX;
            document.documentElement.style.setProperty(variable, `${Math.max(minimum, Math.min(maximum, value))}px`);
        });
        const finish = (event) => {
            if (splitter.hasPointerCapture(event.pointerId))
                splitter.releasePointerCapture(event.pointerId);
            splitter.classList.remove("dragging");
        };
        splitter.addEventListener("pointerup", finish);
        splitter.addEventListener("pointercancel", finish);
    }
    togglePanel(panel, event) {
        const button = event.currentTarget;
        const className = `${panel}-hidden`;
        const hidden = this.workspace.classList.toggle(className);
        button.setAttribute("aria-pressed", String(!hidden));
        this.renderHex();
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
            button.setAttribute("aria-selected", String(artifact.id === this.current.artifactId));
            button.addEventListener("click", () => {
                const offset = artifact.id === this.current.artifactId ? this.current.offset : 0;
                this.navigate({ artifactId: artifact.id, offset, end: Math.min(artifact.size, offset + 1) }, true, true);
            });
            this.artifactTabs.append(button);
        }
    }
    renderTree() {
        this.tree.replaceChildren();
        const fragment = document.createDocumentFragment();
        const appendNode = (node, depth) => {
            const row = document.createElement("div");
            row.className = `tree-row${this.current.nodeId === node.id ? " selected" : ""}`;
            row.role = "treeitem";
            row.dataset.nodeId = node.id;
            row.style.paddingLeft = `${depth * 14 + 3}px`;
            row.setAttribute("aria-level", String(depth + 1));
            row.setAttribute("aria-selected", String(this.current.nodeId === node.id));
            const children = this.children.get(node.id) ?? [];
            const expander = document.createElement("button");
            expander.type = "button";
            expander.className = `tree-expander${children.length ? "" : " empty"}`;
            expander.tabIndex = -1;
            expander.textContent = this.expanded.has(node.id) ? "▾" : "▸";
            expander.setAttribute("aria-label", this.expanded.has(node.id) ? "Collapse" : "Expand");
            expander.addEventListener("click", event => {
                event.stopPropagation();
                if (this.expanded.has(node.id))
                    this.expanded.delete(node.id);
                else
                    this.expanded.add(node.id);
                this.renderTree();
            });
            const mark = document.createElement("span");
            mark.className = "tree-mark";
            mark.style.background = this.categoryColor(node.category);
            const label = document.createElement("span");
            label.className = "tree-label";
            label.textContent = node.label;
            label.title = node.label;
            const kind = document.createElement("span");
            kind.className = "tree-kind";
            kind.textContent = node.kind;
            row.append(expander, mark, label, kind);
            row.addEventListener("click", () => this.selectNode(node.id, true));
            row.addEventListener("mouseenter", () => this.setHovered(node.id));
            row.addEventListener("mouseleave", () => this.setHovered(undefined));
            fragment.append(row);
            if (this.expanded.has(node.id))
                for (const child of children)
                    appendNode(child, depth + 1);
        };
        for (const root of this.roots)
            appendNode(root, 0);
        this.tree.append(fragment);
    }
    selectNode(nodeId, push) {
        const node = this.nodes.get(nodeId);
        if (!node)
            return;
        let span = node.spans.find(value => value.artifact === this.current.artifactId) ?? node.spans[0];
        if (!span) {
            this.navigate({ ...this.current, nodeId }, push, false);
            return;
        }
        this.navigate({ artifactId: span.artifact, offset: span.start, end: span.end, nodeId }, push, true);
    }
    navigate(location, push, scroll) {
        const artifact = this.artifacts.get(location.artifactId);
        if (!artifact)
            return;
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
        if (this.current.nodeId)
            this.revealTree(this.current.nodeId);
        this.offsetInput.value = hex(this.current.offset);
        this.renderTabs();
        this.renderTree();
        this.renderInspector();
        this.updateHistoryButtons();
        if (scroll)
            this.scrollToOffset(this.current.offset);
        else
            this.renderHex();
        this.setStatus(`${artifact.name}  ${hex(this.current.offset)}  ${this.current.end - this.current.offset} byte${this.current.end - this.current.offset === 1 ? "" : "s"}`);
    }
    revealTree(nodeId) {
        let node = this.nodes.get(nodeId);
        while (node?.parent) {
            this.expanded.add(node.parent);
            node = this.nodes.get(node.parent);
        }
        requestAnimationFrame(() => this.tree.querySelector(`[data-node-id="${CSS.escape(nodeId)}"]`)?.scrollIntoView({ block: "nearest" }));
    }
    moveHistory(delta) {
        const next = this.historyIndex + delta;
        if (next < 0 || next >= this.history.length)
            return;
        this.historyIndex = next;
        const location = this.history[next];
        if (location)
            this.navigate(location, false, true);
    }
    updateHistoryButtons() {
        required("back").disabled = this.historyIndex <= 0;
        required("forward").disabled = this.historyIndex >= this.history.length - 1;
    }
    scrollToOffset(offset) {
        const artifact = this.artifacts.get(this.current.artifactId);
        const rowCount = Math.ceil(artifact.size / ROW_BYTES);
        const total = rowCount * ROW_HEIGHT;
        const visual = Math.min(MAX_SCROLL_HEIGHT, Math.max(ROW_HEIGHT, total));
        this.hexSpace.style.height = `${visual}px`;
        if (total <= MAX_SCROLL_HEIGHT) {
            this.hexScroll.scrollTop = Math.floor(offset / ROW_BYTES) * ROW_HEIGHT;
        }
        else {
            const maxScroll = Math.max(1, visual - this.hexScroll.clientHeight);
            this.hexScroll.scrollTop = Math.floor((offset / Math.max(1, artifact.size - 1)) * maxScroll);
        }
        this.renderHex();
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
        for (const values of cover)
            values.sort((a, b) => a.length - b.length || (this.depths.get(b.node.id) ?? 0) - (this.depths.get(a.node.id) ?? 0));
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
        if (primary) {
            cell.classList.add("annotated");
            cell.style.setProperty("--semantic-color", this.categoryColor(primary.node.category));
            if (this.hoveredNode && annotations.some(span => span.node.id === this.hoveredNode))
                cell.classList.add("hovered");
            cell.title = annotations.map(span => span.node.label).filter((value, index, all) => all.indexOf(value) === index).join("\n");
        }
        if (offset === this.current.offset)
            cell.classList.add("selected");
        else if (offset >= this.current.offset && offset < this.current.end)
            cell.classList.add("range-selected");
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
    categoryColor(category) {
        return this.categories.get(category)?.color ?? "#5e6670";
    }
    covering(artifactId, offset) {
        const values = (this.spans.get(artifactId) ?? []).filter(span => span.start <= offset && offset < span.end);
        values.sort((a, b) => a.length - b.length || (this.depths.get(b.node.id) ?? 0) - (this.depths.get(a.node.id) ?? 0));
        return values.map(value => value.node).filter((node, index, all) => all.findIndex(value => value.id === node.id) === index);
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
            if (node.spans.length) {
                const section = this.section("byte spans");
                for (const span of node.spans)
                    section.append(this.spanButton(span));
                this.inspector.append(section);
            }
            if (Object.keys(node.addresses).length) {
                const section = this.section("addresses");
                for (const [name, address] of Object.entries(node.addresses))
                    section.append(this.kv(name, `${address.space}:${hex(address.value)}`));
                this.inspector.append(section);
            }
            if (Object.keys(node.fields).length) {
                const section = this.section("fields");
                for (const [name, field] of Object.entries(node.fields)) {
                    if (field.spans?.length) {
                        const button = document.createElement("button");
                        button.type = "button";
                        button.className = "jump";
                        button.textContent = `${name}  ${field.display ?? displayValue(field.value)}`;
                        button.title = "Navigate to field bytes";
                        button.addEventListener("click", () => this.navigateSpan(field.spans[0], node.id));
                        section.append(button);
                    }
                    else
                        section.append(this.kv(name, field.display ?? displayValue(field.value)));
                }
                this.inspector.append(section);
            }
            if (Object.keys(node.properties).length) {
                const section = this.section();
                for (const [name, value] of Object.entries(node.properties))
                    section.append(this.kv(name, displayValue(value)));
                this.inspector.append(section);
            }
            const outgoing = this.graph.links.filter(link => link.source === node.id);
            const incoming = this.graph.links.filter(link => link.target === node.id);
            if (outgoing.length || incoming.length) {
                const section = this.section("references");
                for (const link of outgoing)
                    section.append(this.referenceButton(link, true));
                for (const link of incoming)
                    section.append(this.referenceButton(link, false));
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
    section(label) {
        const section = document.createElement("section");
        section.className = "inspector-section";
        if (label) {
            const heading = document.createElement("h2");
            heading.textContent = label;
            section.append(heading);
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
    spanButton(span) {
        const artifact = this.artifacts.get(span.artifact);
        const button = document.createElement("button");
        button.type = "button";
        button.className = "jump";
        button.textContent = `${artifact.name}  ${hex(span.start)}–${hex(span.end)}  (${span.end - span.start})`;
        button.addEventListener("click", () => this.navigateSpan(span, this.current.nodeId));
        return button;
    }
    navigateSpan(span, nodeId) {
        this.navigate({ artifactId: span.artifact, offset: span.start, end: span.end, nodeId }, true, true);
    }
    referenceButton(link, outgoing) {
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
    hexKey(event) {
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
        const primary = this.covering(artifact.id, offset)[0];
        this.navigate({ artifactId: artifact.id, offset, end: offset + 1, nodeId: primary?.id }, true, true);
    }
    async copyLocation() {
        const artifact = this.artifacts.get(this.current.artifactId);
        const range = this.current.end > this.current.offset + 1 ? `${hex(this.current.offset)}-${hex(this.current.end)}` : hex(this.current.offset);
        const value = `${artifact.id}:${range}`;
        try {
            await navigator.clipboard.writeText(value);
            this.setStatus(`copied ${value}`);
        }
        catch {
            this.setStatus(`copy unavailable · ${value}`);
        }
    }
    setHovered(nodeId) {
        if (this.hoveredNode === nodeId)
            return;
        this.hoveredNode = nodeId;
        if (nodeId) {
            const node = this.nodes.get(nodeId);
            if (node)
                this.setStatus(`${node.label} · ${node.kind}`);
        }
        else {
            const artifact = this.artifacts.get(this.current.artifactId);
            this.setStatus(`${artifact.name}  ${hex(this.current.offset)}  ${this.current.end - this.current.offset} byte${this.current.end - this.current.offset === 1 ? "" : "s"}`);
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
    if (graph.format !== "npunlock.artifact-graph" || graph.version !== 1)
        throw new Error("unsupported artifact graph format");
    new Viewer(graph);
}
main().catch(error => {
    const app = required("app");
    const element = document.createElement("pre");
    element.className = "error";
    element.textContent = error instanceof Error ? error.message : String(error);
    app.replaceChildren(element);
});
