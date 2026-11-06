export function flattenTree(roots, children, expanded, included, expandAll = false) {
    const result = [];
    const visit = (node, depth) => {
        if (included && !included.has(node.id))
            return;
        result.push({ node, depth });
        if (!expandAll && !expanded.has(node.id))
            return;
        for (const child of children.get(node.id) ?? [])
            visit(child, depth + 1);
    };
    for (const root of roots)
        visit(root, 0);
    return result;
}
export function visibleWindow(count, scrollTop, viewportHeight, rowHeight, overscan) {
    if (count <= 0 || rowHeight <= 0)
        return { start: 0, end: 0 };
    const first = Math.max(0, Math.floor(Math.max(0, scrollTop) / rowHeight) - overscan);
    const visible = Math.ceil(Math.max(0, viewportHeight) / rowHeight);
    return {
        start: Math.min(count, first),
        end: Math.min(count, first + visible + overscan * 2 + 1),
    };
}
