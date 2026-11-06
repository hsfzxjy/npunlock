export interface TreeItem {
  id: string;
}

export interface VisibleTreeItem<T extends TreeItem> {
  node: T;
  depth: number;
}

export interface VisibleWindow {
  start: number;
  end: number;
}

export function flattenTree<T extends TreeItem>(
  roots: readonly T[],
  children: ReadonlyMap<string, readonly T[]>,
  expanded: ReadonlySet<string>,
  included?: ReadonlySet<string>,
  expandAll = false,
): VisibleTreeItem<T>[] {
  const result: VisibleTreeItem<T>[] = [];
  const visit = (node: T, depth: number) => {
    if (included && !included.has(node.id)) return;
    result.push({ node, depth });
    if (!expandAll && !expanded.has(node.id)) return;
    for (const child of children.get(node.id) ?? []) visit(child, depth + 1);
  };
  for (const root of roots) visit(root, 0);
  return result;
}

export function visibleWindow(
  count: number,
  scrollTop: number,
  viewportHeight: number,
  rowHeight: number,
  overscan: number,
): VisibleWindow {
  if (count <= 0 || rowHeight <= 0) return { start: 0, end: 0 };
  const first = Math.max(0, Math.floor(Math.max(0, scrollTop) / rowHeight) - overscan);
  const visible = Math.ceil(Math.max(0, viewportHeight) / rowHeight);
  return {
    start: Math.min(count, first),
    end: Math.min(count, first + visible + overscan * 2 + 1),
  };
}
