export interface SearchEntry<T> {
  id: string;
  label: string;
  detail: string;
  terms: string[];
  value: T;
}

export interface ParsedLocation {
  artifactId: string;
  start: number;
  end: number;
  nodeId?: string;
  fieldName?: string;
  spanIndex?: number;
}

export function formatLocationExpression(location: ParsedLocation): string {
  const range = `0x${location.start.toString(16)}${
    location.end > location.start + 1 ? `-0x${location.end.toString(16)}` : ""
  }`;
  const parameters = new URLSearchParams();
  if (location.nodeId) parameters.set("node", location.nodeId);
  if (location.fieldName) parameters.set("field", location.fieldName);
  if (location.spanIndex !== undefined) parameters.set("span", String(location.spanIndex));
  const query = parameters.toString();
  return `${encodeURIComponent(location.artifactId)}:${range}${query ? `?${query}` : ""}`;
}

export function parseLocationExpression(
  query: string,
  currentArtifactId: string,
  artifactSizes: ReadonlyMap<string, number>,
): ParsedLocation | undefined {
  const trimmed = query.trim();
  const queryIndex = trimmed.indexOf("?");
  const base = queryIndex >= 0 ? trimmed.slice(0, queryIndex) : trimmed;
  const parameterText = queryIndex >= 0 ? trimmed.slice(queryIndex + 1) : "";
  if (parameterText.includes("?")) return undefined;
  const match = /^(?:([^:\s]+):)?(0x[0-9a-f]+|[0-9a-f]+)(?:-(0x[0-9a-f]+|[0-9a-f]+))?$/i.exec(base);
  if (!match) return undefined;
  let requestedArtifact: string;
  try {
    requestedArtifact = match[1] ? decodeURIComponent(match[1]) : currentArtifactId;
  } catch {
    return undefined;
  }
  const artifactId = artifactSizes.has(requestedArtifact)
    ? requestedArtifact
    : [...artifactSizes.keys()].find(value => value.toLowerCase() === requestedArtifact.toLowerCase());
  if (artifactId === undefined) return undefined;
  const size = artifactSizes.get(artifactId);
  if (size === undefined) return undefined;
  const start = Number.parseInt(match[2]!.replace(/^0x/i, ""), 16);
  const end = match[3] ? Number.parseInt(match[3].replace(/^0x/i, ""), 16) : start + 1;
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end)) return undefined;
  if (start < 0 || start >= size || end <= start || end > size) return undefined;
  const parameters = new URLSearchParams(parameterText);
  let unknownParameter = false;
  parameters.forEach((_value, key) => {
    if (!["node", "field", "span"].includes(key)) unknownParameter = true;
  });
  if (unknownParameter) return undefined;
  if (["node", "field", "span"].some(key => parameters.getAll(key).length > 1)) return undefined;
  const nodeId = parameters.get("node") || undefined;
  const fieldName = parameters.get("field") || undefined;
  if (fieldName && !nodeId) return undefined;
  const spanText = parameters.get("span");
  const spanIndex = spanText === null ? undefined : Number(spanText);
  if (spanIndex !== undefined && (!Number.isSafeInteger(spanIndex) || spanIndex < 0 || !nodeId)) return undefined;
  const result: ParsedLocation = { artifactId, start, end };
  if (nodeId) result.nodeId = nodeId;
  if (fieldName) result.fieldName = fieldName;
  if (spanIndex !== undefined) result.spanIndex = spanIndex;
  return result;
}

function scoreText(value: string, query: string, tokens: string[]): number {
  const text = value.toLowerCase();
  if (text === query) return 1000;
  if (text.startsWith(query)) return 700 - Math.min(200, text.length - query.length);
  const index = text.indexOf(query);
  if (index >= 0) return 500 - Math.min(200, index);
  if (tokens.every(token => text.includes(token))) return 300;
  return -1;
}

export function searchEntries<T>(entries: readonly SearchEntry<T>[], query: string, limit = 60): SearchEntry<T>[] {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return entries.slice(0, limit);
  const tokens = normalized.split(/\s+/).filter(Boolean);
  return entries
    .map((entry, order) => {
      const values = [entry.label, ...entry.terms, entry.detail];
      const score = Math.max(...values.map(value => scoreText(value, normalized, tokens)));
      return { entry, order, score };
    })
    .filter(item => item.score >= 0)
    .sort((left, right) => right.score - left.score || left.order - right.order)
    .slice(0, limit)
    .map(item => item.entry);
}
