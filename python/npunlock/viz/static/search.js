export function parseLocationExpression(query, currentArtifactId, artifactSizes) {
    const trimmed = query.trim();
    const match = /^(?:([^:\s]+):)?(0x[0-9a-f]+|[0-9a-f]+)(?:-(0x[0-9a-f]+|[0-9a-f]+))?$/i.exec(trimmed);
    if (!match)
        return undefined;
    const requestedArtifact = match[1] ?? currentArtifactId;
    const artifactId = artifactSizes.has(requestedArtifact)
        ? requestedArtifact
        : [...artifactSizes.keys()].find(value => value.toLowerCase() === requestedArtifact.toLowerCase());
    if (artifactId === undefined)
        return undefined;
    const size = artifactSizes.get(artifactId);
    if (size === undefined)
        return undefined;
    const start = Number.parseInt(match[2].replace(/^0x/i, ""), 16);
    const end = match[3] ? Number.parseInt(match[3].replace(/^0x/i, ""), 16) : start + 1;
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end))
        return undefined;
    if (start < 0 || start >= size || end <= start || end > size)
        return undefined;
    return { artifactId, start, end };
}
function scoreText(value, query, tokens) {
    const text = value.toLowerCase();
    if (text === query)
        return 1000;
    if (text.startsWith(query))
        return 700 - Math.min(200, text.length - query.length);
    const index = text.indexOf(query);
    if (index >= 0)
        return 500 - Math.min(200, index);
    if (tokens.every(token => text.includes(token)))
        return 300;
    return -1;
}
export function searchEntries(entries, query, limit = 60) {
    const normalized = query.trim().toLowerCase();
    if (!normalized)
        return entries.slice(0, limit);
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
