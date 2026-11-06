export interface ArrivalState {
  sourceNodeId: string;
  targetNodeId: string;
  kind: string;
}

export interface LocationState {
  artifactId: string;
  offset: number;
  end: number;
  nodeId?: string;
  fieldName?: string;
  spanIndex?: number;
  arrival?: ArrivalState;
}

export type HistoryMode = "none" | "push" | "replace" | "coalesce";

export interface NavigationSnapshot {
  entries: LocationState[];
  index: number;
}

function sameArrival(left: ArrivalState | undefined, right: ArrivalState | undefined): boolean {
  if (left === undefined || right === undefined) return left === right;
  return left.sourceNodeId === right.sourceNodeId
    && left.targetNodeId === right.targetNodeId
    && left.kind === right.kind;
}

export function sameShareableLocation(left: LocationState, right: LocationState): boolean {
  return left.artifactId === right.artifactId
    && left.offset === right.offset
    && left.end === right.end
    && left.nodeId === right.nodeId
    && left.fieldName === right.fieldName
    && left.spanIndex === right.spanIndex;
}

export function sameLocation(left: LocationState, right: LocationState): boolean {
  return sameShareableLocation(left, right)
    && sameArrival(left.arrival, right.arrival);
}

function copyLocation(value: LocationState): LocationState {
  return value.arrival ? { ...value, arrival: { ...value.arrival } } : { ...value };
}

export class NavigationHistory {
  private entries: LocationState[] = [];
  private index = -1;
  private coalesceKey?: string;

  record(location: LocationState, mode: HistoryMode, key?: string): void {
    if (mode === "none") {
      this.coalesceKey = undefined;
      return;
    }
    const value = copyLocation(location);
    const current = this.entries[this.index];
    if (mode === "replace") {
      if (this.index < 0) {
        this.entries.push(value);
        this.index = 0;
      } else {
        this.entries[this.index] = value;
      }
      this.coalesceKey = undefined;
      return;
    }
    if (current && sameLocation(current, value)) {
      this.coalesceKey = mode === "coalesce" ? key : undefined;
      return;
    }
    if (mode === "coalesce" && key && this.coalesceKey === key && this.index === this.entries.length - 1) {
      this.entries[this.index] = value;
      return;
    }
    this.entries = this.entries.slice(0, this.index + 1);
    this.entries.push(value);
    this.index = this.entries.length - 1;
    this.coalesceKey = mode === "coalesce" ? key : undefined;
  }

  move(delta: number): LocationState | undefined {
    const next = this.index + delta;
    if (next < 0 || next >= this.entries.length) return undefined;
    this.index = next;
    this.coalesceKey = undefined;
    const value = this.entries[this.index];
    return value ? copyLocation(value) : undefined;
  }

  snapshot(limit = 100): NavigationSnapshot {
    if (!Number.isInteger(limit) || limit < 1 || this.index < 0) return { entries: [], index: -1 };
    const count = Math.min(limit, this.entries.length);
    const start = Math.max(0, Math.min(this.index, this.entries.length - count));
    return {
      entries: this.entries.slice(start, start + count).map(copyLocation),
      index: this.index - start,
    };
  }

  restore(snapshot: NavigationSnapshot): boolean {
    if (!Array.isArray(snapshot.entries)
      || snapshot.entries.length < 1
      || !Number.isInteger(snapshot.index)
      || snapshot.index < 0
      || snapshot.index >= snapshot.entries.length) return false;
    this.entries = snapshot.entries.map(copyLocation);
    this.index = snapshot.index;
    this.coalesceKey = undefined;
    return true;
  }

  get canBack(): boolean {
    return this.index > 0;
  }

  get canForward(): boolean {
    return this.index >= 0 && this.index < this.entries.length - 1;
  }

  get length(): number {
    return this.entries.length;
  }
}
