/**
 * Market-regime signals (database/regime.csv, long format).
 *
 * Generated offline by `pipenv run gen-regime`, because the underlying metrics
 * need split detection across consecutive quarters — the browser has no cheap
 * way to do that, and the page only ever reads the result.
 */

import { cachedFetch, fetchCSV } from "./fetch";

export interface RawRegimeRow {
  kind: string;
  quarter: string;
  key: string;
  value: string;
}

/** Per-quarter series, each aligned index-for-index with `quarters`. */
export interface RegimeData {
  quarters: string[];
  /** Change and level metrics keyed by name; null where a quarter has no value. */
  metrics: Record<string, (number | null)[]>;
  mega: { shareIndex: (number | null)[]; weightPct: (number | null)[] };
  sectors: Record<string, (number | null)[]>;
}

export interface SectorChange {
  sector: string;
  from: number;
  to: number;
  delta: number;
}

/** Chronological quarter sort — "2022Q1" strings order correctly as plain text. */
const byQuarter = (a: string, b: string) => a.localeCompare(b);

function numeric(raw: string): number | null {
  if (raw === undefined || raw === null || raw.trim() === "") return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

/**
 * Pivot long-format regime rows into per-quarter series.
 *
 * A metric absent for a quarter stays null rather than becoming 0 — the change
 * metrics have no value in the first quarter (there is no prior quarter to
 * compare against) and charting that as zero would invent a collapse. Kept pure
 * (no fetch) so it is unit-testable.
 */
export function parseRegimeRows(raw: RawRegimeRow[]): RegimeData {
  const quarters = [...new Set(raw.map((r) => r.quarter).filter(Boolean))].sort(byQuarter);
  const slot = new Map(quarters.map((q, i) => [q, i]));
  const blank = () => Array<number | null>(quarters.length).fill(null);

  const metrics: Record<string, (number | null)[]> = {};
  const sectors: Record<string, (number | null)[]> = {};
  const mega = { shareIndex: blank(), weightPct: blank() };

  for (const r of raw) {
    const index = slot.get(r.quarter);
    if (index === undefined) continue;
    const value = numeric(r.value);
    if (value === null) continue;

    if (r.kind === "metric") {
      (metrics[r.key] ??= blank())[index] = value;
    } else if (r.kind === "sector") {
      (sectors[r.key] ??= blank())[index] = value;
    } else if (r.kind === "mega") {
      if (r.key === "share_index") mega.shareIndex[index] = value;
      if (r.key === "weight_pct") mega.weightPct[index] = value;
    }
  }
  return { quarters, metrics, mega, sectors };
}

/**
 * Sector weight change in percentage points between two quarters, biggest move
 * first. Sectors missing from either end are skipped.
 */
export function sectorWindowChange(data: RegimeData, from: string, to: string): SectorChange[] {
  const a = data.quarters.indexOf(from);
  const b = data.quarters.indexOf(to);
  if (a < 0 || b < 0) return [];
  return Object.entries(data.sectors)
    .flatMap(([sector, series]) => {
      const start = series[a];
      const end = series[b];
      if (start === null || start === undefined || end === null || end === undefined) return [];
      return [{ sector, from: start, to: end, delta: end - start }];
    })
    .sort((x, y) => Math.abs(y.delta) - Math.abs(x.delta));
}

/**
 * Rebase the chain-linked mega-cap share index onto a chosen base quarter.
 *
 * The index is chain-linked, so rescaling by the base quarter's level is exact —
 * it is not an approximation of re-running the chain from that point.
 */
export function reindexShareIndex(data: RegimeData, base: string): (number | null)[] {
  const index = data.quarters.indexOf(base);
  const divisor = index < 0 ? null : data.mega.shareIndex[index];
  if (divisor === null || divisor === undefined || divisor === 0) return data.mega.shareIndex;
  return data.mega.shareIndex.map((v) => (v === null ? null : (v / divisor) * 100));
}

export async function getRegime(): Promise<RegimeData> {
  return cachedFetch("regime", async () => {
    const raw = await fetchCSV<RawRegimeRow>("/database/regime.csv", [
      "kind",
      "quarter",
      "key",
      "value",
    ] satisfies readonly (keyof RawRegimeRow)[]);
    return parseRegimeRows(raw);
  });
}
