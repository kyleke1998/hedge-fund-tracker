/**
 * SPY three-month rate of change (database/momentum.csv).
 *
 * One reading per session: the percent change in SPY over a rolling 63-session
 * window. Generated offline by `pipenv run gen-momentum` — the browser has
 * neither the decade of daily bars nor the budget to roll the window itself —
 * so the page only reads it.
 */

import { cachedFetch, fetchCSV } from "./fetch";

export interface RawMomentumRow {
  date: string;
  roc: string;
}

/** One session's reading. `t` is the epoch ms the time axis plots against. */
export interface MomentumPoint {
  date: string;
  t: number;
  roc: number;
}

/**
 * Parse long-format momentum rows into chronological chart points.
 *
 * Rows without a usable date or value are dropped rather than charted as zero:
 * a gap in the series is a session that had no reading, and a flat quarter is a
 * very different claim. Kept pure (no fetch) so it is unit-testable.
 */
export function parseMomentumRows(raw: RawMomentumRow[]): MomentumPoint[] {
  return raw
    .flatMap((row) => {
      const t = Date.parse(`${row.date}T00:00:00Z`);
      const roc = Number(row.roc);
      if (!Number.isFinite(t) || !Number.isFinite(roc) || row.roc.trim() === "") return [];
      return [{ date: row.date, t, roc }];
    })
    .sort((a, b) => a.t - b.t);
}

/**
 * Epoch-ms ticks at the first available session of each calendar year.
 *
 * The axis is a time scale over thousands of sessions, so evenly-spaced
 * automatic ticks land mid-year; anchoring them to years is what makes the span
 * readable.
 */
export function yearTicks(points: MomentumPoint[]): number[] {
  const seen = new Set<string>();
  return points.flatMap((point) => {
    const year = point.date.slice(0, 4);
    if (seen.has(year)) return [];
    seen.add(year);
    return [point.t];
  });
}

/**
 * Y-axis ticks every five points, from the multiple below the trough to the one
 * above the peak, always spanning zero (the line that separates an index up on
 * the quarter from one that is down). A fixed axis would waste height the series
 * has never reached in either direction.
 */
export function rocTicks(points: MomentumPoint[]): number[] {
  if (points.length === 0) return [-5, 0, 5];
  const values = points.map((point) => point.roc);
  const step = 5;
  const low = Math.min(0, Math.floor(Math.min(...values) / step) * step);
  const high = Math.max(0, Math.ceil(Math.max(...values) / step) * step);
  const ticks: number[] = [];
  for (let value = low; value <= high + 1e-9; value += step) ticks.push(value);
  return ticks;
}

export async function getMomentum(): Promise<MomentumPoint[]> {
  return cachedFetch("momentum", async () => {
    const raw = await fetchCSV<RawMomentumRow>("/database/momentum.csv", [
      "date",
      "roc",
    ] satisfies readonly (keyof RawMomentumRow)[]);
    return parseMomentumRows(raw);
  });
}
