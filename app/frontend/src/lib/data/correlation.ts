/**
 * Realized average stock correlation (database/correlation.csv).
 *
 * One reading per session: the mean pairwise correlation of daily returns
 * across the QQQ basket over a rolling three-month window. Generated offline by
 * `pipenv run gen-correlation` — the browser has neither the decade of daily
 * bars nor the budget to roll the window itself — so the page only reads it.
 */

import { cachedFetch, fetchCSV } from "./fetch";

export interface RawCorrelationRow {
  date: string;
  correlation: string;
  stocks: string;
}

/** One session's reading. `t` is the epoch ms the time axis plots against. */
export interface CorrelationPoint {
  date: string;
  t: number;
  correlation: number;
  stocks: number;
}

/**
 * Parse long-format correlation rows into chronological chart points.
 *
 * Rows without a usable date or value are dropped rather than charted as zero:
 * a gap in the series is a session that had no reading, and zero correlation is
 * a very different claim. Kept pure (no fetch) so it is unit-testable.
 */
export function parseCorrelationRows(raw: RawCorrelationRow[]): CorrelationPoint[] {
  return raw
    .flatMap((row) => {
      const t = Date.parse(`${row.date}T00:00:00Z`);
      const correlation = Number(row.correlation);
      if (!Number.isFinite(t) || !Number.isFinite(correlation) || row.correlation.trim() === "")
        return [];
      return [{ date: row.date, t, correlation, stocks: Number(row.stocks) || 0 }];
    })
    .sort((a, b) => a.t - b.t);
}

/**
 * Epoch-ms ticks at the first available session of each calendar year.
 *
 * The axis is a time scale over ~2,500 sessions, so evenly-spaced automatic
 * ticks land mid-year; anchoring them to years is what makes the span readable.
 */
export function yearTicks(points: CorrelationPoint[]): number[] {
  const seen = new Set<string>();
  return points.flatMap((point) => {
    const year = point.date.slice(0, 4);
    if (seen.has(year)) return [];
    seen.add(year);
    return [point.t];
  });
}

/**
 * Y-axis ticks in tenths, from the tenth below the trough to the tenth above
 * the peak (never past 1.0, where every name moves as one). A fixed 0-to-1 axis
 * would waste most of its height on levels the basket has never reached.
 */
export function correlationTicks(points: CorrelationPoint[]): number[] {
  const values = points.map((point) => point.correlation);
  const low = Math.min(0, Math.floor(Math.min(...values, 0) * 10) / 10);
  const high = Math.min(1, Math.max(0.1, Math.ceil(Math.max(...values, 0) * 10) / 10));
  const ticks: number[] = [];
  for (let tenth = Math.round(low * 10); tenth <= Math.round(high * 10); tenth += 1) {
    ticks.push(tenth / 10);
  }
  return ticks;
}

export async function getCorrelation(): Promise<CorrelationPoint[]> {
  return cachedFetch("correlation", async () => {
    const raw = await fetchCSV<RawCorrelationRow>("/database/correlation.csv", [
      "date",
      "correlation",
      "stocks",
    ] satisfies readonly (keyof RawCorrelationRow)[]);
    return parseCorrelationRows(raw);
  });
}
