import { API_BASE } from "@/lib/config";
import { mergeQuarterly } from "@/lib/quarterlyOverlay";

/** One quarter's estimated cost basis for the tracked funds holding a ticker. */
export interface CostBasisPoint {
  quarter: string;
  /** Quarter-end date the estimate is as of (ISO `YYYY-MM-DD`). */
  asOf: string;
  shares: number;
  holders: number;
  /** 10th percentile of the simulated aggregate basis. */
  low: number;
  /** Median of the simulated aggregate basis — the line inside the band. */
  mid: number;
  /** 90th percentile of the simulated aggregate basis. */
  high: number;
  /** Share of the position whose vintage predates our filing history. */
  seededPct: number;
}

/** Fields the band adds to each price point; absent before the first quarter on record. */
export interface CostBasisFields {
  /** `[low, high]`, the pair recharts draws the range area from. */
  costBasisRange: [number, number];
  costBasisMid: number;
  costBasisQuarter: string;
  costBasisSeededPct: number;
}

/**
 * Fetch a ticker's estimated cost-basis band from the local backend.
 *
 * The estimate needs daily prices and every quarter of filings, so it has no
 * static-build equivalent — callers must skip it when there is no API.
 */
export async function fetchCostBasis(ticker: string): Promise<CostBasisPoint[]> {
  if (!API_BASE) throw new Error("offline");
  const res = await fetch(`${API_BASE}/api/stocks/${encodeURIComponent(ticker)}/cost-basis`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = (await res.json()) as { points?: CostBasisPoint[] };
  return data.points ?? [];
}

/**
 * Merge a cost-basis band onto a price series as a step function.
 */
export function withCostBasisBand<T extends { date: string }>(
  series: readonly T[],
  band: readonly CostBasisPoint[],
): (T & Partial<CostBasisFields>)[] {
  return mergeQuarterly(series, band, (point) => ({
    costBasisRange: [point.low, point.high] as [number, number],
    costBasisMid: point.mid,
    costBasisQuarter: point.quarter,
    costBasisSeededPct: point.seededPct,
  }));
}
