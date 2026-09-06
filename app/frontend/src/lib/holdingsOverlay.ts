import type { TickerHoldingsPoint } from "@/lib/dataService";
import { mergeQuarterly } from "@/lib/quarterlyOverlay";

/** Fields the overlay adds to each price point; absent before the first quarter on record. */
export interface HoldingsOverlayFields {
  holdingsShares?: number;
  holdingsValue?: number;
  holdingsHolders?: number;
  holdingsQuarter?: string;
}

/**
 * Merge a quarterly holdings history onto a price series as a step function.
 */
export function withHoldingsOverlay<T extends { date: string }>(
  series: readonly T[],
  history: readonly TickerHoldingsPoint[],
): (T & HoldingsOverlayFields)[] {
  return mergeQuarterly(series, history, (active) => ({
    holdingsShares: active.totalShares,
    holdingsValue: active.totalValue,
    holdingsHolders: active.holderCount,
    holdingsQuarter: active.quarter,
  }));
}
