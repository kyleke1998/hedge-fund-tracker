import { describe, it, expect } from "vitest";
import { withHoldingsOverlay } from "../holdingsOverlay";
import { quarterEndDate } from "../quarters";
import type { TickerHoldingsPoint } from "../dataService";

const point = (
  quarter: TickerHoldingsPoint["quarter"],
  totalShares: number,
  totalValue = 0,
  holderCount = 0,
): TickerHoldingsPoint => ({
  quarter,
  asOf: quarterEndDate(quarter),
  totalShares,
  totalValue,
  holderCount,
});

describe("withHoldingsOverlay", () => {
  const history = [point("2025Q1", 1000, 5_000_000, 3), point("2025Q2", 1500, 9_000_000, 4)];

  it("leaves candles before the first quarter without holdings data", () => {
    const merged = withHoldingsOverlay([{ date: "2025-01-15", close: 10 }], history);

    expect(merged[0].holdingsShares).toBeUndefined();
    expect(merged[0].holdingsQuarter).toBeUndefined();
  });

  it("attaches the quarter's totals on its as-of date", () => {
    const merged = withHoldingsOverlay([{ date: "2025-03-31", close: 10 }], history);

    expect(merged[0].holdingsShares).toBe(1000);
    expect(merged[0].holdingsValue).toBe(5_000_000);
    expect(merged[0].holdingsHolders).toBe(3);
    expect(merged[0].holdingsQuarter).toBe("2025Q1");
  });

  it("forward-fills a quarter until the next one lands", () => {
    const merged = withHoldingsOverlay(
      [
        { date: "2025-04-30", close: 10 },
        { date: "2025-05-30", close: 11 },
        { date: "2025-06-30", close: 12 },
        { date: "2025-07-31", close: 13 },
      ],
      history,
    );

    expect(merged.map((m) => m.holdingsShares)).toEqual([1000, 1000, 1500, 1500]);
    expect(merged.map((m) => m.holdingsQuarter)).toEqual(["2025Q1", "2025Q1", "2025Q2", "2025Q2"]);
  });

  it("preserves the original candle fields", () => {
    const merged = withHoldingsOverlay([{ date: "2025-03-31", close: 10, high: 12 }], history);

    expect(merged[0].date).toBe("2025-03-31");
    expect(merged[0].close).toBe(10);
    expect(merged[0].high).toBe(12);
  });

  it("returns the series untouched when there is no holdings history", () => {
    const series = [{ date: "2025-03-31", close: 10 }];

    const merged = withHoldingsOverlay(series, []);

    expect(merged).toEqual(series);
  });

  it("sorts an out-of-order history before filling", () => {
    const merged = withHoldingsOverlay(
      [{ date: "2025-08-31", close: 10 }],
      [history[1], history[0]],
    );

    expect(merged[0].holdingsShares).toBe(1500);
  });
});
