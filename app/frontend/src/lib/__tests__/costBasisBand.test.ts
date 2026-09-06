import { describe, it, expect } from "vitest";
import { withCostBasisBand, type CostBasisPoint } from "../costBasisBand";

const point = (
  quarter: string,
  asOf: string,
  low: number,
  mid: number,
  high: number,
  seededPct = 0,
): CostBasisPoint => ({
  quarter,
  asOf,
  shares: 1000,
  holders: 3,
  low,
  mid,
  high,
  seededPct,
});

describe("withCostBasisBand", () => {
  const band = [
    point("2025Q1", "2025-03-31", 8, 10, 12),
    point("2025Q2", "2025-06-30", 9, 11, 14, 25),
  ];

  it("leaves points before the first quarter without a band", () => {
    const merged = withCostBasisBand([{ date: "2025-01-15", close: 10 }], band);

    expect(merged[0].costBasisRange).toBeUndefined();
    expect(merged[0].costBasisMid).toBeUndefined();
  });

  it("carries the band as a [low, high] pair for the range area", () => {
    const merged = withCostBasisBand([{ date: "2025-03-31", close: 10 }], band);

    expect(merged[0].costBasisRange).toEqual([8, 12]);
    expect(merged[0].costBasisMid).toBe(10);
    expect(merged[0].costBasisQuarter).toBe("2025Q1");
  });

  it("forward-fills a quarter until the next one lands", () => {
    const merged = withCostBasisBand(
      [
        { date: "2025-04-30", close: 10 },
        { date: "2025-06-30", close: 12 },
        { date: "2025-09-30", close: 13 },
      ],
      band,
    );

    expect(merged.map((m) => m.costBasisMid)).toEqual([10, 11, 11]);
    expect(merged[2].costBasisSeededPct).toBe(25);
  });

  it("preserves the original price fields", () => {
    const merged = withCostBasisBand([{ date: "2025-03-31", close: 10, high: 12 }], band);

    expect(merged[0].date).toBe("2025-03-31");
    expect(merged[0].high).toBe(12);
  });

  it("returns the series untouched when there is no band", () => {
    const series = [{ date: "2025-03-31", close: 10 }];

    expect(withCostBasisBand(series, [])).toEqual(series);
  });

  it("sorts an out-of-order band before filling", () => {
    const merged = withCostBasisBand([{ date: "2025-08-31", close: 10 }], [band[1], band[0]]);

    expect(merged[0].costBasisMid).toBe(11);
  });
});
