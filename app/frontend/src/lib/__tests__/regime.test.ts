import { describe, it, expect } from "vitest";
import {
  parseRegimeRows,
  sectorWindowChange,
  reindexShareIndex,
  type RawRegimeRow,
} from "@/lib/data/regime";

const row = (kind: string, quarter: string, key: string, value: number): RawRegimeRow => ({
  kind,
  quarter,
  key,
  value: String(value),
});

const SAMPLE: RawRegimeRow[] = [
  row("metric", "2022Q1", "positions", 7166),
  row("mega", "2022Q1", "share_index", 100),
  row("mega", "2022Q1", "weight_pct", 12.4),
  row("sector", "2022Q1", "Technology", 26),
  row("sector", "2022Q1", "Healthcare", 13),
  row("metric", "2022Q2", "new_close", 0.74),
  row("metric", "2022Q2", "recycle", 0.67),
  row("metric", "2022Q2", "untouched_pct", 22.14),
  row("metric", "2022Q2", "positions", 6925),
  row("mega", "2022Q2", "share_index", 94.9),
  row("mega", "2022Q2", "weight_pct", 11.06),
  row("sector", "2022Q2", "Technology", 25),
  row("sector", "2022Q2", "Healthcare", 15),
];

describe("parseRegimeRows", () => {
  it("orders quarters chronologically", () => {
    const shuffled = [...SAMPLE].reverse();
    expect(parseRegimeRows(shuffled).quarters).toEqual(["2022Q1", "2022Q2"]);
  });

  it("exposes each metric as a per-quarter series aligned to the quarter list", () => {
    const { metrics } = parseRegimeRows(SAMPLE);
    expect(metrics.new_close).toEqual([null, 0.74]);
    expect(metrics.positions).toEqual([7166, 6925]);
  });

  it("leaves a metric missing for a quarter as null rather than zero", () => {
    // The first quarter has no prior quarter, so change metrics are undefined
    // there — plotting them as 0 would invent a collapse.
    const { metrics } = parseRegimeRows(SAMPLE);
    expect(metrics.recycle[0]).toBeNull();
  });

  it("collects the mega-cap series", () => {
    const { mega } = parseRegimeRows(SAMPLE);
    expect(mega.shareIndex).toEqual([100, 94.9]);
    expect(mega.weightPct).toEqual([12.4, 11.06]);
  });

  it("collects sector weights keyed by sector", () => {
    const { sectors } = parseRegimeRows(SAMPLE);
    expect(sectors.Technology).toEqual([26, 25]);
    expect(sectors.Healthcare).toEqual([13, 15]);
  });

  it("returns empty structures for an empty file", () => {
    const parsed = parseRegimeRows([]);
    expect(parsed.quarters).toEqual([]);
    expect(parsed.sectors).toEqual({});
  });

  it("ignores rows whose value is not numeric", () => {
    const bad = [...SAMPLE, { kind: "metric", quarter: "2022Q2", key: "recycle", value: "" }];
    expect(parseRegimeRows(bad).metrics.recycle).toEqual([null, 0.67]);
  });
});

describe("sectorWindowChange", () => {
  const parsed = parseRegimeRows(SAMPLE);

  it("returns the percentage-point change between two quarters", () => {
    const change = sectorWindowChange(parsed, "2022Q1", "2022Q2");
    expect(change.find((c) => c.sector === "Technology")?.delta).toBeCloseTo(-1);
    expect(change.find((c) => c.sector === "Healthcare")?.delta).toBeCloseTo(2);
  });

  it("sorts by the size of the move so the biggest rotations lead", () => {
    const change = sectorWindowChange(parsed, "2022Q1", "2022Q2");
    expect(change[0].sector).toBe("Healthcare");
  });

  it("is empty when a quarter is unknown", () => {
    expect(sectorWindowChange(parsed, "2022Q1", "1999Q9")).toEqual([]);
  });
});

describe("reindexShareIndex", () => {
  const parsed = parseRegimeRows(SAMPLE);

  it("rebases a chain-linked index onto any base quarter", () => {
    const rebased = reindexShareIndex(parsed, "2022Q2");
    expect(rebased[1]).toBeCloseTo(100);
    expect(rebased[0]).toBeCloseTo((100 / 94.9) * 100);
  });

  it("is a no-op when the base is already the first quarter", () => {
    expect(reindexShareIndex(parsed, "2022Q1")).toEqual([100, 94.9]);
  });

  it("falls back to the raw series for an unknown base", () => {
    expect(reindexShareIndex(parsed, "1999Q9")).toEqual([100, 94.9]);
  });
});
