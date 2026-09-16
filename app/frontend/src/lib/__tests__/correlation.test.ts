import { describe, it, expect } from "vitest";
import {
  correlationTicks,
  parseCorrelationRows,
  yearTicks,
  type RawCorrelationRow,
} from "@/lib/data/correlation";

const row = (date: string, correlation: string, stocks = "95"): RawCorrelationRow => ({
  date,
  correlation,
  stocks,
});

describe("parseCorrelationRows", () => {
  it("orders readings chronologically", () => {
    const points = parseCorrelationRows([row("2024-03-01", "0.31"), row("2024-01-02", "0.44")]);
    expect(points.map((p) => p.date)).toEqual(["2024-01-02", "2024-03-01"]);
  });

  it("exposes each reading as a numeric value with its name count", () => {
    const [point] = parseCorrelationRows([row("2024-01-02", "0.3123", "97")]);
    expect(point.correlation).toBeCloseTo(0.3123, 6);
    expect(point.stocks).toBe(97);
  });

  it("plots dates as UTC epoch milliseconds, so the axis is timezone-stable", () => {
    const [point] = parseCorrelationRows([row("2024-01-02", "0.31")]);
    expect(point.t).toBe(Date.UTC(2024, 0, 2));
  });

  it("drops a session with no reading rather than charting it as zero", () => {
    // A blank value means the window had too few usable names; zero
    // correlation would claim the basket stopped moving together.
    const points = parseCorrelationRows([row("2024-01-02", ""), row("2024-01-03", "0.31")]);
    expect(points.map((p) => p.date)).toEqual(["2024-01-03"]);
  });

  it("drops a row whose date cannot be parsed", () => {
    expect(parseCorrelationRows([row("not-a-date", "0.31")])).toEqual([]);
  });

  it("keeps a negative reading", () => {
    const [point] = parseCorrelationRows([row("2024-01-02", "-0.05")]);
    expect(point.correlation).toBeCloseTo(-0.05, 6);
  });
});

describe("yearTicks", () => {
  it("returns the first session of each calendar year", () => {
    const points = parseCorrelationRows([
      row("2023-01-04", "0.3"),
      row("2023-06-01", "0.4"),
      row("2024-01-03", "0.2"),
      row("2024-02-01", "0.25"),
    ]);
    expect(yearTicks(points)).toEqual([Date.UTC(2023, 0, 4), Date.UTC(2024, 0, 3)]);
  });

  it("has no ticks for an empty series", () => {
    expect(yearTicks([])).toEqual([]);
  });
});

describe("correlationTicks", () => {
  it("spans zero to the next tenth above the peak", () => {
    const points = parseCorrelationRows([row("2024-01-02", "0.31"), row("2024-01-03", "0.66")]);
    expect(correlationTicks(points)).toEqual([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);
  });

  it("extends below zero when the basket decoupled", () => {
    const points = parseCorrelationRows([row("2024-01-02", "-0.04"), row("2024-01-03", "0.12")]);
    expect(correlationTicks(points)).toEqual([-0.1, 0, 0.1, 0.2]);
  });

  it("never draws a tick above perfect correlation", () => {
    const points = parseCorrelationRows([row("2024-01-02", "1")]);
    expect(correlationTicks(points).at(-1)).toBe(1);
  });

  it("falls back to a minimal axis for an empty series", () => {
    expect(correlationTicks([])).toEqual([0, 0.1]);
  });
});
