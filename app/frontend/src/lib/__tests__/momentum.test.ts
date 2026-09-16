import { describe, it, expect } from "vitest";
import { parseMomentumRows, rocTicks, yearTicks, type RawMomentumRow } from "@/lib/data/momentum";

const row = (date: string, roc: string): RawMomentumRow => ({ date, roc });

describe("parseMomentumRows", () => {
  it("orders readings chronologically", () => {
    const points = parseMomentumRows([row("2024-03-01", "3.1"), row("2024-01-02", "-4.4")]);
    expect(points.map((p) => p.date)).toEqual(["2024-01-02", "2024-03-01"]);
  });

  it("exposes each reading as a numeric percent", () => {
    const [point] = parseMomentumRows([row("2024-01-02", "4.6912")]);
    expect(point.roc).toBeCloseTo(4.6912, 6);
  });

  it("plots dates as UTC epoch milliseconds, so the axis is timezone-stable", () => {
    const [point] = parseMomentumRows([row("2024-01-02", "3.1")]);
    expect(point.t).toBe(Date.UTC(2024, 0, 2));
  });

  it("keeps a negative reading rather than clamping it to zero", () => {
    const [point] = parseMomentumRows([row("2024-01-02", "-12.5")]);
    expect(point.roc).toBeCloseTo(-12.5, 6);
  });

  it("drops a session with a blank value rather than charting it as zero", () => {
    const points = parseMomentumRows([row("2024-01-02", ""), row("2024-01-03", "3.1")]);
    expect(points.map((p) => p.date)).toEqual(["2024-01-03"]);
  });

  it("drops a row whose date cannot be parsed", () => {
    expect(parseMomentumRows([row("not-a-date", "3.1")])).toEqual([]);
  });
});

describe("yearTicks", () => {
  it("returns the first session of each calendar year", () => {
    const points = parseMomentumRows([
      row("2023-01-04", "3"),
      row("2023-06-01", "4"),
      row("2024-01-03", "-2"),
      row("2024-02-01", "-1"),
    ]);
    expect(yearTicks(points)).toEqual([Date.UTC(2023, 0, 4), Date.UTC(2024, 0, 3)]);
  });

  it("has no ticks for an empty series", () => {
    expect(yearTicks([])).toEqual([]);
  });
});

describe("rocTicks", () => {
  it("spans the five-point multiples around the range and always crosses zero", () => {
    const points = parseMomentumRows([row("2024-01-02", "3.2"), row("2024-01-03", "12.1")]);
    expect(rocTicks(points)).toEqual([0, 5, 10, 15]);
  });

  it("extends below zero when the index was down on the quarter", () => {
    const points = parseMomentumRows([row("2024-01-02", "-16.4"), row("2024-01-03", "2.1")]);
    expect(rocTicks(points)).toEqual([-20, -15, -10, -5, 0, 5]);
  });

  it("falls back to a minimal axis for an empty series", () => {
    expect(rocTicks([])).toEqual([-5, 0, 5]);
  });
});
