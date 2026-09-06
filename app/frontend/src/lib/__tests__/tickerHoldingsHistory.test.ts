import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { getTickerHoldingsHistory, clearCache } from "../dataService";

const MOCK_STOCKS_CSV = `CUSIP,Ticker,Company,Industry
A123456,TSLA,Tesla Inc,Auto Manufacturers`;

const MOCK_SECTOR_HIERARCHY_CSV = `"Sector","Industry"
"Consumer Cyclical","Auto Manufacturers"
`;

// The latest quarter's view merges 13D/G + Form 4 activity; an empty file keeps
// that path exercised without adding non-quarterly rows to the assertions.
const MOCK_NON_QUARTERLY_CSV = `Fund,CUSIP,Ticker,Company,Shares,Value,Avg_Price,Date,Filing_Date`;

// 2025Q1: Fund A holds 1000, Fund B holds 500 → 1500 shares / $3.5M / 2 holders.
// 2025Q2: Fund A holds 2000, Fund B closed out → 2000 shares / $5M / 1 holder.
const FUND_CSVS: Record<string, string> = {
  "2025Q1/fund_A": `CUSIP,Ticker,Company,Shares,Delta_Shares,Value,Delta_Value,Delta,Portfolio%
A123456,TSLA,Tesla Inc,1000,1000,"$2.5M","$2.5M",NEW,5.0`,
  "2025Q1/fund_B": `CUSIP,Ticker,Company,Shares,Delta_Shares,Value,Delta_Value,Delta,Portfolio%
A123456,TSLA,Tesla Inc,500,500,"$1M","$1M",NEW,3.0`,
  "2025Q2/fund_A": `CUSIP,Ticker,Company,Shares,Delta_Shares,Value,Delta_Value,Delta,Portfolio%
A123456,TSLA,Tesla Inc,2000,1000,"$5M","$2.5M",100.0,8.0`,
  "2025Q2/fund_B": `CUSIP,Ticker,Company,Shares,Delta_Shares,Value,Delta_Value,Delta,Portfolio%
A123456,TSLA,Tesla Inc,0,-500,"$0","-$1M",-100.0,0`,
};

function mockFetch(url: string): Promise<Response> {
  const urlStr = typeof url === "string" ? url : "";
  const csv = (body: string) =>
    Promise.resolve(new Response(body, { headers: { "Content-Type": "text/csv" } }));
  const json = (body: unknown) =>
    Promise.resolve(
      new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json" } }),
    );

  if (urlStr.includes("stocks.csv")) return csv(MOCK_STOCKS_CSV);
  if (urlStr.includes("sector_hierarchy.csv")) return csv(MOCK_SECTOR_HIERARCHY_CSV);
  if (urlStr.includes("non_quarterly.csv")) return csv(MOCK_NON_QUARTERLY_CSV);

  for (const [key, body] of Object.entries(FUND_CSVS)) {
    const [quarter, fund] = key.split("/");
    if (urlStr.includes(`${quarter}/${fund}.csv`)) return csv(body);
  }
  if (urlStr.endsWith("/api/database/quarters")) return json(["2025Q1", "2025Q2"]);
  if (urlStr.includes("/api/database/quarters/")) return json(["fund_A.csv", "fund_B.csv"]);

  return Promise.reject(new Error(`Unmocked URL: ${urlStr}`));
}

describe("getTickerHoldingsHistory", () => {
  beforeEach(() => {
    clearCache();
    vi.stubGlobal("fetch", mockFetch);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns one chronological point per available quarter", async () => {
    const history = await getTickerHoldingsHistory("TSLA");

    expect(history.map((p) => p.quarter)).toEqual(["2025Q1", "2025Q2"]);
    expect(history.map((p) => p.asOf)).toEqual(["2025-03-31", "2025-06-30"]);
  });

  it("sums shares held across every fund in the quarter", async () => {
    const history = await getTickerHoldingsHistory("TSLA");

    expect(history[0].totalShares).toBe(1500);
    expect(history[1].totalShares).toBe(2000);
  });

  it("sums position value across every fund in the quarter", async () => {
    const history = await getTickerHoldingsHistory("TSLA");

    expect(history[0].totalValue).toBeCloseTo(3_500_000, -3);
    expect(history[1].totalValue).toBeCloseTo(5_000_000, -3);
  });

  it("counts only funds still holding shares", async () => {
    const history = await getTickerHoldingsHistory("TSLA");

    expect(history[0].holderCount).toBe(2);
    expect(history[1].holderCount).toBe(1);
  });

  it("omits quarters where no tracked fund held the ticker", async () => {
    const history = await getTickerHoldingsHistory("AAPL");

    expect(history).toEqual([]);
  });
});
