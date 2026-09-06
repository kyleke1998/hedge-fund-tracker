/**
 * Data-layer barrel: the single import surface for CSV/API reads and the
 * client-side analysis pipeline. Implementation lives in `./data/*` (types,
 * fetch/cache plumbing, per-domain loaders, analysis) — mirroring the Python
 * `app/database` package split — while every consumer keeps importing from
 * `@/lib/dataService`.
 */

export type {
  AIModel,
  EnrichedNQFiling,
  ExcludedHedgeFund,
  FundQuarterSnapshot,
  FundTickerHolding,
  HedgeFund,
  NonQuarterlyFiling,
  NumericStockKey,
  PerformanceData,
  PerfSeries,
  PerfWindow,
  QuarterlyHolding,
  RawHedgeFund,
  RawModel,
  RawNonQuarterly,
  RawPerformanceRow,
  RawQuarterlyHolding,
  RawSectorHierarchy,
  RawStock,
  SectorHierarchyEntry,
  Stock,
  StockQuarterAnalysis,
  TickerHoldingsPoint,
} from "./data/types";

export type { Quarter } from "./quarters";

export { clearCache, downloadFile, saveFileToDisk } from "./data/fetch";

export { formatPct, formatValue, formatValueShort, parseValueString } from "./data/format";

export {
  generateAddFundCSV,
  generateDeleteFundCSVs,
  generateExcludedFundsCSV,
  generateHedgeFundsCSV,
  generateRestoreFundCSVs,
  getExcludedHedgeFunds,
  getHedgeFunds,
} from "./data/funds";

export { getSectorHierarchy, getStocks } from "./data/stocks";

export {
  generateModelsCSV,
  getModels,
  MODEL_PROVIDERS,
  PROVIDER_DISPLAY_NAMES,
  type ModelProvider,
} from "./data/models";

export {
  aggregateHoldingsByTicker,
  getAvailableQuarters,
  getFundAvailableQuarters,
  getFundQuarterlyHoldings,
  getLatestQuarter,
  getQuarterFundList,
} from "./data/quarterData";

export { getPerformance, parsePerformanceRows } from "./data/performance";
export { getRegime, parseRegimeRows, reindexShareIndex, sectorWindowChange } from "./data/regime";

export { enrichNQFiling, getEnrichedNQFilings, getNonQuarterlyFilings } from "./data/nonQuarterly";

export {
  aggregateStockLevel,
  fetchQuarterAnalysis,
  getTickerHoldingsHistory,
  mergeNonQuarterlyHoldings,
  runFundAnalysis,
  runQuarterAnalysis,
  runStockAnalysis,
} from "./data/analysis";
