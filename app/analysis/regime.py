"""
Market-regime signals derived from the quarterly 13F panel.

Five measures, all computed from position counts and share counts rather than
reported dollar values, because a bear market moves every dollar figure on its
own: what a fund *did* has to be separated from what the market *did to it*.

Three hazards in the raw filings shape everything here:

  * Stock splits are stored as filed, so a 20:1 split reads as a +1900% purchase
    unless it is detected and neutralised (`detect_splits`).
  * A split landing near quarter-end leaves funds filing on different share bases
    inside the SAME quarter (`harmonise_basis`).
  * Occasional filer denomination errors put a fund's whole book out by three
    orders of magnitude (`drop_anomalous_funds`).

Change metrics run on the funds common to both quarters, so a growing tracked
universe never registers as trading. Level metrics (sector and mega-cap weights)
are shares of that quarter's own reported value.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from app.database import (
    DB_FOLDER,
    SECTOR_HIERARCHY_FILE,
    get_all_quarters,
    load_quarterly_data,
)
from app.database.stocks import load_stocks
from app.utils.logger import get_logger, log_safe
from app.utils.strings import get_numeric

logger = get_logger(__name__)

# Fixed basket, so the series stays comparable across quarters. These are the
# names whose crowding the chart is about; a basket that re-picked itself each
# quarter would measure the pick, not the positioning.
MEGA_CAP_BASKET = ("MSFT", "AMZN", "NVDA", "META", "GOOGL", "AAPL", "TSLA", "AMD")

UNCLASSIFIED = "Unclassified"

# A few broad Yahoo sectors trade as several unrelated themes at once - a fund
# rotating from software into semiconductors, or from big-cap pharma into
# biotech, is invisible at the "Technology" or "Healthcare" level. The regime
# chart rolls those industries up to named subsectors instead. Every industry
# not listed here keeps its parent sector label.
SUBSECTOR_BY_INDUSTRY = {
    # Technology
    "Semiconductors": "Semiconductors",
    "Semiconductor Equipment & Materials": "Semiconductors",
    "Software - Application": "Software",
    "Software - Infrastructure": "Software",
    "Information Technology Services": "IT Services",
    "Communication Equipment": "Tech Hardware",
    "Computer Hardware": "Tech Hardware",
    "Consumer Electronics": "Tech Hardware",
    "Electronic Components": "Tech Hardware",
    "Electronics & Computer Distribution": "Tech Hardware",
    "Scientific & Technical Instruments": "Tech Hardware",
    "Solar": "Tech Hardware",
    # Healthcare
    "Biotechnology": "Biotech",
    "Drug Manufacturers - General": "Pharmaceuticals",
    "Drug Manufacturers - Specialty & Generic": "Pharmaceuticals",
    "Medical Devices": "Medical Devices",
    "Medical Instruments & Supplies": "Medical Devices",
    "Diagnostics & Research": "Medical Devices",
    "Health Information Services": "Healthcare Services",
    "Healthcare Plans": "Healthcare Services",
    "Medical Care Facilities": "Healthcare Services",
    "Medical Distribution": "Healthcare Services",
    "Pharmaceutical Retailers": "Healthcare Services",
    # Communication Services
    "Internet Content & Information": "Interactive Media",
    "Advertising Agencies": "Media & Entertainment",
    "Broadcasting": "Media & Entertainment",
    "Electronic Gaming & Multimedia": "Media & Entertainment",
    "Entertainment": "Media & Entertainment",
    "Publishing": "Media & Entertainment",
    "Telecom Services": "Telecom",
    # Consumer Cyclical
    "Internet Retail": "Internet Retail",
    "Auto & Truck Dealerships": "Automotive",
    "Auto Manufacturers": "Automotive",
    "Auto Parts": "Automotive",
    # Financial Services
    "Banks - Diversified": "Banks",
    "Banks - Regional": "Banks",
    "Asset Management": "Capital Markets",
    "Capital Markets": "Capital Markets",
    "Financial Data & Stock Exchanges": "Capital Markets",
    "Insurance - Diversified": "Insurance",
    "Insurance - Life": "Insurance",
    "Insurance - Property & Casualty": "Insurance",
    "Insurance - Reinsurance": "Insurance",
    "Insurance - Specialty": "Insurance",
    "Insurance Brokers": "Insurance",
    # Industrials
    "Aerospace & Defense": "Aerospace & Defense",
}


def regime_group(industry: str, sector: str) -> str:
    """
    The subsector bucket an industry rolls up to on the regime chart.

    A few broad Yahoo sectors trade as several distinct themes, so their
    industries map to named subsectors (Semiconductors, Software, Biotech, and
    so on). Every other industry keeps its parent sector label.
    """
    return SUBSECTOR_BY_INDUSTRY.get(industry, sector)


# Plausible split factors, plus their reciprocals for reverse splits.
_FACTORS = np.array([2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50], dtype=float)
_CANDIDATES = np.concatenate([_FACTORS, 1 / _FACTORS])

_MIN_PAIRED_HOLDERS = 3
_SPLIT_DEADBAND = (0.72, 1.38)
_FACTOR_TOLERANCE = 1.15
_BASIS_TOLERANCE = 1.10
# pr / k is the stock's true move over the quarter once the split is removed;
# outside this band the share change is trading, not a split.
_TRUE_MOVE_BAND = (0.70, 2.5)
# A book priced this far from consensus for the same securities is misdenominated,
# not merely unusual. Well clear of the 50x ceiling on split-basis differences.
_DENOMINATION_HIGH = 100.0
_DENOMINATION_LOW = 0.01
_MIN_COMPARABLE_POSITIONS = 3


def _to_numeric(series: pd.Series) -> pd.Series:
    """
    Parse a formatted value column ('1.33B') into floats, coercing failures to NaN.
    """

    def parse(raw: object) -> float:
        if not isinstance(raw, str) or not raw.strip():
            return float("nan")
        try:
            return get_numeric(raw.strip())
        except (ValueError, IndexError):
            return float("nan")

    return series.map(parse)


def snap_to_split_factor(ratio: float) -> float | None:
    """
    Nearest plausible split factor to a share ratio, or None if nothing is close.
    """
    factor = float(_CANDIDATES[np.argmin(np.abs(np.log(_CANDIDATES / ratio)))])
    if abs(np.log(factor / ratio)) > np.log(_FACTOR_TOLERANCE):
        return None
    return factor


def harmonise_basis(positions: pd.DataFrame) -> pd.DataFrame:
    """
    Put every fund on the same share basis for a given name within one quarter.

    A split landing near quarter-end (Shopify's 10:1 fell on 29 June 2022) leaves
    some funds filing pre-split and others post-split in the same quarter, so no
    single per-name factor can fix it. Implied price is a market price, so a
    holder whose price is a clean split multiple of the consensus is on the other
    basis: rescale its share count and price, leaving position value untouched.
    """
    if positions.empty:
        return positions
    out = positions.copy()
    consensus = out.groupby("CUSIP")["Px"].transform("median")
    holders = out.groupby("CUSIP")["Px"].transform("size")
    ratio = out["Px"] / consensus

    factor = pd.Series(1.0, index=out.index)
    testable = (holders >= _MIN_PAIRED_HOLDERS) & consensus.gt(0) & ratio.gt(0) & np.isfinite(ratio)
    if testable.any():
        values = np.asarray(ratio[testable], dtype=float)
        nearest = _CANDIDATES[
            np.argmin(np.abs(np.log(values[:, None] / _CANDIDATES[None, :])), axis=1)
        ]
        on_factor = np.abs(np.log(values / nearest)) < np.log(_BASIS_TOLERANCE)
        factor.loc[testable] = np.where(on_factor, nearest, 1.0)

    rescaled = int((factor != 1.0).sum())
    if rescaled:
        logger.debug("Rescaled %d rows onto the consensus share basis", rescaled)
    out["SharesN"] = out["SharesN"] * factor
    out["Px"] = out["Px"] / factor
    return out


def detect_splits(previous: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    """
    Split factors by CUSIP between two consecutive quarters.

    Ratios are computed per fund across holders present in both quarters -
    comparing cross-fund medians instead would move with the holder set rather
    than with the split. A candidate needs a cluster of holders sitting on the
    same factor and, decisively, a commensurate move in implied price: a split
    divides the price as it multiplies the shares, whereas accumulation raises
    the share count at an unchanged price.
    """
    if previous.empty or current.empty:
        return {}
    columns = ["Fund", "CUSIP", "SharesN", "Px"]
    paired = previous[columns].merge(current[columns], on=["Fund", "CUSIP"], suffixes=("_p", "_c"))
    paired = paired[
        (paired["SharesN_p"] > 0)
        & (paired["SharesN_c"] > 0)
        & (paired["Px_p"] > 0)
        & (paired["Px_c"] > 0)
    ]
    if paired.empty:
        return {}
    paired["share_ratio"] = paired["SharesN_c"] / paired["SharesN_p"]
    paired["price_ratio"] = paired["Px_p"] / paired["Px_c"]

    splits: dict[str, float] = {}
    for cusip, group in paired.groupby("CUSIP"):
        if len(group) < _MIN_PAIRED_HOLDERS:
            continue
        share_ratio = float(group["share_ratio"].median())
        if _SPLIT_DEADBAND[0] < share_ratio < _SPLIT_DEADBAND[1]:
            continue
        factor = snap_to_split_factor(share_ratio)
        if factor is None:
            continue
        on_factor = np.abs(np.log(group["share_ratio"] / factor)) < np.log(1.12)
        if int(on_factor.sum()) < 2:
            continue
        price_ratio = float(group["price_ratio"].median())
        true_move = price_ratio / factor
        if not _TRUE_MOVE_BAND[0] < true_move < _TRUE_MOVE_BAND[1]:
            continue
        # Implied price is a market price, so holders must broadly agree on it.
        agreement = float(
            (np.abs(np.log(group["price_ratio"] / price_ratio)) < np.log(1.10)).mean()
        )
        if agreement < 0.6:
            continue
        splits[str(cusip)] = factor
    if splits:
        logger.debug("Detected %d split(s)", len(splits))
    return splits


def drop_anomalous_funds(panels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Remove fund-quarters whose whole book is filed in the wrong denomination.

    Filer errors happen: one fund reported $183B and $72B against a $0.66B book,
    which alone manufactures a $220B outflow and a phantom sector. The test is
    the fund's own pricing of securities other funds also hold - a book priced
    three orders of magnitude away from the consensus for the same CUSIPs is
    misdenominated. Comparing a fund against its own AUM history instead would
    flag four years of ordinary growth, and comparing across funds would flag
    the largest ones.
    """
    cleaned: dict[str, pd.DataFrame] = {}
    for quarter, frame in panels.items():
        if frame.empty:
            cleaned[quarter] = frame
            continue
        holders = frame.groupby("CUSIP")["Px"].transform("size")
        consensus = frame.groupby("CUSIP")["Px"].transform("median")
        comparable = frame[(holders >= _MIN_PAIRED_HOLDERS) & consensus.gt(0)]
        if comparable.empty:
            cleaned[quarter] = frame
            continue
        ratio = comparable["Px"] / consensus[comparable.index]
        per_fund = ratio.groupby(comparable["Fund"]).agg(["median", "size"])
        judged = per_fund[per_fund["size"] >= _MIN_COMPARABLE_POSITIONS]
        bad = set(
            judged.index[
                (judged["median"] > _DENOMINATION_HIGH) | (judged["median"] < _DENOMINATION_LOW)
            ]
        )
        if bad:
            logger.warning(
                "Dropping %d misdenominated fund-quarter(s) in %s: %s",
                len(bad),
                log_safe(quarter),
                log_safe(", ".join(sorted(str(name) for name in bad))),
            )
        cleaned[quarter] = frame[~frame["Fund"].isin(bad)].copy()
    return cleaned


def transition_metrics(
    previous: pd.DataFrame, current: pd.DataFrame, splits: dict[str, float]
) -> dict[str, float]:
    """
    Position churn and capital flow between two consecutive quarters.

    Restricted to funds present in both quarters, so roster growth never reads as
    trading. Prior share counts are scaled by any split factor first, and exits
    are valued at the quarter's consensus price for the name so a closed position
    is measured on the same basis as a trimmed one.
    """
    common = set(previous["Fund"]) & set(current["Fund"])
    prev = previous[previous["Fund"].isin(common)]
    curr = current[current["Fund"].isin(common)]
    if prev.empty or curr.empty:
        return {}

    before = prev[["Fund", "CUSIP", "SharesN", "ValueN"]].rename(
        columns={"SharesN": "shares_before", "ValueN": "value_before"}
    )
    before["shares_before"] = before["shares_before"] * before["CUSIP"].map(splits).fillna(1.0)
    after = curr[["Fund", "CUSIP", "SharesN", "ValueN", "Px", "Ticker"]].rename(
        columns={"SharesN": "shares_after", "ValueN": "value_after"}
    )
    joined = before.merge(after, on=["Fund", "CUSIP"], how="outer")
    joined[["shares_before", "shares_after", "value_before", "value_after"]] = joined[
        ["shares_before", "shares_after", "value_before", "value_after"]
    ].fillna(0)

    consensus = curr.groupby("CUSIP")["Px"].median()
    fallback = joined["value_before"] / joined["shares_before"].replace(0, np.nan)
    joined["Px"] = joined["Px"].fillna(joined["CUSIP"].map(consensus)).fillna(fallback)

    delta = joined["shares_after"] - joined["shares_before"]
    flow = delta * joined["Px"]
    opened = joined["shares_before"] == 0
    closed = joined["shares_after"] == 0

    new_count = int(opened.sum())
    closed_count = int(closed.sum())
    new_capital = float(joined.loc[opened, "value_after"].sum())
    released = float(joined.loc[closed, "value_before"].sum())
    gross_buy = float(flow[flow > 0].sum())
    gross_sell = float(-flow[flow < 0].sum())
    book = float(curr["ValueN"].sum())

    return {
        "new_positions": new_count,
        "closed_positions": closed_count,
        "new_close": new_count / closed_count if closed_count else float("nan"),
        "untouched_pct": float(100 * (delta == 0).mean()),
        "recycle": new_capital / released if released else float("nan"),
        "gross_buy": gross_buy,
        "gross_sell": gross_sell,
        "turnover_pct": 100 * (gross_buy + gross_sell) / book if book else float("nan"),
        "net_buyer_pct": float(
            100 * (joined.assign(flow=flow).groupby("Fund")["flow"].sum() > 0).mean()
        ),
        "mega_share_growth": _mega_share_growth(joined, delta),
    }


def _mega_share_growth(joined: pd.DataFrame, delta: pd.Series) -> float:
    """
    Growth in mega-cap shares held between two quarters, value-weighted.

    Weighted by each name's prior value so the eight names combine on a common
    footing rather than by raw share count, which would let the cheapest stock
    dominate. Returns 1.0 when the basket is absent.
    """
    basket = joined["Ticker"].isin(MEGA_CAP_BASKET)
    if not basket.any():
        return 1.0
    rows = joined[basket]
    weights = rows["value_before"]
    before = rows["shares_before"]
    if weights.sum() <= 0 or before.sum() <= 0:
        return 1.0
    # Per-name growth, weighted by prior value: sum(w * after/before) / sum(w).
    ratio = (rows["shares_before"] + delta[basket]) / before.replace(0, np.nan)
    usable = ratio.notna() & (weights > 0)
    if not usable.any():
        return 1.0
    return float((ratio[usable] * weights[usable]).sum() / weights[usable].sum())


def quarter_levels(positions: pd.DataFrame) -> dict:
    """
    Composition of the tracked book at one quarter-end.

    Sector weights are shares of *classified* value - unclassified value is
    excluded rather than bucketed into a residual, so the percentages describe
    only what is actually known.
    """
    if positions.empty:
        return {"sectors": {}, "mega_weight_pct": float("nan"), "positions": 0, "aum": 0.0}
    total = float(positions["ValueN"].sum())
    classified = positions[positions["Sector"] != UNCLASSIFIED]
    classified_total = float(classified["ValueN"].sum())
    sectors = (
        (100 * classified.groupby("Sector")["ValueN"].sum() / classified_total).to_dict()
        if classified_total > 0
        else {}
    )
    mega = positions[positions["Ticker"].isin(MEGA_CAP_BASKET)]
    return {
        "sectors": {str(k): float(v) for k, v in sectors.items()},
        "mega_weight_pct": 100 * float(mega["ValueN"].sum()) / total if total else float("nan"),
        "positions": int(len(positions)),
        "aum": total,
        "funds": int(positions["Fund"].nunique()),
        "median_positions": float(positions.groupby("Fund").size().median()),
        "classified_pct": 100 * classified_total / total if total else float("nan"),
    }


def build_regime_rows(panels: dict[str, pd.DataFrame]) -> list[dict]:
    """
    Long-format rows describing every quarter: metrics, mega-cap and sectors.

    The mega-cap share index is chain-linked - each quarter multiplies the
    previous index by growth measured on funds common to both quarters - so a
    fund joining the tracked universe cannot move it.
    """
    quarters = sorted(panels)
    rows: list[dict] = []
    share_index = 100.0

    for position, quarter in enumerate(quarters):
        frame = panels[quarter]
        levels = quarter_levels(frame)

        if position:
            previous = panels[quarters[position - 1]]
            splits = detect_splits(previous, frame)
            metrics = transition_metrics(previous, frame, splits)
            share_index *= metrics.get("mega_share_growth", 1.0)
            for key, value in metrics.items():
                if key == "mega_share_growth" or value is None or not np.isfinite(value):
                    continue
                rows.append(
                    {"kind": "metric", "quarter": quarter, "key": key, "value": round(value, 6)}
                )

        for key in ("positions", "aum", "funds", "median_positions", "classified_pct"):
            value = levels.get(key)
            if value is not None and np.isfinite(value):
                rows.append(
                    {"kind": "metric", "quarter": quarter, "key": key, "value": round(value, 6)}
                )

        rows.append(
            {
                "kind": "mega",
                "quarter": quarter,
                "key": "share_index",
                "value": round(share_index, 6),
            }
        )
        if np.isfinite(levels["mega_weight_pct"]):
            rows.append(
                {
                    "kind": "mega",
                    "quarter": quarter,
                    "key": "weight_pct",
                    "value": round(levels["mega_weight_pct"], 6),
                }
            )
        for sector, weight in sorted(levels["sectors"].items()):
            rows.append(
                {"kind": "sector", "quarter": quarter, "key": sector, "value": round(weight, 6)}
            )
    return rows


def load_panel(quarter: str, sectors: dict[str, str]) -> pd.DataFrame:
    """
    One quarter's positions with numeric columns, sectors and a harmonised basis.
    """
    frame = load_quarterly_data(quarter)
    if frame.empty:
        return pd.DataFrame(
            columns=["Fund", "CUSIP", "Ticker", "SharesN", "ValueN", "Px", "Sector"]
        )
    frame = frame.copy()
    frame["ValueN"] = _to_numeric(frame["Value"])
    frame["SharesN"] = pd.to_numeric(frame["Shares"], errors="coerce")
    frame = frame[(frame["SharesN"] > 0) & frame["ValueN"].gt(0)]
    frame["Px"] = frame["ValueN"] / frame["SharesN"]
    frame["Sector"] = frame["CUSIP"].map(sectors).fillna(UNCLASSIFIED)
    frame["Ticker"] = frame["Ticker"].fillna("")
    return harmonise_basis(
        frame[["Fund", "CUSIP", "Ticker", "SharesN", "ValueN", "Px", "Sector"]].reset_index(
            drop=True
        )
    )


def _sector_map() -> dict[str, str]:
    """
    CUSIP to regime group, resolved through stocks.csv industries and the
    hierarchy, then rolled up to a named subsector where one applies.
    """
    stocks = load_stocks()
    if stocks.empty:
        return {}
    hierarchy = pd.read_csv(Path(DB_FOLDER) / SECTOR_HIERARCHY_FILE, dtype=str)
    industry_to_sector = dict(zip(hierarchy["Industry"], hierarchy["Sector"], strict=False))
    # load_stocks() indexes by CUSIP rather than carrying it as a column.
    return {
        str(cusip): regime_group(industry, industry_to_sector[industry])
        for cusip, industry in zip(stocks.index, stocks["Industry"], strict=False)
        if industry in industry_to_sector
    }


def compute_regime_signals(quarters: list[str] | None = None) -> list[dict]:
    """
    Build the full regime-signal table from the quarterly database.
    """
    selected = sorted(quarters or get_all_quarters())
    sectors = _sector_map()
    panels = {quarter: load_panel(quarter, sectors) for quarter in selected}
    panels = {quarter: frame for quarter, frame in panels.items() if not frame.empty}
    logger.progress("Loaded %d quarter(s) of positions", len(panels))
    panels = drop_anomalous_funds(panels)
    return build_regime_rows(panels)
