"""
Build and evaluate price / institutional-accumulation divergence signals.

Reads the cached bars written by ``scripts/fetch_divergence_prices.py``, assembles
one point-in-time observation per (quarter, ticker), scores each one, and writes
the evidence to ``__reports__/divergence/``: per-feature information coefficients,
quantile monotonicity tables, portfolio track records with win rates and
drawdowns, and the live screen for the most recent quarter.

Everything is evaluated as of the entry date — quarter-end plus the 13F filing
lag — so no observation uses a number that was not public when the trade is
assumed to happen.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.backtest.divergence import (  # noqa: E402
    ACCUMULATION_COLUMNS,
    PRICE_STRENGTH_COLUMNS,
    accumulation_features,
    buy_gate,
    divergence_score,
    hit_rate,
    max_drawdown,
    price_features,
    rank_ic,
    rank_z,
    sell_gate,
    summarize_windows,
)
from app.backtest.divergence_cache import CLOSE_FILE, SPLITS_FILE, VOLUME_FILE  # noqa: E402
from app.backtest.engine import quarter_entry_date, quarter_smart_scores  # noqa: E402
from app.backtest.holdings_panel import apply_split_adjustment, build_holdings_panel  # noqa: E402
from app.backtest.study import (  # noqa: E402
    daily_equity,
    double_sort_table,
    forward_returns,
    neutralize,
    quantile_table,
)
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

OUTPUT_DIR = Path("__reports__") / "divergence"

# Tradability filters. A signal that only fires on illiquid sub-$3 names is not a
# signal, it is a bid-ask spread.
MIN_COMMON_HOLDERS = 5
MIN_DOLLAR_VOLUME = 2_000_000.0
MIN_PRICE = 3.0

LOOKBACK_QUARTERS = 4
TOP_N = 30
BENCHMARK = "SPY"

# Holding horizons in quarters. A quiet-accumulation thesis is supposed to need
# time, so testing it at one quarter alone would be a strawman; the longer
# windows overlap, which inflates their t-statistics and is flagged in the report.
HORIZONS = (1, 2, 4)

# Features whose standalone predictive content is reported alongside the composite.
FEATURE_COLUMNS = [
    *ACCUMULATION_COLUMNS,
    *PRICE_STRENGTH_COLUMNS,
    "Share_Chg_Pct",
    f"Share_Chg_Pct_{LOOKBACK_QUARTERS}q",
    "Breadth",
    "Holder_Chg",
    "Avg_Portfolio_Pct",
    "Ret_1q",
    "Ret_12m",
    "Drawdown_52w",
    "Vol_63",
    "Accumulation_Z",
    "Price_Strength_Z",
    "Divergence",
    "Divergence_Gated",
    "Confirmation",
    "Smart_Score",
]

# Scores whose quarter-by-quarter IC is written out, so a mean IC can be checked
# for stability instead of being taken on trust.
STABILITY_SCORES = [
    "Divergence",
    "Accumulation_Z",
    "Price_Strength_Z",
    "Avg_Portfolio_Pct",
    "Smart_Score",
]


def load_cache() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the cached close / volume / split files, failing loudly if absent.
    """
    if not CLOSE_FILE.exists():
        raise SystemExit("Price cache missing — run scripts/fetch_divergence_prices.py first.")
    close = pd.read_parquet(CLOSE_FILE)
    volume = pd.read_parquet(VOLUME_FILE)
    splits = pd.read_csv(SPLITS_FILE, dtype={"ticker": str, "date": str})
    return close.sort_index(), volume.sort_index(), splits


def sector_map() -> pd.Series:
    """
    Ticker -> Yahoo sector, derived by joining stocks.csv's Industry against the
    sector hierarchy (the project stores Industry and derives Sector at read time).
    """
    from app.database import load_sector_hierarchy, load_stocks

    stocks = load_stocks().reset_index()
    hierarchy = load_sector_hierarchy()
    merged = stocks.merge(hierarchy, on="Industry", how="left")
    return merged.dropna(subset=["Ticker"]).drop_duplicates("Ticker").set_index("Ticker")["Sector"]


def build_observations(
    panel: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, today: date
) -> pd.DataFrame:
    """
    One row per (quarter, ticker): features known at the entry date plus the
    return realised over the following holding window.

    The final quarter has no successor, so its rows carry features and no forward
    return — that is the live screen rather than a backtest observation.
    """
    quarters = sorted(panel["Quarter"].unique())
    sectors = sector_map()
    frames = []

    for position, quarter in enumerate(quarters):
        if position == 0:
            continue
        entry = quarter_entry_date(quarter)
        if entry > today:
            continue

        features = accumulation_features(panel, quarter, lookback=LOOKBACK_QUARTERS)
        if features.empty:
            continue
        prices = price_features(close, volume, entry)
        merged = features.merge(prices, on="Ticker", how="inner")

        current = panel[panel["Quarter"] == quarter]
        conviction = (
            current.groupby("Ticker")
            .agg(
                Avg_Portfolio_Pct=("Portfolio_Pct", "mean"),
                Max_Portfolio_Pct=("Portfolio_Pct", "max"),
                Total_Value=("Value", "sum"),
                Holder_Count_All=("Fund", "nunique"),
            )
            .reset_index()
        )
        previous = panel[panel["Quarter"] == quarters[position - 1]]
        prior_holders = previous.groupby("Ticker")["Fund"].nunique().rename("Prior_Holders")
        merged = merged.merge(conviction, on="Ticker", how="left").merge(
            prior_holders, on="Ticker", how="left"
        )
        merged["Prior_Holders"] = merged["Prior_Holders"].fillna(0)
        merged["Holder_Chg"] = merged["Holder_Count_All"] - merged["Prior_Holders"]

        # Dollar flow the tracked funds pushed through the tape, in days of median
        # trading volume: the same percentage add is a far stronger tell in a name
        # that barely trades than in a mega-cap.
        merged["Acc_Days_Of_Volume"] = (
            (merged["Curr_Shares"] - merged["Prev_Shares"]) * merged["Px_Entry"]
        ) / merged["Dollar_Vol_63"].replace(0.0, np.nan)

        merged = merged[
            (merged["Holders"] >= MIN_COMMON_HOLDERS)
            & (merged["Dollar_Vol_63"] >= MIN_DOLLAR_VOLUME)
            & (merged["Px_Entry"] >= MIN_PRICE)
            & merged["Ret_6m"].notna()
            & merged["Trend_Tstat"].notna()
        ].copy()
        if merged.empty:
            continue

        merged["Accumulation_Z"] = pd.concat(
            [rank_z(merged[c]) for c in ACCUMULATION_COLUMNS], axis=1
        ).mean(axis=1)
        merged["Price_Strength_Z"] = pd.concat(
            [rank_z(merged[c]) for c in PRICE_STRENGTH_COLUMNS], axis=1
        ).mean(axis=1)
        merged["Divergence"] = divergence_score(merged)
        # The inverse reading of the same two legs: funds adding to a name the
        # market is already marking up. Tested alongside the thesis because a
        # divergence score that predicts negatively is, mechanically, a claim that
        # its opposite predicts positively — and that claim deserves its own test.
        merged["Confirmation"] = merged["Accumulation_Z"] + merged["Price_Strength_Z"]
        # The project's published smart score, joined rather than recomputed: its
        # components are percentiles over the quarter's whole universe, so scoring
        # this study's liquidity-filtered frame would not be the same number.
        merged["Smart_Score"] = merged["Ticker"].map(quarter_smart_scores(quarter))
        merged["Buy_Gate"] = buy_gate(merged, min_holders=MIN_COMMON_HOLDERS)
        merged["Sell_Gate"] = sell_gate(merged, min_holders=MIN_COMMON_HOLDERS)
        # The gated score keeps the ranking but only inside the names that pass the
        # literal thesis, so the two readings can be compared on one axis.
        merged["Divergence_Gated"] = merged["Divergence"].where(merged["Buy_Gate"])

        merged.insert(0, "Quarter", quarter)
        merged["Entry_Date"] = entry.isoformat()
        merged["Sector"] = merged["Ticker"].map(sectors)

        for horizon in HORIZONS:
            target = position + horizon
            if target >= len(quarters):
                continue
            exit_date = quarter_entry_date(quarters[target])
            if exit_date > today:
                continue
            realised = forward_returns(close, entry, exit_date)
            suffix = "" if horizon == 1 else f"_{horizon}q"
            merged[f"Exit_Date{suffix}"] = exit_date.isoformat()
            merged[f"Fwd_Return{suffix}"] = merged["Ticker"].map(realised)
            bench = realised.get(BENCHMARK, np.nan)
            merged[f"Bench_Return{suffix}"] = bench
            merged[f"Fwd_Excess{suffix}"] = merged[f"Fwd_Return{suffix}"] - bench
            # Demeaning inside the quarter removes the market move, so the score is
            # judged purely on picking winners out of the same opportunity set.
            merged[f"Fwd_Demeaned{suffix}"] = (
                merged[f"Fwd_Return{suffix}"] - merged[f"Fwd_Return{suffix}"].mean()
            )

        frames.append(merged)

    return pd.concat(frames, ignore_index=True)


def information_coefficients(observations: pd.DataFrame) -> pd.DataFrame:
    """
    Per-feature rank IC against forward returns, by holding horizon.

    This is the properly powered test of the thesis: it uses every stock-quarter
    rather than the handful of portfolio windows the sample allows. Three outcome
    definitions are reported side by side — the raw return, the return demeaned
    within the quarter (which strips out whether the market rose), and the return
    demeaned within quarter *and* sector (which strips out sector bets too), so a
    signal that is really a market or sector tilt shows up as one that survives
    the first column and dies in the third.
    """
    rows = []
    for horizon in HORIZONS:
        suffix = "" if horizon == 1 else f"_{horizon}q"
        outcome = f"Fwd_Return{suffix}"
        if outcome not in observations.columns:
            continue
        scored = observations[observations[outcome].notna()].copy()
        scored[f"Fwd_Sector_Neutral{suffix}"] = neutralize(
            scored, [f"Fwd_Demeaned{suffix}"], "Sector"
        )[f"Fwd_Demeaned{suffix}"]
        for column in FEATURE_COLUMNS:
            if column not in scored.columns:
                continue
            per_quarter = (
                scored.groupby("Quarter")
                .apply(
                    lambda g, c=column, sfx=suffix: pd.Series(
                        {
                            "ic_raw": rank_ic(g[c], g[f"Fwd_Return{sfx}"]),
                            "ic_demeaned": rank_ic(g[c], g[f"Fwd_Demeaned{sfx}"]),
                            "ic_sector_neutral": rank_ic(g[c], g[f"Fwd_Sector_Neutral{sfx}"]),
                            "n": int(g[[c, f"Fwd_Return{sfx}"]].dropna().shape[0]),
                        }
                    ),
                    include_groups=False,
                )
                .dropna(subset=["ic_demeaned"])
            )
            if per_quarter.empty:
                continue
            ic = per_quarter["ic_demeaned"]
            rows.append(
                {
                    "horizon_q": horizon,
                    "feature": column,
                    "quarters": len(ic),
                    "avg_obs": float(per_quarter["n"].mean()),
                    "mean_ic": float(ic.mean()),
                    "median_ic": float(ic.median()),
                    "std_ic": float(ic.std(ddof=1)),
                    "t_stat": float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))))
                    if len(ic) > 1 and ic.std(ddof=1) > 0
                    else np.nan,
                    "pct_positive": hit_rate(ic.tolist()),
                    "mean_ic_raw": float(per_quarter["ic_raw"].mean()),
                    "mean_ic_sector_neutral": float(per_quarter["ic_sector_neutral"].mean()),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["horizon_q", "mean_ic"], ascending=[True, False])
        .reset_index(drop=True)
    )


def conditional_tables(observations: pd.DataFrame) -> pd.DataFrame:
    """
    Double sorts that answer the thesis directly rather than through a blend.

    Accumulation against price strength asks what buying is worth *given* the
    trend — the divergence claim is that the high-accumulation / weak-price corner
    is the best cell. Accumulation against conviction asks whether accumulation
    adds anything once position size is known.
    """
    scored = observations[observations["Fwd_Return"].notna()]
    pairs = [
        ("Accumulation_Z", "Price_Strength_Z"),
        ("Accumulation_Z", "Avg_Portfolio_Pct"),
        ("Divergence", "Avg_Portfolio_Pct"),
        ("Smart_Score", "Divergence"),
    ]
    frames = []
    for row_col, col_col in pairs:
        table = double_sort_table(scored, row_col, col_col, "Fwd_Demeaned", bins=5)
        if not table:
            continue
        melted = table["mean_return"].stack().rename("mean_return").reset_index()
        melted.columns = ["row_bucket", "col_bucket", "mean_return"]
        melted["n"] = table["count"].stack().to_numpy()
        melted.insert(0, "col_feature", col_col)
        melted.insert(0, "row_feature", row_col)
        frames.append(melted)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ic_time_series(observations: pd.DataFrame) -> pd.DataFrame:
    """
    Quarter-by-quarter IC for the headline scores.

    A mean IC hides whether an edge was steady or came from one lucky quarter, and
    with sixteen quarters that distinction decides whether the average means
    anything at all.
    """
    scored = observations[observations["Fwd_Return"].notna()]
    rows = []
    for quarter, group in scored.groupby("Quarter"):
        row = {"quarter": quarter, "n": len(group)}
        for column in STABILITY_SCORES:
            row[column] = rank_ic(group[column], group["Fwd_Demeaned"])
        rows.append(row)
    return pd.DataFrame(rows)


def _select(group: pd.DataFrame, column: str, *, ascending: bool, top_n: int) -> pd.DataFrame:
    """
    Best ``top_n`` rows of a quarter by one column.
    """
    return group.dropna(subset=[column]).sort_values(column, ascending=ascending).head(top_n)


def _weights(selected: pd.DataFrame, scheme: str) -> dict[str, float]:
    """
    Position weights for a selected screen, normalised to sum to one.
    """
    if selected.empty:
        return {}
    if scheme == "conviction":
        raw = selected.set_index("Ticker")["Avg_Portfolio_Pct"].clip(lower=0.0)
        if raw.sum() <= 0:
            raw = pd.Series(1.0, index=selected["Ticker"])
    else:
        raw = pd.Series(1.0, index=selected["Ticker"])
    return (raw / raw.sum()).to_dict()


def portfolio_definitions() -> list[dict]:
    """
    The screens under test, including the ablations and the reference score.

    ``ACC_ONLY`` and ``PX_WEAK`` isolate the halves of the composite: if either
    matches the divergence screen on its own, the divergence framing adds nothing.
    ``CONVICTION_ONLY`` plays the same role for the smart score, whose strongest
    component is conviction — if ranking on that alone does as well, the score's
    other two components are carrying nothing.
    The two ``SMART_SCORE`` rows are the project's existing published screen —
    equal-weighted for comparability with everything else here, and
    conviction-weighted to reproduce exactly what ``gen-strategy`` trades.
    """
    return [
        {
            "id": "SMART_SCORE",
            "label": "Smart Score (top 30, EW)",
            "column": "Smart_Score",
            "ascending": False,
            "scheme": "equal",
        },
        {
            "id": "SMART_SCORE_CONV",
            "label": "Smart Score (top 30, conviction-weighted)",
            "column": "Smart_Score",
            "ascending": False,
            "scheme": "conviction",
        },
        {
            "id": "DIV_LONG",
            "label": "Divergence long (top 30, EW)",
            "column": "Divergence",
            "ascending": False,
            "scheme": "equal",
        },
        {
            "id": "DIV_LONG_CONV",
            "label": "Divergence long (top 30, conviction-weighted)",
            "column": "Divergence",
            "ascending": False,
            "scheme": "conviction",
        },
        {
            "id": "DIV_SHORT",
            "label": "Divergence short leg (bottom 30, EW)",
            "column": "Divergence",
            "ascending": True,
            "scheme": "equal",
        },
        {
            "id": "CONFIRM_LONG",
            "label": "Confirmation long (accumulation + strong price, top 30)",
            "column": "Confirmation",
            "ascending": False,
            "scheme": "equal",
        },
        {
            "id": "CONVICTION_ONLY",
            "label": "Ablation: conviction only (top 30)",
            "column": "Avg_Portfolio_Pct",
            "ascending": False,
            "scheme": "equal",
        },
        {
            "id": "ACC_ONLY",
            "label": "Ablation: accumulation only (top 30)",
            "column": "Accumulation_Z",
            "ascending": False,
            "scheme": "equal",
        },
        {
            "id": "PX_STRONG",
            "label": "Ablation: strong price only (top 30)",
            "column": "Price_Strength_Z",
            "ascending": False,
            "scheme": "equal",
        },
        {
            "id": "PX_WEAK",
            "label": "Ablation: weak price only (bottom 30)",
            "column": "Price_Strength_Z",
            "ascending": True,
            "scheme": "equal",
        },
        {
            "id": "GATE_BUY",
            "label": "Buy gate (all qualifying names, EW)",
            "column": "Divergence",
            "ascending": False,
            "scheme": "equal",
            "gate": "Buy_Gate",
            "top_n": 10_000,
        },
        {
            "id": "GATE_BUY_TOP",
            "label": "Buy gate, best 30 by divergence",
            "column": "Divergence",
            "ascending": False,
            "scheme": "equal",
            "gate": "Buy_Gate",
        },
        {
            "id": "GATE_SELL",
            "label": "Sell gate (all qualifying names, EW)",
            "column": "Divergence",
            "ascending": True,
            "scheme": "equal",
            "gate": "Sell_Gate",
            "top_n": 10_000,
        },
        {
            "id": "UNIVERSE_EW",
            "label": "Universe equal-weight (screen-free control)",
            "column": "Divergence",
            "ascending": False,
            "scheme": "equal",
            "top_n": 10_000,
        },
    ]


def run_portfolios(
    observations: pd.DataFrame, close: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trade every screen quarter by quarter and return its windows and daily curves.
    """
    scored = observations[observations["Fwd_Return"].notna()]
    windows: list[dict] = []
    curves: list[pd.Series] = []

    # Every screen is judged over the same quarters. A gate that only fires from
    # 2022Q4 would otherwise show a cumulative return that simply skips the two
    # worst quarters of the sample.
    tradable = sorted(scored["Quarter"].unique())

    for spec in portfolio_definitions():
        chained: list[pd.Series] = []
        previous: set[str] = set()
        for quarter in tradable:
            group = scored[scored["Quarter"] == quarter]
            pool = group[group[spec["gate"]]] if spec.get("gate") else group
            entry = date.fromisoformat(group["Entry_Date"].iloc[0])
            exit_date = date.fromisoformat(group["Exit_Date"].iloc[0])
            selected = _select(
                pool, spec["column"], ascending=spec["ascending"], top_n=spec.get("top_n", TOP_N)
            )
            priced = (
                selected.set_index("Ticker")["Fwd_Return"].dropna()
                if not selected.empty
                else pd.Series(dtype="float64")
            )
            if priced.empty:
                # A gate with no qualifying name is in cash for the window. Skipping
                # it instead would quietly drop the quarters the screen found hardest.
                windows.append(
                    {
                        "strategy": spec["id"],
                        "label": spec["label"],
                        "quarter": quarter,
                        "entry_date": entry.isoformat(),
                        "exit_date": exit_date.isoformat(),
                        "n_selected": 0,
                        "n_priced": 0,
                        "coverage": np.nan,
                        "window_return": 0.0,
                        "bench_return": float(group["Bench_Return"].iloc[0]),
                        "position_hit_rate": np.nan,
                        "median_position_return": np.nan,
                        "turnover": 0.0,
                    }
                )
                chained.append(
                    pd.Series(
                        1.0,
                        index=close.loc[
                            (close.index >= pd.Timestamp(entry))
                            & (close.index <= pd.Timestamp(exit_date))
                        ].index,
                    )
                )
                previous = set()
                continue

            weights = _weights(selected, spec["scheme"])
            active = {t: w for t, w in weights.items() if t in priced.index}
            total = sum(active.values())
            window_return = float(sum(w / total * priced[t] for t, w in active.items()))

            curve = daily_equity(close, weights, entry, exit_date)
            if not curve.empty:
                chained.append(curve)

            names = set(priced.index)
            windows.append(
                {
                    "strategy": spec["id"],
                    "label": spec["label"],
                    "quarter": quarter,
                    "entry_date": entry.isoformat(),
                    "exit_date": exit_date.isoformat(),
                    "n_selected": len(selected),
                    "n_priced": len(priced),
                    "coverage": len(priced) / len(selected),
                    "window_return": window_return,
                    "bench_return": float(selected["Bench_Return"].iloc[0]),
                    "position_hit_rate": hit_rate(priced.tolist()),
                    "median_position_return": float(priced.median()),
                    "turnover": 0.0 if not previous else 1.0 - len(names & previous) / len(names),
                }
            )
            previous = names

        curves.append(_chain(chained).rename(spec["id"]))

    curve_frame = pd.concat(curves, axis=1) if curves else pd.DataFrame()
    return pd.DataFrame(windows), curve_frame


def _chain(curves: list[pd.Series]) -> pd.Series:
    """
    Splice consecutive within-window curves into one continuous equity series.
    """
    if not curves:
        return pd.Series(dtype="float64")
    level = 1.0
    pieces = []
    for curve in curves:
        scaled = curve * level
        pieces.append(scaled.iloc[1:] if pieces else scaled)
        level = float(scaled.iloc[-1])
    return pd.concat(pieces)


def summarize(windows: pd.DataFrame, curves: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """
    Headline statistics per screen: win rate, position hit rate, drawdown, risk.

    Both drawdowns are reported. The quarterly figure is what a quarter-end
    statement would show; the daily figure is what an investor actually lived
    through, and it is always the larger of the two.
    """
    rows = []
    for strategy, group in windows.groupby("strategy"):
        group = group.sort_values("quarter")
        summary = summarize_windows(
            group["window_return"].tolist(), group["bench_return"].tolist(), periods_per_year=4
        )
        curve = curves[strategy].dropna() if strategy in curves.columns else pd.Series(dtype=float)
        daily_dd = max_drawdown(curve.pct_change().dropna().tolist()) if len(curve) > 1 else np.nan
        rows.append(
            {
                "strategy": strategy,
                "label": group["label"].iloc[0],
                "windows": summary["n_windows"],
                "avg_names": float(group["n_priced"].mean()),
                "cum_return": summary["cum_return"],
                "cagr": summary["cagr"],
                "avg_window": summary["avg_window"],
                "win_rate": summary["win_rate"],
                "beat_spy_rate": summary["beat_benchmark_rate"],
                "position_hit_rate": float(
                    (group["position_hit_rate"].fillna(0) * group["n_priced"]).sum()
                    / group["n_priced"].sum()
                ),
                "cash_windows": int((group["n_priced"] == 0).sum()),
                "max_drawdown_q": summary["max_drawdown"],
                "max_drawdown_daily": daily_dd,
                "volatility": summary["volatility"],
                "sharpe": summary["sharpe"],
                "avg_excess": summary["avg_excess"],
                "information_ratio": summary["information_ratio"],
                "avg_turnover": float(group["turnover"].mean()),
                "avg_coverage": float(group["coverage"].mean()),
            }
        )

    bench = windows.drop_duplicates("quarter").sort_values("quarter")["bench_return"]
    bench_summary = summarize_windows(bench.tolist(), periods_per_year=4)
    bench_curve = close[BENCHMARK].dropna()
    if len(windows):
        span = (bench_curve.index >= pd.Timestamp(windows["entry_date"].min())) & (
            bench_curve.index <= pd.Timestamp(windows["exit_date"].max())
        )
        bench_curve = bench_curve[span]
    rows.append(
        {
            "strategy": BENCHMARK,
            "label": "S&P 500 (benchmark)",
            "windows": bench_summary["n_windows"],
            "avg_names": 1.0,
            "cum_return": bench_summary["cum_return"],
            "cagr": bench_summary["cagr"],
            "avg_window": bench_summary["avg_window"],
            "win_rate": bench_summary["win_rate"],
            "beat_spy_rate": np.nan,
            "position_hit_rate": np.nan,
            "max_drawdown_q": bench_summary["max_drawdown"],
            "max_drawdown_daily": max_drawdown(bench_curve.pct_change().dropna().tolist()),
            "volatility": bench_summary["volatility"],
            "sharpe": bench_summary["sharpe"],
            "avg_excess": np.nan,
            "information_ratio": np.nan,
            "avg_turnover": 0.0,
            "avg_coverage": 1.0,
            "cash_windows": 0,
        }
    )
    return pd.DataFrame(rows).sort_values("cum_return", ascending=False).reset_index(drop=True)


def main() -> None:
    """
    Run the whole study and write every table to ``__reports__/divergence/``.
    """
    close, volume, splits = load_cache()
    panel = apply_split_adjustment(build_holdings_panel(), splits)
    logger.info("Panel: %d rows, %d quarters", len(panel), panel["Quarter"].nunique())

    observations = build_observations(panel, close, volume, date.today())
    logger.success(
        "Observations: %d rows, %d scored",
        len(observations),
        observations["Fwd_Return"].notna().sum(),
    )

    ics = information_coefficients(observations)
    conditionals = conditional_tables(observations)
    ic_series = ic_time_series(observations)
    windows, curves = run_portfolios(observations, close)
    summary = summarize(windows, curves, close)
    buckets = pd.concat(
        [
            quantile_table(
                observations[observations["Fwd_Return"].notna()], column, "Fwd_Demeaned", bins=5
            ).assign(feature=column)
            for column in ["Divergence", "Accumulation_Z", "Price_Strength_Z"]
        ],
        ignore_index=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observations.to_csv(OUTPUT_DIR / "observations.csv", index=False)
    ics.to_csv(OUTPUT_DIR / "information_coefficients.csv", index=False)
    windows.to_csv(OUTPUT_DIR / "windows.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    buckets.to_csv(OUTPUT_DIR / "quantiles.csv", index=False)
    conditionals.to_csv(OUTPUT_DIR / "conditional_sorts.csv", index=False)
    ic_series.to_csv(OUTPUT_DIR / "ic_by_quarter.csv", index=False)
    curves.to_csv(OUTPUT_DIR / "equity_curves.csv")

    live = observations[observations["Fwd_Return"].isna()]
    if not live.empty:
        live.sort_values("Divergence", ascending=False).to_csv(
            OUTPUT_DIR / "live_screen.csv", index=False
        )

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        logger.info("\n=== Information coefficients ===\n%s", ics.to_string(index=False))
        logger.info("\n=== Quantiles ===\n%s", buckets.to_string(index=False))
        for (row_feature, col_feature), block in conditionals.groupby(
            ["row_feature", "col_feature"], sort=False
        ):
            grid = block.pivot(index="row_bucket", columns="col_bucket", values="mean_return")
            logger.info(
                "\n=== %s (rows) x %s (cols): mean quarter-demeaned forward return ===\n%s",
                row_feature,
                col_feature,
                (grid * 100).round(2).to_string(),
            )
        logger.info(
            "\n=== IC by quarter (quarter-demeaned, 1q horizon) ===\n%s",
            ic_series.round(3).to_string(index=False),
        )
        logger.info("\n=== Portfolio summary ===\n%s", summary.to_string(index=False))
    logger.success("Wrote study output to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
