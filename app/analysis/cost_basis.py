"""
Estimated institutional average cost basis for a single ticker.

13F filings say what a fund held at each quarter end, never what it paid. The
gap between those two facts is the whole problem here: a quarter's net share
change tells us *how many* shares were bought, and the quarter's price path
tells us the *range* of prices those shares could have been bought at, but the
execution schedule inside the quarter is unobservable. So the basis is not a
number, it is a distribution — and this module estimates it by simulation
rather than pretending a single point estimate exists.

The generative model, per fund and quarter:

  * Net buys are the only flows we can see (intra-quarter round trips are
    invisible to a quarter-end snapshot), so a positive share delta is treated
    as one accumulation and a negative delta as a sale.
  * The buy is executed over the quarter with an unknown weight vector `w` on
    that quarter's trading days, giving an execution price `X = Σ w_d p_d`.
    Given only "spread over the quarter, more volume means more fill", the
    maximum-entropy prior on `w` is a Dirichlet centred on the quarter's volume
    profile: `w ~ Dir(α · m)`, `m_d = v_d / Σv`. Its mean is the quarter VWAP
    and its variance is the volume-weighted price variance divided by `α + 1`.
  * `α + 1` is pinned to data instead of taste: a fund cannot buy `B` shares
    faster than `PARTICIPATION_RATE` of daily volume, so the order needs at
    least `B / (ρ · ADV)` days. Small positions could be a single block trade
    anywhere in the quarter (wide band); positions large against the tape must
    be worked over most of it (band collapses onto the VWAP).
  * Accumulation is not timing-neutral. Funds initiate on catalysts and add
    with momentum, so fills land systematically above the plain VWAP. The
    volume profile is exponentially tilted toward higher-priced days
    (`TIMING_TILT`), shifting the expected fill by roughly `TIMING_TILT`
    price standard deviations while — unlike an additive drift — never
    letting a draw leave the window's traded range.
  * Funds are not independent — they buy the same names on the same news — so
    each fund's execution deviation is split into a quarter-common component
    and an idiosyncratic one (`CO_MOVEMENT`), which keeps the aggregate band
    from averaging itself away as holders are added.

Positions are then carried forward under average-cost accounting (buys move the
average, sales do not) and aggregated across funds share-weighted. The reported
band is the 10th–90th percentile of the simulated aggregate; the line is its
median.

Two caveats travel with every number this produces: holdings present in the
earliest quarter on record have unknown vintage and are priced under a diffuse
prior over the preceding `SEED_LOOKBACK_QUARTERS` quarters — each path prices
the whole legacy position at one unknown day of that window, so a large legacy
holding keeps a wide band instead of collapsing onto a multi-year VWAP via the
participation cap (`seeded_pct` says how much of the current position that is)
— and the whole estimate covers tracked funds only, not the full institutional
float.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from app.analysis.regime import harmonise_basis, snap_to_split_factor
from app.database import get_all_quarters, load_quarterly_data
from app.database.stocks import load_stocks
from app.stocks.bar_history import Bar, load_daily_bars
from app.utils.logger import get_logger, log_safe
from app.utils.pd import get_numeric_series
from app.utils.strings import get_quarter, get_quarter_date, parse_quarter

logger = get_logger(__name__)

# A fund cannot take more than this share of a day's volume without moving the
# tape against itself; it sets the minimum number of days an order needs.
PARTICIPATION_RATE = 0.10
# Fraction of each fund's execution risk that is common to every buyer in the
# quarter. Institutions crowd into the same names on the same catalysts, so
# their fills are correlated; independent draws would understate the band.
CO_MOVEMENT = 0.5
# Systematic buy-above-VWAP timing bias, as an exponential tilt of the day
# weights per unit of standardised price: the expected fill sits ~this many
# price standard deviations above the plain VWAP.
TIMING_TILT = 0.25
# Days an order is assumed to take when the price source carries no volume.
DEFAULT_SPREAD_DAYS = 5
# Vintage window for positions already open in the first quarter on record.
SEED_LOOKBACK_QUARTERS = 12
BAND_PERCENTILES = (10.0, 90.0)
DEFAULT_SIMULATIONS = 800


@dataclass(frozen=True)
class CostBasisPoint:
    """
    The estimated aggregate cost basis of tracked funds at one quarter end.
    """

    quarter: str
    as_of: str
    shares: float
    holders: int
    low: float
    mid: float
    high: float
    seeded_pct: float


@dataclass(frozen=True)
class _Window:
    """
    The price/volume profile an order is executed against.
    """

    prices: np.ndarray
    weights: np.ndarray
    adv: float

    @property
    def mean(self) -> float:
        """
        Volume-weighted average price of the window.
        """
        return float(self.weights @ self.prices)


def _quarter_start(quarter: str) -> date:
    """
    First calendar day of a quarter.
    """
    year, q = parse_quarter(quarter)
    return date(year, (q - 1) * 3 + 1, 1)


def _shift_quarters(quarter: str, back: int) -> str:
    """
    The quarter `back` quarters before the given one.
    """
    year, q = parse_quarter(quarter)
    index = year * 4 + (q - 1) - back
    return f"{index // 4}Q{index % 4 + 1}"


def _make_window(bars: Sequence[Bar]) -> _Window | None:
    """
    Build an execution window from bars, weighting days by volume where known.
    """
    if not bars:
        return None
    prices = np.array([b.price for b in bars], dtype=float)
    volumes = np.array([max(b.volume, 0.0) for b in bars], dtype=float)
    valid = np.isfinite(prices) & (prices > 0)
    if not valid.any():
        return None
    prices, volumes = prices[valid], volumes[valid]
    total = volumes.sum()
    if total > 0:
        weights = volumes / total
        adv = float(volumes.mean())
    else:
        weights = np.full(prices.size, 1.0 / prices.size)
        adv = 0.0
    return _Window(prices=prices, weights=weights, adv=adv)


def _effective_days(window: _Window, shares: float) -> int:
    """
    How many days an order of `shares` needs at the participation cap.

    Without volume there is nothing to size against, so the order is assumed to
    take `DEFAULT_SPREAD_DAYS` — the honest middle between a single block and a
    quarter-long programme.
    """
    n_days = window.prices.size
    if window.adv <= 0:
        return min(DEFAULT_SPREAD_DAYS, n_days)
    needed = int(np.ceil(shares / (PARTICIPATION_RATE * window.adv))) if shares > 0 else 1
    return int(np.clip(needed, 1, n_days))


def _timing_tilted_weights(window: _Window) -> np.ndarray:
    """
    The volume profile tilted toward the window's higher-priced days.

    Funds do not fill at the plain VWAP: they initiate on catalysts and add
    with momentum, so executions cluster on the stronger days. The exponential
    tilt encodes that bias while keeping every draw inside the traded range,
    which an additive drift on the sampled price would not.
    """
    mean = window.mean
    variance = float(window.weights @ (window.prices - mean) ** 2)
    if variance <= 0:
        return window.weights
    tilted = window.weights * np.exp(TIMING_TILT * (window.prices - mean) / np.sqrt(variance))
    return tilted / tilted.sum()


def _execution_samples(
    window: _Window, shares: float, n_sims: int, rng: np.random.Generator, *, diffuse: bool = False
) -> np.ndarray:
    """
    Draw `n_sims` execution prices for an order of `shares` against a window.

    A `diffuse` order has no observable footprint to size against — the
    unknown vintage of a pre-history position — so the participation cap must
    not collapse it onto the window VWAP; each path prices it at one unknown
    day of the window instead.
    """
    n_days = window.prices.size
    weights = _timing_tilted_weights(window)
    mean = float(weights @ window.prices)
    days = 1 if diffuse else _effective_days(window, shares)
    if n_days == 1 or days >= n_days:
        return np.full(n_sims, mean)
    if days == 1:
        return window.prices[rng.choice(n_days, size=n_sims, p=weights)]

    alpha = np.maximum((days - 1) * weights, 1e-9)
    gamma = rng.gamma(shape=np.broadcast_to(alpha, (n_sims, n_days)))
    totals = gamma.sum(axis=1, keepdims=True)
    weights = np.divide(gamma, totals, out=np.full_like(gamma, 1.0 / n_days), where=totals > 0)
    return weights @ window.prices


def _standardise(samples: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    Split samples into their mean, standard deviation and unit-variance shape.
    """
    mean = float(samples.mean())
    sd = float(samples.std())
    if sd <= 0:
        return mean, 0.0, np.zeros_like(samples)
    return mean, sd, (samples - mean) / sd


def normalise_share_basis(panel: pd.DataFrame, market_price: Mapping[str, float]) -> pd.DataFrame:
    """
    Restate filed share counts onto the split basis of the price series.

    Share counts in the quarter CSVs are stored exactly as filed while market
    prices are split-adjusted at source, so a pre-split quarter would otherwise
    read as an enormous position that was later "sold". Each fund's filed value
    per share is a market price, which makes the fix measurable rather than
    guessed: stragglers inside a quarter are pulled onto the consensus basis
    first, then the quarter's consensus price is compared with the market close
    and rescaled when the ratio lands on a plausible split factor.
    """
    if panel.empty:
        return panel

    frame = panel.rename(columns={"Shares": "SharesN", "Price": "Px"}).assign(
        CUSIP=lambda df: df["Quarter"]
    )
    harmonised = pd.concat(
        [harmonise_basis(group) for _, group in frame.groupby("Quarter", sort=False)],
        ignore_index=True,
    )

    for quarter, group in harmonised.groupby("Quarter", sort=False):
        close = market_price.get(str(quarter))
        consensus = float(group["Px"].median())
        if not close or close <= 0 or not np.isfinite(consensus) or consensus <= 0:
            continue
        factor = snap_to_split_factor(consensus / close)
        if factor is None:
            continue
        logger.debug(
            "Restating %s onto the post-split basis (factor %.4g)", log_safe(str(quarter)), factor
        )
        rows = harmonised["Quarter"] == quarter
        harmonised.loc[rows, "SharesN"] *= factor
        harmonised.loc[rows, "Px"] /= factor

    return harmonised.drop(columns="CUSIP").rename(columns={"SharesN": "Shares", "Px": "Price"})[
        ["Quarter", "Fund", "Shares", "Price"]
    ]


def _windows_by_quarter(bars: Sequence[Bar]) -> dict[str, _Window]:
    """
    Group bars into one execution window per calendar quarter.
    """
    grouped: dict[str, list[Bar]] = {}
    for bar in bars:
        grouped.setdefault(get_quarter(bar.day.isoformat()), []).append(bar)
    windows = {q: _make_window(group) for q, group in grouped.items()}
    return {q: w for q, w in windows.items() if w is not None}


def _seed_window(bars: Sequence[Bar], first_quarter: str) -> _Window | None:
    """
    The price window a position already open in `first_quarter` was built in.

    Its vintage is unknowable from the filings, so it is spread over the years
    preceding our record. With no earlier prices on file the first quarter's own
    window stands in — a floor on the uncertainty, not a claim about vintage.
    """
    start = _quarter_start(_shift_quarters(first_quarter, SEED_LOOKBACK_QUARTERS))
    cutoff = _quarter_start(first_quarter)
    return _make_window([b for b in bars if start <= b.day < cutoff])


def estimate_cost_basis(
    panel: pd.DataFrame,
    bars: Sequence[Bar],
    quarters: Sequence[str] | None = None,
    *,
    n_sims: int = DEFAULT_SIMULATIONS,
    seed: int = 0,
) -> list[CostBasisPoint]:
    """
    Simulate the aggregate cost basis of every tracked holder, quarter by quarter.

    Args:
        panel: One row per (Quarter, Fund) with `Shares` held and the filed
            `Price` per share, already on the price series' split basis.
        bars: Daily price/volume bars covering the panel (and ideally the years
            before it, for seeding pre-history positions).
        quarters: Ordered quarters to walk. Defaults to those in the panel;
            pass the full sequence when a quarter with no holders sits between
            two that have them, so the flows either side are not merged.
        n_sims: Monte Carlo paths. The band is a percentile of these.
        seed: RNG seed, so a given panel always yields the same band.

    Returns:
        One point per quarter that had at least one holder.
    """
    if panel.empty:
        return []

    ordered = list(quarters) if quarters is not None else sorted(panel["Quarter"].unique())
    if not ordered:
        return []

    rng = np.random.default_rng(seed)
    windows = _windows_by_quarter(bars)
    seed_window = _seed_window(bars, ordered[0]) or windows.get(ordered[0])

    held: dict[str, float] = {}
    seeded: dict[str, float] = {}
    basis: dict[str, np.ndarray] = {}
    points: list[CostBasisPoint] = []

    for index, quarter in enumerate(ordered):
        rows = panel[panel["Quarter"] == quarter]
        shares_now = {
            str(fund): float(value)
            for fund, value in zip(rows["Fund"], rows["Shares"], strict=True)
        }
        buys = {
            fund: shares_now.get(fund, 0.0) - held.get(fund, 0.0)
            for fund in set(shares_now) | set(held)
        }
        buys = {fund: delta for fund, delta in buys.items() if delta > 0}

        window = windows.get(quarter)
        if window is None:
            filed = rows.loc[rows["Price"] > 0, "Price"]
            window = (
                _Window(np.array([float(filed.median())]), np.array([1.0]), 0.0)
                if not filed.empty
                else None
            )
        if index == 0:
            window = seed_window or window

        if buys and window is not None:
            # First-quarter buys are pre-history positions of unknown vintage:
            # priced under the diffuse prior, never sized by participation.
            diffuse = index == 0
            _, _, common = _standardise(
                _execution_samples(window, sum(buys.values()), n_sims, rng, diffuse=diffuse)
            )
            for fund, bought in buys.items():
                mean, sd, shape = _standardise(
                    _execution_samples(window, bought, n_sims, rng, diffuse=diffuse)
                )
                price = mean + sd * (
                    np.sqrt(CO_MOVEMENT) * common + np.sqrt(1.0 - CO_MOVEMENT) * shape
                )
                previous = held.get(fund, 0.0)
                previous_basis = basis.get(fund, np.zeros(n_sims))
                basis[fund] = (previous_basis * previous + price * bought) / (previous + bought)
                if index == 0:
                    seeded[fund] = bought

        # A fund absent from a quarter has not filed, it has not sold: exits are
        # explicit CLOSE rows (shares 0). Zeroing the absentees instead would
        # invent a round trip and reprice the position at a later quarter.
        for fund, shares in shares_now.items():
            previous = held.get(fund, 0.0)
            if shares < previous and previous > 0:
                seeded[fund] = seeded.get(fund, 0.0) * shares / previous
            held[fund] = shares

        total = sum(held.values())
        if total <= 0:
            continue
        aggregate = sum(held[f] * basis[f] for f in held if held[f] > 0 and f in basis)
        if not isinstance(aggregate, np.ndarray):
            continue
        aggregate = aggregate / total
        low, high = np.percentile(aggregate, BAND_PERCENTILES)
        points.append(
            CostBasisPoint(
                quarter=quarter,
                as_of=get_quarter_date(quarter),
                shares=total,
                holders=sum(1 for v in held.values() if v > 0),
                low=float(low),
                mid=float(np.median(aggregate)),
                high=float(high),
                seeded_pct=100.0 * sum(seeded.get(f, 0.0) for f in held) / total,
            )
        )

    return points


def build_position_panel(ticker: str, quarters: Sequence[str]) -> pd.DataFrame:
    """
    Every tracked fund's position in one ticker, quarter by quarter.

    Positions are matched on CUSIP rather than the ticker written in the filing,
    so a name filed under several CUSIPs (share classes, post-merger issues)
    collapses into the single position the fund actually holds. The filed value
    per share is carried alongside: it is the fund's own quarter-end mark, and
    the split-basis normalisation is measured against it.

    Args:
        ticker: The ticker to build the panel for.
        quarters: Quarters to scan.

    Returns:
        pd.DataFrame: Columns `Quarter`, `Fund`, `Shares`, `Price` — empty when
            no tracked fund held the ticker in any of the quarters.
    """
    cusips = set(load_stocks().index[load_stocks()["Ticker"] == ticker])
    if not cusips:
        return pd.DataFrame(columns=["Quarter", "Fund", "Shares", "Price"])

    frames: list[pd.DataFrame] = []
    for quarter in quarters:
        df_quarter = load_quarterly_data(quarter)
        if df_quarter.empty:
            continue
        rows = df_quarter[df_quarter["CUSIP"].isin(cusips)].copy()
        if rows.empty:
            continue
        rows["Shares"] = pd.to_numeric(rows["Shares"], errors="coerce").fillna(0.0)
        rows["Value_Num"] = get_numeric_series(rows["Value"]).fillna(0.0)
        grouped = rows.groupby("Fund", as_index=False)[["Shares", "Value_Num"]].sum()
        grouped["Quarter"] = quarter
        frames.append(grouped)

    if not frames:
        return pd.DataFrame(columns=["Quarter", "Fund", "Shares", "Price"])

    panel = pd.concat(frames, ignore_index=True)
    panel["Price"] = np.where(
        panel["Shares"] > 0, panel["Value_Num"] / panel["Shares"].replace(0, np.nan), 0.0
    )
    return panel[["Quarter", "Fund", "Shares", "Price"]].fillna({"Price": 0.0})


def _market_prices(bars: Sequence[Bar], quarters: Sequence[str]) -> dict[str, float]:
    """
    Last traded price on or before each quarter end.
    """
    prices: dict[str, float] = {}
    for quarter in quarters:
        cutoff = date.fromisoformat(get_quarter_date(quarter))
        earlier = [b.price for b in bars if b.day <= cutoff]
        if earlier:
            prices[quarter] = earlier[-1]
    return prices


def ticker_cost_basis(
    ticker: str, *, n_sims: int = DEFAULT_SIMULATIONS, seed: int = 0
) -> list[CostBasisPoint]:
    """
    Estimate the cost-basis band of every tracked holder of a ticker.

    Args:
        ticker: The ticker to estimate.
        n_sims: Monte Carlo paths behind each quarter's band.
        seed: RNG seed, so the same filings always yield the same band.

    Returns:
        list[CostBasisPoint]: One point per quarter with at least one holder,
            oldest first (empty when the ticker was never held or no price
            history could be loaded).
    """
    quarters = sorted(get_all_quarters())
    panel = build_position_panel(ticker, quarters)
    if panel.empty:
        return []

    held = sorted(panel["Quarter"].unique())
    start = _quarter_start(_shift_quarters(str(held[0]), SEED_LOOKBACK_QUARTERS))
    bars = load_daily_bars(ticker, start)
    if not bars:
        logger.warning("No price history for %s: skipping cost basis", log_safe(ticker))
        return []

    panel = normalise_share_basis(panel, _market_prices(bars, held))
    walk = [q for q in quarters if q >= str(held[0])]
    return estimate_cost_basis(panel, bars, walk, n_sims=n_sims, seed=seed)
