"""Replay the live screener's own selection criteria against real history.

Walks day by day through a symbol's actual historical option chain (from the
community DoltHub dataset, via voldata.py) applying the exact same filters and
compute_score ranking chain.py/screen_oom.py use live, "papers" the top
candidate, holds to expiration, and resolves the outcome against the real
historical stock price. Reports whether the entry criteria actually predicted
good outcomes - it does not execute or recommend trades.

Limitations (see voldata.py's module docstring for the data-source story):
  - No open-interest/volume in the historical dataset - those filters can't be
    replicated. Spread% (from bid/ask) is applied since that data does exist.
  - Earnings-date exclusion is best-effort (ticker.get_earnings_dates) and is
    silently skipped for a symbol/period where that history isn't available -
    reported in the output, not hidden.
  - Every cycle is evaluated independently, as if freshly buying the stock at
    that day's spot price - this measures whether the entry criteria predict
    good per-trade outcomes, not a compounded, continuously-held portfolio
    return across a wheel chain.
  - EOD data, lagged about a trading day.
"""
import itertools
import pickle
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yfinance as yf

import chain as chain_mod
import voldata

from config import (  # noqa: E402
    MIN_DTE, MAX_DTE, IV_THRESHOLD, IV_RV_RATIO_MIN, IV_RANK_MIN,
    DELTA_MIN, DELTA_MAX, MAX_SPREAD_PCT, RISK_FREE_RATE, SHARES_PER_CONTRACT,
    ASSIGNMENT_PREFERENCE,
)

BACKTEST_CACHE_DIR = Path(__file__).resolve().parent / "backtest_cache"
BACKTEST_MAX_WORKERS = 4
DEFAULT_BACKTEST_MONTHS = 12
# volatility_history's earliest date; option_chain starts around the same time.
EARLIEST_AVAILABLE_DATE = "2019-02-09"
RV_LOOKBACK_BUFFER_DAYS = 130  # calendar days, covers the 63-trading-day RV window


def backtest_cache_path(symbol):
    return BACKTEST_CACHE_DIR / f"{symbol.upper()}.pkl"


def load_backtest_cache(symbol, min_dte, max_dte):
    """Per-symbol cache of single-day historical chain snapshots. No TTL - a
    past trading day's data never changes. Keyed on the DTE window too: if
    --min-dte/--max-dte differ from what's cached, that's a different set of
    contracts entirely, so the stale cache is ignored (not deleted - the next
    save overwrites it once the new window has been fetched)."""
    path = backtest_cache_path(symbol)
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except Exception:
        return {}
    if payload.get("min_dte") != min_dte or payload.get("max_dte") != max_dte:
        return {}
    return payload.get("days", {})


def save_backtest_cache(symbol, min_dte, max_dte, days):
    try:
        BACKTEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with backtest_cache_path(symbol).open("wb") as f:
            pickle.dump({"min_dte": min_dte, "max_dte": max_dte, "days": days}, f)
    except Exception:
        pass


def get_price_series(symbol, start_date, end_date):
    """Daily close prices for symbol, with enough lookback before start_date
    for the realized-vol window, through end_date. Indexed by trading date."""
    ticker = yf.Ticker(symbol)
    buffer_start = (pd.Timestamp(start_date) - pd.Timedelta(days=RV_LOOKBACK_BUFFER_DAYS)).strftime("%Y-%m-%d")
    fetch_end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    hist = ticker.history(start=buffer_start, end=fetch_end, interval="1d", auto_adjust=False)
    if hist.empty:
        return hist
    hist = hist.reset_index()
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None).dt.date
    return hist


def get_earnings_dates_history(symbol):
    """Best-effort historical earnings dates. Empty set (not an error) if
    yfinance doesn't have this symbol's earnings history - the earnings-window
    exclusion is then simply not applied for this run, which the caller
    reports rather than silently pretending it happened."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.get_earnings_dates(limit=40)
        if df is None or df.empty:
            return set()
        idx = pd.to_datetime(df.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        return set(idx.date)
    except Exception:
        return set()


def fetch_chain_for_days(symbol, trading_days, min_dte, max_dte, use_cache=True):
    """Ensure every trading day has a cached (possibly empty) chain snapshot,
    fetching only what's missing. One network request per missing day - see
    voldata.fetch_option_chain_history's docstring for why this can't be one
    bulk range query."""
    cache = load_backtest_cache(symbol, min_dte, max_dte) if use_cache else {}
    missing = [d for d in trading_days if d.isoformat() not in cache]
    if not missing:
        return cache

    print(f"Fetching {len(missing)} new day(s) of historical chain data for {symbol} "
          f"({len(trading_days) - len(missing)} already cached)...")
    lock = threading.Lock()
    completed = 0
    failed = 0

    def fetch_one(day):
        date_str = day.isoformat()
        rows = voldata.fetch_option_chain_history(
            symbol, date_str, date_str, call_put="Call", min_dte=min_dte, max_dte=max_dte,
        )
        return date_str, rows

    with ThreadPoolExecutor(max_workers=BACKTEST_MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_one, d): d for d in missing}
        for future in as_completed(futures):
            date_str, rows = future.result()
            with lock:
                completed += 1
                if rows is not None:
                    cache[date_str] = rows
                else:
                    failed += 1
                if completed % 25 == 0 or completed == len(missing):
                    print(f"  {completed}/{len(missing)} fetched" + (f" ({failed} failed)" if failed else ""))

    save_backtest_cache(symbol, min_dte, max_dte, cache)
    return cache


def _resolve_cycle(position, day, exit_spot):
    strike = position["strike"]
    entry_spot = position["entry_spot"]
    fill_dollars = position["fill_dollars"]
    cash_required = entry_spot * SHARES_PER_CONTRACT
    assigned = exit_spot >= strike
    if assigned:
        pl_dollars = fill_dollars + (strike - entry_spot) * SHARES_PER_CONTRACT
    else:
        pl_dollars = fill_dollars
    realized_pct = pl_dollars / cash_required * 100
    holding_days = max((day - position["entry_date"]).days, 1)
    return {
        "entry_date": position["entry_date"],
        "expiration": position["expiration_date"],
        "strike": strike,
        "entry_spot": entry_spot,
        "exit_spot": exit_spot,
        "premium": fill_dollars,
        "outcome": "ASSIGNED" if assigned else "EXPIRED",
        "pl_dollars": pl_dollars,
        "realized_pct": realized_pct,
        "realized_ann": realized_pct * (365.0 / holding_days),
        "holding_days": holding_days,
        "score_at_entry": position["score"],
        "iv_at_entry": position["iv"],
        "iv_rv_at_entry": position["iv_rv"],
        "iv_rank_at_entry": position["iv_rank"],
        "delta_at_entry": position["delta"],
        "prob_itm_at_entry": position["prob_itm"],
    }


def _best_candidate(rows, day, spot, rv, iv_rank_today, earnings_dates,
                    iv_threshold, iv_rv_ratio_min, iv_rank_min, assignment_preference):
    best = None
    for row in rows:
        try:
            strike = float(row["strike"])
            bid = float(row["bid"]) if row["bid"] is not None else None
            ask = float(row["ask"]) if row["ask"] is not None else None
            iv = float(row["vol"]) if row["vol"] is not None else None
            delta = float(row["delta"]) if row["delta"] is not None else None
            expiration_date = pd.Timestamp(row["expiration"]).date()
        except (TypeError, ValueError, KeyError):
            continue
        if bid is None or ask is None or iv is None or delta is None or bid <= 0:
            continue
        if not (iv > iv_threshold):
            continue
        if not (DELTA_MIN <= delta <= DELTA_MAX):
            continue
        mid = (bid + ask) / 2
        if mid <= 0 or (ask - bid) / mid > MAX_SPREAD_PCT:
            continue
        if rv:
            iv_rv = iv / rv
            if iv_rv < iv_rv_ratio_min:
                continue
        else:
            iv_rv = None
        if iv_rank_today is not None and iv_rank_today < iv_rank_min:
            continue
        if any(day <= ed <= expiration_date for ed in earnings_dates):
            continue

        dte = (expiration_date - day).days
        years_to_expiry = max(dte, 1) / 365.0
        prob_itm = chain_mod.bs_prob_itm_exp(spot, strike, years_to_expiry, RISK_FREE_RATE, iv)
        if prob_itm is None:
            continue
        fill = chain_mod.modeled_fill_price(bid, ask)
        fill_dollars = fill * SHARES_PER_CONTRACT
        cash_required = spot * SHARES_PER_CONTRACT
        ann_factor = 365.0 / max(dte, 1)
        pct_if_expired_ann = fill_dollars / cash_required * 100 * ann_factor
        pct_if_called_ann = (fill_dollars + (strike - spot) * SHARES_PER_CONTRACT) / cash_required * 100 * ann_factor
        score = chain_mod.compute_score(pct_if_expired_ann, pct_if_called_ann, prob_itm, assignment_preference)

        if best is None or score > best["score"]:
            best = {
                "strike": strike, "expiration_date": expiration_date, "fill_dollars": fill_dollars,
                "score": score, "iv": iv, "iv_rv": iv_rv, "iv_rank": iv_rank_today,
                "delta": delta, "prob_itm": prob_itm,
            }
    return best


def fetch_backtest_inputs(symbol, start_date, end_date, min_dte=None, max_dte=None, use_cache=True):
    """Everything a backtest needs that's independent of iv/iv-rv/iv-rank/
    assign - price history, earnings dates, the IV-rank series, and the
    per-day option chain (itself disk-cached, keyed on the DTE window).
    Fetched ONCE per (symbol, window, DTE) and reused across every threshold
    combination in a sweep - those thresholds are pure local filtering and
    scoring, they never change any of this. Returns None (after printing why)
    if there's no price history or no trading days in the window.
    """
    min_dte = MIN_DTE if min_dte is None else min_dte
    max_dte = MAX_DTE if max_dte is None else max_dte
    symbol = symbol.strip().upper()

    hist = get_price_series(symbol, start_date, end_date)
    if hist.empty:
        print(f"No price history for {symbol} in this window.")
        return None

    price_by_date = dict(zip(hist["Date"], hist["Close"]))
    trading_days = sorted(d for d in price_by_date if start_date <= d.isoformat() <= end_date)
    if not trading_days:
        print("No trading days in the requested window.")
        return None

    earnings_dates = get_earnings_dates_history(symbol)

    iv_rank_rows = voldata.fetch_iv_rank_history(symbol, start_date, end_date)
    iv_rank_by_date = {}
    if iv_rank_rows:
        for row in iv_rank_rows:
            try:
                iv_rank_by_date[pd.Timestamp(row["date"]).date()] = row["iv_rank"]
            except Exception:
                continue

    chain_cache = fetch_chain_for_days(symbol, trading_days, min_dte, max_dte, use_cache)

    return {
        "symbol": symbol, "hist": hist, "price_by_date": price_by_date,
        "trading_days": trading_days, "earnings_dates": earnings_dates,
        "iv_rank_by_date": iv_rank_by_date, "iv_rank_available": bool(iv_rank_rows),
        "chain_cache": chain_cache,
    }


def simulate_cycles(inputs, iv_threshold, iv_rv_ratio_min, iv_rank_min, assignment_preference):
    """Pure local simulation given pre-fetched `inputs` (from
    fetch_backtest_inputs) - no network calls, so this is what actually
    varies per threshold combination in a sweep; everything about the
    symbol/window/DTE is already fixed by the time this runs."""
    hist = inputs["hist"]
    price_by_date = inputs["price_by_date"]
    earnings_dates = inputs["earnings_dates"]
    iv_rank_by_date = inputs["iv_rank_by_date"]
    chain_cache = inputs["chain_cache"]

    cycles = []
    state = "flat"
    position = None

    for day in inputs["trading_days"]:
        if state == "in_trade":
            if day < position["expiration_date"]:
                continue
            exit_spot = price_by_date.get(day)
            if exit_spot is not None:
                cycles.append(_resolve_cycle(position, day, exit_spot))
            state, position = "flat", None
            continue

        rows = chain_cache.get(day.isoformat())
        if not rows:
            continue
        spot = price_by_date.get(day)
        if spot is None or spot <= 0:
            continue

        closes_upto = hist.loc[hist["Date"] <= day, "Close"]
        rv, _, _ = chain_mod.blended_realized_volatility(closes_upto)
        iv_rank_today = iv_rank_by_date.get(day)

        best = _best_candidate(
            rows, day, spot, rv, iv_rank_today, earnings_dates,
            iv_threshold, iv_rv_ratio_min, iv_rank_min, assignment_preference,
        )
        if best is not None:
            position = dict(best)
            position["entry_date"] = day
            position["entry_spot"] = spot
            state = "in_trade"

    return cycles


def run_backtest(symbol, start_date, end_date, iv_threshold=None, iv_rv_ratio_min=None,
                 iv_rank_min=None, min_dte=None, max_dte=None, assignment_preference=None,
                 use_cache=True):
    iv_threshold = IV_THRESHOLD if iv_threshold is None else iv_threshold
    iv_rv_ratio_min = IV_RV_RATIO_MIN if iv_rv_ratio_min is None else iv_rv_ratio_min
    iv_rank_min = IV_RANK_MIN if iv_rank_min is None else iv_rank_min
    assignment_preference = ASSIGNMENT_PREFERENCE if assignment_preference is None else assignment_preference

    inputs = fetch_backtest_inputs(symbol, start_date, end_date, min_dte, max_dte, use_cache)
    if inputs is None:
        return [], False, False

    cycles = simulate_cycles(inputs, iv_threshold, iv_rv_ratio_min, iv_rank_min, assignment_preference)
    return cycles, bool(inputs["earnings_dates"]), inputs["iv_rank_available"]


def print_backtest_results(symbol, cycles, start_date, end_date, iv_threshold, iv_rv_ratio_min,
                           iv_rank_min, min_dte, max_dte, earnings_available, iv_rank_available=True):
    print()
    print(f"Backtest: {symbol}  {start_date} -> {end_date}")
    iv_rv_desc = f"IV/RV >= {iv_rv_ratio_min:.2f}" if iv_rv_ratio_min > 0 else "IV/RV off"
    iv_rank_desc = f"IV rank >= {iv_rank_min:.0f}" if iv_rank_min > 0 else "IV rank off"
    print(f"Screen replayed: {min_dte}-{max_dte} DTE, IV > {iv_threshold:.2f}, {iv_rv_desc}, {iv_rank_desc}, "
          f"delta {DELTA_MIN:.2f}-{DELTA_MAX:.2f}, spread <= {MAX_SPREAD_PCT * 100:.0f}% of mid")
    print("Not modeled (unavailable in this historical dataset): open interest, volume")
    if not earnings_available:
        print("Earnings-date history unavailable for this symbol - earnings-window exclusion NOT applied")
    if iv_rank_min > 0 and not iv_rank_available:
        print("IV-rank history unavailable for this window (third-party source didn't respond) - "
              "IV rank filtering was NOT applied despite --iv-rank, results may differ from a run "
              "where it was available")
    print()

    if not cycles:
        print("No cycles: no candidate ever passed this screen in this window.")
        return

    df = pd.DataFrame(cycles)
    display = pd.DataFrame({
        "entry": df["entry_date"].astype(str),
        "exp": df["expiration"].astype(str),
        "strike": df["strike"],
        "spot_in": df["entry_spot"],
        "spot_out": df["exit_spot"],
        "prem$": df["premium"],
        "outcome": df["outcome"],
        "real_ann": df["realized_ann"],
        "score": df["score_at_entry"],
    })

    def color(col, value):
        if col == "outcome":
            return chain_mod.ANSI_RED if value == "ASSIGNED" else chain_mod.ANSI_GREEN
        if col == "real_ann" and pd.notna(value):
            return chain_mod.ANSI_GREEN if value >= 0 else chain_mod.ANSI_RED
        return None

    print("Cycles:")
    chain_mod.print_colored_table(
        display, ["strike", "spot_in", "spot_out", "prem$", "real_ann", "score"],
        color, min_width=8, na_rep="",
    )
    print()

    n = len(df)
    win_rate = (df["outcome"] == "EXPIRED").mean() * 100
    assign_rate = (df["outcome"] == "ASSIGNED").mean() * 100
    corr = df["score_at_entry"].corr(df["realized_pct"]) if n >= 3 else None
    best = df.loc[df["pl_dollars"].idxmax()]
    worst = df.loc[df["pl_dollars"].idxmin()]

    print(f"Cycles: {n}")
    print(f"Win rate (expired OTM): {win_rate:.1f}%   Assigned: {assign_rate:.1f}%")
    print(f"Avg realized annualized return: {df['realized_ann'].mean():.2f}%   "
          f"Avg entry score: {df['score_at_entry'].mean():.2f}")
    if corr is not None and pd.notna(corr):
        print(f"Correlation(entry score, realized return): {corr:.2f}  "
              f"(positive = score was actually predictive here; near zero or negative = it wasn't)")
    else:
        print("Correlation(entry score, realized return): not enough cycles to compute")
    print(f"Best cycle:  {best['entry_date']} {best['strike']:.2f}c -> {best['outcome']}  "
          f"P&L ${best['pl_dollars']:+.2f}")
    print(f"Worst cycle: {worst['entry_date']} {worst['strike']:.2f}c -> {worst['outcome']}  "
          f"P&L ${worst['pl_dollars']:+.2f}")


# --- Parameter sweep / optimizer ----------------------------------------------
#
# Deliberately does NOT sweep min_dte/max_dte by default. run_backtest's
# per-day chain fetch is cached by (symbol, min_dte, max_dte) - every
# combination of iv_threshold/iv_rv_ratio_min/iv_rank_min/assignment_preference
# replays against that SAME cached data (they're pure local filtering/scoring,
# never change what got fetched), so a sweep over those is nearly free once
# the first combination has populated the cache. A DTE sweep would need a
# fresh network fetch per distinct window - orders of magnitude more
# expensive - so it's opt-in only, one call at a time, not multiplied
# silently inside a grid.
SWEEP_DEFAULT_GRID = {
    "iv_threshold": [0.0, 0.20, IV_THRESHOLD, 0.40],
    "iv_rv_ratio_min": [0.0, 0.60, IV_RV_RATIO_MIN, 1.00],
    "iv_rank_min": [0, 15, 25, 50],
    "assignment_preference": ["keep", "neutral"],
}
SWEEP_RANK_METRICS = ("win_rate", "avg_realized_ann", "correlation", "assign_rate")


def sweep_backtest(symbol, start_date, end_date, param_grid=None, min_dte=None, max_dte=None,
                   min_cycles=5, use_cache=True, progress=True):
    """Grid-search the screener's free-to-vary parameters against ONE cached
    historical dataset for `symbol`. Returns (results, earnings_available,
    iv_rank_available): `results` is a DataFrame, one row per combination
    that produced at least `min_cycles` cycles, with columns
    iv_threshold/iv_rv_ratio_min/iv_rank_min/assignment_preference/n_cycles/
    win_rate/assign_rate/avg_realized_ann/avg_score/correlation - unsorted;
    the caller picks the ranking metric. The two availability flags apply to
    the WHOLE sweep (fetched once, shared by every combination) - if
    iv_rank_available is False, every combination in this call ran with IV
    rank filtering silently unable to exclude anything, which materially
    changes results vs. a run where it was available. This DOES happen: the
    historical IV-rank range query is a third-party call over a wide date
    span and measurably flakier than the rest of this pipeline - retried a
    few times internally, but not guaranteed to succeed.

    min_cycles matters more than it looks: a combination that only ever
    entered one trade can show a 100% win rate on pure luck. Dropping
    thin-sample combinations, and always reporting n_cycles next to every
    other number, is what keeps this from quietly recommending a filter so
    tight it rarely trades at all.

    param_grid overrides SWEEP_DEFAULT_GRID per-key; a key you don't specify
    keeps its default range. Every axis is de-duplicated, order-preserved.
    """
    inputs = fetch_backtest_inputs(symbol, start_date, end_date, min_dte, max_dte, use_cache)
    if inputs is None:
        return pd.DataFrame(), False, False

    grid = {k: list(v) for k, v in SWEEP_DEFAULT_GRID.items()}
    if param_grid:
        grid.update(param_grid)
    for key, values in grid.items():
        seen = []
        for v in values:
            if v not in seen:
                seen.append(v)
        grid[key] = seen

    combos = list(itertools.product(
        grid["iv_threshold"], grid["iv_rv_ratio_min"], grid["iv_rank_min"], grid["assignment_preference"],
    ))
    if progress:
        print(f"Sweeping {len(combos)} combination(s) for {symbol} ({start_date} -> {end_date}) "
              f"- data fetched once, everything from here is local...")

    rows = []
    for i, (iv, iv_rv, iv_rank, assign) in enumerate(combos, start=1):
        cycles = simulate_cycles(inputs, iv, iv_rv, iv_rank, assign)
        n = len(cycles)
        if n < min_cycles:
            continue
        cdf = pd.DataFrame(cycles)
        win_rate = (cdf["outcome"] == "EXPIRED").mean() * 100
        assign_rate = (cdf["outcome"] == "ASSIGNED").mean() * 100
        corr = cdf["score_at_entry"].corr(cdf["realized_pct"]) if n >= 3 else float("nan")
        rows.append({
            "iv_threshold": iv, "iv_rv_ratio_min": iv_rv, "iv_rank_min": iv_rank,
            "assignment_preference": assign, "n_cycles": n, "win_rate": win_rate,
            "assign_rate": assign_rate, "avg_realized_ann": cdf["realized_ann"].mean(),
            "avg_score": cdf["score_at_entry"].mean(), "correlation": corr,
        })
        if progress and (i % 25 == 0 or i == len(combos)):
            print(f"  {i}/{len(combos)} combinations tested ({len(rows)} usable so far)")

    return pd.DataFrame(rows), bool(inputs["earnings_dates"]), inputs["iv_rank_available"]


def find_default_combo_row(results):
    """The sweep row matching config.py's ACTUAL current defaults - what the
    live screener is set up to trade with right now - if that combination is
    present in the grid. None otherwise. Lets a caller show where "business
    as usual" falls within the full spread of alternatives that were tested,
    not just the cherry-picked top of a ranked list (SWEEP_DEFAULT_GRID
    always includes the config defaults on every axis, so this is present
    unless the caller passed a custom param_grid that excluded them).
    """
    if results.empty:
        return None
    match = results[
        (results["iv_threshold"] == IV_THRESHOLD) &
        (results["iv_rv_ratio_min"] == IV_RV_RATIO_MIN) &
        (results["iv_rank_min"] == IV_RANK_MIN) &
        (results["assignment_preference"] == ASSIGNMENT_PREFERENCE)
    ]
    return match.iloc[0] if not match.empty else None


def print_sweep_distribution(results, metric, bins=10):
    """ASCII histogram of `metric` across every combination in `results`
    (already min_cycles-filtered), with the bucket containing the current
    config's combination (see find_default_combo_row) flagged - so a ranked
    top-N table (survivorship-biased by construction) sits next to the
    actual shape of the full distribution it was drawn from.
    """
    values = results[metric].dropna()
    if values.empty:
        print(f"No data to show a distribution for {metric}.")
        return

    default_row = find_default_combo_row(results)
    default_value = default_row[metric] if default_row is not None else None

    lo, hi = float(values.min()), float(values.max())
    print(f"Distribution of {metric} across {len(values)} combination(s) "
          f"(min {lo:.2f}, median {values.median():.2f}, max {hi:.2f}):")
    if lo == hi:
        print(f"  All combinations landed on the same value: {lo:.2f}")
        return

    width = (hi - lo) / bins
    counts = [0] * bins
    default_bin = None
    for v in values:
        idx = min(int((v - lo) / width), bins - 1)
        counts[idx] += 1
    if default_value is not None and pd.notna(default_value):
        default_bin = min(int((float(default_value) - lo) / width), bins - 1)
        percentile = (values <= default_value).mean() * 100

    max_count = max(counts)
    for i, count in enumerate(counts):
        bucket_lo, bucket_hi = lo + i * width, lo + (i + 1) * width
        bar = "#" * (round((count / max_count) * 40) if max_count else 0)
        marker = "  <- current config" if i == default_bin else ""
        print(f"  [{bucket_lo:8.2f}, {bucket_hi:8.2f})  {count:3d}  {bar}{marker}")

    if default_value is not None and pd.notna(default_value):
        print(f"Current config (iv={IV_THRESHOLD:.2f} iv-rv={IV_RV_RATIO_MIN:.2f} "
              f"iv-rank={IV_RANK_MIN} assign={ASSIGNMENT_PREFERENCE}): {metric}={default_value:.2f}, "
              f"{percentile:.0f}th percentile of all tested combinations")
    else:
        print("Current config combination wasn't in this sweep's grid (custom --param-grid?) - "
              "no percentile to report.")


def print_sweep_results(results, symbol, start_date, end_date, rank_by, min_cycles, top_n=15,
                        earnings_available=True, iv_rank_available=True):
    print()
    print(f"Sweep: {symbol}  {start_date} -> {end_date}  (ranked by {rank_by}, >= {min_cycles} cycles required)")
    print("CAUTION: this ranks combinations by how well each WOULD have done in this exact window.")
    print("The more combinations tested, the higher the chance the top one just fit this window's")
    print("noise (multiple-comparisons risk) - treat it as a hypothesis to check on a different date")
    print("range, not a conclusion. n_cycles is shown for every row - a high win rate on a handful of")
    print("trades is not the same claim as a high win rate on fifty.")
    if not earnings_available:
        print("Earnings-date history unavailable for this symbol - earnings-window exclusion NOT applied")
    if not iv_rank_available:
        print("IV-rank history unavailable for this window (third-party source didn't respond) - EVERY "
              "combination below ran with IV rank filtering unable to exclude anything, regardless of "
              "its iv_rank_min value - re-run to try again, results may differ once it's available")
    print()

    if results.empty:
        print(f"No combination produced >= {min_cycles} cycles - try a wider date range, a lower "
              f"--min-cycles, or check the symbol has data in this window.")
        return

    ranked = results.sort_values(rank_by, ascending=False).reset_index(drop=True)
    display = ranked.head(top_n).copy()

    def color(col, value):
        if col in ("win_rate", "avg_realized_ann") and pd.notna(value):
            return chain_mod.ANSI_GREEN if value >= 0 else chain_mod.ANSI_RED
        if col == "correlation" and pd.notna(value):
            return chain_mod.ANSI_GREEN if value >= 0.3 else (chain_mod.ANSI_YELLOW if value >= 0 else chain_mod.ANSI_RED)
        return None

    print(f"Top {len(display)} of {len(ranked)} combination(s) that cleared {min_cycles} cycles:")
    chain_mod.print_colored_table(
        display,
        ["iv_threshold", "iv_rv_ratio_min", "iv_rank_min", "n_cycles", "win_rate",
         "assign_rate", "avg_realized_ann", "avg_score", "correlation"],
        color, min_width=8, na_rep="",
    )
    print()
    print_sweep_distribution(results, rank_by)

    best = ranked.iloc[0]
    print()
    print(f"Best by {rank_by}: iv={best['iv_threshold']:.2f}  iv-rv={best['iv_rv_ratio_min']:.2f}  "
          f"iv-rank={best['iv_rank_min']:.0f}  assign={best['assignment_preference']}")
    corr_text = f"{best['correlation']:.2f}" if pd.notna(best['correlation']) else "n/a"
    print(f"  {int(best['n_cycles'])} cycles, {best['win_rate']:.1f}% win rate, "
          f"{best['avg_realized_ann']:.1f}% avg ann. return, correlation {corr_text}")
    print(f"  Reproduce live:    python chain.py {symbol} --iv {best['iv_threshold']:.2f} "
          f"--iv-rv {best['iv_rv_ratio_min']:.2f} --iv-rank {best['iv_rank_min']:.0f} "
          f"--assign {best['assignment_preference']}")
    print(f"  Verify on backtest: python backtest.py {symbol} --from {start_date} --to {end_date} "
          f"--iv {best['iv_threshold']:.2f} --iv-rv {best['iv_rv_ratio_min']:.2f} "
          f"--iv-rank {best['iv_rank_min']:.0f} --assign {best['assignment_preference']}")


def extract_date_flag(argv, flag):
    if flag not in argv:
        return None
    i = argv.index(flag)
    if i + 1 >= len(argv):
        raise ValueError(f"{flag} requires a date in YYYY-MM-DD format")
    value = argv[i + 1].strip()
    try:
        pd.Timestamp(value)
    except Exception:
        raise ValueError(f"{flag} requires a date in YYYY-MM-DD format, got '{value}'")
    del argv[i:i + 2]
    return value


def resolve_date_range(argv):
    from_date = extract_date_flag(argv, "--from")
    to_date = extract_date_flag(argv, "--to")
    today = pd.Timestamp.now(tz="US/Pacific").date().isoformat()
    if to_date is None:
        to_date = today
    if from_date is None:
        from_date = (pd.Timestamp(to_date) - pd.DateOffset(months=DEFAULT_BACKTEST_MONTHS)).date().isoformat()
    if from_date < EARLIEST_AVAILABLE_DATE:
        from_date = EARLIEST_AVAILABLE_DATE
    return from_date, to_date


def print_usage():
    print(f"""Usage:
  python backtest.py SYMBOL [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--iv N] [--iv-rv N]
                            [--iv-rank N] [--min-dte N] [--max-dte N] [--assign MODE] [--refresh]

      Replays the live screener's own criteria day-by-day against SYMBOL's real
      historical option chain, "papers" the top-scoring candidate on each day
      you'd be flat, holds to expiration, and resolves the outcome against the
      real historical stock price. Reports win rate, realized annualized
      return, and whether the entry score actually predicted good outcomes.

      --from DATE    Start of the backtest window (default: {DEFAULT_BACKTEST_MONTHS} months
                     before --to). Earliest available data is {EARLIEST_AVAILABLE_DATE}.
      --to DATE      End of the backtest window (default: today).
      --iv N         Absolute implied-vol floor (default {IV_THRESHOLD:.2f}).
      --iv-rv N      Minimum implied/realized vol ratio (default {IV_RV_RATIO_MIN:.2f}).
      --iv-rank N    Minimum IV rank, 0-100 (default {IV_RANK_MIN}).
      --min-dte N    Minimum days to expiration (default {MIN_DTE}).
      --max-dte N    Maximum days to expiration (default {MAX_DTE}).
      --assign MODE  'keep' (default) or 'neutral' - see chain.py --help.
      --refresh      Force refetching every day in the window, ignoring the
                     local cache (backtest_cache/SYMBOL.pkl). Historical days
                     never change, so this is rarely needed.

      NOT modeled: open interest / volume (absent from the historical dataset).
      Earnings-window exclusion is best-effort and reported if unavailable.

      Example: python backtest.py AMD --from 2024-01-01 --to 2026-08-01
      Example: python backtest.py AMD --iv-rank 50 --assign neutral

  python backtest.py sweep SYMBOL [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--min-dte N] [--max-dte N]
                                  [--rank-by METRIC] [--min-cycles N] [--top N] [--refresh]

      Grid-searches iv/iv-rv/iv-rank/assign-mode against ONE cached historical
      fetch and ranks every combination that cleared --min-cycles. Nearly free
      after the first combination populates the cache (those parameters are
      pure local filtering/scoring - they don't change what gets fetched).
      --min-dte/--max-dte are NOT swept - a different DTE window needs its own
      fresh network fetch, so re-run with a different pair explicitly if you
      want to explore that dimension too.

      --rank-by METRIC   One of {", ".join(SWEEP_RANK_METRICS)} (default win_rate).
      --min-cycles N     Drop combinations with fewer cycles than this (default 5) -
                         a 100% win rate from one lucky trade is not a signal.
      --top N            How many rows to print (default 15).
      --refresh          Force refetching this symbol/DTE window once, before sweeping.

      CAUTION: ranks by fit to ONE historical window - the more combinations
      tested, the more likely the top one just fit that window's noise.
      Treat the winner as a hypothesis to verify on a different date range
      (`backtest.py SYMBOL --from ... --to ...` with those exact flags), not
      a conclusion.

      Example: python backtest.py sweep AMD --from 2024-01-01 --to 2026-08-01
      Example: python backtest.py sweep AMD --rank-by avg_realized_ann --min-cycles 10
""")


def cmd_sweep(argv):
    if not argv or argv[0].startswith("--"):
        print("Usage: python backtest.py sweep SYMBOL [--from YYYY-MM-DD] [--to YYYY-MM-DD] [...]")
        print("Run 'python backtest.py help' for the full option list.")
        return

    symbol = argv[0].strip().upper()
    argv = argv[1:]
    try:
        from_date, to_date = resolve_date_range(argv)
        min_dte = chain_mod.extract_int_flag(argv, "--min-dte")
        max_dte = chain_mod.extract_int_flag(argv, "--max-dte")
        rank_by = chain_mod.extract_choice_flag(argv, "--rank-by", SWEEP_RANK_METRICS)
        min_cycles = chain_mod.extract_int_flag(argv, "--min-cycles")
        top_n = chain_mod.extract_int_flag(argv, "--top")
        refresh = chain_mod.extract_bool_flag(argv, "--refresh")
    except ValueError as exc:
        print(exc)
        return

    min_dte = MIN_DTE if min_dte is None else min_dte
    max_dte = MAX_DTE if max_dte is None else max_dte
    rank_by = rank_by or "win_rate"
    min_cycles = 5 if min_cycles is None else min_cycles
    top_n = 15 if top_n is None else top_n

    results, earnings_available, iv_rank_available = sweep_backtest(
        symbol, from_date, to_date, min_dte=min_dte, max_dte=max_dte,
        min_cycles=min_cycles, use_cache=not refresh,
    )
    print_sweep_results(results, symbol, from_date, to_date, rank_by, min_cycles, top_n,
                        earnings_available, iv_rank_available)


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("help", "-h", "--help"):
        print_usage()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        cmd_sweep(sys.argv[2:])
        return
    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        print("Usage: python backtest.py SYMBOL [--from YYYY-MM-DD] [--to YYYY-MM-DD] [...]")
        print("Run 'python backtest.py help' for the full option list.")
        return

    symbol = sys.argv[1].strip().upper()
    argv = sys.argv[2:]
    try:
        from_date, to_date = resolve_date_range(argv)
        iv = chain_mod.extract_float_flag(argv, "--iv")
        iv_rv = chain_mod.extract_float_flag(argv, "--iv-rv")
        iv_rank = chain_mod.extract_float_flag(argv, "--iv-rank")
        min_dte = chain_mod.extract_int_flag(argv, "--min-dte")
        max_dte = chain_mod.extract_int_flag(argv, "--max-dte")
        assign = chain_mod.extract_choice_flag(argv, "--assign", ("keep", "neutral"))
        refresh = chain_mod.extract_bool_flag(argv, "--refresh")
    except ValueError as exc:
        print(exc)
        return

    iv_threshold = IV_THRESHOLD if iv is None else iv
    iv_rv_ratio_min = IV_RV_RATIO_MIN if iv_rv is None else iv_rv
    iv_rank_min = IV_RANK_MIN if iv_rank is None else iv_rank
    min_dte = MIN_DTE if min_dte is None else min_dte
    max_dte = MAX_DTE if max_dte is None else max_dte
    assignment_preference = ASSIGNMENT_PREFERENCE if assign is None else assign

    print(f"Backtesting {symbol}: {from_date} -> {to_date}")
    print("First run fetches one request per new trading day and may take a few minutes; "
          "fully cached after (re-running with different thresholds costs no new requests).")

    cycles, earnings_available, iv_rank_available = run_backtest(
        symbol, from_date, to_date, iv_threshold, iv_rv_ratio_min, iv_rank_min,
        min_dte, max_dte, assignment_preference, use_cache=not refresh,
    )
    print_backtest_results(
        symbol, cycles, from_date, to_date, iv_threshold, iv_rv_ratio_min,
        iv_rank_min, min_dte, max_dte, earnings_available, iv_rank_available,
    )


if __name__ == "__main__":
    main()
