import json
import math
import pickle
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import NormalDist

import pandas as pd
import yfinance as yf

import chain as chain_mod

# Screening criteria live in config.py so this and chain.py cannot drift apart.
from config import (  # noqa: E402
    MIN_DTE, MAX_DTE, IV_THRESHOLD, IV_RV_RATIO_MIN, TRADING_DAYS_PER_YEAR,
    RV_SHORT_TRADING_DAYS, RV_LONG_TRADING_DAYS, RV_SHORT_WEIGHT,
    TARGET_DELTA, DELTA_MIN, DELTA_MAX, PREMIUM_THRESHOLD, TOTAL_PREMIUM_THRESHOLD,
    MIN_OPEN_INTEREST, MAX_SPREAD_PCT, MIN_VOLUME, FILL_HAIRCUT,
    RISK_FREE_RATE, ASSIGNMENT_PREFERENCE,
    SHARES_PER_CONTRACT, CACHE_TTL_HOURS, CHAIN_CACHE_TTL_MINUTES, IV_RANK_MIN,
)

# voldata is entirely optional (see voldata.py) - chain_mod already resolves it
# to None if it fails to import, so aliasing it here rather than importing it
# a second time keeps that single point of truth.
voldata = chain_mod.voldata
fetch_iv_rank = chain_mod.fetch_iv_rank

DEFAULT_TICKER_FILE = "tickers.txt"
DEFAULT_TICKERS = [
    "AMD", "NVDA", "AAPL", "MSFT", "AMZN", "META", "TSLA", "GOOGL", "NFLX", "AVGO"
]
TOP_TICKERS = 50
MAX_SHARE_CAPITAL = 100000
MAX_WORKERS = 3
LOOKUP_CACHE_FILE = Path(__file__).resolve().parent / "lookup_cache.json"
# Raw option chains are cached so re-screening with different thresholds
# (--iv / --iv-rv) is pure local computation and hits no Yahoo endpoints.
CHAIN_CACHE_DIR = Path(__file__).resolve().parent / "chain_cache"
nd = NormalDist()


def bs_d1_d2(spot, strike, years_to_expiry, rate, sigma):
    if years_to_expiry <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return None, None
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma**2) * years_to_expiry) / (sigma * math.sqrt(years_to_expiry))
    d2 = d1 - sigma * math.sqrt(years_to_expiry)
    return d1, d2


def bs_call_delta(spot, strike, years_to_expiry, rate, sigma):
    d1, _ = bs_d1_d2(spot, strike, years_to_expiry, rate, sigma)
    if d1 is None:
        return None
    return nd.cdf(d1)


def bs_prob_itm_exp(spot, strike, years_to_expiry, rate, sigma):
    _, d2 = bs_d1_d2(spot, strike, years_to_expiry, rate, sigma)
    if d2 is None:
        return None
    return nd.cdf(d2) * 100


def approx_prob_touch(delta):
    if delta is None or pd.isna(delta):
        return None
    return min(100.0, max(0.0, 2 * float(delta) * 100))


def load_symbols_from_file(path_str):
    path = Path(path_str)
    if not path.exists():
        return None

    symbols = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            symbol = line.strip().upper()
            if not symbol or symbol.startswith("#"):
                continue
            symbols.append(symbol)

    deduped = []
    seen = set()
    for symbol in symbols:
        if symbol not in seen:
            seen.add(symbol)
            deduped.append(symbol)
    return deduped


def extract_float_flag(argv, flag):
    """Pull `--flag VALUE` out of argv (mutating it) and return VALUE, or None."""
    if flag not in argv:
        return None
    i = argv.index(flag)
    if i + 1 >= len(argv):
        raise ValueError(f"{flag} requires a numeric value")
    try:
        value = float(argv[i + 1])
    except ValueError:
        raise ValueError(f"{flag} requires a numeric value, got '{argv[i + 1]}'")
    del argv[i:i + 2]
    return value


def extract_bool_flag(argv, flag):
    if flag in argv:
        argv.remove(flag)
        return True
    return False


def parse_threshold_flags(argv):
    """Strip --iv / --iv-rv / --iv-rank / --premium / --min-dte / --max-dte /
    --assign / --refresh from argv, return effective settings."""
    iv = extract_float_flag(argv, "--iv")
    iv_rv = extract_float_flag(argv, "--iv-rv")
    iv_rank = extract_float_flag(argv, "--iv-rank")
    premium = extract_float_flag(argv, "--premium")
    min_dte = chain_mod.extract_int_flag(argv, "--min-dte")
    max_dte = chain_mod.extract_int_flag(argv, "--max-dte")
    assign = chain_mod.extract_choice_flag(argv, "--assign", ("keep", "neutral"))
    refresh = extract_bool_flag(argv, "--refresh")
    return (
        IV_THRESHOLD if iv is None else iv,
        IV_RV_RATIO_MIN if iv_rv is None else iv_rv,
        IV_RANK_MIN if iv_rank is None else iv_rank,
        TOTAL_PREMIUM_THRESHOLD if premium is None else premium,
        MIN_DTE if min_dte is None else min_dte,
        MAX_DTE if max_dte is None else max_dte,
        not refresh,
        ASSIGNMENT_PREFERENCE if assign is None else assign,
    )


def get_cash_to_invest():
    if len(sys.argv) > 2 and sys.argv[2].strip():
        return float(sys.argv[2].strip())
    return None


def get_symbols():
    script_dir = Path(__file__).resolve().parent
    user_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    default_path = script_dir / DEFAULT_TICKER_FILE

    if user_path is not None:
        symbols = load_symbols_from_file(user_path)
        if symbols:
            return symbols, str(user_path), "file"

    symbols = load_symbols_from_file(default_path)
    if symbols:
        return symbols, str(default_path), "file"

    return DEFAULT_TICKERS, "hardcoded DEFAULT_TICKERS", "default"


# Shared with chain.py so both scripts fetch/interpret spot price and day change
# identically - see chain.py for why this doesn't use the old camelCase lookups.
get_spot_price = chain_mod.get_spot_price
get_day_change = chain_mod.get_day_change


def get_company_name(ticker, symbol):
    try:
        info = ticker.get_info()
    except Exception:
        return symbol
    name = info.get("longName") or info.get("shortName")
    return name if name else symbol


# Realized-vol / resistance helpers are shared with chain.py so the two
# screeners cannot disagree about how "rich" premium is or where resistance
# sits. get_price_context used to be duplicated verbatim here; now it's the
# one definition in chain.py, reused directly.
realized_volatility = chain_mod.realized_volatility
blended_realized_volatility = chain_mod.blended_realized_volatility
modeled_fill_price = chain_mod.modeled_fill_price
compute_score = chain_mod.compute_score
get_price_context = chain_mod.get_price_context
find_oi_wall = chain_mod.find_oi_wall


def get_earnings_and_dividend(ticker):
    earnings_dates = []
    ex_div_date = None
    try:
        calendar = ticker.calendar
        if isinstance(calendar, dict):
            raw_earnings = calendar.get("Earnings Date")
            if raw_earnings:
                raw_earnings = raw_earnings if isinstance(raw_earnings, (list, tuple)) else [raw_earnings]
                earnings_dates = [d for d in raw_earnings if d is not None]
            ex_div_date = calendar.get("Ex-Dividend Date")
    except Exception:
        pass

    div_amount = None
    try:
        dividends = ticker.dividends
        if not dividends.empty:
            div_amount = float(dividends.iloc[-1])
    except Exception:
        pass

    return earnings_dates, ex_div_date, div_amount


def load_lookup_cache():
    if LOOKUP_CACHE_FILE.exists():
        try:
            with LOOKUP_CACHE_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_lookup_cache(cache):
    try:
        with LOOKUP_CACHE_FILE.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def is_cache_entry_fresh(entry):
    if not entry or "fetched_at" not in entry:
        return False
    fetched_at = pd.to_datetime(entry["fetched_at"])
    age_hours = (pd.Timestamp.now(tz="UTC") - fetched_at).total_seconds() / 3600
    return age_hours < CACHE_TTL_HOURS


def get_cached_lookup(symbol, ticker, cache, cache_lock):
    with cache_lock:
        entry = cache.get(symbol)
        fresh = is_cache_entry_fresh(entry)
        if fresh:
            cached = dict(entry)

    if fresh:
        earnings_dates = [pd.to_datetime(d).date() for d in cached["earnings_dates"]]
        ex_div_date = pd.to_datetime(cached["ex_div_date"]).date() if cached["ex_div_date"] else None
        return cached["company_name"], earnings_dates, ex_div_date, cached["div_amount"]

    company_name = get_company_name(ticker, symbol)
    earnings_dates, ex_div_date, div_amount = get_earnings_and_dividend(ticker)
    with cache_lock:
        cache[symbol] = {
            "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "company_name": company_name,
            "earnings_dates": [d.isoformat() for d in earnings_dates],
            "ex_div_date": ex_div_date.isoformat() if ex_div_date else None,
            "div_amount": div_amount,
        }
    return company_name, earnings_dates, ex_div_date, div_amount


def format_columns(df):
    two_decimal_cols = [
        "dte", "stk", "spot", "hi6", "d_spot", "d_hi6", "hit6",
        "last", "bid", "ask", "mid", "fill", "fill$", "called$", "spr", "vol", "oi", "sh$",
        "iv", "iv_rv", "score", "delta", "p_spot", "p_hi6", "p_itm", "p_touch",
        "p_exp", "p_call", "p_exp_ann", "p_call_ann", "div_amt",
        "contracts", "cash_used", "cash_left", "tot_prem"
    ]

    for col in two_decimal_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    return df


def chain_cache_path(symbol):
    return CHAIN_CACHE_DIR / f"{symbol.upper()}.oom.pkl"


def load_chain_cache(symbol, min_dte, max_dte):
    path = chain_cache_path(symbol)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
    age_minutes = (pd.Timestamp.now(tz="UTC") - payload["fetched_at"]).total_seconds() / 60
    if age_minutes > CHAIN_CACHE_TTL_MINUTES:
        return None
    if payload.get("min_dte") != min_dte or payload.get("max_dte") != max_dte:
        return None
    if payload.get("trade_date") != pd.Timestamp.now(tz="US/Pacific").date():
        return None
    return payload


def save_chain_cache(symbol, payload):
    try:
        CHAIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with chain_cache_path(symbol).open("wb") as f:
            pickle.dump(payload, f)
    except Exception:
        pass


def fetch_symbol_data(symbol, cash_to_invest, cache, cache_lock, use_cache=True,
                      min_dte=None, max_dte=None):
    """Everything requiring network access. Threshold-independent, so re-screening
    with different --iv / --iv-rv values reuses this untouched."""
    min_dte = MIN_DTE if min_dte is None else min_dte
    max_dte = MAX_DTE if max_dte is None else max_dte

    def affordable(spot):
        share_capital = spot * SHARES_PER_CONTRACT
        if share_capital > MAX_SHARE_CAPITAL:
            return False
        if cash_to_invest is not None and share_capital > cash_to_invest:
            return False
        return True

    if use_cache:
        cached = load_chain_cache(symbol, min_dte, max_dte)
        if cached is not None:
            if not affordable(cached["spot"]):
                return None
            cached["from_cache"] = True
            return cached

    ticker = yf.Ticker(symbol)
    spot = get_spot_price(ticker)
    # Early exit before the expensive chain fetch keeps unaffordable tickers cheap.
    if not affordable(spot):
        return None

    company_name, earnings_dates, ex_div_date, div_amount = get_cached_lookup(symbol, ticker, cache, cache_lock)
    context = get_price_context(ticker, spot)
    today = pd.Timestamp.now(tz="UTC").tz_convert("US/Pacific").date()

    expirations = []
    for exp in ticker.options:
        exp_date = pd.to_datetime(exp).date()
        dte = (exp_date - today).days
        if not (min_dte <= dte <= max_dte):
            continue
        if any(today <= ed <= exp_date for ed in earnings_dates):
            continue
        calls = ticker.option_chain(exp).calls.copy()
        if calls.empty:
            continue
        expirations.append({"exp": exp, "exp_date": exp_date, "dte": dte, "calls": calls})

    payload = {
        "fetched_at": pd.Timestamp.now(tz="UTC"),
        "trade_date": today,
        "min_dte": min_dte,
        "max_dte": max_dte,
        "spot": spot,
        "context": context,
        "company_name": company_name,
        "ex_div_date": ex_div_date,
        "div_amount": div_amount,
        "expirations": expirations,
        "from_cache": False,
    }
    save_chain_cache(symbol, payload)
    return payload


def best_contract_for_symbol(symbol, cash_to_invest=None, cache=None, cache_lock=None,
                             iv_threshold=None, iv_rv_ratio_min=None, use_cache=True,
                             assignment_preference=None, premium_threshold=None,
                             min_dte=None, max_dte=None, iv_rank_min=None,
                             voldata_cache=None, voldata_lock=None):
    iv_threshold = IV_THRESHOLD if iv_threshold is None else iv_threshold
    iv_rv_ratio_min = IV_RV_RATIO_MIN if iv_rv_ratio_min is None else iv_rv_ratio_min
    assignment_preference = ASSIGNMENT_PREFERENCE if assignment_preference is None else assignment_preference
    premium_threshold = TOTAL_PREMIUM_THRESHOLD if premium_threshold is None else premium_threshold
    iv_rank_min = IV_RANK_MIN if iv_rank_min is None else iv_rank_min
    if cache is None:
        cache = {}
    if cache_lock is None:
        cache_lock = threading.Lock()
    if voldata_cache is None:
        voldata_cache = {}
    if voldata_lock is None:
        voldata_lock = threading.Lock()
    try:
        data = fetch_symbol_data(symbol, cash_to_invest, cache, cache_lock, use_cache, min_dte, max_dte)
        if data is None:
            return None
        spot = data["spot"]
        share_capital = spot * SHARES_PER_CONTRACT
        company_name = data["company_name"]
        ex_div_date = data["ex_div_date"]
        div_amount = data["div_amount"]
        context = data["context"]
        today = data["trade_date"]
        matches = []

        # Symbol-level, not per-contract, so it's fetched once. Missing/
        # unavailable never excludes a candidate unless iv_rank_min > 0.
        iv_rank_ctx = fetch_iv_rank(symbol, voldata_cache, voldata_lock)
        iv_rank_value = iv_rank_ctx["iv_rank"]
        iv_rank_ok = iv_rank_value is None or iv_rank_value >= iv_rank_min

        # If you already hold shares, flag (not exclude) strikes below your
        # cost basis - see chain_mod.get_cost_basis for why this matters.
        cost_basis = chain_mod.get_cost_basis(symbol)

        for entry in data["expirations"]:
            exp = entry["exp"]
            exp_date = entry["exp_date"]
            dte = entry["dte"]
            calls = entry["calls"].copy()

            years_to_expiry = max(dte, 1) / 365.0
            calls["iv"] = pd.to_numeric(calls["impliedVolatility"], errors="coerce")
            calls["bid"] = pd.to_numeric(calls["bid"], errors="coerce")
            calls["ask"] = pd.to_numeric(calls["ask"], errors="coerce")
            calls["lastPrice"] = pd.to_numeric(calls["lastPrice"], errors="coerce")
            calls["volume"] = pd.to_numeric(calls["volume"], errors="coerce")
            calls["openInterest"] = pd.to_numeric(calls["openInterest"], errors="coerce")
            oi_wall_strike, oi_wall_oi = find_oi_wall(calls, spot)
            calls["delta"] = calls.apply(
                lambda row: bs_call_delta(spot, float(row["strike"]), years_to_expiry, RISK_FREE_RATE, float(row["iv"])) if pd.notna(row["iv"]) else None,
                axis=1,
            )
            calls["prob_itm_exp"] = calls.apply(
                lambda row: bs_prob_itm_exp(spot, float(row["strike"]), years_to_expiry, RISK_FREE_RATE, float(row["iv"])) if pd.notna(row["iv"]) else None,
                axis=1,
            )
            calls["prob_touch"] = calls["delta"].apply(approx_prob_touch)
            calls["mid"] = (calls["bid"] + calls["ask"]) / 2
            calls["spread"] = calls["ask"] - calls["bid"]
            calls["spread_pct"] = calls["spread"] / calls["mid"]
            calls["dist_to_target_delta"] = (calls["delta"] - TARGET_DELTA).abs()
            # All downstream premium/return/score math uses the modeled fill, not mid.
            calls["fill"] = modeled_fill_price(calls["bid"], calls["ask"])
            calls["fill_dollars"] = calls["fill"] * 100

            realized_vol = context.get("realized_vol")
            if realized_vol:
                calls["iv_rv_ratio"] = calls["iv"] / realized_vol
            else:
                calls["iv_rv_ratio"] = float("nan")
            # Missing realized vol should not silently reject every contract.
            iv_rich_enough = calls["iv_rv_ratio"].isna() | (calls["iv_rv_ratio"] >= iv_rv_ratio_min)

            filtered = calls[
                (calls["iv"] > iv_threshold) &
                iv_rich_enough &
                iv_rank_ok &
                (calls["delta"] >= DELTA_MIN) &
                (calls["delta"] <= DELTA_MAX) &
                (calls["bid"] > 0) &
                (calls["openInterest"] >= MIN_OPEN_INTEREST) &
                (calls["volume"] >= MIN_VOLUME) &
                (calls["spread_pct"] <= MAX_SPREAD_PCT)
            ].copy()

            if filtered.empty:
                continue

            filtered["symbol"] = symbol
            filtered["company_name"] = company_name
            filtered["expiration"] = exp
            filtered["dte"] = dte
            filtered["premium_over_1"] = (filtered["bid"] > PREMIUM_THRESHOLD) | (filtered["ask"] > PREMIUM_THRESHOLD)
            filtered["highlight"] = filtered["premium_over_1"].map({True: "***", False: ""})
            filtered["spot_price"] = spot
            filtered["iv_rank"] = iv_rank_value
            filtered["share_capital"] = share_capital
            filtered["high_6wk"] = context["resistance_price"]
            filtered["oi_wall"] = oi_wall_strike if oi_wall_strike is not None else float("nan")
            filtered["oi_wall_oi"] = oi_wall_oi if oi_wall_oi is not None else float("nan")
            filtered["oi_wall_ok"] = (
                filtered["strike"] >= oi_wall_strike if oi_wall_strike is not None else pd.NA
            )
            filtered["distance_to_spot"] = filtered["strike"] - spot
            filtered["pct_to_spot"] = (filtered["distance_to_spot"] / spot) * 100
            filtered["distance_to_6wk_high"] = filtered["strike"] - context["resistance_price"]
            filtered["pct_to_6wk_high"] = (filtered["distance_to_6wk_high"] / context["resistance_price"]) * 100
            filtered["resistance_hits_6wk"] = context["resistance_touches"]
            filtered["p_hi6_block"] = (filtered["pct_to_6wk_high"] < 0).astype(int)
            filtered["day_change_pct"] = context.get("day_change_pct")

            filtered["called_dollars"] = filtered["fill_dollars"] + filtered["distance_to_spot"] * 100
            filtered["pct_if_expired"] = filtered["fill_dollars"] / share_capital * 100
            filtered["pct_if_called"] = filtered["called_dollars"] / share_capital * 100
            filtered["pct_if_expired_ann"] = filtered["pct_if_expired"] * (365.0 / dte)
            filtered["pct_if_called_ann"] = filtered["pct_if_called"] * (365.0 / dte)
            filtered["score"] = compute_score(
                filtered["pct_if_expired_ann"], filtered["pct_if_called_ann"],
                filtered["prob_itm_exp"], assignment_preference,
            )

            ex_div_in_window = ex_div_date is not None and today <= ex_div_date <= exp_date
            intrinsic = (spot - filtered["strike"]).clip(lower=0)
            extrinsic_now = filtered["fill"] - intrinsic
            if ex_div_in_window and div_amount is not None:
                filtered["div_risk"] = extrinsic_now < div_amount
            else:
                filtered["div_risk"] = False
            filtered["div_amount"] = div_amount if div_amount is not None else float("nan")

            filtered["cost_basis"] = cost_basis if cost_basis is not None else float("nan")
            filtered["below_basis"] = (filtered["strike"] < cost_basis) if cost_basis is not None else False

            if cash_to_invest is not None:
                contracts = int(cash_to_invest // share_capital)
                cash_used = contracts * share_capital
                filtered["contracts"] = contracts
                filtered["cash_used"] = cash_used
                filtered["cash_left"] = cash_to_invest - cash_used
                filtered["tot_premium"] = filtered["fill_dollars"] * contracts
                filtered["premium_over_threshold"] = filtered["tot_premium"] >= premium_threshold
            else:
                # No cash figure to size against, so the threshold applies to a single contract.
                filtered["premium_over_threshold"] = filtered["fill_dollars"] >= premium_threshold

            matches.append(filtered)

        if not matches:
            return None

        all_matches = pd.concat(matches, ignore_index=True)
        all_matches = all_matches[all_matches["premium_over_1"] & all_matches["premium_over_threshold"]].copy()
        if all_matches.empty:
            return None
        best = all_matches.sort_values(
            ["score", "p_hi6_block", "fill_dollars"],
            ascending=[False, True, False]
        ).head(1).copy()

        columns = ["highlight", "symbol", "company_name", "contractSymbol"]
        if cash_to_invest is not None:
            columns += ["contracts"]
        columns += ["fill_dollars"]
        columns += [
            "expiration", "dte", "score",
            "strike", "spot_price", "day_change_pct", "share_capital", "high_6wk",
            "distance_to_spot", "pct_to_spot", "distance_to_6wk_high", "pct_to_6wk_high", "resistance_hits_6wk",
            "oi_wall", "oi_wall_oi", "oi_wall_ok",
            "lastPrice", "bid", "ask", "mid", "fill", "called_dollars", "spread", "volume", "openInterest",
            "iv", "iv_rv_ratio", "iv_rank", "delta", "prob_itm_exp", "prob_touch", "premium_over_1",
            "premium_over_threshold",
            "pct_if_expired", "pct_if_called", "pct_if_expired_ann", "pct_if_called_ann",
            "div_risk", "div_amount", "cost_basis", "below_basis",
        ]
        if cash_to_invest is not None:
            columns += ["cash_used", "cash_left", "tot_premium"]
        return best[columns]
    except Exception as exc:
        print(f"  Skipping {symbol}: {exc}")
        return None


def run_screener(symbols, cash_to_invest=None, iv_threshold=None, iv_rv_ratio_min=None,
                 use_cache=True, assignment_preference=None, premium_threshold=None,
                 min_dte=None, max_dte=None, iv_rank_min=None):
    rows = []
    total = len(symbols)
    cache = load_lookup_cache()
    cache_lock = threading.Lock()
    voldata_cache = voldata.load_cache() if voldata else {}
    voldata_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_symbol = {
            executor.submit(
                best_contract_for_symbol, symbol, cash_to_invest, cache, cache_lock,
                iv_threshold, iv_rv_ratio_min, use_cache, assignment_preference, premium_threshold,
                min_dte, max_dte, iv_rank_min, voldata_cache, voldata_lock,
            ): symbol
            for symbol in symbols
        }
        for completed, future in enumerate(as_completed(future_to_symbol), start=1):
            symbol = future_to_symbol[future]
            print(f"Scanning {completed}/{total}: {symbol}")
            row = future.result()
            if row is not None and not row.empty:
                rows.append(row)
    save_lookup_cache(cache)
    if voldata:
        voldata.save_cache(voldata_cache)

    if not rows:
        return None

    results = pd.concat(rows, ignore_index=True)
    results["p_hi6_block"] = (results["pct_to_6wk_high"] < 0).astype(int)
    results = results.sort_values(
        ["score", "p_hi6_block", "fill_dollars"],
        ascending=[False, True, False]
    ).head(TOP_TICKERS).drop(columns=["p_hi6_block"])
    results = results.rename(columns={
        "highlight": "*",
        "symbol": "sym",
        "company_name": "name",
        "contractSymbol": "contract",
        "expiration": "exp",
        "dte": "dte",
        "strike": "stk",
        "spot_price": "spot",
        "day_change_pct": "day_chg",
        "share_capital": "sh$",
        "high_6wk": "hi6",
        "distance_to_spot": "d_spot",
        "pct_to_spot": "p_spot",
        "distance_to_6wk_high": "d_hi6",
        "pct_to_6wk_high": "p_hi6",
        "resistance_hits_6wk": "hit6",
        "oi_wall": "oi_wall",
        "oi_wall_oi": "oi_wall_oi",
        "oi_wall_ok": "oi_wall_ok",
        "lastPrice": "last",
        "bid": "bid",
        "ask": "ask",
        "mid": "mid",
        "fill": "fill",
        "fill_dollars": "fill$",
        "called_dollars": "called$",
        "spread": "spr",
        "volume": "vol",
        "openInterest": "oi",
        "iv": "iv",
        "iv_rv_ratio": "iv_rv",
        "iv_rank": "iv_rank",
        "score": "score",
        "delta": "delta",
        "prob_itm_exp": "p_itm",
        "prob_touch": "p_touch",
        "premium_over_1": "gt1",
        "premium_over_threshold": "gt1k",
        "pct_if_expired": "p_exp",
        "pct_if_called": "p_call",
        "pct_if_expired_ann": "p_exp_ann",
        "pct_if_called_ann": "p_call_ann",
        "div_risk": "div_risk",
        "div_amount": "div_amt",
        "cost_basis": "basis",
        "below_basis": "below_basis",
        "contracts": "contracts",
        "cash_used": "cash_used",
        "cash_left": "cash_left",
        "tot_premium": "tot_prem",
    })
    # Left numeric on purpose: the CLI colors it at print time and the GUI styles it.
    return results


def print_results(results, symbols, source_label, source_type, cash_to_invest=None,
                  iv_threshold=None, iv_rv_ratio_min=None, assignment_preference=None,
                  premium_threshold=None, min_dte=None, max_dte=None, iv_rank_min=None):
    iv_threshold = IV_THRESHOLD if iv_threshold is None else iv_threshold
    iv_rv_ratio_min = IV_RV_RATIO_MIN if iv_rv_ratio_min is None else iv_rv_ratio_min
    assignment_preference = ASSIGNMENT_PREFERENCE if assignment_preference is None else assignment_preference
    premium_threshold = TOTAL_PREMIUM_THRESHOLD if premium_threshold is None else premium_threshold
    min_dte = MIN_DTE if min_dte is None else min_dte
    max_dte = MAX_DTE if max_dte is None else max_dte
    iv_rank_min = IV_RANK_MIN if iv_rank_min is None else iv_rank_min
    print()
    print(f"Source: {source_label}")
    if source_type == "default":
        print("Ticker file not found or empty, using hardcoded defaults.")
    print(f"Universe size: {len(symbols)}")
    iv_rv_desc = f"IV/RV >= {iv_rv_ratio_min:.2f}" if iv_rv_ratio_min > 0 else "IV/RV off"
    iv_rank_desc = f"IV rank >= {iv_rank_min:.0f}" if iv_rank_min > 0 else "IV rank off"
    print(f"Screen: {min_dte}-{max_dte} DTE, IV > {iv_threshold:.2f}, {iv_rv_desc}, {iv_rank_desc}, "
          f"delta {DELTA_MIN:.2f}-{DELTA_MAX:.2f}, "
          f"100-share cost <= ${MAX_SHARE_CAPITAL:,.0f}, OI >= {MIN_OPEN_INTEREST}, "
          f"vol >= {MIN_VOLUME}, spread <= {MAX_SPREAD_PCT * 100:.0f}% of mid, "
          f"earnings-window expirations excluded")
    score_desc = ("(1-p) x ann. if expired + p x ann. if called  [neutral on assignment]"
                  if assignment_preference == "neutral"
                  else "ann. return if expired x (1 - prob ITM)  [prefers keeping shares]")
    print(f"Ranked by score = {score_desc}")
    print(f"Premiums modeled at mid minus {FILL_HAIRCUT:.0%} of the half-spread (column 'fill')")
    if cash_to_invest is not None:
        print(f"Cash to invest: ${cash_to_invest:,.2f} (tickers requiring more than this per contract are excluded)")
        print(f"Total premium (fill x contracts that cash affords) must be >= ${premium_threshold:,.2f}")
    else:
        print(f"Premium per contract must be >= ${premium_threshold:,.2f} (no cash figure given, so this is per-contract)")
    print()

    if results is None or results.empty:
        print("No tickers matched the current screen.")
        return

    results = chain_mod.reorder_columns(results)

    print("Best contract per ticker:")
    numeric_cols = [
        "dte", "score", "stk", "spot", "day_chg", "sh$", "hi6", "d_spot", "p_spot", "d_hi6", "p_hi6", "hit6",
        "oi_wall", "oi_wall_oi", "last", "bid", "ask", "mid", "fill", "fill$", "called$", "spr", "vol", "oi",
        "iv", "iv_rv", "iv_rank", "delta", "p_itm", "p_touch",
        "p_exp", "p_call", "p_exp_ann", "p_call_ann", "div_amt", "basis",
        "contracts", "cash_used", "cash_left", "tot_prem",
    ]
    best_score = results["score"].max() if "score" in results.columns else None
    # Column names differ from chain.py's; map them onto the shared decision semantics.
    alias = {"p_exp_ann": "pct_exp_ann", "p_itm": "prob_itm", "p_hi6": "pct_hi6"}
    chain_mod.print_colored_table(
        results, numeric_cols,
        lambda col, value: chain_mod.candidate_cell_color(alias.get(col, col), value, best_score),
        min_width=8, na_rep="",
    )
    print()
    print("Read these first: "
          f"{chain_mod.ANSI_GREEN}score{chain_mod.ANSI_RESET} (best trade) | "
          f"p_exp_ann (annualized income) | p_itm (assignment risk) | "
          f"iv_rv (is premium rich vs. recent moves?) | "
          f"iv_rank (is premium rich vs. this stock's own year?) | "
          f"p_hi6 (strike vs nearest resistance zone) | "
          f"oi (can you fill it?) | oi_wall_ok (strike beyond the heaviest call OI) | "
          f"div_risk (early assignment) | "
          f"below_basis (locks in a stock loss if assigned)")
    print(f"{chain_mod.ANSI_GREEN}green{chain_mod.ANSI_RESET} = favorable   "
          f"{chain_mod.ANSI_YELLOW}amber{chain_mod.ANSI_RESET} = check it   "
          f"{chain_mod.ANSI_RED}red{chain_mod.ANSI_RESET} = warning")


def print_usage():
    print(f"""Usage:
  python screen_oom.py [TICKER_FILE] [CASH_TO_INVEST] [--iv N] [--iv-rv N] [--iv-rank N] [--premium N] [--min-dte N] [--max-dte N] [--refresh]
      Scan a watchlist for the best covered-call candidate per ticker.
      TICKER_FILE      One symbol per line, # for comments (default {DEFAULT_TICKER_FILE}).
      CASH_TO_INVEST   Excludes tickers costing more than this per contract.
      --iv N           Absolute implied-vol floor (default {IV_THRESHOLD:.2f}).
      --iv-rv N        Minimum implied/realized vol ratio (default {IV_RV_RATIO_MIN:.2f}).
                       Use 0 to disable the ratio test entirely.
      --iv-rank N      Minimum IV rank, 0-100 (default {IV_RANK_MIN}): where today's IV
                       sits in each stock's own 52-week IV range. Sourced from a
                       community-maintained dataset (dolthub.com/post-no-preference/
                       options) - if it's unreachable, iv_rank is blank for every
                       ticker and never excludes one regardless of this setting.
                       0 disables.
      --premium N      Minimum total premium in dollars (default {TOTAL_PREMIUM_THRESHOLD:,.2f}).
                       With CASH_TO_INVEST given, this is fill x contracts that cash
                       affords; otherwise it's a single contract's premium. Use 0 to
                       disable.
      --min-dte N      Minimum days to expiration (default {MIN_DTE}).
      --max-dte N      Maximum days to expiration (default {MAX_DTE}).
      --assign MODE    'keep' (default) ranks to avoid assignment; 'neutral' ranks
                       by expected value across both outcomes.
      --refresh        Force a refetch. Chain data is otherwise cached for
                       {CHAIN_CACHE_TTL_MINUTES} minutes, so re-screening the same
                       universe with different thresholds costs no Yahoo API calls.
      Example: python screen_oom.py sp500.txt 50000
      Example: python screen_oom.py sp500.txt --iv 0.60 --iv-rv 0
      Example: python screen_oom.py sp500.txt 50000 --premium 500
      Example: python screen_oom.py sp500.txt --min-dte 10 --max-dte 21
      Example: python screen_oom.py sp500.txt --iv-rank 50
""")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("help", "-h", "--help"):
        print_usage()
    else:
        try:
            iv_threshold, iv_rv_ratio_min, iv_rank_min, premium_threshold, min_dte, max_dte, \
                use_cache, assignment_preference = parse_threshold_flags(sys.argv)
        except ValueError as exc:
            print(exc)
            sys.exit(1)
        symbols, source_label, source_type = get_symbols()
        cash_to_invest = get_cash_to_invest()
        results = run_screener(symbols, cash_to_invest, iv_threshold, iv_rv_ratio_min,
                               use_cache, assignment_preference, premium_threshold,
                               min_dte, max_dte, iv_rank_min)
        print_results(results, symbols, source_label, source_type, cash_to_invest,
                      iv_threshold, iv_rv_ratio_min, assignment_preference, premium_threshold,
                      min_dte, max_dte, iv_rank_min)