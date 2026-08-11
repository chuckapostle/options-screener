"""Optional enrichment from a community-maintained third-party dataset:
https://www.dolthub.com/repositories/post-no-preference/options

This is outside our control - the repo could go private, rename its tables, or
disappear entirely at any point. Every public function here is designed to
fail soft: on ANY error (network, HTTP, malformed response, symbol the dataset
has never heard of) it returns the same "unavailable" shape and never raises.
chain.py and screen_oom.py treat a missing iv_rank exactly like a missing
iv_rv - it never disqualifies a candidate unless a threshold is explicitly
set (IV_RANK_MIN in config.py, default 0 = informational only).

If this dataset ever disappears for good: delete this file, drop the
`import voldata` / `fetch_iv_rank` call sites in chain.py, and remove
IV_RANK_* from config.py. Nothing else in the app depends on it.
"""
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

from config import IV_RANK_CACHE_TTL_HOURS

DOLTHUB_OWNER = "post-no-preference"
DOLTHUB_REPO = "options"
DOLTHUB_BRANCH = "master"
DOLTHUB_API_URL = f"https://www.dolthub.com/api/v1alpha1/{DOLTHUB_OWNER}/{DOLTHUB_REPO}/{DOLTHUB_BRANCH}"
REQUEST_TIMEOUT_SECONDS = 10

CACHE_FILE = Path(__file__).resolve().parent / "voldata_cache.json"

# Tickers only ever look like this; reject anything else before it goes near
# a hand-built SQL string (this API takes SQL as a query param, not bind params).
_SYMBOL_RE = re.compile(r"^[A-Z0-9.\-]{1,10}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

UNAVAILABLE = {
    "available": False,
    "as_of": None,
    "iv_current": None,
    "iv_rank": None,
    "iv_year_high": None,
    "iv_year_low": None,
    "hv_current": None,
}


def _run_query(sql, retries=1, retry_delay=1.5):
    """GET a read-only query against the public DoltHub SQL API. Returns the
    row list on success, or None if every attempt fails - bad status,
    timeout, connection error, unexpected JSON shape. Never raises.

    retries>1 is for the wider-range queries backtesting uses (e.g.
    fetch_iv_rank_history over several months) - measured directly to be
    meaningfully more prone to transient timeouts than the small single-day/
    single-symbol queries the live screener makes (which keep the default
    of no retry, since most of their "failures" are genuinely no data for
    that day, not a network hiccup worth retrying).
    """
    for attempt in range(retries):
        try:
            resp = requests.get(DOLTHUB_API_URL, params={"q": sql}, timeout=REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("query_execution_status") == "Success":
                return payload.get("rows")
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(retry_delay)
    return None


def _to_float(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _compute_iv_rank(iv_current, iv_year_low, iv_year_high):
    """0-100: where iv_current sits within this stock's OWN trailing 52-week
    IV range. Distinct from IV/RV (which compares IV to recent realized
    moves) - this is the standard "is it a good time to sell premium on THIS
    stock" timing signal, and needs no local price history to compute."""
    if iv_current is None or iv_year_low is None or iv_year_high is None:
        return None
    spread = iv_year_high - iv_year_low
    if spread <= 0:
        return None
    return max(0.0, min(100.0, (iv_current - iv_year_low) / spread * 100))


def load_cache():
    if CACHE_FILE.exists():
        try:
            with CACHE_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache):
    try:
        with CACHE_FILE.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def _cache_entry_fresh(entry):
    if not entry or "fetched_at" not in entry:
        return False
    fetched_at = pd.to_datetime(entry["fetched_at"])
    age_hours = (pd.Timestamp.now(tz="UTC") - fetched_at).total_seconds() / 3600
    return age_hours < IV_RANK_CACHE_TTL_HOURS


def get_iv_rank_context(symbol, cache, cache_lock, use_cache=True):
    """Best-effort IV-rank context for `symbol`. Always returns a dict shaped
    like UNAVAILABLE (available=False, every value None) if the dataset is
    unreachable or has never heard of this symbol - callers never need to
    special-case a broken data source, only check `available`.

    `cache`/`cache_lock` are a plain dict and threading.Lock the caller owns
    (mirrors the lookup_cache pattern already used for company/earnings data),
    so a multi-symbol scan shares one cache across threads instead of hitting
    the network once per contract.
    """
    symbol = symbol.strip().upper()
    if not _SYMBOL_RE.match(symbol):
        return dict(UNAVAILABLE)

    with cache_lock:
        entry = cache.get(symbol)
        fresh = use_cache and _cache_entry_fresh(entry)
        cached_result = dict(entry["result"]) if fresh else None
    if cached_result is not None:
        return cached_result

    sql = (
        "SELECT date, iv_current, iv_year_high, iv_year_low, hv_current "
        f"FROM volatility_history WHERE act_symbol='{symbol}' "
        "ORDER BY date DESC LIMIT 1"
    )
    rows = _run_query(sql)
    if not rows:
        result = dict(UNAVAILABLE)
    else:
        row = rows[0]
        iv_current = _to_float(row.get("iv_current"))
        iv_year_high = _to_float(row.get("iv_year_high"))
        iv_year_low = _to_float(row.get("iv_year_low"))
        result = {
            "available": True,
            "as_of": row.get("date"),
            "iv_current": iv_current,
            "iv_rank": _compute_iv_rank(iv_current, iv_year_low, iv_year_high),
            "iv_year_high": iv_year_high,
            "iv_year_low": iv_year_low,
            "hv_current": _to_float(row.get("hv_current")),
        }

    with cache_lock:
        cache[symbol] = {"fetched_at": pd.Timestamp.now(tz="UTC").isoformat(), "result": result}
    return result


# --- Backtesting (used by backtest.py) ----------------------------------------
#
# IMPORTANT query-shape constraint, found by testing the live API: option_chain
# is keyed (date, act_symbol, expiration, strike, call_put) - date leads, not
# symbol - so a multi-date RANGE query has to scan every symbol's rows across
# the whole range before it can filter down to one symbol, and the free query
# API times out on that even for a ~40-day window. A single EXACT date + symbol
# query is fast (~0.5-1s) because it only has to filter within that one date's
# slice. backtest.py therefore always calls this with start_date == end_date,
# looping one trading day at a time - never with a wide date range.
# volatility_history (fetch_iv_rank_history below) is a much smaller table and
# tolerates multi-year single-symbol range queries fine - no such constraint
# there.
def fetch_option_chain_history(symbol, start_date=None, end_date=None,
                               call_put=None, min_dte=None, max_dte=None, limit=5000):
    """Historical EOD contract rows for `symbol`, optionally bounded by
    [start_date, end_date] (ISO 'YYYY-MM-DD' strings), and optionally
    restricted to `call_put` ('Call'/'Put') and a DTE window (`min_dte`/
    `max_dte`, inclusive - evaluated server-side as
    `DATEDIFF(expiration, date) BETWEEN min_dte AND max_dte`, i.e. relative to
    each row's own `date`, not "today"). Returns a list of row dicts (date,
    expiration, strike, call_put, bid, ask, vol, delta, gamma, theta, vega,
    rho), or None if the data source is unavailable or an argument is
    malformed. EOD and lagged by about a trading day - this is a research
    feed for backtesting, never a substitute for a live quote.

    Pass a single exact date (start_date == end_date) for backtesting - see
    the module-level note above for why a wide range isn't safe here.
    """
    symbol = symbol.strip().upper()
    if not _SYMBOL_RE.match(symbol):
        return None

    clauses = [f"act_symbol='{symbol}'"]
    for op, value in ((">=", start_date), ("<=", end_date)):
        if value is None:
            continue
        if not _DATE_RE.match(value):
            return None
        clauses.append(f"date {op} '{value}'")
    if call_put is not None:
        if call_put not in ("Call", "Put"):
            return None
        clauses.append(f"call_put='{call_put}'")
    if min_dte is not None and max_dte is not None:
        clauses.append(f"DATEDIFF(expiration, date) BETWEEN {int(min_dte)} AND {int(max_dte)}")

    sql = (
        "SELECT date, expiration, strike, call_put, bid, ask, vol, delta, gamma, theta, vega, rho "
        f"FROM option_chain WHERE {' AND '.join(clauses)} "
        f"ORDER BY date, expiration, strike LIMIT {int(limit)}"
    )
    return _run_query(sql)


def fetch_iv_rank_history(symbol, start_date, end_date):
    """Daily IV-rank/IV/HV series for `symbol` over [start_date, end_date]
    (ISO 'YYYY-MM-DD' strings, both required). One range query against
    volatility_history - unlike option_chain above, this table tolerates
    multi-year single-symbol range queries fine. Returns a list of
    {date, iv_rank, iv_current, hv_current} dicts (iv_rank computed the same
    way get_iv_rank_context does), or None on failure/malformed input.
    """
    symbol = symbol.strip().upper()
    if not _SYMBOL_RE.match(symbol):
        return None
    if not (_DATE_RE.match(start_date) and _DATE_RE.match(end_date)):
        return None

    sql = (
        "SELECT date, iv_current, iv_year_high, iv_year_low, hv_current FROM volatility_history "
        f"WHERE act_symbol='{symbol}' AND date >= '{start_date}' AND date <= '{end_date}' "
        "ORDER BY date"
    )
    rows = _run_query(sql, retries=3)
    if rows is None:
        return None

    series = []
    for row in rows:
        iv_current = _to_float(row.get("iv_current"))
        iv_year_high = _to_float(row.get("iv_year_high"))
        iv_year_low = _to_float(row.get("iv_year_low"))
        series.append({
            "date": row.get("date"),
            "iv_rank": _compute_iv_rank(iv_current, iv_year_low, iv_year_high),
            "iv_current": iv_current,
            "hv_current": _to_float(row.get("hv_current")),
        })
    return series
