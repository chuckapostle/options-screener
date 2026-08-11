import ctypes
import json
import math
import os
import pickle
import re
import sys
import threading
from pathlib import Path
from statistics import NormalDist

import pandas as pd
import yfinance as yf


def enable_ansi_on_windows():
    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


enable_ansi_on_windows()

# Screening criteria live in config.py so this and screen_oom.py cannot drift apart.
from config import (  # noqa: E402
    MIN_DTE, MAX_DTE, IV_THRESHOLD, IV_RV_RATIO_MIN, TRADING_DAYS_PER_YEAR,
    RV_SHORT_TRADING_DAYS, RV_LONG_TRADING_DAYS, RV_SHORT_WEIGHT,
    TARGET_DELTA, DELTA_MIN, DELTA_MAX, PREMIUM_THRESHOLD,
    MIN_OPEN_INTEREST, MAX_SPREAD_PCT, MIN_VOLUME, FILL_HAIRCUT,
    RISK_FREE_RATE, ASSIGNMENT_PREFERENCE,
    TAKE_PROFIT_PCT, MANAGE_DTE, ROLL_WATCH_DELTA, ROLL_WATCH_PROXIMITY_PCT,
    CACHE_TTL_HOURS, CHAIN_CACHE_TTL_MINUTES, IV_RANK_MIN,
    RESISTANCE_LOOKBACK_DAYS, RESISTANCE_SWING_WINDOW, RESISTANCE_CLUSTER_PCT,
)

# voldata wraps a third-party community dataset and is entirely optional - if
# it fails to import (missing dependency, file removed) the screen still runs
# with a blank iv_rank column. See voldata.py for the full story.
try:
    import voldata
except Exception:
    voldata = None

_IV_RANK_UNAVAILABLE = {
    "available": False, "as_of": None, "iv_current": None,
    "iv_rank": None, "iv_year_high": None, "iv_year_low": None, "hv_current": None,
}


def fetch_iv_rank(symbol, cache, cache_lock):
    """Thin, never-raising wrapper around the optional voldata module. Returns
    the same "unavailable" shape whether voldata failed to import, the
    network is down, or the community dataset just doesn't cover this symbol
    - callers only ever need to check result["available"]."""
    if voldata is None:
        return dict(_IV_RANK_UNAVAILABLE)
    try:
        return voldata.get_iv_rank_context(symbol, cache, cache_lock)
    except Exception:
        return dict(_IV_RANK_UNAVAILABLE)

DEFAULT_SYMBOL = "AMD"
TOP_ROWS = 10
POSITIONS_FILE = Path(__file__).resolve().parent / "positions.csv"
POSITIONS_COLUMNS = [
    "entry_date", "symbol", "contract", "option_type", "strike", "expiration",
    "contracts", "shares", "stock_price", "premium_collected",
    "status", "close_date", "close_premium", "realized_pl",
    # Assignment closes the stock leg too, at the strike. Blank for close/roll,
    # where the shares are still held and nothing is realized on them yet.
    "stock_realized_pl",
    # Screen metrics captured at entry, so you can later test which criteria
    # actually predicted good outcomes. Blank if the lookup failed.
    "entry_score", "entry_iv", "entry_iv_rv", "entry_delta", "entry_prob_itm", "entry_pct_hi6",
    # lot_id groups every leg against the same continuous share-holding - the
    # source of truth for cumulative returns on a stock (see get_or_open_lot),
    # robust to gaps where you closed and later logged a fresh call instead of
    # rolling. rolled_from/cumulative_premium remain a finer-grained record of
    # same-day roll chains within a lot.
    "lot_id", "rolled_from", "cumulative_premium",
]
LOOKUP_CACHE_FILE = Path(__file__).resolve().parent / "lookup_cache.json"
# Raw option chains are cached so that re-running with different thresholds
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


def modeled_fill_price(bid, ask):
    """Realistic sellable price: partway from mid toward the bid.

    Selling at the exact midpoint is optimistic, and the error scales with the
    spread. Works elementwise on scalars or Series.
    """
    mid = (bid + ask) / 2
    return mid - FILL_HAIRCUT * ((ask - bid) / 2)


def compute_score(ann_if_expired, ann_if_called, prob_itm, preference=None):
    """Ranking score under an explicit assignment preference.

    "keep"    - assignment is a bad outcome, so weight only the expire-worthless
                branch by its probability.
    "neutral" - true expected value across both branches. Favors closer-to-money
                strikes, since being called away usually pays more than expiring
                worthless; it also ignores upside forgone above the strike.

    Vectorized: accepts scalars or Series.
    """
    preference = ASSIGNMENT_PREFERENCE if preference is None else preference
    p = prob_itm / 100.0
    if preference == "neutral":
        return (1 - p) * ann_if_expired + p * ann_if_called
    return ann_if_expired * (1 - p)


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


def extract_int_flag(argv, flag):
    """Pull `--flag VALUE` out of argv (mutating it) and return int(VALUE), or None."""
    if flag not in argv:
        return None
    i = argv.index(flag)
    if i + 1 >= len(argv):
        raise ValueError(f"{flag} requires an integer value")
    try:
        value = int(argv[i + 1])
    except ValueError:
        raise ValueError(f"{flag} requires an integer value, got '{argv[i + 1]}'")
    del argv[i:i + 2]
    return value


def extract_bool_flag(argv, flag):
    if flag in argv:
        argv.remove(flag)
        return True
    return False


def extract_choice_flag(argv, flag, choices):
    if flag not in argv:
        return None
    i = argv.index(flag)
    if i + 1 >= len(argv):
        raise ValueError(f"{flag} requires one of: {', '.join(choices)}")
    value = argv[i + 1].strip().lower()
    if value not in choices:
        raise ValueError(f"{flag} must be one of: {', '.join(choices)} (got '{argv[i + 1]}')")
    del argv[i:i + 2]
    return value


def parse_threshold_flags(argv):
    """Strip --iv / --iv-rv / --iv-rank / --min-dte / --max-dte / --assign /
    --refresh from argv, return effective settings."""
    iv = extract_float_flag(argv, "--iv")
    iv_rv = extract_float_flag(argv, "--iv-rv")
    iv_rank = extract_float_flag(argv, "--iv-rank")
    min_dte = extract_int_flag(argv, "--min-dte")
    max_dte = extract_int_flag(argv, "--max-dte")
    assign = extract_choice_flag(argv, "--assign", ("keep", "neutral"))
    refresh = extract_bool_flag(argv, "--refresh")
    return (
        IV_THRESHOLD if iv is None else iv,
        IV_RV_RATIO_MIN if iv_rv is None else iv_rv,
        IV_RANK_MIN if iv_rank is None else iv_rank,
        MIN_DTE if min_dte is None else min_dte,
        MAX_DTE if max_dte is None else max_dte,
        not refresh,
        ASSIGNMENT_PREFERENCE if assign is None else assign,
    )


def get_symbol():
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return sys.argv[1].strip().upper()
    return DEFAULT_SYMBOL


def get_cash_to_invest():
    if len(sys.argv) > 2 and sys.argv[2].strip():
        return float(sys.argv[2].strip())
    return None


def parse_occ_contract(contract):
    match = re.match(r"^([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$", contract.strip().upper())
    if not match:
        raise ValueError(f"Could not parse option contract symbol: {contract}")
    root, yy, mm, dd, opt_type, strike_raw = match.groups()
    expiration = f"20{yy}-{mm}-{dd}"
    strike = int(strike_raw) / 1000.0
    return root, expiration, opt_type, strike


def backfill_lot_ids(df):
    """One-time migration for rows written before lot_id existed. For any
    symbol with at least one blank lot_id, recompute lot_id for that symbol's
    ENTIRE history from scratch, walking rows in NATURAL (append) order - not
    resorted by entry_date, which only has day granularity and ties whenever
    two legs land on the same calendar day (e.g. a roll immediately followed
    by an assign), which would make a date-based sort pick an unpredictable
    order. Rows are always appended in true chronological order as trades
    happen, so natural order is the reliable signal, exactly like
    cmd_close/cmd_assign/cmd_roll's own `df[mask].index[-1]` pattern. A lot
    continues across consecutive legs unless the previous one was 'assigned'
    (shares gone - a fresh purchase starts a new lot). Idempotent - once
    every row has a lot_id, this is a no-op. Returns (df, changed: bool).
    """
    df["lot_id"] = df["lot_id"].fillna("").astype(str)
    blank = df["lot_id"].str.strip() == ""
    if not blank.any():
        return df, False

    df = df.copy()
    for symbol in df.loc[blank, "symbol"].dropna().unique():
        lot_num = 0
        lot_open = False
        for i in df.index[df["symbol"] == symbol]:
            if not lot_open:
                lot_num += 1
            df.at[i, "lot_id"] = f"{symbol}-{lot_num}"
            lot_open = df.at[i, "status"] != "assigned"
    return df, True


def get_or_open_lot(df, symbol, force_new=False):
    """lot_id for a new leg on `symbol`: continues the most recent lot if its
    shares are still held (the last leg on it wasn't 'assigned'), otherwise
    opens a new one - a fresh purchase starts a new lot. Pass force_new=True
    to always open a new lot regardless (e.g. you sold the shares outright,
    outside this tool, without ever running `assign` - this tool has no way
    to detect that on its own).

    "Most recent" means the last matching row by DataFrame position, not by
    parsing entry_date - entry_date only has day granularity, so same-day
    legs (a roll then an assign, say) tie under a date sort. Rows are always
    appended in true chronological order, so position is the reliable
    signal - same convention as cmd_close/cmd_assign/cmd_roll's own
    `df[mask].index[-1]`.
    """
    symbol_rows = df[df["symbol"] == symbol]
    if symbol_rows.empty:
        return f"{symbol}-1"

    last_row = symbol_rows.iloc[-1]
    last_lot_id = str(last_row["lot_id"] or f"{symbol}-1")
    last_status = last_row["status"]

    if force_new or last_status == "assigned":
        try:
            n = int(last_lot_id.rsplit("-", 1)[-1])
        except ValueError:
            n = symbol_rows["lot_id"].nunique()
        return f"{symbol}-{n + 1}"
    return last_lot_id


def lot_cumulative_premium(df, lot_id, new_premium):
    """Net premium economics of `lot_id` so far: realized P&L of every
    already-closed leg in the lot, plus the fresh premium just collected on
    the newly opened leg. Works whether legs were connected via roll or via
    an independent close-then-log with a gap in between - lot_id is the
    single source of truth now, not the rolled_from chain."""
    lot_rows = df[df["lot_id"] == lot_id]
    prior_realized = pd.to_numeric(lot_rows["realized_pl"], errors="coerce").fillna(0).sum()
    return float(prior_realized) + float(new_premium)


def load_positions():
    if not POSITIONS_FILE.exists():
        return pd.DataFrame(columns=POSITIONS_COLUMNS)
    df = pd.read_csv(POSITIONS_FILE, dtype={"contract": str})
    # Files written before newer columns existed still load; missing fields blank.
    for col in POSITIONS_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    # An all-empty text column reads back as float64; writing a string into it then
    # raises a pandas dtype FutureWarning (an error in future versions).
    for col in ["entry_date", "symbol", "contract", "option_type", "expiration",
                "status", "close_date", "rolled_from", "lot_id"]:
        df[col] = df[col].astype(object)
    extras = [c for c in df.columns if c not in POSITIONS_COLUMNS]
    df = df[POSITIONS_COLUMNS + extras]

    if not df.empty:
        df, changed = backfill_lot_ids(df)
        if changed:
            save_positions(df)
            print(f"(one-time migration: backfilled lot_id across {df['symbol'].nunique()} "
                  f"symbol(s) in positions.csv - see 'lots' command)")
    return df


def save_positions(df):
    df.to_csv(POSITIONS_FILE, index=False)


def cmd_log(args):
    args = list(args)
    force_new_lot = extract_bool_flag(args, "--new-lot")
    if len(args) != 6:
        print("Usage: python chain.py log SYMBOL CONTRACT STOCK_PRICE SHARES PREMIUM_COLLECTED CONTRACTS [--new-lot]")
        print("  --new-lot forces a fresh lot even if you still hold shares from an earlier")
        print("  entry on this symbol - use when you sold those shares outside this tool")
        print("  (e.g. a manual broker sale) without ever running `assign`.")
        return

    symbol, contract, stock_price, shares, premium_collected, contracts = args
    contract = contract.strip().upper()
    try:
        root, expiration, opt_type, strike = parse_occ_contract(contract)
    except ValueError as exc:
        print(exc)
        return

    if root != symbol.strip().upper():
        print(f"Warning: contract root '{root}' does not match symbol '{symbol.strip().upper()}'")

    symbol = symbol.strip().upper()
    premium_collected = float(premium_collected)
    df = load_positions()
    lot_id = get_or_open_lot(df, symbol, force_new=force_new_lot)
    continuing_lot = not df.empty and (df["lot_id"] == lot_id).any()

    row = {
        "entry_date": pd.Timestamp.now(tz="US/Pacific").date().isoformat(),
        "symbol": symbol,
        "contract": contract,
        "option_type": opt_type,
        "strike": strike,
        "expiration": expiration,
        "contracts": int(contracts),
        "shares": int(shares),
        "stock_price": float(stock_price),
        "premium_collected": premium_collected,
        "status": "open",
        "close_date": "",
        "close_premium": "",
        "realized_pl": "",
        "lot_id": lot_id,
        "rolled_from": "",
        "cumulative_premium": lot_cumulative_premium(df, lot_id, premium_collected),
    }
    row.update(get_contract_metrics(symbol, contract))

    new_row = pd.DataFrame([row])
    df = new_row if df.empty else pd.concat([df, new_row], ignore_index=True)
    save_positions(df)
    print(f"Logged position: {row['symbol']} {row['contract']} (strike {row['strike']}, exp {row['expiration']})")
    if continuing_lot:
        print(f"  Lot: {lot_id} (continuing - cumulative premium across this lot so far: "
              f"{row['cumulative_premium']:.2f})")
    else:
        print(f"  Lot: {lot_id} (new)")
    if row.get("entry_score") != "":
        print(f"  Entry metrics: score={row['entry_score']} iv={row['entry_iv']} "
              f"iv_rv={row['entry_iv_rv']} delta={row['entry_delta']} "
              f"prob_itm={row['entry_prob_itm']} pct_hi6={row['entry_pct_hi6']}")
    else:
        print("  Entry metrics unavailable (contract not in current chain) - recorded blank.")


def cmd_close(args):
    if len(args) != 2:
        print("Usage: python chain.py close CONTRACT CLOSE_PREMIUM")
        print("  CLOSE_PREMIUM is the total $ paid to close (0 if it expired worthless).")
        return

    contract, close_premium = args
    contract = contract.strip().upper()
    df = load_positions()
    mask = (df["contract"] == contract) & (df["status"] == "open")
    if not mask.any():
        print(f"No open position found for contract {contract}")
        return

    idx = df[mask].index[-1]
    close_premium = float(close_premium)
    premium_collected = float(df.at[idx, "premium_collected"])
    realized_pl = premium_collected - close_premium

    df.at[idx, "status"] = "closed"
    df.at[idx, "close_date"] = pd.Timestamp.now(tz="US/Pacific").date().isoformat()
    df.at[idx, "close_premium"] = close_premium
    df.at[idx, "realized_pl"] = realized_pl
    save_positions(df)
    print(f"Closed {contract}: realized P&L = {realized_pl:.2f}")
    print("  (Option leg only - you still hold the shares.)")


def cmd_assign(args):
    """Record the call being exercised and the shares called away.

    Assignment closes BOTH legs, which is why it is not the same as `close`:
      - option: you pay nothing to close, so you keep 100% of the premium
      - stock:  the shares are sold at the strike, realizing gain or loss
                against your cost basis
    Recording this as a plain `close` would silently discard the stock leg,
    which is usually the larger number.
    """
    if len(args) != 1:
        print("Usage: python chain.py assign CONTRACT")
        print("  Use when the call was exercised and your shares were called away.")
        print("  No price argument: the sale price is the strike, by definition.")
        return

    contract = args[0].strip().upper()
    df = load_positions()
    mask = (df["contract"] == contract) & (df["status"] == "open")
    if not mask.any():
        print(f"No open position found for contract {contract}")
        return

    idx = df[mask].index[-1]
    premium_collected = float(df.at[idx, "premium_collected"])
    strike = float(df.at[idx, "strike"])
    stock_price = float(df.at[idx, "stock_price"])
    shares = int(df.at[idx, "shares"])

    option_pl = premium_collected          # nothing paid to close
    stock_pl = (strike - stock_price) * shares
    total = option_pl + stock_pl

    df.at[idx, "status"] = "assigned"
    df.at[idx, "close_date"] = pd.Timestamp.now(tz="US/Pacific").date().isoformat()
    df.at[idx, "close_premium"] = 0.0
    df.at[idx, "realized_pl"] = option_pl
    df.at[idx, "stock_realized_pl"] = stock_pl
    save_positions(df)

    print(f"Assigned {contract}: shares called away at {strike:.2f}")
    print(f"  Option leg (premium kept in full): {option_pl:+.2f}")
    print(f"  Stock leg ({shares} sh: {strike:.2f} - {stock_price:.2f} basis): {stock_pl:+.2f}")
    print(f"  Total realized: {total:+.2f}")


def cmd_roll(args):
    """Close the current call and open a replacement as one linked pair.

    Rolling is how most covered calls actually end, so the two legs are kept
    connected (rolled_from) and premium is accumulated across the whole chain
    rather than each leg looking like an unrelated trade.
    """
    if len(args) != 4:
        print("Usage: python chain.py roll OLD_CONTRACT CLOSE_PREMIUM NEW_CONTRACT NEW_PREMIUM")
        print("  CLOSE_PREMIUM is the total $ paid to buy back the old call.")
        print("  NEW_PREMIUM is the total $ received for the new call.")
        return

    old_contract, close_premium, new_contract, new_premium = args
    old_contract = old_contract.strip().upper()
    new_contract = new_contract.strip().upper()

    try:
        root, expiration, opt_type, strike = parse_occ_contract(new_contract)
    except ValueError as exc:
        print(exc)
        return

    df = load_positions()
    mask = (df["contract"] == old_contract) & (df["status"] == "open")
    if not mask.any():
        print(f"No open position found for contract {old_contract}")
        return

    idx = df[mask].index[-1]
    close_premium = float(close_premium)
    new_premium = float(new_premium)
    old_premium = float(df.at[idx, "premium_collected"])
    realized_pl = old_premium - close_premium

    # Close the old leg.
    df.at[idx, "status"] = "rolled"
    df.at[idx, "close_date"] = pd.Timestamp.now(tz="US/Pacific").date().isoformat()
    df.at[idx, "close_premium"] = close_premium
    df.at[idx, "realized_pl"] = realized_pl

    # The stock leg is untouched by a roll, so cost basis, share count, and
    # lot_id all carry over - a roll always continues the same lot.
    symbol = str(df.at[idx, "symbol"])
    stock_price = float(df.at[idx, "stock_price"])
    lot_id = str(df.at[idx, "lot_id"] or f"{symbol}-1")
    new_row = {
        "entry_date": pd.Timestamp.now(tz="US/Pacific").date().isoformat(),
        "symbol": symbol,
        "contract": new_contract,
        "option_type": opt_type,
        "strike": strike,
        "expiration": expiration,
        "contracts": int(df.at[idx, "contracts"]),
        "shares": int(df.at[idx, "shares"]),
        "stock_price": stock_price,
        "premium_collected": new_premium,
        "status": "open",
        "close_date": "",
        "close_premium": "",
        "realized_pl": "",
        "lot_id": lot_id,
        "rolled_from": old_contract,
        "cumulative_premium": lot_cumulative_premium(df, lot_id, new_premium),
    }
    new_row.update(get_contract_metrics(symbol, new_contract))

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_positions(df)

    net_credit = new_premium - close_premium
    print(f"Rolled {old_contract} -> {new_contract}")
    print(f"  Closed old leg: realized {realized_pl:+.2f}")
    print(f"  Net credit on roll: {net_credit:+.2f}")
    print(f"  Lot: {lot_id}  cumulative premium: {new_row['cumulative_premium']:.2f}")
    if net_credit < 0:
        print(f"  {ANSI_RED}Warning: this is a net DEBIT of {abs(net_credit):.2f} - you paid to roll "
              f"rather than collecting a credit. Fine if you're deliberately defending the "
              f"position, but a debit roll chasing a loser rarely recovers it.{ANSI_RESET}")
    if strike < stock_price:
        print(f"  {ANSI_RED}Warning: new strike {strike:.2f} is BELOW your cost basis "
              f"{stock_price:.2f} - assignment would realize a stock loss.{ANSI_RESET}")
    if root != symbol:
        print(f"  Warning: new contract root '{root}' does not match position symbol '{symbol}'")


def fetch_current_option_quote(symbol, expiration, opt_type, contract):
    """Current mid and IV for one contract, or (None, None) if unavailable.

    IV is needed to compute live delta for the roll-watch warning.
    """
    try:
        # yfinance requires ISO YYYY-MM-DD. positions.csv can come back in another
        # format (e.g. Excel rewrites it as M/D/YYYY), which would otherwise make
        # this lookup fail silently and blank out every live position value.
        expiration = pd.to_datetime(expiration).strftime("%Y-%m-%d")
        ticker = yf.Ticker(symbol)
        chain = ticker.option_chain(expiration)
        table = chain.calls if opt_type == "C" else chain.puts
        match = table[table["contractSymbol"] == contract]
        if match.empty:
            return None, None
        row = match.iloc[0]
        mid = (float(row["bid"]) + float(row["ask"])) / 2
        iv = float(row["impliedVolatility"]) if pd.notna(row["impliedVolatility"]) else None
        return mid, iv
    except Exception:
        return None, None


def get_open_positions(symbol=None):
    df = load_positions()
    open_df = df[df["status"] == "open"]
    if symbol is not None:
        open_df = open_df[open_df["symbol"] == symbol.strip().upper()]
    return open_df


def get_cost_basis(symbol):
    """Share-weighted average cost basis across all open stock legs for
    `symbol`, or None if you hold no open position in it. Multiple open rows
    happen when shares were added or reassigned at different prices; a
    weighted average is the closest thing to a single "your basis" number.
    """
    held = get_open_positions(symbol)
    if held.empty:
        return None
    shares = held["shares"].astype(float)
    total_shares = shares.sum()
    if total_shares <= 0:
        return None
    return float((shares * held["stock_price"].astype(float)).sum() / total_shares)


def build_history_table(symbol=None):
    """Closed and rolled legs, newest first. No network calls - pure local read,
    so history is always available even offline or rate-limited."""
    df = load_positions()
    closed_df = df[df["status"].isin(["closed", "rolled", "assigned"])].copy()
    if symbol is not None:
        closed_df = closed_df[closed_df["symbol"] == symbol.strip().upper()]
    if closed_df.empty:
        return pd.DataFrame()

    # A "rolled" leg's story isn't complete without what it became.
    rolled_into = {r["rolled_from"]: r["contract"] for _, r in df.iterrows() if r.get("rolled_from")}

    rows = []
    for _, pos in closed_df.iterrows():
        entry_date = pd.to_datetime(pos["entry_date"]).date()
        close_date_raw = pos.get("close_date") or ""
        close_date = pd.to_datetime(close_date_raw).date() if close_date_raw else None
        premium_collected = float(pos["premium_collected"])
        realized_pl = pd.to_numeric(pos.get("realized_pl"), errors="coerce")
        stock_realized = pd.to_numeric(pos.get("stock_realized_pl"), errors="coerce")
        # Total across both legs; the stock leg only exists on assignment.
        total_realized = (0 if pd.isna(realized_pl) else realized_pl) + \
                         (0 if pd.isna(stock_realized) else stock_realized)
        if pd.isna(realized_pl) and pd.isna(stock_realized):
            total_realized = None
        rows.append({
            "symbol": pos["symbol"],
            "contract": pos["contract"],
            "status": pos["status"],
            "entry_date": pos["entry_date"],
            "close_date": close_date_raw,
            "days_held": (close_date - entry_date).days if close_date else None,
            "strike": float(pos["strike"]),
            # Screen fundamentals captured when the trade was logged - blank for
            # trades logged before that feature existed, never backfilled/guessed.
            "entry_score": pd.to_numeric(pos.get("entry_score"), errors="coerce"),
            "entry_iv": pd.to_numeric(pos.get("entry_iv"), errors="coerce"),
            "entry_iv_rv": pd.to_numeric(pos.get("entry_iv_rv"), errors="coerce"),
            "entry_delta": pd.to_numeric(pos.get("entry_delta"), errors="coerce"),
            "entry_prob_itm": pd.to_numeric(pos.get("entry_prob_itm"), errors="coerce"),
            "entry_pct_hi6": pd.to_numeric(pos.get("entry_pct_hi6"), errors="coerce"),
            "premium_collected": premium_collected,
            "close_premium": pd.to_numeric(pos.get("close_premium"), errors="coerce"),
            "realized_pl": realized_pl,
            # Only populated on assignment, where the shares were sold at the strike.
            "stock_realized_pl": stock_realized,
            "total_realized": total_realized,
            # Outcome metric to pair against the entry fundamentals above: did this
            # trade actually deliver, normalized so differently sized trades compare.
            "pct_captured": (realized_pl / premium_collected * 100)
                            if pd.notna(realized_pl) and premium_collected else None,
            "rolled_into": rolled_into.get(pos["contract"], ""),
        })
    return pd.DataFrame(rows).sort_values("close_date", ascending=False).reset_index(drop=True)


def position_action(pct_max_profit, dte_left, current_spot, strike, current_delta=None):
    # Past expiration the option no longer exists; the row needs closing out,
    # not managing. Checked first so it cannot be masked by TESTED/MANAGE.
    if dte_left < 0:
        return "EXPIRED"
    if current_spot >= strike:
        return "TESTED"
    # Early warning before the strike is breached, so rolling is a decision rather
    # than a reaction. Either a rising delta or simple price proximity triggers it.
    approaching = strike > 0 and (strike - current_spot) / strike * 100 <= ROLL_WATCH_PROXIMITY_PCT
    if (current_delta is not None and current_delta >= ROLL_WATCH_DELTA) or approaching:
        return "ROLL WATCH"
    if pct_max_profit is not None and pct_max_profit >= TAKE_PROFIT_PCT:
        return "TAKE PROFIT"
    if dte_left <= MANAGE_DTE:
        return "MANAGE"
    return "HOLD"


def compute_position_pnl(pos, current_spot):
    symbol = pos["symbol"]
    current_mid, current_iv = fetch_current_option_quote(
        symbol, pos["expiration"], pos["option_type"], pos["contract"]
    )
    contracts = int(pos["contracts"])
    shares = int(pos["shares"])
    stock_price = float(pos["stock_price"])
    strike = float(pos["strike"])
    premium_collected = float(pos["premium_collected"])
    # Total premium kept across an entire roll chain; falls back to this leg alone.
    cumulative_premium = pd.to_numeric(pos.get("cumulative_premium"), errors="coerce")
    if pd.isna(cumulative_premium):
        cumulative_premium = premium_collected
    investment_cash = stock_price * shares

    stock_pl = (current_spot - stock_price) * shares
    if current_mid is not None:
        opt_value_now = current_mid * 100 * contracts
        option_pl = premium_collected - opt_value_now
        total_pl = stock_pl + option_pl
    else:
        opt_value_now = None
        option_pl = None
        total_pl = None

    if_expired_pl = premium_collected
    if_called_pl = premium_collected + (strike - stock_price) * shares
    pct_if_expired = if_expired_pl / investment_cash * 100 if investment_cash else None
    pct_if_called = if_called_pl / investment_cash * 100 if investment_cash else None

    entry_date = pd.to_datetime(pos["entry_date"]).date()
    expiration_date = pd.to_datetime(pos["expiration"]).date()
    today = pd.Timestamp.now(tz="US/Pacific").date()
    holding_days = max((expiration_date - entry_date).days, 1)

    ann_if_expired = pct_if_expired * (365.0 / holding_days) if pct_if_expired is not None else None
    ann_if_called = pct_if_called * (365.0 / holding_days) if pct_if_called is not None else None

    dte_left = (expiration_date - today).days
    pct_max_profit = (option_pl / premium_collected * 100) if (option_pl is not None and premium_collected) else None
    current_delta = None
    if current_iv is not None and dte_left >= 0:
        current_delta = bs_call_delta(
            current_spot, strike, max(dte_left, 1) / 365.0, RISK_FREE_RATE, current_iv
        )
    action = position_action(pct_max_profit, dte_left, current_spot, strike, current_delta)

    return {
        "symbol": symbol,
        "contract": pos["contract"],
        "action": action,
        "delta_now": current_delta,
        "pct_max": pct_max_profit,
        "days_held": (today - entry_date).days,
        "dte_left": dte_left,
        "stock_entry": stock_price,
        "stock_now": current_spot,
        "stock_pl": stock_pl,
        "prem_coll": premium_collected,
        "cum_prem": cumulative_premium,
        "opt_now": opt_value_now,
        "opt_pl": option_pl,
        "total_pl": total_pl,
        "if_exp_pl": if_expired_pl,
        "pct_if_exp": pct_if_expired,
        "if_call_pl": if_called_pl,
        "pct_if_call": pct_if_called,
        "ann_if_exp": ann_if_expired,
        "ann_if_call": ann_if_called,
    }


ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_YELLOW = "\033[33m"
ANSI_RESET = "\033[0m"
COLOR_PNL_COLUMNS = {"total_pl","if_exp_pl", "if_call_pl"}
ACTION_COLORS = {
    "EXPIRED": ANSI_RED,
    "TESTED": ANSI_RED,
    "ROLL WATCH": ANSI_YELLOW,
    "TAKE PROFIT": ANSI_GREEN,
    "MANAGE": ANSI_YELLOW,
}
MIN_COL_WIDTH = 12


def colorize_pnl(col, plain_text, raw_value):
    if col == "action":
        color = ACTION_COLORS.get(raw_value)
        return f"{color}{plain_text}{ANSI_RESET}" if color else plain_text
    if col not in COLOR_PNL_COLUMNS or pd.isna(raw_value):
        return plain_text
    if raw_value > 0:
        return f"{ANSI_GREEN}{plain_text}{ANSI_RESET}"
    if raw_value < 0:
        return f"{ANSI_RED}{plain_text}{ANSI_RESET}"
    return plain_text


def print_colored_table(df, numeric_cols, color_fn, min_width=MIN_COL_WIDTH, na_rep="N/A"):
    """Render a DataFrame with ANSI colors, computing widths from the *uncolored*
    text so escape sequences never break column alignment."""
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    plain = {}
    for col in df.columns:
        if col in numeric_cols:
            plain[col] = df[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else na_rep)
        else:
            plain[col] = df[col].astype(str)

    widths = {c: max([len(c)] + [len(v) for v in plain[c]] + [min_width]) for c in df.columns}
    print(" ".join(c.rjust(widths[c]) for c in df.columns))
    for i in range(len(df)):
        cells = []
        for col in df.columns:
            text = plain[col].iloc[i].rjust(widths[col])
            color = color_fn(col, df[col].iloc[i]) if color_fn else None
            cells.append(f"{color}{text}{ANSI_RESET}" if color else text)
        print(" ".join(cells))


# The columns that actually decide a covered-call trade. Everything else in the
# table is supporting detail. Green = favorable, amber = check it, red = warning.
DECISION_COLUMNS = [
    "score", "pct_exp_ann", "prob_itm", "delta", "iv_rv", "iv_rank", "pct_hi6", "oi", "div_risk", "day_chg",
]

# Left-to-right display priority for the candidates table. Each entry is a
# tuple of equivalent column names - chain.py and screen_oom.py abbreviate
# some of the same concepts differently (e.g. "prob_itm" vs "p_itm") - so one
# list drives both CLIs and the GUI. Not listed = keeps its current relative
# position, appended after everything listed here.
COLUMN_PRIORITY = [
    # identity: minimum needed to know what this row even is
    ("*",), ("held",), ("contract",), ("sym",), ("name",),
    ("exp",), ("dte",), ("strike", "stk"), ("spot",),
    # decision indicators: color-coded, most important first
    ("score",),
    ("pct_exp_ann", "p_exp_ann"),
    ("prob_itm", "p_itm"),
    ("delta",),
    ("iv_rv",),
    ("iv_rank",),
    ("below_basis",),
    ("div_risk",),
    ("pct_hi6", "p_hi6"),
    ("oi",),
    ("oi_wall_ok",),
    ("day_chg",),
]


def reorder_columns(df, priority=COLUMN_PRIORITY):
    """Decision indicators first (in priority order), everything else keeps
    its existing relative order, appended after. Silently skips any name not
    present in df - callers never need to filter the list themselves."""
    ordered = [name for group in priority for name in group if name in df.columns]
    remainder = [c for c in df.columns if c not in ordered]
    return df[ordered + remainder]


def candidate_cell_color(col, value, best_score=None):
    # NOTE: never compare a DataFrame-derived boolean with `is True`/`is False`.
    # A value pulled from a pandas column is numpy.bool_, not the Python `bool`
    # singleton, and `numpy.bool_(True) is True` is False - the comparison
    # silently fails and falls through to the "else" branch every time. Plain
    # truthiness (`if value:`) works correctly for both. This bit us for real:
    # div_risk/below_basis/oi_wall_ok all used `is True` and none of them were
    # actually coloring on a true value until this was found and fixed.
    if col == "div_risk":
        return ANSI_RED if value else None
    if col == "below_basis":
        return ANSI_RED if value else None
    if col == "oi_wall_ok":
        if pd.isna(value):
            return None
        return ANSI_GREEN if value else ANSI_RED
    if col == "day_chg":
        # Not a pass/fail gate like the others, just a same-day-entry caution.
        return ANSI_RED if value < 0 else None
    if col not in DECISION_COLUMNS or pd.isna(value):
        return None
    if col == "score":
        # Relative: highlight the top of this particular table.
        if best_score and value >= best_score * 0.95:
            return ANSI_GREEN
        return None
    if col == "pct_exp_ann":
        if value >= 30:
            return ANSI_GREEN
        return ANSI_YELLOW if value >= 15 else None
    if col == "prob_itm":
        if value <= 25:
            return ANSI_GREEN
        return ANSI_YELLOW if value <= 40 else ANSI_RED
    if col == "delta":
        # DELTA_MIN/MAX already gate every row shown to this narrow band, so
        # "in range" tells you nothing - color proximity to TARGET_DELTA
        # instead, normalized by the distance to whichever edge it leans
        # toward (stays correct if those config values are ever retuned).
        span = (DELTA_MAX - TARGET_DELTA) if value >= TARGET_DELTA else (TARGET_DELTA - DELTA_MIN)
        fraction = abs(value - TARGET_DELTA) / span if span > 0 else 0
        if fraction <= 1 / 3:
            return ANSI_GREEN
        return ANSI_YELLOW if fraction <= 2 / 3 else ANSI_RED
    if col == "iv_rv":
        if value >= 1.0:
            return ANSI_GREEN
        return ANSI_YELLOW if value >= 0.90 else ANSI_RED
    if col == "iv_rank":
        # Where IV sits in this stock's own 52-week range - a timing signal,
        # not a hard gate, so the bands are wider than iv_rv's.
        if value >= 50:
            return ANSI_GREEN
        return ANSI_YELLOW if value >= 25 else ANSI_RED
    if col == "pct_hi6":
        # Strike above the nearest resistance zone means you are selling through it.
        return ANSI_GREEN if value > 0 else ANSI_RED
    if col == "oi":
        if value >= 500:
            return ANSI_GREEN
        return ANSI_YELLOW if value < 100 else None
    return None


def print_candidates_table(results):
    numeric_cols = [
        "dte", "score", "strike", "spot", "hi6", "dist_spot", "pct_spot", "dist_hi6", "pct_hi6",
        "hit6", "oi_wall", "oi_wall_oi", "last", "bid", "ask", "mid", "fill", "fill$", "called$", "cash_req",
        "pct_exp", "pct_call", "pct_exp_ann", "pct_call_ann", "div_amt", "basis", "sprd", "volum", "oi",
        "impvol", "iv_rv", "iv_rank", "delta", "prob_itm", "prob_touch", "premium",
    ]
    best_score = results["score"].max() if "score" in results.columns else None
    print_colored_table(
        results, numeric_cols,
        lambda col, value: candidate_cell_color(col, value, best_score),
        min_width=8, na_rep="",
    )


def position_cell_color(col, value):
    if col == "action":
        return ACTION_COLORS.get(value)
    if col not in COLOR_PNL_COLUMNS or pd.isna(value):
        return None
    if value > 0:
        return ANSI_GREEN
    if value < 0:
        return ANSI_RED
    return None


def print_position_pnl_table(rows):
    df = pd.DataFrame(rows)
    numeric_cols = [
        "stock_entry", "stock_now", "stock_pl", "prem_coll", "cum_prem", "opt_now", "opt_pl",
        "total_pl", "if_exp_pl", "pct_if_exp", "if_call_pl", "pct_if_call",
        "ann_if_exp", "ann_if_call", "pct_max", "delta_now",
    ]
    print_colored_table(df, numeric_cols, position_cell_color)


def cmd_positions():
    open_positions = get_open_positions()
    if open_positions.empty:
        print("No open positions.")
        return

    spot_cache = {}
    rows = []
    for _, pos in open_positions.iterrows():
        symbol = pos["symbol"]
        if symbol not in spot_cache:
            spot_cache[symbol] = get_spot_price(yf.Ticker(symbol))
        rows.append(compute_position_pnl(pos, spot_cache[symbol]))

    print("Open positions:")
    print_position_pnl_table(rows)


def history_cell_color(col, value):
    if col in ("realized_pl", "stock_realized_pl", "total_realized", "pct_captured") and pd.notna(value):
        if value > 0:
            return ANSI_GREEN
        if value < 0:
            return ANSI_RED
        return None
    if col == "status":
        if value == "rolled":
            return ANSI_YELLOW
        if value == "assigned":
            return ANSI_YELLOW
    # Same thresholds as the live candidates screen (candidate_cell_color), so a
    # closed trade's entry fundamentals read the same way they did when you picked it.
    if col == "entry_iv_rv" and pd.notna(value):
        if value >= 1.0:
            return ANSI_GREEN
        return ANSI_YELLOW if value >= 0.90 else ANSI_RED
    if col == "entry_prob_itm" and pd.notna(value):
        if value <= 25:
            return ANSI_GREEN
        return ANSI_YELLOW if value <= 40 else ANSI_RED
    if col == "entry_pct_hi6" and pd.notna(value):
        return ANSI_GREEN if value > 0 else ANSI_RED
    return None


def cmd_history(args):
    symbol = args[0].strip().upper() if args else None
    history = build_history_table(symbol)
    if history.empty:
        scope = f" for {symbol}" if symbol else ""
        print(f"No closed or rolled positions{scope}.")
        return

    print(f"Position history{f' for {symbol}' if symbol else ''}:")
    numeric_cols = [
        "days_held", "strike", "entry_score", "entry_iv", "entry_iv_rv", "entry_delta",
        "entry_prob_itm", "entry_pct_hi6", "premium_collected", "close_premium",
        "realized_pl", "stock_realized_pl", "total_realized", "pct_captured",
    ]
    print_colored_table(history, numeric_cols, history_cell_color)
    blank_fundamentals = history["entry_score"].isna().sum()
    if blank_fundamentals:
        print(f"({blank_fundamentals} leg(s) logged before entry-fundamental capture existed - blank above)")

    opt = pd.to_numeric(history["realized_pl"], errors="coerce").dropna()
    stock = pd.to_numeric(history["stock_realized_pl"], errors="coerce").dropna()
    total = pd.to_numeric(history["total_realized"], errors="coerce").dropna()
    if not total.empty:
        print()
        print(f"Realized across {len(total)} leg(s):  option {opt.sum():+.2f}"
              + (f"  |  stock (assigned) {stock.sum():+.2f}" if not stock.empty else "")
              + f"  |  TOTAL {total.sum():+.2f}")


def build_lots_table(symbol=None):
    """Cumulative return per lot - a continuous share-holding, regardless of
    how many legs it took to get there or whether they were connected via
    `roll` or an independent `close` then a later `log` with a gap in
    between. Complements build_history_table, which is per-leg; this is
    per-lot, which is what "cumulative return on this stock" actually means.
    """
    df = load_positions()
    if symbol is not None:
        df = df[df["symbol"] == symbol.strip().upper()]
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["realized_pl_num"] = pd.to_numeric(df["realized_pl"], errors="coerce").fillna(0)
    df["stock_realized_pl_num"] = pd.to_numeric(df["stock_realized_pl"], errors="coerce").fillna(0)
    df["premium_collected_num"] = pd.to_numeric(df["premium_collected"], errors="coerce").fillna(0)
    df["entry_sort"] = pd.to_datetime(df["entry_date"], errors="coerce")

    rows = []
    for lot_id, group in df.groupby("lot_id", sort=False):
        group = group.sort_values("entry_sort")
        open_legs = group[group["status"] == "open"]
        option_pl = group["realized_pl_num"].sum()
        stock_pl = group["stock_realized_pl_num"].sum()
        if not open_legs.empty:
            status = "open"
        elif (group["status"] == "assigned").any():
            status = "assigned"
        else:
            status = "closed"
        rows.append({
            "lot_id": lot_id,
            "symbol": group["symbol"].iloc[0],
            "legs": len(group),
            "opened": group["entry_date"].iloc[0],
            "status": status,
            "option_pl": option_pl,
            "stock_pl": stock_pl,
            "total_realized": option_pl + stock_pl,
            "open_premium": open_legs["premium_collected_num"].sum() if not open_legs.empty else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("opened", ascending=False).reset_index(drop=True)


def lot_cell_color(col, value):
    if col in ("option_pl", "stock_pl", "total_realized") and pd.notna(value):
        if value > 0:
            return ANSI_GREEN
        if value < 0:
            return ANSI_RED
        return None
    if col == "status" and value in ("open", "assigned"):
        return ANSI_YELLOW
    return None


def cmd_lots(args):
    symbol = args[0].strip().upper() if args else None
    lots = build_lots_table(symbol)
    if lots.empty:
        scope = f" for {symbol}" if symbol else ""
        print(f"No positions{scope}.")
        return

    print(f"Lots{f' for {symbol}' if symbol else ''} - cumulative return per continuous "
          f"share-holding, across every close/roll/re-log against it:")
    numeric_cols = ["legs", "option_pl", "stock_pl", "total_realized", "open_premium"]
    print_colored_table(lots, numeric_cols, lot_cell_color, na_rep="")
    print()
    closed = pd.to_numeric(lots.loc[lots["status"] != "open", "total_realized"], errors="coerce").dropna()
    if not closed.empty:
        print(f"Realized across {len(closed)} closed/assigned lot(s): {closed.sum():+.2f}")


def cmd_backfill(args):
    """Repair pass over positions.csv: fills BLANK entry_* screen fundamentals
    (via the historical DoltHub dataset, for contracts that left the live
    chain before get_contract_metrics could capture them) and BLANK
    cumulative_premium cells (via the lot-aware running total). By default,
    never overwrites an already-populated cell - a value that predates
    lot-aware accounting and looks stale is a correction, not a backfill.
    Pass --recalc to also correct those: recompute cumulative_premium for
    every row in a lot (not just blanks) from the running total, overwriting
    any that don't match.
    """
    args = list(args)
    recalc = extract_bool_flag(args, "--recalc")
    symbol_filter = args[0].strip().upper() if args else None
    df = load_positions()
    if df.empty:
        print("No positions.")
        return

    target_idx = df.index if symbol_filter is None else df.index[df["symbol"] == symbol_filter]
    filled_metrics = 0
    skipped_metrics = 0

    for i in target_idx:
        cell = df.at[i, "entry_score"]
        # Blank reads back as NaN (float) for a numeric column, or "" if ever
        # written as a bare string - a plain str(x) != "" check misses the
        # NaN case entirely (str(nan) == "nan", which "isn't blank").
        already_filled = not (pd.isna(cell) if not isinstance(cell, str) else cell.strip() == "")
        if already_filled:
            continue
        entry_date_iso = pd.Timestamp(df.at[i, "entry_date"]).strftime("%Y-%m-%d")
        metrics = get_historical_contract_metrics(df.at[i, "symbol"], df.at[i, "contract"], entry_date_iso)
        if metrics["entry_score"] == "":
            skipped_metrics += 1
            print(f"  Could not backfill entry metrics for {df.at[i, 'contract']} "
                  f"(entry {df.at[i, 'entry_date']}) - no historical coverage for that date/strike.")
            continue
        for k, v in metrics.items():
            df.at[i, k] = v
        filled_metrics += 1
        print(f"  Filled entry metrics for {df.at[i, 'contract']} (entry {df.at[i, 'entry_date']}): "
              f"score={metrics['entry_score']} iv={metrics['entry_iv']} delta={metrics['entry_delta']}")

    filled_cumulative = 0
    corrected_cumulative = 0
    for lot_id, group in df.groupby("lot_id", sort=False):
        if symbol_filter is not None and not (group["symbol"] == symbol_filter).any():
            continue
        running_realized = 0.0
        for i in group.index:
            cum = pd.to_numeric(df.at[i, "cumulative_premium"], errors="coerce")
            premium = pd.to_numeric(df.at[i, "premium_collected"], errors="coerce")
            premium = 0.0 if pd.isna(premium) else float(premium)
            correct_value = round(running_realized + premium, 2)
            if pd.isna(cum):
                df.at[i, "cumulative_premium"] = correct_value
                filled_cumulative += 1
                print(f"  Filled cumulative_premium for {df.at[i, 'contract']} (lot {lot_id}): {correct_value}")
            elif recalc and abs(float(cum) - correct_value) > 0.005:
                old_value = float(cum)
                df.at[i, "cumulative_premium"] = correct_value
                corrected_cumulative += 1
                print(f"  Corrected cumulative_premium for {df.at[i, 'contract']} (lot {lot_id}): "
                      f"{old_value} -> {correct_value}")
            realized = pd.to_numeric(df.at[i, "realized_pl"], errors="coerce")
            running_realized += 0.0 if pd.isna(realized) else float(realized)

    if filled_metrics or filled_cumulative or corrected_cumulative:
        save_positions(df)
        corrected_note = f", corrected {corrected_cumulative} stale cell(s)" if corrected_cumulative else ""
        print(f"\nBackfilled {filled_metrics} entry-metric row(s) and {filled_cumulative} "
              f"cumulative_premium cell(s){corrected_note}. Saved to positions.csv.")
    else:
        scope = f" for {symbol_filter}" if symbol_filter else ""
        reason = "no blank fields found" if not recalc else "no blank or stale fields found"
        print(f"Nothing to backfill{scope} - {reason}.")
    if skipped_metrics:
        print(f"({skipped_metrics} row(s) left blank - historical data unavailable for that date/contract)")


def get_spot_price(ticker):
    # yfinance's FastInfo only exposes snake_case attributes (last_price, not
    # lastPrice/regularMarketPrice - the latter don't exist on this object at
    # all). A prior version of this function tried camelCase names, which meant
    # this always missed and silently fell through to the history() fetch below
    # on every single call - not wrong (that fallback happens to be accurate),
    # but a wasted API round-trip every time.
    try:
        value = ticker.fast_info.last_price
        if value is not None:
            return float(value)
    except Exception:
        pass
    hist = ticker.history(period="5d", auto_adjust=False)
    return float(hist["Close"].dropna().iloc[-1])


def get_day_change(ticker):
    """Today's move vs. the prior close, as a percentage. None fields if unavailable.

    Reuses ticker.fast_info, which get_spot_price has typically already fetched
    and yfinance caches on the Ticker object - this costs no extra API call.
    """
    try:
        info = ticker.fast_info
        last = info.last_price
        prev = info.previous_close
        if last is None or prev is None or prev == 0:
            return {"change_pct": None, "last_price": None, "previous_close": None}
        return {
            "change_pct": (float(last) - float(prev)) / float(prev) * 100,
            "last_price": float(last),
            "previous_close": float(prev),
        }
    except Exception:
        return {"change_pct": None, "last_price": None, "previous_close": None}


def realized_volatility(closes, trading_days=None):
    """Annualized close-to-close realized vol over the last `trading_days` bars."""
    closes = closes.dropna()
    if trading_days is not None:
        closes = closes.tail(trading_days + 1)  # n+1 closes yield n returns
    if len(closes) < 3:
        return None
    log_returns = (closes / closes.shift(1)).dropna().apply(math.log)
    if len(log_returns) < 2:
        return None
    vol = float(log_returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR))
    return vol if vol > 0 else None


def blended_realized_volatility(closes):
    """Horizon-matched RV for short-dated options.

    A single long window (e.g. 3 months) keeps an old gap in the sample long after
    the market has stopped pricing a repeat, inflating RV and making every option
    look artificially cheap. The short window fixes the horizon mismatch; the long
    window damps its noise. Returns (blended, short, long).
    """
    rv_short = realized_volatility(closes, RV_SHORT_TRADING_DAYS)
    rv_long = realized_volatility(closes, RV_LONG_TRADING_DAYS)
    if rv_short is None and rv_long is None:
        return None, None, None
    if rv_short is None:
        return rv_long, None, rv_long
    if rv_long is None:
        return rv_short, rv_short, None
    blended = RV_SHORT_WEIGHT * rv_short + (1 - RV_SHORT_WEIGHT) * rv_long
    return blended, rv_short, rv_long


def find_swing_highs(hist, swing_window):
    """Local peaks: a day's High exceeds every High `swing_window` days on
    each side. `hist` must be sorted by Date ascending with Date/High
    columns (Date as python date objects). Returns [(date, high), ...]. The
    most recent `swing_window` days can never be confirmed peaks yet (not
    enough days after them) - an inherent lag in swing detection, not a bug.
    """
    highs = hist["High"].to_numpy()
    dates = hist["Date"].to_numpy()
    n = len(highs)
    peaks = []
    for i in range(swing_window, n - swing_window):
        window = highs[i - swing_window:i + swing_window + 1]
        if highs[i] == window.max() and (window == highs[i]).sum() == 1:
            peaks.append((dates[i], float(highs[i])))
    return peaks


def find_swing_lows(hist, swing_window):
    """Local troughs: the mirror of find_swing_highs, for support instead of
    resistance. A day's Low sits below every Low `swing_window` days on each
    side. `hist` must be sorted by Date ascending with Date/Low columns
    (Date as python date objects). Returns [(date, low), ...].
    """
    lows = hist["Low"].to_numpy()
    dates = hist["Date"].to_numpy()
    n = len(lows)
    troughs = []
    for i in range(swing_window, n - swing_window):
        window = lows[i - swing_window:i + swing_window + 1]
        if lows[i] == window.min() and (window == lows[i]).sum() == 1:
            troughs.append((dates[i], float(lows[i])))
    return troughs


def cluster_resistance_zones(peaks, cluster_pct):
    """Merge peaks within cluster_pct of each other into zones. Works the
    same whether `peaks` are swing highs (resistance) or swing lows
    (support) - it's just clustering (date, price) pairs by proximity.
    Returns [{"price": mean_of_members, "touches": count, "last_date": ...},
    ...] sorted by price ascending."""
    if not peaks:
        return []
    ordered = sorted(peaks, key=lambda p: p[1])
    zones, current = [], [ordered[0]]
    for date, price in ordered[1:]:
        zone_avg = sum(p for _, p in current) / len(current)
        if abs(price - zone_avg) / zone_avg <= cluster_pct:
            current.append((date, price))
        else:
            zones.append(current)
            current = [(date, price)]
    zones.append(current)
    return [
        {"price": sum(p for _, p in z) / len(z), "touches": len(z), "last_date": max(d for d, _ in z)}
        for z in zones
    ]


def nearest_resistance(hist, spot, as_of=None, lookback_days=None, swing_window=None, cluster_pct=None):
    """Nearest overhead swing-high zone above `spot`, from clustering peaks
    over the trailing `lookback_days` of `hist` (Date/High columns, Date as
    python date objects) as of `as_of` (defaults to hist's latest date -
    pass an explicit date to evaluate historically, e.g. at a past
    position's entry date). Returns {"price", "touches", "days_since_touch"},
    or None if no zone sits above spot within the lookback - a genuine
    breakout beyond recent history, reported as unavailable rather than
    guessed. Replaces the old "single trailing 6-week max" measure, which
    was one noisy point (one spike day set the whole level) and couldn't
    distinguish a level tested once from one rejected five times.
    """
    lookback_days = RESISTANCE_LOOKBACK_DAYS if lookback_days is None else lookback_days
    swing_window = RESISTANCE_SWING_WINDOW if swing_window is None else swing_window
    cluster_pct = RESISTANCE_CLUSTER_PCT if cluster_pct is None else cluster_pct
    if hist.empty:
        return None

    as_of_date = hist["Date"].max() if as_of is None else as_of
    cutoff = as_of_date - pd.Timedelta(days=lookback_days)
    window = hist[(hist["Date"] > cutoff) & (hist["Date"] <= as_of_date)].sort_values("Date").reset_index(drop=True)
    if len(window) < swing_window * 2 + 1:
        return None

    zones = cluster_resistance_zones(find_swing_highs(window, swing_window), cluster_pct)
    above = [z for z in zones if z["price"] >= spot]
    if not above:
        return None
    nearest = min(above, key=lambda z: z["price"])
    days_since = (as_of_date - nearest["last_date"]).days
    return {"price": nearest["price"], "touches": nearest["touches"], "days_since_touch": int(days_since)}


def nearest_support(hist, spot, as_of=None, lookback_days=None, swing_window=None, cluster_pct=None):
    """Nearest swing-low zone below `spot` - the mirror of nearest_resistance,
    for the "floor" instead of the "ceiling". Same clustering, same lookback,
    just swing lows instead of swing highs and "below spot" instead of
    "above spot". Returns {"price", "touches", "days_since_touch"}, or None
    if no zone sits below spot within the lookback.
    """
    lookback_days = RESISTANCE_LOOKBACK_DAYS if lookback_days is None else lookback_days
    swing_window = RESISTANCE_SWING_WINDOW if swing_window is None else swing_window
    cluster_pct = RESISTANCE_CLUSTER_PCT if cluster_pct is None else cluster_pct
    if hist.empty:
        return None

    as_of_date = hist["Date"].max() if as_of is None else as_of
    cutoff = as_of_date - pd.Timedelta(days=lookback_days)
    window = hist[(hist["Date"] > cutoff) & (hist["Date"] <= as_of_date)].sort_values("Date").reset_index(drop=True)
    if len(window) < swing_window * 2 + 1:
        return None

    zones = cluster_resistance_zones(find_swing_lows(window, swing_window), cluster_pct)
    below = [z for z in zones if z["price"] <= spot]
    if not below:
        return None
    nearest = max(below, key=lambda z: z["price"])
    days_since = (as_of_date - nearest["last_date"]).days
    return {"price": nearest["price"], "touches": nearest["touches"], "days_since_touch": int(days_since)}


def find_oi_wall(calls, spot):
    """Strike (and its OI) with the heaviest open interest among OTM calls
    (strike > spot) in this expiration's chain - a rough proxy for
    dealer-hedging resistance: as spot approaches a strike where dealers are
    short a lot of calls, their delta hedging becomes real sell pressure.
    Caveat: open interest alone doesn't distinguish bought-to-open from
    sold-to-open, so this is a proxy, not a certainty - informational only,
    no filter threshold. Returns (wall_strike, wall_oi), or (None, None) if
    no OTM call in the chain has any open interest.
    """
    otm = calls[(calls["strike"] > spot) & (calls["openInterest"] > 0)]
    if otm.empty:
        return None, None
    idx = otm["openInterest"].idxmax()
    return float(otm.at[idx, "strike"]), int(otm.at[idx, "openInterest"])


def get_resistance_chart_data(symbol, spot=None, lookback_days=None):
    """Price history plus EVERY detected resistance zone (not just the
    nearest one above spot, which is all get_price_context needs) - for
    charting. A fresh, on-demand fetch, independent of the cached screener
    pipeline, since this is only called when a user actually opens the
    chart, not on every screen. Returns (hist, zones, spot, support):
      hist    - DataFrame with Date (python date)/Close/High/Low columns
      zones   - cluster_resistance_zones' output, sorted by price ascending
      spot    - the current spot price used (fetched if not passed in)
      support - nearest_support's output (single nearest floor below spot),
                or None. Only the nearest support is surfaced (unlike zones
                above) - one floor line is the useful signal here, and every
                swing low plotted alongside every swing high is exactly the
                clutter this chart already had to climb out of once.
    """
    lookback_days = RESISTANCE_LOOKBACK_DAYS if lookback_days is None else lookback_days
    ticker = yf.Ticker(symbol)
    if spot is None:
        spot = get_spot_price(ticker)
    hist = ticker.history(period="9mo", interval="1d", auto_adjust=False).reset_index()
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None).dt.date
    if hist.empty:
        return hist, [], spot, None

    as_of_date = hist["Date"].max()
    cutoff = as_of_date - pd.Timedelta(days=lookback_days)
    window = hist[(hist["Date"] > cutoff) & (hist["Date"] <= as_of_date)].sort_values("Date").reset_index(drop=True)
    zones = cluster_resistance_zones(find_swing_highs(window, RESISTANCE_SWING_WINDOW), RESISTANCE_CLUSTER_PCT)
    support = nearest_support(hist, spot, lookback_days=lookback_days)
    return hist, zones, spot, support


def get_price_context(ticker, spot):
    # Fetch 9 months so swing-high resistance zones (RESISTANCE_LOOKBACK_DAYS,
    # ~6 months) have comfortable margin plus edge buffer for the swing-window
    # check, and realized vol (needs RV_LONG_TRADING_DAYS, ~3 months) has a
    # stable sample from the same single API call.
    hist = ticker.history(period="9mo", interval="1d", auto_adjust=False).reset_index()
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None).dt.date
    resistance = nearest_resistance(hist, spot)
    support = nearest_support(hist, spot)
    rv_blended, rv_short, rv_long = blended_realized_volatility(hist["Close"])
    day_change = get_day_change(ticker)
    return {
        "resistance_price": resistance["price"] if resistance else float("nan"),
        "resistance_touches": resistance["touches"] if resistance else float("nan"),
        "resistance_days_since_touch": resistance["days_since_touch"] if resistance else float("nan"),
        "support_price": support["price"] if support else float("nan"),
        "support_touches": support["touches"] if support else float("nan"),
        "support_days_since_touch": support["days_since_touch"] if support else float("nan"),
        "realized_vol": rv_blended,
        "realized_vol_short": rv_short,
        "realized_vol_long": rv_long,
        "day_change_pct": day_change["change_pct"],
        "day_previous_close": day_change["previous_close"],
    }


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


def get_company_name(ticker, symbol):
    try:
        info = ticker.get_info()
    except Exception:
        return symbol
    name = info.get("longName") or info.get("shortName")
    return name if name else symbol


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
        "dte", "strike", "spot", "hi6", "dist_spot", "pct_spot", "dist_hi6", "pct_hi6", "hit6",
        "last", "bid", "ask", "mid", "fill", "fill$", "called$", "cash_req", "pct_exp", "pct_call",
        "pct_exp_ann", "pct_call_ann", "div_amt", "sprd", "volum", "oi",
        "impvol", "iv_rv", "score", "delta", "prob_itm", "prob_touch", "premium"
    ]

    for col in two_decimal_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    return df


def chain_cache_path(symbol):
    return CHAIN_CACHE_DIR / f"{symbol.upper()}.pkl"


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
    # The cached slice is DTE-window dependent; refetch if that window moved.
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


def fetch_chain_data(symbol, use_cache=True, min_dte=None, max_dte=None):
    """Fetch everything that requires network access. Threshold-independent."""
    min_dte = MIN_DTE if min_dte is None else min_dte
    max_dte = MAX_DTE if max_dte is None else max_dte
    if use_cache:
        cached = load_chain_cache(symbol, min_dte, max_dte)
        if cached is not None:
            cached["from_cache"] = True
            return cached

    ticker = yf.Ticker(symbol)
    spot = get_spot_price(ticker)
    context = get_price_context(ticker, spot)
    cache = load_lookup_cache()
    cache_lock = threading.Lock()
    company_name, earnings_dates, ex_div_date, div_amount = get_cached_lookup(symbol, ticker, cache, cache_lock)
    save_lookup_cache(cache)
    today = pd.Timestamp.now(tz="UTC").tz_convert("US/Pacific").date()

    expirations = []
    expirations_skipped_earnings = 0
    for exp in ticker.options:
        exp_date = pd.to_datetime(exp).date()
        dte = (exp_date - today).days
        if not (min_dte <= dte <= max_dte):
            continue
        if any(today <= ed <= exp_date for ed in earnings_dates):
            expirations_skipped_earnings += 1
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
        "symbol": symbol,
        "spot": spot,
        "context": context,
        "company_name": company_name,
        "ex_div_date": ex_div_date,
        "div_amount": div_amount,
        "expirations": expirations,
        "expirations_skipped_earnings": expirations_skipped_earnings,
        "from_cache": False,
    }
    save_chain_cache(symbol, payload)
    return payload


def get_contract_metrics(symbol, contract):
    """Screen metrics for one specific contract, ignoring all thresholds.

    Captured at entry so you can later test which criteria actually predicted
    good outcomes. Returns blanks rather than raising: a data hiccup must never
    prevent a real trade from being recorded.
    """
    blank = {k: "" for k in
             ["entry_score", "entry_iv", "entry_iv_rv", "entry_delta", "entry_prob_itm", "entry_pct_hi6"]}
    try:
        data = fetch_chain_data(symbol)
        spot = data["spot"]
        realized_vol = data["context"].get("realized_vol")
        high_6wk = data["context"]["resistance_price"]

        for entry in data["expirations"]:
            calls = entry["calls"]
            match = calls[calls["contractSymbol"] == contract]
            if match.empty:
                continue
            row = match.iloc[0]
            dte = entry["dte"]
            years = max(dte, 1) / 365.0
            iv = float(row["impliedVolatility"])
            strike = float(row["strike"])
            bid, ask = float(row["bid"]), float(row["ask"])
            fill = modeled_fill_price(bid, ask)

            delta = bs_call_delta(spot, strike, years, RISK_FREE_RATE, iv)
            prob_itm = bs_prob_itm_exp(spot, strike, years, RISK_FREE_RATE, iv)
            cash_required = spot * 100
            annualize = 365.0 / max(dte, 1)
            pct_if_expired_ann = (fill * 100) / cash_required * 100 * annualize
            pct_if_called_ann = ((fill * 100 + (strike - spot) * 100) / cash_required) * 100 * annualize
            score = (compute_score(pct_if_expired_ann, pct_if_called_ann, prob_itm)
                     if prob_itm is not None else None)

            return {
                "entry_score": round(score, 2) if score is not None else "",
                "entry_iv": round(iv, 4),
                "entry_iv_rv": round(iv / realized_vol, 3) if realized_vol else "",
                "entry_delta": round(delta, 3) if delta is not None else "",
                "entry_prob_itm": round(prob_itm, 2) if prob_itm is not None else "",
                "entry_pct_hi6": round((strike - high_6wk) / high_6wk * 100, 2) if pd.notna(high_6wk) else "",
            }
        return blank
    except Exception:
        return blank


def get_historical_contract_metrics(symbol, contract, entry_date):
    """Best-effort get_contract_metrics for a contract that already left the
    live chain (closed or expired) before its entry_* fundamentals could be
    captured - sourced from the historical DoltHub dataset (voldata.py)
    instead of a live yfinance chain. `entry_date` is a 'YYYY-MM-DD' string.
    Returns the same blanks dict as get_contract_metrics on any failure -
    missing historical coverage for that symbol/date is expected sometimes,
    never a reason to touch existing data.

    Unlike get_contract_metrics (which derives delta itself via Black-Scholes,
    since yfinance's chain doesn't provide it), the historical dataset
    supplies delta directly - used as-is rather than re-derived.
    """
    blank = {k: "" for k in
             ["entry_score", "entry_iv", "entry_iv_rv", "entry_delta", "entry_prob_itm", "entry_pct_hi6"]}
    if voldata is None:
        return blank
    try:
        root, expiration, opt_type, strike = parse_occ_contract(contract)
        if opt_type != "C":
            return blank  # this tool only ever writes calls

        rows = voldata.fetch_option_chain_history(symbol, entry_date, entry_date, call_put="Call")
        if not rows:
            return blank
        match = next(
            (r for r in rows if r.get("expiration") == expiration
             and r.get("bid") is not None and r.get("ask") is not None and r.get("vol") is not None
             and abs(float(r["strike"]) - strike) < 0.005),
            None,
        )
        if match is None:
            return blank

        entry_date_obj = pd.Timestamp(entry_date).date()
        ticker = yf.Ticker(symbol)
        # Buffer covers RESISTANCE_LOOKBACK_DAYS (~6mo) plus swing-window edge
        # margin, so historical zone detection as of entry_date has the same
        # comfortable margin get_price_context gives the live path.
        hist = ticker.history(
            start=(pd.Timestamp(entry_date) - pd.Timedelta(days=230)).strftime("%Y-%m-%d"),
            end=(pd.Timestamp(entry_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d", auto_adjust=False,
        )
        if hist.empty:
            return blank
        hist = hist.reset_index()
        hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None).dt.date
        closes_upto = hist.loc[hist["Date"] <= entry_date_obj, "Close"]
        if closes_upto.empty:
            return blank
        spot = float(closes_upto.iloc[-1])
        realized_vol, _, _ = blended_realized_volatility(closes_upto)

        resistance = nearest_resistance(hist, spot, as_of=entry_date_obj)
        high_6wk = resistance["price"] if resistance else None

        iv = float(match["vol"])
        delta = float(match["delta"])
        bid, ask = float(match["bid"]), float(match["ask"])
        expiration_date = pd.Timestamp(expiration).date()
        dte = (expiration_date - entry_date_obj).days
        years = max(dte, 1) / 365.0
        prob_itm = bs_prob_itm_exp(spot, strike, years, RISK_FREE_RATE, iv)
        fill = modeled_fill_price(bid, ask)
        cash_required = spot * 100
        annualize = 365.0 / max(dte, 1)
        pct_if_expired_ann = (fill * 100) / cash_required * 100 * annualize
        pct_if_called_ann = ((fill * 100 + (strike - spot) * 100) / cash_required) * 100 * annualize
        score = compute_score(pct_if_expired_ann, pct_if_called_ann, prob_itm) if prob_itm is not None else None

        return {
            "entry_score": round(score, 2) if score is not None else "",
            "entry_iv": round(iv, 4),
            "entry_iv_rv": round(iv / realized_vol, 3) if realized_vol else "",
            "entry_delta": round(delta, 3),
            "entry_prob_itm": round(prob_itm, 2) if prob_itm is not None else "",
            "entry_pct_hi6": round((strike - high_6wk) / high_6wk * 100, 2) if high_6wk is not None else "",
        }
    except Exception:
        return blank


def find_candidates(symbol, iv_threshold=None, iv_rv_ratio_min=None, use_cache=True,
                    assignment_preference=None, min_dte=None, max_dte=None, iv_rank_min=None):
    iv_threshold = IV_THRESHOLD if iv_threshold is None else iv_threshold
    iv_rv_ratio_min = IV_RV_RATIO_MIN if iv_rv_ratio_min is None else iv_rv_ratio_min
    assignment_preference = ASSIGNMENT_PREFERENCE if assignment_preference is None else assignment_preference
    iv_rank_min = IV_RANK_MIN if iv_rank_min is None else iv_rank_min

    data = fetch_chain_data(symbol, use_cache=use_cache, min_dte=min_dte, max_dte=max_dte)
    spot = data["spot"]
    context = dict(data["context"])
    context["from_cache"] = data.get("from_cache", False)
    context["fetched_at"] = data.get("fetched_at")
    company_name = data["company_name"]
    ex_div_date = data["ex_div_date"]
    div_amount = data["div_amount"]
    today = data["trade_date"]
    expirations_skipped_earnings = data["expirations_skipped_earnings"]
    candidates = []
    expirations_checked = 0

    # Symbol-level, not per-contract, so it's fetched once up front. Missing/
    # unavailable never excludes a candidate unless iv_rank_min > 0.
    voldata_cache = voldata.load_cache() if voldata else {}
    iv_rank_ctx = fetch_iv_rank(symbol, voldata_cache, threading.Lock())
    if voldata is not None:
        voldata.save_cache(voldata_cache)
    context["iv_rank_ctx"] = iv_rank_ctx

    # If you already hold shares, flag (not exclude - realizing the loss can be
    # intentional) any strike below your cost basis. The Wheel's classic trap:
    # selling calls below basis just to collect premium locks in a stock loss
    # if assigned.
    cost_basis = get_cost_basis(symbol)
    context["cost_basis"] = cost_basis
    iv_rank_value = iv_rank_ctx["iv_rank"]
    iv_rank_ok = iv_rank_value is None or iv_rank_value >= iv_rank_min

    for entry in data["expirations"]:
        exp = entry["exp"]
        exp_date = entry["exp_date"]
        dte = entry["dte"]
        expirations_checked += 1
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

        filtered["expiration"] = exp
        filtered["dte"] = dte
        filtered["premium_over_1"] = (filtered["bid"] > PREMIUM_THRESHOLD) | (filtered["ask"] > PREMIUM_THRESHOLD)
        filtered["highlight"] = filtered["premium_over_1"].map({True: "***", False: ""})
        filtered["spot_price"] = spot
        filtered["iv_rank"] = iv_rank_value
        filtered["high_6wk"] = context["resistance_price"]
        filtered["oi_wall"] = oi_wall_strike if oi_wall_strike is not None else float("nan")
        filtered["oi_wall_oi"] = oi_wall_oi if oi_wall_oi is not None else float("nan")
        filtered["oi_wall_ok"] = (
            filtered["strike"] >= oi_wall_strike if oi_wall_strike is not None else pd.NA
        )
        filtered["distance_to_spot"] = filtered["strike"] - spot
        filtered["pct_to_spot"] = (filtered["distance_to_spot"] / spot) * 100
        filtered["called_dollars"] = filtered["fill_dollars"] + filtered["distance_to_spot"] * 100
        filtered["cash_required"] = spot * 100
        filtered["pct_if_expired"] = filtered["fill_dollars"] / filtered["cash_required"] * 100
        filtered["pct_if_called"] = filtered["called_dollars"] / filtered["cash_required"] * 100
        filtered["pct_if_expired_ann"] = filtered["pct_if_expired"] * (365.0 / dte)
        filtered["pct_if_called_ann"] = filtered["pct_if_called"] * (365.0 / dte)
        filtered["score"] = compute_score(
            filtered["pct_if_expired_ann"], filtered["pct_if_called_ann"],
            filtered["prob_itm_exp"], assignment_preference,
        )
        filtered["distance_to_6wk_high"] = filtered["strike"] - context["resistance_price"]
        filtered["pct_to_6wk_high"] = (filtered["distance_to_6wk_high"] / context["resistance_price"]) * 100
        filtered["resistance_hits_6wk"] = context["resistance_touches"]

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

        candidates.append(filtered[[
            "highlight", "contractSymbol", "expiration", "dte", "score", "strike", "spot_price", "high_6wk",
            "distance_to_spot", "pct_to_spot", "distance_to_6wk_high", "pct_to_6wk_high", "resistance_hits_6wk",
            "oi_wall", "oi_wall_oi", "oi_wall_ok",
            "lastPrice", "bid", "ask", "mid", "fill", "fill_dollars", "called_dollars", "cash_required",
            "pct_if_expired", "pct_if_called", "pct_if_expired_ann", "pct_if_called_ann",
            "div_risk", "div_amount", "cost_basis", "below_basis", "spread", "volume", "openInterest",
            "iv", "iv_rv_ratio", "iv_rank", "delta", "prob_itm_exp", "prob_touch", "premium_over_1"
        ]])

    if not candidates:
        return spot, context, expirations_checked, expirations_skipped_earnings, company_name, None

    results = pd.concat(candidates, ignore_index=True)
    results = results.sort_values(["score", "bid"], ascending=[False, False]).head(TOP_ROWS)
    results = results.rename(columns={
        "highlight": "*",
        "contractSymbol": "contract",
        "expiration": "exp",
        "dte": "dte",
        "strike": "strike",
        "spot_price": "spot",
        "high_6wk": "hi6",
        "distance_to_spot": "dist_spot",
        "pct_to_spot": "pct_spot",
        "distance_to_6wk_high": "dist_hi6",
        "pct_to_6wk_high": "pct_hi6",
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
        "cash_required": "cash_req",
        "pct_if_expired": "pct_exp",
        "pct_if_called": "pct_call",
        "pct_if_expired_ann": "pct_exp_ann",
        "pct_if_called_ann": "pct_call_ann",
        "div_risk": "div_risk",
        "div_amount": "div_amt",
        "cost_basis": "basis",
        "below_basis": "below_basis",
        "spread": "sprd",
        "volume": "volum",
        "openInterest": "oi",
        "iv": "impvol",
        "iv_rv_ratio": "iv_rv",
        "iv_rank": "iv_rank",
        "score": "score",
        "delta": "delta",
        "prob_itm_exp": "prob_itm",
        "prob_touch": "prob_touch",
        "premium_over_1": "premium",
    })
    # Left numeric on purpose: the CLI colors it at print time and the GUI styles it.
    return spot, context, expirations_checked, expirations_skipped_earnings, company_name, results


def print_results(iv_threshold=None, iv_rv_ratio_min=None, use_cache=True, assignment_preference=None,
                  min_dte=None, max_dte=None, iv_rank_min=None):
    iv_threshold = IV_THRESHOLD if iv_threshold is None else iv_threshold
    iv_rv_ratio_min = IV_RV_RATIO_MIN if iv_rv_ratio_min is None else iv_rv_ratio_min
    assignment_preference = ASSIGNMENT_PREFERENCE if assignment_preference is None else assignment_preference
    min_dte = MIN_DTE if min_dte is None else min_dte
    max_dte = MAX_DTE if max_dte is None else max_dte
    iv_rank_min = IV_RANK_MIN if iv_rank_min is None else iv_rank_min
    symbol = get_symbol()
    cash_to_invest = get_cash_to_invest()
    spot, context, expirations_checked, expirations_skipped_earnings, company_name, results = find_candidates(
        symbol, iv_threshold, iv_rv_ratio_min, use_cache=use_cache,
        assignment_preference=assignment_preference, min_dte=min_dte, max_dte=max_dte,
        iv_rank_min=iv_rank_min,
    )

    print(f"Symbol: {symbol} ({company_name})")
    print(f"Spot: {spot:.1f}")
    day_change_pct = context.get("day_change_pct")
    if day_change_pct is not None:
        prev_close = context.get("day_previous_close")
        if day_change_pct < 0:
            print(f"{ANSI_RED}Trading DOWN {abs(day_change_pct):.2f}% today (prev close {prev_close:.2f}) - "
                  f"you've noted this usually isn't an entry day.{ANSI_RESET}")
        else:
            print(f"Day change: {ANSI_GREEN}+{day_change_pct:.2f}%{ANSI_RESET} (prev close {prev_close:.2f})")
    cash_required = spot * 100
    print(f"Cash required per contract: {cash_required:.2f}")
    if cash_to_invest is not None:
        contracts = int(cash_to_invest // cash_required)
        leftover = cash_to_invest - contracts * cash_required
        print(f"Cash to invest: {cash_to_invest:.2f} -> {contracts} contract(s) (leftover: {leftover:.2f})")
    if pd.notna(context["resistance_price"]):
        print(f"Resistance: {context['resistance_price']:.1f} "
              f"({context['resistance_touches']:.0f} touches, "
              f"last seen {context['resistance_days_since_touch']:.0f}d ago)")
    else:
        print(f"Resistance: none identified in trailing {RESISTANCE_LOOKBACK_DAYS}d "
              f"(price above all recent swing highs)")
    realized_vol = context.get("realized_vol")
    rv_short, rv_long = context.get("realized_vol_short"), context.get("realized_vol_long")
    if realized_vol:
        parts = f"{RV_SHORT_TRADING_DAYS}d {rv_short:.2f}" if rv_short else "short N/A"
        parts += f" / {RV_LONG_TRADING_DAYS}d {rv_long:.2f}" if rv_long else " / long N/A"
        print(f"Realized vol (blended): {realized_vol:.2f}  [{parts}]")
    else:
        print("Realized vol: N/A")
    iv_rank_ctx = context.get("iv_rank_ctx") or {}
    if iv_rank_ctx.get("available"):
        print(f"IV rank: {iv_rank_ctx['iv_rank']:.0f}/100 (52wk range {iv_rank_ctx['iv_year_low']:.2f}-"
              f"{iv_rank_ctx['iv_year_high']:.2f}, as of {iv_rank_ctx['as_of']}, "
              f"source: community DoltHub dataset)")
    else:
        print("IV rank: N/A (community data source unavailable or doesn't cover this symbol)")
    print(f"Expirations checked: {expirations_checked:.1f}")
    if expirations_skipped_earnings:
        print(f"Expirations skipped (earnings in window): {expirations_skipped_earnings}")
    iv_rv_desc = f"IV/RV >= {iv_rv_ratio_min:.2f}" if iv_rv_ratio_min > 0 else "IV/RV off"
    iv_rank_desc = f"IV rank >= {iv_rank_min:.0f}" if iv_rank_min > 0 else "IV rank off"
    print(f"Screen: {min_dte}-{max_dte} DTE, IV > {iv_threshold:.2f}, {iv_rv_desc}, {iv_rank_desc}, "
          f"delta {DELTA_MIN:.2f}-{DELTA_MAX:.2f}, OI >= {MIN_OPEN_INTEREST}, "
          f"vol >= {MIN_VOLUME}, spread <= {MAX_SPREAD_PCT * 100:.0f}% of mid")
    score_desc = ("(1-p) x ann. if expired + p x ann. if called  [neutral on assignment]"
                  if assignment_preference == "neutral"
                  else "ann. return if expired x (1 - prob ITM)  [prefers keeping shares]")
    print(f"Ranked by score = {score_desc}")
    print(f"Premiums modeled at mid minus {FILL_HAIRCUT:.0%} of the half-spread (column 'fill')")
    if context.get("from_cache"):
        age_min = (pd.Timestamp.now(tz="UTC") - context["fetched_at"]).total_seconds() / 60
        print(f"Chain data: cached {age_min:.1f} min ago (use --refresh to refetch)")
    print()

    held = get_open_positions(symbol)
    held_contracts = set(held["contract"]) if not held.empty else set()
    if not held.empty:
        print(f"Held positions in {symbol}:")
        print_position_pnl_table([compute_position_pnl(pos, spot) for _, pos in held.iterrows()])
        print()

    cost_basis = context.get("cost_basis")
    if cost_basis is not None:
        print(f"Cost basis: {cost_basis:.2f} (share-weighted across open lots) - "
              f"candidates below this are flagged 'below_basis' (assignment would realize a stock loss)")
        print()

    if results is None or results.empty:
        print("No option candidates matched the current screen.")
        return

    if "contract" in results.columns:
        results.insert(1, "held", results["contract"].map(lambda c: "HOLD" if c in held_contracts else ""))

    results = reorder_columns(results)

    print("Candidates:")
    print_candidates_table(results)
    print()
    print("Read these first: "
          f"{ANSI_GREEN}score{ANSI_RESET} (best trade) | "
          f"pct_exp_ann (annualized income) | "
          f"prob_itm (assignment risk) | "
          f"iv_rv (is premium rich vs. recent moves?) | "
          f"iv_rank (is premium rich vs. this stock's own year) | "
          f"pct_hi6 (strike vs nearest resistance zone) | "
          f"oi (can you fill it?) | "
          f"oi_wall_ok (strike beyond the heaviest call OI) | "
          f"div_risk (early assignment) | "
          f"below_basis (locks in a stock loss if assigned)")
    print(f"{ANSI_GREEN}green{ANSI_RESET} = favorable   "
          f"{ANSI_YELLOW}amber{ANSI_RESET} = check it   "
          f"{ANSI_RED}red{ANSI_RESET} = warning")


def print_usage():
    print(f"""Usage:
  python chain.py [SYMBOL] [CASH_TO_INVEST] [--iv N] [--iv-rv N] [--iv-rank N] [--min-dte N] [--max-dte N] [--assign MODE] [--refresh]
      Screen SYMBOL (default {DEFAULT_SYMBOL}) for covered-call candidates.
      CASH_TO_INVEST (optional) shows how many contracts that cash covers.
      --iv N        Absolute implied-vol floor (default {IV_THRESHOLD:.2f}).
      --iv-rv N     Minimum implied/realized vol ratio (default {IV_RV_RATIO_MIN:.2f}).
                    1.00 = IV exactly matches how the stock actually moves.
                    Use 0 to disable the ratio test entirely.
      --iv-rank N   Minimum IV rank, 0-100 (default {IV_RANK_MIN}): where today's IV
                    sits in this stock's own 52-week IV range. Sourced from a
                    community-maintained dataset (dolthub.com/post-no-preference/
                    options) - if it's unreachable, iv_rank is blank and never
                    excludes a candidate regardless of this setting. 0 disables.
      --min-dte N   Minimum days to expiration (default {MIN_DTE}).
      --max-dte N   Maximum days to expiration (default {MAX_DTE}).
      --assign MODE 'keep' (default) ranks to avoid assignment; 'neutral' ranks by
                    true expected value across both outcomes, which favors
                    closer-to-the-money strikes.
      --refresh     Force a refetch. Chain data is otherwise cached for
                    {CHAIN_CACHE_TTL_MINUTES} minutes, so re-running with different
                    thresholds costs no Yahoo API calls at all.
      Example: python chain.py AMD 5000
      Example: python chain.py AMD --iv 0.60 --iv-rv 0     (original pre-IV/RV screen)
      Example: python chain.py AMD --assign neutral        (indifferent to assignment)
      Example: python chain.py AMD --min-dte 10 --max-dte 21
      Example: python chain.py AMD --iv-rank 50             (only sell when IV is rich for AMD)

  python chain.py log SYMBOL CONTRACT STOCK_PRICE SHARES PREMIUM_COLLECTED CONTRACTS [--new-lot]
      Record a new covered-call position after you buy the stock and sell the call.
      PREMIUM_COLLECTED is the total $ received (mid * 100 * contracts), not a per-share quote.
      If you still hold shares from an earlier entry on this symbol (its last leg wasn't
      'assigned'), this automatically continues that lot - see `lots` below - rather than
      starting a fresh one, even after a gap where you used `close` instead of `roll`.
      --new-lot forces a fresh lot anyway (e.g. you sold those shares outside this tool,
      without ever running `assign`, so this tool has no way to know they're gone).
      Example: python chain.py log AMD AMD260116C00150000 145.32 100 320.00 1

  python chain.py positions
      Show all open positions with current stock/option prices and unrealized P&L.

  python chain.py close CONTRACT CLOSE_PREMIUM
      You bought the call back, or it expired worthless. You KEEP the shares.
      CLOSE_PREMIUM is the total $ paid to close (0 if it expired worthless).
      Example: python chain.py close AMD260116C00150000 45.00

  python chain.py assign CONTRACT
      The call was exercised and your shares were CALLED AWAY. Records both legs:
      the full premium kept, plus the stock sold at the strike against your basis.
      Do not use `close` for this - it would discard the stock leg entirely.
      Example: python chain.py assign AMD260116C00150000

  python chain.py roll OLD_CONTRACT CLOSE_PREMIUM NEW_CONTRACT NEW_PREMIUM
      Close the current call and open a replacement as one linked pair, carrying the
      stock cost basis over and accumulating premium across the whole roll chain.
      Example: python chain.py roll AMD260814C00545000 210 AMD260821C00560000 380

  python chain.py history [SYMBOL]
      Show closed and rolled positions with realized P&L, one row per leg. No
      network calls. SYMBOL (optional) filters to one ticker.
      Example: python chain.py history AMD

  python chain.py lots [SYMBOL]
      Cumulative return per lot - a continuous share-holding, summed across every
      leg against it (logged, rolled, or closed), regardless of gaps. This is
      "how has this stock actually done overall" - `history` above is per-leg;
      this is per-lot. No network calls. SYMBOL (optional) filters to one ticker.
      Example: python chain.py lots AMD

  python chain.py backfill [SYMBOL] [--recalc]
      Fills BLANK entry_* screen fundamentals (via the historical DoltHub
      dataset, for contracts that left the live chain before they could be
      captured at log time) and BLANK cumulative_premium cells (via the
      lot-aware running total). By default never touches an already-populated
      cell. --recalc additionally corrects cumulative_premium cells that ARE
      populated but predate lot-aware accounting (e.g. logged with a gap
      before this tool understood lots) and no longer match the running total.
      SYMBOL (optional) limits it to one ticker.
      Example: python chain.py backfill AMD
      Example: python chain.py backfill AMD --recalc

  python chain.py help | -h | --help
      Show this message.
""")


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("help", "-h", "--help"):
        print_usage()
        return

    try:
        iv_threshold, iv_rv_ratio_min, iv_rank_min, min_dte, max_dte, use_cache, assignment_preference = \
            parse_threshold_flags(sys.argv)
    except ValueError as exc:
        print(exc)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "log":
        cmd_log(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "close":
        cmd_close(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "assign":
        cmd_assign(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "roll":
        cmd_roll(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "positions":
        cmd_positions()
    elif len(sys.argv) > 1 and sys.argv[1] == "history":
        cmd_history(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "lots":
        cmd_lots(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "backfill":
        cmd_backfill(sys.argv[2:])
    else:
        print_results(iv_threshold, iv_rv_ratio_min, use_cache, assignment_preference, min_dte, max_dte, iv_rank_min)


if __name__ == "__main__":
    main()
