import contextlib
import io
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import chain as chain_mod
import screen_oom as oom_mod
import backtest as backtest_mod

st.set_page_config(page_title="Covered Call Screener", layout="wide")


def usd(amount):
    """Money for Streamlit markdown, with the '$' escaped.

    Streamlit treats '$...$' as LaTeX math delimiters, so a caption containing two
    unescaped dollar amounts silently renders everything between them as math and
    swallows the surrounding formatting. Escaping keeps it literal.
    """
    return f"\\${amount:,.2f}"


def as_markdown_block(text):
    """Multi-line CLI output -> markdown that keeps its line breaks.

    Markdown collapses single newlines into one paragraph, and indented lines can
    turn into code blocks, so strip each line and join with explicit hard breaks.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return "  \n".join(lines)

POSITION_FLOAT_COLS = [
    "stock_entry", "stock_now", "stock_pl", "prem_coll", "opt_now", "opt_pl", "total_pl",
    "if_exp_pl", "pct_if_exp", "if_call_pl", "pct_if_call", "ann_if_exp", "ann_if_call",
    "pct_max", "delta_now",
]
PNL_COLOR_COLUMNS = ["stock_pl", "opt_pl", "total_pl", "if_exp_pl", "if_call_pl"]
ACTION_STYLES = {
    "EXPIRED": "color: #e74c3c; font-weight: bold",
    "TESTED": "color: #e74c3c; font-weight: bold",
    "ROLL WATCH": "color: #f39c12; font-weight: bold",
    "TAKE PROFIT": "color: #2ecc71; font-weight: bold",
    "MANAGE": "color: #f39c12; font-weight: bold",
}


def color_pnl(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if v > 0:
        return "color: #2ecc71"
    if v < 0:
        return "color: #e74c3c"
    return ""


def color_action(value):
    return ACTION_STYLES.get(value, "")


# Full-length column headers for the GUI. The CLI keeps the short names (chain.py /
# screen_oom.py print to a fixed-width terminal); this dict is GUI-only, applied via
# st.column_config so the underlying DataFrame columns - and every styling function
# above that references them by short name - are untouched.
COLUMN_LABELS = {
    "*": "Flag", "held": "Held", "sym": "Symbol", "symbol": "Symbol", "name": "Company",
    "contract": "Contract", "exp": "Expiration", "dte": "Days to Expiry", "dte_left": "Days to Expiry",
    "score": "Score", "strike": "Strike", "stk": "Strike", "spot": "Spot Price",
    "day_chg": "Day Change %",
    "hi6": "Resistance", "sh$": "Cash Required", "cash_req": "Cash Required",
    "dist_spot": "Dist. to Spot", "d_spot": "Dist. to Spot",
    "pct_spot": "% to Spot", "p_spot": "% to Spot",
    "dist_hi6": "Dist. to Resistance", "d_hi6": "Dist. to Resistance",
    "pct_hi6": "% to Resistance", "p_hi6": "% to Resistance",
    "hit6": "Resistance Touches", "last": "Last Price", "bid": "Bid", "ask": "Ask", "mid": "Mid Price",
    "oi_wall": "Call OI Wall", "oi_wall_oi": "OI at Wall", "oi_wall_ok": "Above OI Wall",
    "mid$": "Premium at Mid ($)", "fill": "Modeled Fill", "fill$": "Premium ($)",
    "called$": "If Called ($)", "delta_now": "Delta Now",
    "pct_exp": "Return if Expired %", "p_exp": "Return if Expired %",
    "pct_call": "Return if Called %", "p_call": "Return if Called %",
    "pct_exp_ann": "Ann. Return if Expired %", "p_exp_ann": "Ann. Return if Expired %",
    "pct_call_ann": "Ann. Return if Called %", "p_call_ann": "Ann. Return if Called %",
    "div_risk": "Dividend Risk", "div_amt": "Dividend ($)",
    "basis": "Cost Basis ($)", "below_basis": "Below Cost Basis",
    "sprd": "Spread", "spr": "Spread", "volum": "Volume", "vol": "Volume", "oi": "Open Interest",
    "impvol": "Implied Vol", "iv": "Implied Vol", "iv_rv": "IV / Realized Vol",
    "iv_rank": "IV Rank",
    "delta": "Delta", "prob_itm": "Prob. ITM %", "p_itm": "Prob. ITM %",
    "prob_touch": "Prob. Touch %", "p_touch": "Prob. Touch %",
    "premium": "Premium > $1", "gt1": "Premium > $1",
    "gt1k": "Total Premium >= $1000",
    "contracts": "# Contracts", "cash_used": "Cash Used", "cash_left": "Cash Left",
    "tot_prem": "Total Premium ($)",
    "action": "Action", "pct_max": "% of Max Profit", "days_held": "Days Held",
    "stock_entry": "Stock Entry ($)", "stock_now": "Stock Now ($)", "stock_pl": "Stock P&L ($)",
    "prem_coll": "Premium Collected ($)", "cum_prem": "Cumulative Premium ($)",
    "opt_now": "Option Value Now ($)", "opt_pl": "Option P&L ($)", "total_pl": "Total P&L ($)",
    "if_exp_pl": "P&L if Expired ($)", "pct_if_exp": "% if Expired",
    "if_call_pl": "P&L if Called ($)", "pct_if_call": "% if Called",
    "ann_if_exp": "Ann. % if Expired", "ann_if_call": "Ann. % if Called",
    # History table
    "status": "Status", "entry_date": "Entry Date", "close_date": "Close Date",
    "premium_collected": "Premium Collected ($)", "close_premium": "Close Premium ($)",
    "realized_pl": "Option P&L ($)", "rolled_into": "Rolled Into",
    "stock_realized_pl": "Stock P&L if Assigned ($)", "total_realized": "Total Realized ($)",
    "pct_captured": "% of Premium Captured",
    "entry_score": "Entry Score", "entry_iv": "Entry IV", "entry_iv_rv": "Entry IV / RV",
    "entry_delta": "Entry Delta", "entry_prob_itm": "Entry Prob. ITM %",
    "entry_pct_hi6": "Entry % to Resistance",
    # Lots table
    "lot_id": "Lot", "legs": "Legs", "opened": "Opened",
    "option_pl": "Option P&L ($)", "stock_pl": "Stock P&L ($)",
    "open_premium": "Open Premium at Risk ($)",
    # Backtest table
    "expiration": "Expiration", "entry_spot": "Spot at Entry", "exit_spot": "Spot at Expiration",
    "premium": "Premium ($)", "outcome": "Outcome", "pl_dollars": "P&L ($)",
    "realized_pct": "Realized Return %", "realized_ann": "Realized Ann. Return %",
    "holding_days": "Days Held", "score_at_entry": "Entry Score",
    "iv_at_entry": "Entry IV", "iv_rv_at_entry": "Entry IV/RV", "iv_rank_at_entry": "Entry IV Rank",
    "delta_at_entry": "Entry Delta", "prob_itm_at_entry": "Entry Prob. ITM %",
    # Sweep table
    "iv_threshold": "IV Floor", "iv_rv_ratio_min": "Min IV/RV", "iv_rank_min": "Min IV Rank",
    "assignment_preference": "Assign Mode", "n_cycles": "Cycles", "win_rate": "Win Rate %",
    "assign_rate": "Assigned %", "avg_realized_ann": "Avg Realized Ann. Return %",
    "avg_score": "Avg Entry Score", "correlation": "Score-Return Correlation",
}
# Columns you asked to be able to spot at a glance: how many contracts, and the
# dollar premium involved. Background highlight is deliberately a neutral color,
# distinct from the green/amber/red value-based coloring used elsewhere, so the
# two visual systems ("this column matters" vs "this value is good/bad") don't
# get confused with each other.
HIGHLIGHT_PROPS = {"background-color": "rgba(52, 152, 219, 0.18)", "font-weight": "600"}
CANDIDATE_HIGHLIGHT_COLS = ["fill$", "called$", "contracts", "tot_prem"]
POSITION_HIGHLIGHT_COLS = ["prem_coll", "cum_prem"]


def get_column_config(df):
    return {col: st.column_config.Column(label=COLUMN_LABELS[col])
            for col in df.columns if col in COLUMN_LABELS}


GREEN, AMBER, RED = "color: #2ecc71", "color: #f39c12", "color: #e74c3c"
# No HTML entities here: st.caption renders markdown but not raw HTML, so &nbsp;
# would show up literally.
LEGEND_MD = (
    "**Read these first:** Score = best risk-adjusted trade | "
    "Ann. Return = annualized income | Prob. ITM = assignment risk | "
    "IV / Realized Vol = is the premium rich vs. recent moves? | "
    "IV Rank = is the premium rich vs. this stock's own year? | "
    "% to Resistance = strike vs nearest overhead swing-high zone | "
    "Open Interest = can you fill it? | Above OI Wall = strike beyond the heaviest call OI | "
    "Dividend Risk = early assignment | "
    "Below Cost Basis = locks in a stock loss if assigned.  \n"
    ":green[green] = favorable, :orange[amber] = check it, :red[red] = warning, "
    ":blue[blue highlight] = premium / contract size"
)
# Numeric formatting for the candidates table (kept numeric upstream so it can be styled).
CANDIDATE_FLOAT_COLS = [
    "dte", "score", "strike", "spot", "hi6", "dist_spot", "pct_spot", "dist_hi6", "pct_hi6",
    "hit6", "oi_wall", "oi_wall_oi", "last", "bid", "ask", "mid", "fill", "fill$", "called$", "cash_req",
    "pct_exp", "pct_call", "pct_exp_ann", "pct_call_ann", "div_amt", "basis", "sprd", "volum", "oi",
    "impvol", "iv_rv", "iv_rank", "delta", "prob_itm", "prob_touch", "premium",
    # screen_oom column names
    "stk", "sh$", "d_spot", "p_spot", "d_hi6", "p_hi6", "spr", "vol", "iv",
    "p_itm", "p_touch", "p_exp", "p_call", "p_exp_ann", "p_call_ann",
    "contracts", "cash_used", "cash_left", "tot_prem", "day_chg",
]


def _num(value):
    try:
        v = float(value)
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return None


# Shared decision-color rules. Module-level (not nested in style_candidates_df) so
# style_history_df can apply the exact same thresholds to a closed trade's entry
# fundamentals as were used when it was a live candidate - a trade should read the
# same way in hindsight as it did at the moment it was picked.
def ann_return_style(v):
    n = _num(v)
    if n is None:
        return ""
    return GREEN if n >= 30 else (AMBER if n >= 15 else "")


def prob_itm_style(v):
    n = _num(v)
    if n is None:
        return ""
    return GREEN if n <= 25 else (AMBER if n <= 40 else RED)


def iv_rv_style(v):
    n = _num(v)
    if n is None:
        return ""
    return GREEN if n >= 1.0 else (AMBER if n >= 0.90 else RED)


def iv_rank_style(v):
    # Where IV sits in this stock's own 52-week range - wider bands than iv_rv
    # since it's a timing signal, not a hard gate. Blank when the optional
    # community data source (voldata.py) is unavailable.
    n = _num(v)
    if n is None:
        return ""
    return GREEN if n >= 50 else (AMBER if n >= 25 else RED)


def resistance_style(v):
    # Strike above the nearest overhead swing-high resistance zone.
    n = _num(v)
    if n is None:
        return ""
    return GREEN if n > 0 else RED


def oi_wall_ok_style(v):
    # Strike at/beyond the heaviest OTM call OI in this expiration - a proxy
    # for dealer-hedging resistance. Blank (not red) when no wall was found,
    # since a missing signal must never read as a bad one.
    #
    # NOTE: a boolean pulled out of a DataFrame column is numpy.bool_, not the
    # Python `bool` singleton, so `v is True`/`v is False` silently never
    # matches - it always fell through here. Check pd.isna first (a value
    # here can genuinely be pd.NA when no wall exists), then use plain
    # truthiness, which works correctly for numpy.bool_.
    if pd.isna(v):
        return ""
    return GREEN if v else RED


def oi_style(v):
    n = _num(v)
    if n is None:
        return ""
    return GREEN if n >= 500 else (AMBER if n < 100 else "")


def delta_style(v):
    # DELTA_MIN/MAX already gate every row shown to a narrow band, so "in
    # range" tells you nothing - color proximity to TARGET_DELTA instead,
    # normalized by the distance to whichever edge it leans toward. Mirrors
    # chain.candidate_cell_color's delta branch.
    n = _num(v)
    if n is None:
        return ""
    target, dmin, dmax = chain_mod.TARGET_DELTA, chain_mod.DELTA_MIN, chain_mod.DELTA_MAX
    span = (dmax - target) if n >= target else (target - dmin)
    fraction = abs(n - target) / span if span > 0 else 0
    return GREEN if fraction <= 1 / 3 else (AMBER if fraction <= 2 / 3 else RED)


def div_risk_style(v):
    # See oi_wall_ok_style's note: `is True` never matches a DataFrame-derived
    # numpy.bool_, so this silently never colored a real warning red before.
    if pd.isna(v):
        return ""
    return f"{RED}; font-weight: bold" if v else ""


def below_basis_style(v):
    if pd.isna(v):
        return ""
    return f"{RED}; font-weight: bold" if v else ""


def day_chg_style(v):
    # Not a pass/fail gate like the others, just a same-day-entry caution.
    n = _num(v)
    return RED if (n is not None and n < 0) else ""


def style_candidates_df(df):
    """Color only the columns that actually decide the trade; everything else is detail."""
    df = chain_mod.reorder_columns(df)
    best_score = None
    if "score" in df.columns:
        scores = pd.to_numeric(df["score"], errors="coerce")
        best_score = scores.max() if not scores.isna().all() else None

    def score_style(v):
        n = _num(v)
        return GREEN if (n is not None and best_score and n >= best_score * 0.95) else ""

    rules = {
        "score": score_style,
        "pct_exp_ann": ann_return_style, "p_exp_ann": ann_return_style,
        "prob_itm": prob_itm_style, "p_itm": prob_itm_style,
        "delta": delta_style,
        "iv_rv": iv_rv_style,
        "iv_rank": iv_rank_style,
        "pct_hi6": resistance_style, "p_hi6": resistance_style,
        "oi": oi_style,
        "oi_wall_ok": oi_wall_ok_style,
        "div_risk": div_risk_style,
        "below_basis": below_basis_style,
        "day_chg": day_chg_style,
    }

    float_cols = [c for c in CANDIDATE_FLOAT_COLS if c in df.columns]
    styler = df.style.format({c: "{:.2f}" for c in float_cols}, na_rep="")
    for col, fn in rules.items():
        if col in df.columns:
            styler = styler.map(fn, subset=[col])
    highlight_cols = [c for c in CANDIDATE_HIGHLIGHT_COLS if c in df.columns]
    if highlight_cols:
        styler = styler.set_properties(subset=highlight_cols, **HIGHLIGHT_PROPS)
    return styler


def style_positions_df(df):
    float_cols = [c for c in POSITION_FLOAT_COLS if c in df.columns]
    color_cols = [c for c in PNL_COLOR_COLUMNS if c in df.columns]
    fmt = {c: "{:.2f}" for c in float_cols}
    styler = df.style.format(fmt, na_rep="N/A")
    if color_cols:
        styler = styler.map(color_pnl, subset=color_cols)
    if "action" in df.columns:
        styler = styler.map(color_action, subset=["action"])
    highlight_cols = [c for c in POSITION_HIGHLIGHT_COLS if c in df.columns]
    if highlight_cols:
        styler = styler.set_properties(subset=highlight_cols, **HIGHLIGHT_PROPS)
    return styler


HISTORY_FLOAT_COLS = [
    "days_held", "strike", "entry_score", "entry_iv", "entry_iv_rv", "entry_delta",
    "entry_prob_itm", "entry_pct_hi6", "premium_collected", "close_premium",
    "realized_pl", "stock_realized_pl", "total_realized", "pct_captured",
]
HISTORY_HIGHLIGHT_COLS = ["premium_collected", "close_premium"]


def status_style(value):
    return "color: #f39c12; font-weight: bold" if value in ("rolled", "assigned") else ""


def style_history_df(df):
    float_cols = [c for c in HISTORY_FLOAT_COLS if c in df.columns]
    styler = df.style.format({c: "{:.2f}" for c in float_cols}, na_rep="")
    # Entry fundamentals reuse the live-screen thresholds so a closed trade reads
    # the same way in hindsight as it did when it was a candidate.
    history_rules = {
        "realized_pl": color_pnl,
        "stock_realized_pl": color_pnl,
        "total_realized": color_pnl,
        "pct_captured": color_pnl,
        "status": status_style,
        "entry_iv_rv": iv_rv_style,
        "entry_prob_itm": prob_itm_style,
        "entry_pct_hi6": resistance_style,
    }
    for col, fn in history_rules.items():
        if col in df.columns:
            styler = styler.map(fn, subset=[col])
    highlight_cols = [c for c in HISTORY_HIGHLIGHT_COLS if c in df.columns]
    if highlight_cols:
        styler = styler.set_properties(subset=highlight_cols, **HIGHLIGHT_PROPS)
    return styler


LOTS_FLOAT_COLS = ["legs", "option_pl", "stock_pl", "total_realized", "open_premium"]
LOTS_HIGHLIGHT_COLS = ["open_premium"]


def lot_status_style(value):
    return "color: #f39c12; font-weight: bold" if value in ("open", "assigned") else ""


def style_lots_df(df):
    float_cols = [c for c in LOTS_FLOAT_COLS if c in df.columns]
    styler = df.style.format({c: "{:.2f}" for c in float_cols}, na_rep="")
    lots_rules = {
        "option_pl": color_pnl,
        "stock_pl": color_pnl,
        "total_realized": color_pnl,
        "status": lot_status_style,
    }
    for col, fn in lots_rules.items():
        if col in df.columns:
            styler = styler.map(fn, subset=[col])
    highlight_cols = [c for c in LOTS_HIGHLIGHT_COLS if c in df.columns]
    if highlight_cols:
        styler = styler.set_properties(subset=highlight_cols, **HIGHLIGHT_PROPS)
    return styler


BACKTEST_FLOAT_COLS = [
    "strike", "entry_spot", "exit_spot", "premium", "pl_dollars", "realized_pct",
    "realized_ann", "holding_days", "score_at_entry", "iv_at_entry", "iv_rv_at_entry",
    "iv_rank_at_entry", "delta_at_entry", "prob_itm_at_entry",
]
BACKTEST_HIGHLIGHT_COLS = ["premium", "pl_dollars"]


def outcome_style(value):
    return "color: #e74c3c; font-weight: bold" if value == "ASSIGNED" else "color: #2ecc71; font-weight: bold"


def style_backtest_df(df):
    float_cols = [c for c in BACKTEST_FLOAT_COLS if c in df.columns]
    styler = df.style.format({c: "{:.2f}" for c in float_cols}, na_rep="")
    rules = {"outcome": outcome_style, "realized_ann": color_pnl, "pl_dollars": color_pnl}
    for col, fn in rules.items():
        if col in df.columns:
            styler = styler.map(fn, subset=[col])
    highlight_cols = [c for c in BACKTEST_HIGHLIGHT_COLS if c in df.columns]
    if highlight_cols:
        styler = styler.set_properties(subset=highlight_cols, **HIGHLIGHT_PROPS)
    return styler


SWEEP_FLOAT_COLS = [
    "iv_threshold", "iv_rv_ratio_min", "iv_rank_min", "n_cycles", "win_rate",
    "assign_rate", "avg_realized_ann", "avg_score", "correlation",
]
SWEEP_HIGHLIGHT_COLS = ["win_rate"]


def win_rate_style(v):
    # Unlike P&L, win_rate is never negative, so color_pnl's >0/<0 split would
    # just paint everything green - tiered bands instead, same idea as
    # backtest.py's CLI sweep coloring.
    n = _num(v)
    if n is None:
        return ""
    return GREEN if n >= 60 else (AMBER if n >= 40 else RED)


def correlation_style(v):
    n = _num(v)
    if n is None:
        return ""
    return GREEN if n >= 0.3 else (AMBER if n >= 0 else RED)


def style_sweep_df(df):
    float_cols = [c for c in SWEEP_FLOAT_COLS if c in df.columns]
    styler = df.style.format({c: "{:.2f}" for c in float_cols}, na_rep="")
    rules = {"win_rate": win_rate_style, "avg_realized_ann": color_pnl, "correlation": correlation_style}
    for col, fn in rules.items():
        if col in df.columns:
            styler = styler.map(fn, subset=[col])
    highlight_cols = [c for c in SWEEP_HIGHLIGHT_COLS if c in df.columns]
    if highlight_cols:
        styler = styler.set_properties(subset=highlight_cols, **HIGHLIGHT_PROPS)
    return styler


def build_sweep_distribution_chart(results, metric, default_value=None):
    """Histogram of `metric` across every swept combination - the ranked
    table only ever shows the cherry-picked top of this, which hides whether
    the winner is a clear standout or barely distinguishable from the pack.
    The current-config combination (find_default_combo_row), if present in
    the grid, gets a highlighted vertical rule - same emphasis pattern as
    the resistance chart's active zone (gray = context, amber = the one
    point that's actually "us")."""
    label = COLUMN_LABELS.get(metric, metric)
    data = results[[metric]].dropna().rename(columns={metric: "value"})

    hist = alt.Chart(data).mark_bar(color=RESISTANCE_LINE_COLOR).encode(
        x=alt.X("value:Q", bin=alt.Bin(maxbins=20), title=label),
        y=alt.Y("count():Q", title="Combinations"),
        tooltip=[alt.Tooltip("count():Q", title="Combinations")],
    )
    layers = [hist]

    if default_value is not None and pd.notna(default_value):
        rule_df = pd.DataFrame([{"value": float(default_value)}])
        layers.append(
            alt.Chart(rule_df).mark_rule(color=RESISTANCE_ACTIVE_COLOR, strokeWidth=2.5).encode(
                x="value:Q", tooltip=[alt.Tooltip("value:Q", title="Current config", format=".2f")],
            )
        )

    return alt.layer(*layers).properties(height=280)


RESISTANCE_LINE_COLOR = "#95a5a6"   # context zones - gray, the rest-are-context form
RESISTANCE_ACTIVE_COLOR = "#f39c12"  # the one zone actually driving pct_hi6 - amber, matches AMBER elsewhere
SUPPORT_ACTIVE_COLOR = "#16a085"    # the floor below spot - teal, distinct from every other line on the chart
OI_WALL_COLOR = "#8e44ad"           # a different signal family from price resistance - kept visually distinct
STRIKE_COLOR = "#2ecc71"            # matches GREEN used for "favorable" everywhere else
SPOT_COLOR = "#2980b9"
PRICE_LINE_COLOR = "#2c3e50"
MAX_CONTEXT_ZONES = 4  # every swing high above spot, drawn at once, read as a plate of spaghetti - keep only the most-touched few
CONTEXT_RELEVANCE_PCT = 0.35  # a pre-runup/crash swing high 60% away from spot isn't context, it's an axis-wrecking outlier

# One shared vocabulary for every reference line so they can live in a single
# combined dataframe below and Altair renders one real legend for all of them,
# instead of the hand-rolled markdown caption this used to need.
_LEVEL_STYLE = {
    "Other swing highs": {"color": RESISTANCE_LINE_COLOR, "dash": [1, 4]},
    "Call OI wall": {"color": OI_WALL_COLOR, "dash": [6, 3]},
    "Best candidate strike": {"color": STRIKE_COLOR, "dash": [1, 0]},
    "Nearest support": {"color": SUPPORT_ACTIVE_COLOR, "dash": [1, 0]},
    "Nearest resistance": {"color": RESISTANCE_ACTIVE_COLOR, "dash": [1, 0]},
    "Spot": {"color": SPOT_COLOR, "dash": [1, 0]},
}


def build_resistance_chart(
    hist, zones, spot, resistance_price=None, oi_wall=None, best_strike=None, support_price=None,
    history_days=90,
):
    """Price history with the current spot, the swing-high resistance zone
    actually nearest-above-spot (driving pct_hi6 in the candidates table), the
    nearest swing-low support zone below spot, the call-OI wall (a different
    signal family), and the best candidate's strike, each a horizontal
    reference line. Every other detected swing high is context, capped at
    MAX_CONTEXT_ZONES (by touch count) and drawn thin and faint so it doesn't
    compete with the lines that actually matter. Support only ever shows its
    single nearest zone - plotting every swing low too would undo the very
    decluttering this chart needed in the first place.

    `hist` comes in as the full ~9mo fetch (zone detection needs that much
    history to find swing highs) but the chart itself only plots the trailing
    `history_days` - the rest is just clutter that predates anything the
    trade cares about.
    """
    price_df = hist[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
    price_df["date"] = pd.to_datetime(price_df["date"])
    if history_days:
        cutoff = price_df["date"].max() - pd.Timedelta(days=history_days)
        price_df = price_df[price_df["date"] > cutoff]

    def _is_relevant(price):
        return abs(price - spot) / spot <= CONTEXT_RELEVANCE_PCT

    rows = []
    if zones:
        zones_df = pd.DataFrame(zones)
        zones_df["last_date"] = pd.to_datetime(zones_df["last_date"])
        is_active = pd.Series(False, index=zones_df.index)
        if resistance_price is not None and pd.notna(resistance_price):
            is_active = (zones_df["price"] - resistance_price).abs() < 0.01

        for _, z in zones_df[is_active].iterrows():
            rows.append({
                "kind": "Nearest resistance", "price": float(z["price"]),
                "detail": f"{int(z['touches'])} touches · last touched {z['last_date']:%b %d}",
                "width": 2.5, "opacity": 0.95,
            })

        # Relevance-filtered before ranking, not after - a stale pre-runup/crash
        # swing high 60% away from spot has nothing to do with today's decision,
        # and once it's in the top-MAX_CONTEXT_ZONES-by-touches it drags the
        # whole y-axis out to include it, bunching the actually-relevant price
        # action into a sliver at one edge.
        context_zones = zones_df[~is_active]
        context_zones = context_zones[context_zones["price"].apply(_is_relevant)]
        context_zones = context_zones.sort_values("touches", ascending=False).head(MAX_CONTEXT_ZONES)
        max_touches = context_zones["touches"].max() if not context_zones.empty else 0
        for _, z in context_zones.iterrows():
            frac = (z["touches"] / max_touches) if max_touches else 0.5
            rows.append({
                "kind": "Other swing highs", "price": float(z["price"]),
                "detail": f"{int(z['touches'])} touches · last touched {z['last_date']:%b %d}",
                "width": 0.75 + 0.5 * frac, "opacity": 0.18 + 0.22 * frac,
            })

    if oi_wall is not None and pd.notna(oi_wall) and _is_relevant(oi_wall):
        rows.append({
            "kind": "Call OI wall", "price": float(oi_wall), "detail": "dealer-hedging proxy",
            "width": 1.75, "opacity": 0.85,
        })
    if best_strike is not None and pd.notna(best_strike) and _is_relevant(best_strike):
        rows.append({
            "kind": "Best candidate strike", "price": float(best_strike), "detail": "top-ranked call",
            "width": 1.75, "opacity": 0.85,
        })
    if support_price is not None and pd.notna(support_price):
        rows.append({
            "kind": "Nearest support", "price": float(support_price), "detail": "nearest swing-low floor",
            "width": 2.5, "opacity": 0.95,
        })
    rows.append({
        "kind": "Spot", "price": float(spot), "detail": "current price",
        "width": 1.5, "opacity": 0.8,
    })

    # An explicit y-domain, sized to whatever's actually being plotted (price
    # history + every reference line), instead of letting Altair pick one.
    # The area mark below has an implicit baseline at 0, which would otherwise
    # drag the auto-computed domain all the way down to $0 and squeeze the
    # entire meaningful price range into a sliver at the top of the chart.
    all_prices = list(price_df["close"]) + [r["price"] for r in rows]
    lo, hi = min(all_prices), max(all_prices)
    pad = max((hi - lo) * 0.08, 0.5)
    y_scale = alt.Scale(domain=[lo - pad, hi + pad], zero=False, nice=False)

    price_area = alt.Chart(price_df).mark_area(color=PRICE_LINE_COLOR, opacity=0.08, clip=True).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("close:Q", title="Price ($)", scale=y_scale),
    )
    price_line = alt.Chart(price_df).mark_line(color=PRICE_LINE_COLOR, strokeWidth=1.75).encode(
        x="date:T",
        y=alt.Y("close:Q", scale=y_scale),
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("close:Q", title="Close", format="$.2f")],
    )

    levels_df = pd.DataFrame(rows)
    present = [k for k in _LEVEL_STYLE if k in levels_df["kind"].unique()]
    levels_layer = alt.Chart(levels_df).mark_rule().encode(
        y="price:Q",
        color=alt.Color(
            "kind:N", sort=present,
            scale=alt.Scale(domain=present, range=[_LEVEL_STYLE[k]["color"] for k in present]),
            legend=alt.Legend(title=None, orient="bottom", direction="horizontal", symbolType="stroke"),
        ),
        strokeDash=alt.StrokeDash(
            "kind:N", sort=present,
            scale=alt.Scale(domain=present, range=[_LEVEL_STYLE[k]["dash"] for k in present]), legend=None,
        ),
        strokeWidth=alt.StrokeWidth("width:Q", scale=None, legend=None),
        opacity=alt.Opacity("opacity:Q", scale=None, legend=None),
        tooltip=[
            alt.Tooltip("kind:N", title="Level"),
            alt.Tooltip("price:Q", title="Price", format="$.2f"),
            alt.Tooltip("detail:N", title="Detail"),
        ],
    )

    return (
        alt.layer(price_area, price_line, levels_layer)
        .properties(height=380)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=True, gridOpacity=0.12, domain=False, tickSize=3)
        .configure_legend(labelFontSize=11, symbolStrokeWidth=2.5)
        .interactive()
    )


def get_positions_with_pnl(symbol=None):
    open_positions = chain_mod.get_open_positions(symbol)
    if open_positions.empty:
        return pd.DataFrame()
    spot_cache = {}
    rows = []
    for _, pos in open_positions.iterrows():
        sym = pos["symbol"]
        if sym not in spot_cache:
            spot_cache[sym] = chain_mod.get_spot_price(chain_mod.yf.Ticker(sym))
        rows.append(chain_mod.compute_position_pnl(pos, spot_cache[sym]))
    return pd.DataFrame(rows)


def run_captured(func, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = func(*args, **kwargs)
    return result, buf.getvalue()


st.title("Covered Call Screener")

tab_screener, tab_watchlist, tab_positions, tab_backtest = st.tabs(
    ["Screener", "Watchlist Screener", "Positions", "Backtest"]
)

with tab_screener:
    # A ticker selected in the Watchlist tab preloads this tab and reruns (see
    # below) - by the time this pass starts, screener_result is already set,
    # so this only needs to surface a confirmation that it arrived.
    screener_flash = st.session_state.pop("screener_flash", None)
    if screener_flash:
        st.info(screener_flash)

    st.subheader("Single-Symbol Screener")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        symbol_input = st.text_input("Symbol", value=chain_mod.DEFAULT_SYMBOL, key="screener_symbol")
    with col2:
        cash_input = st.number_input(
            "Cash to invest (optional, 0 = ignore)", min_value=0.0, value=0.0, step=100.0, key="screener_cash"
        )
    with col3:
        st.write("")
        st.write("")
        run_clicked = st.button("Run Screener", type="primary")

    thr1, thr2, thr3, thr4, thr5 = st.columns(5)
    with thr1:
        iv_input = st.number_input(
            "IV floor", min_value=0.0, max_value=5.0, value=float(chain_mod.IV_THRESHOLD),
            step=0.05, format="%.2f", key="screener_iv",
            help="Absolute implied-vol floor. Set 0.60 to reproduce the original screen.",
        )
    with thr2:
        iv_rv_input = st.number_input(
            "Min IV/RV ratio", min_value=0.0, max_value=5.0, value=float(chain_mod.IV_RV_RATIO_MIN),
            step=0.05, format="%.2f", key="screener_ivrv",
            help="Implied vol as a multiple of realized vol. 1.00 = paid exactly what the "
                 "stock actually moves. 0 disables the test.",
        )
    with thr3:
        iv_rank_input = st.number_input(
            "Min IV Rank", min_value=0.0, max_value=100.0, value=float(chain_mod.IV_RANK_MIN),
            step=5.0, format="%.0f", key="screener_ivrank",
            help="Where today's IV sits (0-100) in this stock's own 52-week IV range. "
                 "From a community-maintained dataset - blank/unfiltered if it's "
                 "unreachable. 0 disables.",
        )
    with thr4:
        min_dte_input = st.number_input(
            "Min DTE", min_value=0, value=int(chain_mod.MIN_DTE), step=1, key="screener_min_dte",
            help="Minimum days to expiration.",
        )
    with thr5:
        max_dte_input = st.number_input(
            "Max DTE", min_value=0, value=int(chain_mod.MAX_DTE), step=1, key="screener_max_dte",
            help="Maximum days to expiration.",
        )

    if run_clicked:
        symbol = symbol_input.strip().upper()
        if not symbol:
            st.error("Enter a symbol.")
        elif min_dte_input > max_dte_input:
            st.error("Min DTE cannot exceed Max DTE.")
        else:
            with st.spinner(f"Scanning {symbol}..."):
                try:
                    spot, context, expirations_checked, expirations_skipped_earnings, company_name, results = (
                        chain_mod.find_candidates(
                            symbol, iv_input, iv_rv_input,
                            min_dte=int(min_dte_input), max_dte=int(max_dte_input),
                            iv_rank_min=iv_rank_input,
                        )
                    )
                    chart_hist, chart_zones, _, chart_support = chain_mod.get_resistance_chart_data(symbol, spot)
                    st.session_state["screener_result"] = {
                        "symbol": symbol,
                        "spot": spot,
                        "context": context,
                        "expirations_checked": expirations_checked,
                        "expirations_skipped_earnings": expirations_skipped_earnings,
                        "company_name": company_name,
                        "results": results,
                        "chart_hist": chart_hist,
                        "chart_zones": chart_zones,
                        "chart_support": chart_support,
                    }
                except Exception as exc:
                    st.error(f"Failed to screen {symbol}: {exc}")

    result = st.session_state.get("screener_result")
    if result:
        symbol = result["symbol"]
        spot = result["spot"]
        context = result["context"]
        cash_required = spot * 100

        st.markdown(f"### {symbol} — {result['company_name']}")
        m1, m2, m3, m4, m5 = st.columns(5)
        day_change_pct = context.get("day_change_pct")
        m1.metric(
            "Spot", f"${spot:.2f}",
            delta=f"{day_change_pct:+.2f}%" if day_change_pct is not None else None,
        )
        m2.metric("Cash required / contract", f"${cash_required:,.2f}")
        resistance_price = context.get("resistance_price")
        m3.metric("Resistance", f"${resistance_price:.2f}" if pd.notna(resistance_price) else "None found")
        support_price = context.get("support_price")
        m4.metric("Support", f"${support_price:.2f}" if pd.notna(support_price) else "None found")
        m5.metric("Expirations checked", result["expirations_checked"])

        if day_change_pct is not None and day_change_pct < 0:
            prev_close = context.get("day_previous_close")
            st.warning(
                f"**{symbol} is trading down {abs(day_change_pct):.2f}% today** "
                f"(prev close {usd(prev_close)}) - you've noted this usually isn't an entry day."
            )

        if cash_input:
            contracts = int(cash_input // cash_required)
            leftover = cash_input - contracts * cash_required
            st.info(f"{usd(cash_input)} covers **{contracts} contract(s)**, leftover {usd(leftover)}")

        if result["expirations_skipped_earnings"]:
            st.warning(
                f"{result['expirations_skipped_earnings']} expiration(s) skipped due to earnings falling in the window."
            )

        held = chain_mod.get_open_positions(symbol)
        if not held.empty:
            st.markdown("#### Held Positions")
            held_rows = [chain_mod.compute_position_pnl(pos, spot) for _, pos in held.iterrows()]
            held_df = pd.DataFrame(held_rows)
            st.dataframe(style_positions_df(held_df), column_config=get_column_config(held_df), width="stretch")

        st.markdown("#### Candidates")
        candidates = result["results"]
        if candidates is None or candidates.empty:
            st.info("No option candidates matched the current screen.")
        else:
            held_contracts = set(held["contract"]) if not held.empty else set()
            display_df = candidates.copy()
            if "contract" in display_df.columns:
                display_df.insert(
                    1, "held", display_df["contract"].map(lambda c: "HOLD" if c in held_contracts else "")
                )
            st.dataframe(
                style_candidates_df(display_df), column_config=get_column_config(display_df), width="stretch"
            )
            st.caption(LEGEND_MD)

        st.markdown("#### Price & Resistance")
        chart_hist = result.get("chart_hist")
        chart_zones = result.get("chart_zones")
        chart_support = result.get("chart_support")
        if chart_hist is not None and not chart_hist.empty:
            cand_df = result["results"]
            best_strike = None
            oi_wall_price = None
            if cand_df is not None and not cand_df.empty:
                if "strike" in cand_df.columns:
                    best_strike = float(cand_df.iloc[0]["strike"])
                if "oi_wall" in cand_df.columns and pd.notna(cand_df.iloc[0].get("oi_wall")):
                    oi_wall_price = float(cand_df.iloc[0]["oi_wall"])
            chart = build_resistance_chart(
                chart_hist, chart_zones, spot,
                resistance_price=context.get("resistance_price"),
                oi_wall=oi_wall_price, best_strike=best_strike,
                support_price=chart_support["price"] if chart_support else None,
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("No price history available to chart.")

with tab_watchlist:
    st.subheader("Watchlist Screener")

    def load_universe_text(source_label):
        if source_label == "Hardcoded defaults (10 tickers)":
            return "\n".join(oom_mod.DEFAULT_TICKERS)
        path = APP_DIR / UNIVERSE_FILES[source_label]
        if path.exists():
            return path.read_text(encoding="utf-8")
        return "\n".join(oom_mod.DEFAULT_TICKERS)

    UNIVERSE_FILES = {
        "tickers.txt": oom_mod.DEFAULT_TICKER_FILE,
        "sp500.txt": "sp500.txt",
    }
    universe_options = list(UNIVERSE_FILES.keys()) + ["Hardcoded defaults (10 tickers)", "Custom / paste below"]

    selected_universe = st.selectbox("Universe source", options=universe_options, key="universe_source_select")

    if st.session_state.get("_last_universe_source") != selected_universe and selected_universe != "Custom / paste below":
        st.session_state["watchlist_tickers"] = load_universe_text(selected_universe)
        st.session_state["_last_universe_source"] = selected_universe

    if "watchlist_tickers" not in st.session_state:
        st.session_state["watchlist_tickers"] = load_universe_text("tickers.txt")

    tickers_text = st.text_area(
        "Tickers (one per line, # for comments)", height=200, key="watchlist_tickers"
    )
    wl1, wl2, wl3, wl4 = st.columns(4)
    with wl1:
        cash_input_wl = st.number_input(
            "Cash to invest (optional, 0 = ignore)", min_value=0.0, value=0.0, step=100.0, key="watchlist_cash"
        )
    with wl2:
        iv_input_wl = st.number_input(
            "IV floor", min_value=0.0, max_value=5.0, value=float(oom_mod.IV_THRESHOLD),
            step=0.05, format="%.2f", key="watchlist_iv",
            help="Absolute implied-vol floor. Set 0.60 to reproduce the original screen.",
        )
    with wl3:
        iv_rv_input_wl = st.number_input(
            "Min IV/RV ratio", min_value=0.0, max_value=5.0, value=float(oom_mod.IV_RV_RATIO_MIN),
            step=0.05, format="%.2f", key="watchlist_ivrv",
            help="Implied vol as a multiple of realized vol. 1.00 = paid exactly what the "
                 "stock actually moves. 0 disables the test.",
        )
    with wl4:
        premium_input_wl = st.number_input(
            "Min total premium ($)", min_value=0.0, value=float(oom_mod.TOTAL_PREMIUM_THRESHOLD),
            step=100.0, format="%.2f", key="watchlist_premium",
            help="With cash to invest set, this is fill x contracts that cash affords; "
                 "otherwise it's a single contract's premium. 0 disables the test.",
        )
    wl5, wl6, wl7 = st.columns(3)
    with wl5:
        min_dte_input_wl = st.number_input(
            "Min DTE", min_value=0, value=int(oom_mod.MIN_DTE), step=1, key="watchlist_min_dte",
            help="Minimum days to expiration.",
        )
    with wl6:
        max_dte_input_wl = st.number_input(
            "Max DTE", min_value=0, value=int(oom_mod.MAX_DTE), step=1, key="watchlist_max_dte",
            help="Maximum days to expiration.",
        )
    with wl7:
        iv_rank_input_wl = st.number_input(
            "Min IV Rank", min_value=0.0, max_value=100.0, value=float(oom_mod.IV_RANK_MIN),
            step=5.0, format="%.0f", key="watchlist_ivrank",
            help="Where today's IV sits (0-100) in each stock's own 52-week IV range. "
                 "From a community-maintained dataset - blank/unfiltered if it's "
                 "unreachable. 0 disables.",
        )
    run_wl = st.button("Run Watchlist Screener", type="primary")

    if run_wl:
        symbols = []
        seen = set()
        for line in tickers_text.splitlines():
            sym = line.strip().upper()
            if not sym or sym.startswith("#"):
                continue
            if sym not in seen:
                seen.add(sym)
                symbols.append(sym)

        if not symbols:
            st.error("Enter at least one ticker.")
        elif min_dte_input_wl > max_dte_input_wl:
            st.error("Min DTE cannot exceed Max DTE.")
        else:
            cash_to_invest = cash_input_wl if cash_input_wl else None
            with st.spinner(f"Scanning {len(symbols)} tickers..."):
                try:
                    results, log_output = run_captured(
                        oom_mod.run_screener, symbols, cash_to_invest, iv_input_wl, iv_rv_input_wl,
                        True, None, premium_input_wl, int(min_dte_input_wl), int(max_dte_input_wl),
                        iv_rank_input_wl,
                    )
                except Exception as exc:
                    st.error(f"Watchlist scan failed: {exc}")
                    results, log_output = None, ""
            st.session_state["watchlist_result"] = {
                "results": results,
                "log": log_output,
                "symbols": symbols,
                "cash": cash_to_invest,
                "iv": iv_input_wl,
                "iv_rv": iv_rv_input_wl,
            }

    wl_result = st.session_state.get("watchlist_result")
    if wl_result:
        caption = f"Universe size: {len(wl_result['symbols'])}"
        if wl_result["cash"]:
            caption += f" | Cash to invest: {usd(wl_result['cash'])}"
        iv_rv_txt = f"{wl_result['iv_rv']:.2f}" if wl_result.get("iv_rv") else "off"
        caption += f" | IV > {wl_result.get('iv', 0):.2f} | IV/RV {iv_rv_txt}"
        st.caption(caption)

        results = wl_result["results"]
        if results is None or results.empty:
            st.info("No tickers matched the current screen.")
        else:
            st.caption("Click a row to open that ticker in the Screener tab.")
            select_event = st.dataframe(
                style_candidates_df(results), column_config=get_column_config(results), width="stretch",
                on_select="rerun", selection_mode="single-row", key="watchlist_results_select",
            )
            st.caption(LEGEND_MD)

            selected_rows = select_event.selection.rows if hasattr(select_event, "selection") else []
            if selected_rows:
                picked_symbol = str(results.iloc[selected_rows[0]]["sym"])
                # Dataframe row selection persists across reruns. Gating on "is this
                # a NEW pick" - rather than trusting that clearing the widget's own
                # selection state works - means a stale/persisted selection can
                # never re-trigger this block and loop st.rerun() forever.
                if st.session_state.get("_watchlist_last_pick") != picked_symbol:
                    st.session_state["_watchlist_last_pick"] = picked_symbol
                    # st.tabs has no API to switch the active tab from code, so the
                    # nearest thing to "open in the Screener tab" is: preload that
                    # tab's state and results here, then tell the user to click over.
                    with st.spinner(f"Loading {picked_symbol} in the Screener tab..."):
                        try:
                            iv_used = wl_result.get("iv")
                            iv_rv_used = wl_result.get("iv_rv")
                            spot, context, checked, skipped, name, cand_results = chain_mod.find_candidates(
                                picked_symbol, iv_used, iv_rv_used
                            )
                            chart_hist, chart_zones, _, chart_support = chain_mod.get_resistance_chart_data(
                                picked_symbol, spot
                            )
                            st.session_state["screener_result"] = {
                                "symbol": picked_symbol,
                                "spot": spot,
                                "context": context,
                                "expirations_checked": checked,
                                "expirations_skipped_earnings": skipped,
                                "company_name": name,
                                "results": cand_results,
                                "chart_hist": chart_hist,
                                "chart_zones": chart_zones,
                                "chart_support": chart_support,
                            }
                            st.session_state["screener_symbol"] = picked_symbol
                            if iv_used is not None:
                                st.session_state["screener_iv"] = float(iv_used)
                            if iv_rv_used is not None:
                                st.session_state["screener_ivrv"] = float(iv_rv_used)
                            st.session_state["screener_flash"] = (
                                f"Loaded **{picked_symbol}** from the watchlist scan."
                            )
                        except Exception as exc:
                            st.session_state["screener_flash"] = f"Failed to load {picked_symbol}: {exc}"
                    st.rerun()
        with st.expander("Scan log"):
            st.text(wl_result["log"] or "(no output)")

with tab_positions:
    # A log/close/assign/roll action reruns the page (see the `st.rerun()` calls
    # below) before the user's browser ever renders the success message from that
    # action's own script pass - st.rerun() aborts execution immediately, so a
    # message shown right before it is never actually seen. Stashing it in
    # session_state and showing it here, at the top of the FOLLOWING run, is what
    # actually gets it in front of the user.
    pending_message = st.session_state.pop("positions_flash", None)
    if pending_message:
        st.success(as_markdown_block(pending_message))

    st.subheader("Open Positions")
    if st.button("Refresh Positions", type="primary") or "positions_df" not in st.session_state:
        with st.spinner("Fetching current prices..."):
            st.session_state["positions_df"] = get_positions_with_pnl()

    positions_df = st.session_state.get("positions_df", pd.DataFrame())
    if positions_df.empty:
        st.info("No open positions.")
    else:
        st.dataframe(
            style_positions_df(positions_df), column_config=get_column_config(positions_df), width="stretch"
        )

    with st.expander("Position History (closed & rolled)"):
        history_df = chain_mod.build_history_table()
        if history_df.empty:
            st.info("No closed or rolled positions yet.")
        else:
            st.dataframe(
                style_history_df(history_df), column_config=get_column_config(history_df), width="stretch"
            )
            opt = pd.to_numeric(history_df["realized_pl"], errors="coerce").dropna()
            stock = pd.to_numeric(history_df["stock_realized_pl"], errors="coerce").dropna()
            total = pd.to_numeric(history_df["total_realized"], errors="coerce").dropna()
            if not total.empty:
                caption = f"Realized across {len(total)} leg(s): option **{usd(opt.sum())}**"
                if not stock.empty:
                    caption += f" | stock (assigned) **{usd(stock.sum())}**"
                caption += f" | **TOTAL {usd(total.sum())}**"
                st.caption(caption)

    with st.expander("Lots (cumulative return per stock)"):
        st.caption(
            "One row per continuous share-holding, summing every leg against it - "
            "logged, rolled, or closed - even across a gap where you closed and later "
            "logged a fresh call instead of rolling."
        )
        lots_df = chain_mod.build_lots_table()
        if lots_df.empty:
            st.info("No positions yet.")
        else:
            st.dataframe(
                style_lots_df(lots_df), column_config=get_column_config(lots_df), width="stretch"
            )
            closed_total = pd.to_numeric(
                lots_df.loc[lots_df["status"] != "open", "total_realized"], errors="coerce"
            ).dropna()
            if not closed_total.empty:
                st.caption(f"Realized across {len(closed_total)} closed/assigned lot(s): "
                           f"**{usd(closed_total.sum())}**")

    st.markdown("---")
    col_log, col_close = st.columns(2)

    with col_log:
        st.markdown("#### Log New Position")
        with st.form("log_form"):
            log_symbol = st.text_input("Symbol")
            log_contract = st.text_input("Contract (OCC symbol)", placeholder="AMD260814C00545000")
            log_stock_price = st.number_input("Stock purchase price", min_value=0.0, step=0.01)
            log_shares = st.number_input("Shares", min_value=0, step=100, value=100)
            log_premium = st.number_input("Premium collected (total $)", min_value=0.0, step=1.0)
            log_contracts = st.number_input("Contracts", min_value=1, step=1, value=1)
            log_new_lot = st.checkbox(
                "Force new lot", value=False,
                help="Normally, if you still hold shares from an earlier entry on this symbol "
                     "(its last leg wasn't 'assigned'), this continues that lot automatically - "
                     "see the Lots expander above. Check this only if you sold those shares "
                     "outside this tool (e.g. a manual broker sale) without ever using Assign, "
                     "so this tool has no way to know they're gone.",
            )
            submitted_log = st.form_submit_button("Log Position")

        if submitted_log:
            args = [
                log_symbol, log_contract, str(log_stock_price),
                str(int(log_shares)), str(log_premium), str(int(log_contracts)),
            ]
            if log_new_lot:
                args.append("--new-lot")
            _, output = run_captured(chain_mod.cmd_log, args)
            if "Logged position" in output:
                st.session_state["positions_flash"] = output
                st.session_state.pop("positions_df", None)
                st.rerun()
            else:
                st.error(output.strip() or "Failed to log position.")

    open_positions_for_action = chain_mod.get_open_positions()
    contract_options = (
        open_positions_for_action["contract"].tolist() if not open_positions_for_action.empty else []
    )

    with col_close:
        st.markdown("#### Close Position")
        st.caption("You bought the call back or it expired worthless. **You keep the shares.**")
        if not contract_options:
            st.info("No open positions to close.")
        else:
            with st.form("close_form"):
                close_contract = st.selectbox("Contract", options=contract_options, key="close_pick")
                close_premium = st.number_input(
                    "Close premium (total $ paid, 0 if expired worthless)", min_value=0.0, step=1.0
                )
                submitted_close = st.form_submit_button("Close Position")

            if submitted_close:
                args = [close_contract, str(close_premium)]
                _, output = run_captured(chain_mod.cmd_close, args)
                if "Closed" in output:
                    st.session_state["positions_flash"] = output
                    st.session_state.pop("positions_df", None)
                    st.rerun()
                else:
                    st.error(output.strip() or "Failed to close position.")

    col_assign, col_roll = st.columns(2)

    with col_assign:
        st.markdown("#### Called Away (Assigned)")
        st.caption(
            "The call was exercised and your **shares were sold at the strike**. "
            "Records the premium kept *and* the stock leg - don't use Close for this."
        )
        if not contract_options:
            st.info("No open positions to assign.")
        else:
            with st.form("assign_form"):
                assign_contract = st.selectbox("Contract", options=contract_options, key="assign_pick")
                st.caption("No price needed: the sale price is the strike, by definition.")
                submitted_assign = st.form_submit_button("Record Assignment")

            if submitted_assign:
                _, output = run_captured(chain_mod.cmd_assign, [assign_contract])
                if "Assigned" in output:
                    st.session_state["positions_flash"] = output
                    st.session_state.pop("positions_df", None)
                    st.rerun()
                else:
                    st.error(output.strip() or "Failed to record assignment.")

    with col_roll:
        st.markdown("#### Roll Position")
        st.caption(
            "Buy back the current call and sell a new one. Carries your stock cost "
            "basis over and links the legs so premium accumulates across the chain."
        )
        if not contract_options:
            st.info("No open positions to roll.")
        else:
            with st.form("roll_form"):
                roll_old = st.selectbox("Current contract", options=contract_options, key="roll_pick")
                roll_close_premium = st.number_input(
                    "Paid to buy back old call (total $)", min_value=0.0, step=1.0
                )
                roll_new = st.text_input("New contract (OCC symbol)", placeholder="AMD260828C00560000")
                roll_new_premium = st.number_input(
                    "Received for new call (total $)", min_value=0.0, step=1.0
                )
                submitted_roll = st.form_submit_button("Roll Position")

            if submitted_roll:
                if not roll_new.strip():
                    st.error("Enter the new contract symbol.")
                else:
                    args = [roll_old, str(roll_close_premium), roll_new.strip(), str(roll_new_premium)]
                    _, output = run_captured(chain_mod.cmd_roll, args)
                    if "Rolled" in output:
                        st.session_state["positions_flash"] = output
                        st.session_state.pop("positions_df", None)
                        st.rerun()
                    else:
                        st.error(output.strip() or "Failed to roll position.")

with tab_backtest:
    st.subheader("Backtest")
    st.caption(
        "Replays the live screener's own criteria day-by-day against a symbol's real "
        "historical option chain: on every day you'd be flat, it scores that day's actual "
        "chain the same way the Screener tab would, papers the top candidate, holds to "
        "expiration, and checks what the stock actually did. Answers \"did the entry "
        "criteria predict good outcomes\" - it does not execute or recommend trades."
    )

    bt1, bt2, bt3 = st.columns(3)
    with bt1:
        bt_symbol = st.text_input("Symbol", value=chain_mod.DEFAULT_SYMBOL, key="backtest_symbol")
    with bt2:
        bt_default_to = pd.Timestamp.now(tz="US/Pacific").date()
        bt_from = st.date_input(
            "From",
            value=(bt_default_to - pd.DateOffset(months=backtest_mod.DEFAULT_BACKTEST_MONTHS)).date(),
            min_value=pd.Timestamp(backtest_mod.EARLIEST_AVAILABLE_DATE).date(), max_value=bt_default_to,
            key="backtest_from",
        )
    with bt3:
        bt_to = st.date_input(
            "To", value=bt_default_to,
            min_value=pd.Timestamp(backtest_mod.EARLIEST_AVAILABLE_DATE).date(), max_value=bt_default_to,
            key="backtest_to",
        )

    bt4, bt5, bt6, bt7, bt8, bt9 = st.columns(6)
    with bt4:
        bt_iv = st.number_input(
            "IV floor", min_value=0.0, max_value=5.0, value=float(backtest_mod.IV_THRESHOLD),
            step=0.05, format="%.2f", key="backtest_iv",
        )
    with bt5:
        bt_iv_rv = st.number_input(
            "Min IV/RV", min_value=0.0, max_value=5.0, value=float(backtest_mod.IV_RV_RATIO_MIN),
            step=0.05, format="%.2f", key="backtest_ivrv",
        )
    with bt6:
        bt_iv_rank = st.number_input(
            "Min IV Rank", min_value=0.0, max_value=100.0, value=float(backtest_mod.IV_RANK_MIN),
            step=5.0, format="%.0f", key="backtest_ivrank",
        )
    with bt7:
        bt_min_dte = st.number_input(
            "Min DTE", min_value=0, value=int(backtest_mod.MIN_DTE), step=1, key="backtest_min_dte",
        )
    with bt8:
        bt_max_dte = st.number_input(
            "Max DTE", min_value=0, value=int(backtest_mod.MAX_DTE), step=1, key="backtest_max_dte",
        )
    with bt9:
        bt_assign = st.selectbox("Assign mode", options=["keep", "neutral"], key="backtest_assign")

    st.caption(
        "First run fetches one request per new trading day and may take a few minutes; "
        "fully cached after (re-running with different thresholds costs no new requests). "
        "Not modeled: open interest/volume (absent from the historical dataset)."
    )
    run_bt = st.button("Run Backtest", type="primary")

    if run_bt:
        symbol = bt_symbol.strip().upper()
        if not symbol:
            st.error("Enter a symbol.")
        elif bt_min_dte > bt_max_dte:
            st.error("Min DTE cannot exceed Max DTE.")
        elif bt_from > bt_to:
            st.error("From date cannot be after To date.")
        else:
            with st.spinner(f"Backtesting {symbol} ({bt_from} -> {bt_to})..."):
                try:
                    (cycles, earnings_available, iv_rank_available), log_output = run_captured(
                        backtest_mod.run_backtest, symbol, bt_from.isoformat(), bt_to.isoformat(),
                        bt_iv, bt_iv_rv, bt_iv_rank, int(bt_min_dte), int(bt_max_dte), bt_assign,
                    )
                except Exception as exc:
                    st.error(f"Backtest failed: {exc}")
                    cycles, earnings_available, iv_rank_available, log_output = [], False, False, ""
            st.session_state["backtest_result"] = {
                "symbol": symbol, "from": bt_from.isoformat(), "to": bt_to.isoformat(),
                "cycles": cycles, "earnings_available": earnings_available,
                "iv_rank_available": iv_rank_available, "iv_rank_min": bt_iv_rank, "log": log_output,
            }

    bt_result = st.session_state.get("backtest_result")
    if bt_result:
        st.caption(f"{bt_result['symbol']}: {bt_result['from']} -> {bt_result['to']}")
        if not bt_result["earnings_available"]:
            st.warning("Earnings-date history unavailable for this symbol - earnings-window exclusion NOT applied.")
        if bt_result["iv_rank_min"] > 0 and not bt_result["iv_rank_available"]:
            st.warning(
                "IV-rank history unavailable for this window (third-party source didn't respond) - "
                "IV rank filtering was NOT applied despite the Min IV Rank setting, results may differ "
                "from a run where it was available."
            )

        cycles = bt_result["cycles"]
        if not cycles:
            st.info("No cycles - no candidate ever passed this screen in this window.")
        else:
            df = pd.DataFrame(cycles)
            display_cols = [
                "entry_date", "expiration", "strike", "entry_spot", "exit_spot", "premium",
                "outcome", "realized_ann", "score_at_entry", "holding_days",
            ]
            display = df[display_cols].copy()
            display["entry_date"] = display["entry_date"].astype(str)
            display["expiration"] = display["expiration"].astype(str)

            n = len(df)
            win_rate = (df["outcome"] == "EXPIRED").mean() * 100
            assign_rate = (df["outcome"] == "ASSIGNED").mean() * 100
            corr = df["score_at_entry"].corr(df["realized_pct"]) if n >= 3 else None
            best = df.loc[df["pl_dollars"].idxmax()]
            worst = df.loc[df["pl_dollars"].idxmin()]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cycles", n)
            m2.metric("Win rate (expired OTM)", f"{win_rate:.1f}%")
            m3.metric("Assigned", f"{assign_rate:.1f}%")
            m4.metric("Avg realized ann. return", f"{df['realized_ann'].mean():.1f}%")
            m5, m6, m7 = st.columns(3)
            m5.metric("Avg entry score", f"{df['score_at_entry'].mean():.1f}")
            m6.metric(
                "Score-return correlation", f"{corr:.2f}" if corr is not None and pd.notna(corr) else "N/A",
                help="Positive = the entry score was actually predictive in this window; "
                     "near zero or negative = it wasn't.",
            )
            m7.metric("Best / worst cycle", f"${best['pl_dollars']:,.2f} / ${worst['pl_dollars']:,.2f}")

            st.dataframe(
                style_backtest_df(display), column_config=get_column_config(display), width="stretch"
            )

        with st.expander("Backtest log"):
            st.text(bt_result["log"] or "(no output)")

    st.markdown("---")
    st.markdown("#### Parameter Sweep (Optimizer)")
    st.caption(
        "Grid-searches IV floor / Min IV-RV / Min IV Rank / Assign mode against ONE cached "
        "historical fetch and ranks every combination that cleared Min Cycles. DTE isn't swept - "
        "a different Min/Max DTE needs its own fresh fetch, so re-run this with different DTE "
        "values if you want to explore that dimension too."
    )

    sw1, sw2, sw3 = st.columns(3)
    with sw1:
        sw_symbol = st.text_input("Symbol", value=chain_mod.DEFAULT_SYMBOL, key="sweep_symbol")
    with sw2:
        sw_default_to = pd.Timestamp.now(tz="US/Pacific").date()
        sw_from = st.date_input(
            "From",
            value=(sw_default_to - pd.DateOffset(months=backtest_mod.DEFAULT_BACKTEST_MONTHS)).date(),
            min_value=pd.Timestamp(backtest_mod.EARLIEST_AVAILABLE_DATE).date(), max_value=sw_default_to,
            key="sweep_from",
        )
    with sw3:
        sw_to = st.date_input(
            "To", value=sw_default_to,
            min_value=pd.Timestamp(backtest_mod.EARLIEST_AVAILABLE_DATE).date(), max_value=sw_default_to,
            key="sweep_to",
        )

    sw4, sw5, sw6, sw7, sw8 = st.columns(5)
    with sw4:
        sw_min_dte = st.number_input(
            "Min DTE", min_value=0, value=int(backtest_mod.MIN_DTE), step=1, key="sweep_min_dte",
        )
    with sw5:
        sw_max_dte = st.number_input(
            "Max DTE", min_value=0, value=int(backtest_mod.MAX_DTE), step=1, key="sweep_max_dte",
        )
    with sw6:
        sw_rank_by = st.selectbox(
            "Rank by", options=list(backtest_mod.SWEEP_RANK_METRICS), key="sweep_rank_by",
        )
    with sw7:
        sw_min_cycles = st.number_input(
            "Min cycles", min_value=1, value=5, step=1, key="sweep_min_cycles",
            help="Combinations with fewer cycles than this are dropped - a 100% win rate from "
                 "one lucky trade is not a signal.",
        )
    with sw8:
        sw_top_n = st.number_input("Show top N", min_value=1, value=15, step=1, key="sweep_top_n")

    run_sweep = st.button("Run Sweep", type="primary")

    if run_sweep:
        symbol = sw_symbol.strip().upper()
        if not symbol:
            st.error("Enter a symbol.")
        elif sw_min_dte > sw_max_dte:
            st.error("Min DTE cannot exceed Max DTE.")
        elif sw_from > sw_to:
            st.error("From date cannot be after To date.")
        else:
            with st.spinner(f"Sweeping {symbol} ({sw_from} -> {sw_to})..."):
                try:
                    (results, earnings_available, iv_rank_available), log_output = run_captured(
                        backtest_mod.sweep_backtest, symbol, sw_from.isoformat(), sw_to.isoformat(),
                        min_dte=int(sw_min_dte), max_dte=int(sw_max_dte), min_cycles=int(sw_min_cycles),
                    )
                except Exception as exc:
                    st.error(f"Sweep failed: {exc}")
                    results, earnings_available, iv_rank_available, log_output = pd.DataFrame(), False, False, ""
            st.session_state["sweep_result"] = {
                "symbol": symbol, "from": sw_from.isoformat(), "to": sw_to.isoformat(),
                "results": results, "rank_by": sw_rank_by, "min_cycles": int(sw_min_cycles),
                "top_n": int(sw_top_n), "earnings_available": earnings_available,
                "iv_rank_available": iv_rank_available, "log": log_output,
            }

    sweep_result = st.session_state.get("sweep_result")
    if sweep_result:
        if not sweep_result["earnings_available"]:
            st.warning("Earnings-date history unavailable for this symbol - earnings-window exclusion NOT applied.")
        if not sweep_result["iv_rank_available"]:
            st.warning(
                "IV-rank history unavailable for this window (third-party source didn't respond) - "
                "every combination below ran with IV rank filtering unable to exclude anything, "
                "regardless of its Min IV Rank value. Re-run to try again."
            )
        st.caption(
            f"{sweep_result['symbol']}: {sweep_result['from']} -> {sweep_result['to']}  "
            f"(ranked by {sweep_result['rank_by']}, >= {sweep_result['min_cycles']} cycles required)"
        )
        results = sweep_result["results"]
        if results is None or results.empty:
            st.info(
                f"No combination produced >= {sweep_result['min_cycles']} cycles - try a wider "
                f"date range or a lower Min Cycles."
            )
        else:
            st.warning(
                "This ranks combinations by how well each WOULD have done in this exact window. "
                "The more combinations tested, the higher the chance the top one just fit this "
                "window's noise (multiple-comparisons risk) - treat it as a hypothesis to verify "
                "with a fresh backtest above on a different date range, not a conclusion."
            )

            ranked = results.sort_values(sweep_result["rank_by"], ascending=False).reset_index(drop=True)
            best = ranked.iloc[0]

            sb1, sb2, sb3, sb4 = st.columns(4)
            sb1.metric("Best combo cycles", int(best["n_cycles"]))
            sb2.metric("Win rate", f"{best['win_rate']:.1f}%")
            sb3.metric("Avg realized ann. return", f"{best['avg_realized_ann']:.1f}%")
            corr_val = best["correlation"]
            sb4.metric("Correlation", f"{corr_val:.2f}" if pd.notna(corr_val) else "N/A")

            st.markdown(
                f"**Best by {sweep_result['rank_by']}:** IV floor {best['iv_threshold']:.2f}  |  "
                f"Min IV/RV {best['iv_rv_ratio_min']:.2f}  |  Min IV Rank {best['iv_rank_min']:.0f}  |  "
                f"Assign `{best['assignment_preference']}`"
            )
            st.code(
                f"python chain.py {sweep_result['symbol']} --iv {best['iv_threshold']:.2f} "
                f"--iv-rv {best['iv_rv_ratio_min']:.2f} --iv-rank {best['iv_rank_min']:.0f} "
                f"--assign {best['assignment_preference']}",
                language="bash",
            )
            st.code(
                f"python backtest.py {sweep_result['symbol']} --from {sweep_result['from']} "
                f"--to {sweep_result['to']} --iv {best['iv_threshold']:.2f} "
                f"--iv-rv {best['iv_rv_ratio_min']:.2f} --iv-rank {best['iv_rank_min']:.0f} "
                f"--assign {best['assignment_preference']}",
                language="bash",
            )

            st.markdown(f"##### Distribution of {sweep_result['rank_by']} across all tested combinations")
            st.caption(
                "The table above only ever shows the cherry-picked top - this is the full spread it "
                "was drawn from, so you can see whether the winner is a clear standout or barely "
                "distinguishable from the pack."
            )
            default_row = backtest_mod.find_default_combo_row(results)
            default_value = default_row[sweep_result["rank_by"]] if default_row is not None else None
            st.altair_chart(
                build_sweep_distribution_chart(results, sweep_result["rank_by"], default_value),
                use_container_width=True,
            )
            if default_value is not None and pd.notna(default_value):
                percentile = (results[sweep_result["rank_by"]] <= default_value).mean() * 100
                st.caption(
                    f":orange[**│ Current config**] (iv={backtest_mod.IV_THRESHOLD:.2f} "
                    f"iv-rv={backtest_mod.IV_RV_RATIO_MIN:.2f} iv-rank={backtest_mod.IV_RANK_MIN} "
                    f"assign={backtest_mod.ASSIGNMENT_PREFERENCE}): "
                    f"{sweep_result['rank_by']}={default_value:.2f}, **{percentile:.0f}th percentile** "
                    f"of all {len(results)} tested combinations"
                )
            else:
                st.caption(
                    "Current config combination wasn't in this sweep's grid - no percentile to report."
                )

            display = ranked.head(sweep_result["top_n"])
            st.dataframe(
                style_sweep_df(display), column_config=get_column_config(display), width="stretch"
            )

        with st.expander("Sweep log"):
            st.text(sweep_result["log"] or "(no output)")
