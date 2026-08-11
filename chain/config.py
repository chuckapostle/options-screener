"""Shared screening criteria.

Single source of truth for every parameter that defines what counts as a valid
covered-call candidate. chain.py (single symbol) and screen_oom.py (watchlist)
both import from here so they can never silently disagree about what qualifies.

Per-run overrides for the two vol thresholds are available via --iv / --iv-rv.
"""

# --- Expiration window -------------------------------------------------------
# Defaults only - both scripts accept --min-dte / --max-dte to override per run.
MIN_DTE = 10
MAX_DTE = 21

# --- Volatility --------------------------------------------------------------
# Absolute IV floor. Set to 0.60 to reproduce the original pre-IV/RV screen.
IV_THRESHOLD = 0.30
# Realized-vol lookback, in trading days. This MUST be roughly horizon-matched to
# the options being sold: comparing a 3-month trailing RV against a ~2-week forward
# IV systematically overstates RV (one old gap stays in the window long after the
# market has stopped pricing a repeat), which biases every IV/RV ratio downward.
# The short window is horizon-relevant but noisy; the long window is a stability
# anchor. They are blended, and both components are reported for transparency.
RV_SHORT_TRADING_DAYS = 21   # ~1 month, closest stable match to a 10-16 DTE option
RV_LONG_TRADING_DAYS = 63    # ~3 months
RV_SHORT_WEIGHT = 0.70
# Implied vol as a multiple of blended realized vol. 1.00 is par - you are paid
# exactly what the stock has actually been moving. Above 1.00 is genuinely rich.
#
# Measured 2026-08-06 across AAPL/NVDA/TSLA/MSTR/AMD/MSFT/META/GOOGL, the ratio
# ran 0.57-1.11 with a median of 0.78: i.e. IV is broadly BELOW recent realized
# vol, which is an unfavorable environment for selling premium. The default below
# is therefore a pragmatic concession, not an endorsement - it is well under par.
# Caveat: RV is backward-looking, so IV < RV may just mean the market correctly
# expects calm after a volatile stretch, not that options are mispriced.
# Set to 0 to disable; raise to 1.00+ to only sell demonstrably rich premium.
IV_RV_RATIO_MIN = 0.80
TRADING_DAYS_PER_YEAR = 252

# --- Strike selection --------------------------------------------------------
TARGET_DELTA = 0.30
DELTA_MIN = 0.25
DELTA_MAX = 0.35

# --- Liquidity / quality -----------------------------------------------------
PREMIUM_THRESHOLD = 1.00
# Total $ premium for the position actually sized against cash_to_invest (i.e.
# fill_dollars x however many contracts that cash affords). With no cash figure
# given, this falls back to a single contract's premium. This is a "worth the
# trouble" filter, distinct from PREMIUM_THRESHOLD above which is a per-contract
# liquidity floor that screens out effectively-worthless quotes.
TOTAL_PREMIUM_THRESHOLD = 1000.00
MIN_OPEN_INTEREST = 50
MAX_SPREAD_PCT = 0.10
# Open interest says a contract exists; volume says it actually trades today.
# A contract with high OI but no prints can have a stale, unfillable quote.
MIN_VOLUME = 10
# Fills are not at mid. This models execution partway from mid toward the bid:
# 0.0 = optimistic mid fill, 0.5 = halfway to bid, 1.0 = hit the bid outright.
# Every premium, return and score figure is computed from this modeled fill, not
# from mid, so the numbers are not silently optimistic.
FILL_HAIRCUT = 0.50

# --- Price context -----------------------------------------------------------
RISK_FREE_RATE = 0.04
SHARES_PER_CONTRACT = 100

# --- Resistance ----------------------------------------------------------------
# Swing-high zone clustering: local peaks (a High exceeding the highs
# RESISTANCE_SWING_WINDOW days on either side) over the trailing lookback
# window are grouped into zones when they land within RESISTANCE_CLUSTER_PCT
# of each other. The nearest zone ABOVE current spot is reported as
# resistance, along with how many times it's been touched and how recently.
# Replaces the old "max High in the trailing 6 weeks" measure, which was a
# single noisy point (one spike day set the whole level) over an arbitrarily
# short window, and couldn't distinguish a level tested once from one
# rejected five times.
RESISTANCE_LOOKBACK_DAYS = 180   # ~6 months of daily highs to detect swing points from
RESISTANCE_SWING_WINDOW = 3      # trading days on each side a High must exceed to count as a local peak
RESISTANCE_CLUSTER_PCT = 0.02    # peaks within this % of each other merge into one zone

# --- Strike-selection objective ----------------------------------------------
# What the ranking score optimizes for. This is a preference, not a fact:
#   "keep"    - you want to retain the shares. Assignment is a bad outcome, so
#               score = annualized return if expired x (1 - prob ITM).
#   "neutral" - you are indifferent to assignment. Score is the true expected
#               value: (1-p) x ann. return if expired + p x ann. return if called.
#               NOTE this systematically favors closer-to-the-money strikes,
#               because being called away usually pays more than expiring
#               worthless. It also ignores upside forgone above the strike.
ASSIGNMENT_PREFERENCE = "keep"

# --- Exit management ---------------------------------------------------------
# MANAGE_DTE is tuned to the 10-16 DTE entry window; the textbook 21-DTE rule
# assumes ~45 DTE entries and would fire on every position immediately.
TAKE_PROFIT_PCT = 50.0
MANAGE_DTE = 5
# Early warning before the strike is actually breached, so a roll is a decision
# rather than a reaction. Fires on either condition.
ROLL_WATCH_DELTA = 0.45
ROLL_WATCH_PROXIMITY_PCT = 1.0  # spot within this % below the strike

# --- Caching -----------------------------------------------------------------
CACHE_TTL_HOURS = 24          # company name / earnings / dividend lookups
CHAIN_CACHE_TTL_MINUTES = 15  # raw option chains

# --- IV Rank (optional third-party enrichment, see voldata.py) ---------------
# Where today's IV sits (0-100) within this stock's OWN trailing 52-week IV
# range, sourced from a community-maintained DoltHub dataset. Distinct from
# IV_RV_RATIO_MIN above, which compares IV to recent realized moves rather
# than to the stock's own IV history. This data source is outside our
# control and can go offline at any time - when it does, iv_rank is simply
# blank for every candidate and nothing else in the screen is affected.
IV_RANK_CACHE_TTL_HOURS = 20  # source updates once per trading day
# Minimum acceptable IV rank (0-100). Missing data (source down, or a symbol
# it doesn't cover) never excludes a candidate - only an explicit, present
# iv_rank below this threshold does.
#
# Measured 2026-08-11 across AMD/NVDA/AAPL/MSFT/AMZN/META/TSLA/GOOGL/NFLX/AVGO,
# iv_rank ran 2.7-52.4 with a median of 33.5: only 1 of 10 cleared the
# textbook "sell premium above IV Rank 50" rule, the same unfavorable-for-
# sellers finding as IV_RV_RATIO_MIN above. A 50 floor would gut the
# candidate pool in conditions like these, so this isn't set as an "only
# sell when rich" gate. It's set low, just to catch true bottom-of-year
# outliers - that day's TSLA sat at 2.7, near the cheapest IV of its whole
# year despite an absolute IV (0.40) that clears IV_THRESHOLD easily, a
# blind spot nothing else in the screen would have caught.
IV_RANK_MIN = 15
