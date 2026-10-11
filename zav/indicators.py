"""Pull well-known technical indicators (daily timeframe) via tvscreener and derive simple signals."""
import pandas as pd
from tvscreener import StockScreener, StockField as F, FilterOperator as Op, Market

# (output column, StockField). Missing fields are skipped at runtime and reported.
FIELD_MAP = [
    ("symbol", "NAME"), ("exchange", "EXCHANGE"), ("price", "PRICE"), ("chg_pct", "CHANGE_PERCENT"),
    ("rsi14", "RELATIVE_STRENGTH_INDEX_14"),
    ("macd", "MACD_LEVEL_12_26"), ("macd_signal", "MACD_SIGNAL_12_26"), ("macd_hist", "MACD_HIST"),
    ("sma50", "SIMPLE_MOVING_AVERAGE_50"), ("sma200", "SIMPLE_MOVING_AVERAGE_200"),
    ("ema20", "EXPONENTIAL_MOVING_AVERAGE_20"),
    ("bb_upper", "BOLLINGER_UPPER_BAND_20"), ("bb_lower", "BOLLINGER_LOWER_BAND_20"),
    ("stoch_k", "STOCHASTIC_PERCENTK_14_3_3"), ("stoch_d", "STOCHASTIC_PERCENTD_14_3_3"),
    ("adx", "AVERAGE_DIRECTIONAL_INDEX_14"), ("atr", "AVERAGE_TRUE_RANGE_14"),
    ("rel_volume", "RELATIVE_VOLUME"), ("tech_rating", "TECHNICAL_RATING"),
]
ALLOWED_EXCHANGES = {"NASDAQ", "NYSE"}

def resolve_fields():
    used, missing = [], []
    for col, name in FIELD_MAP:
        (used if name in F.__members__ else missing).append((col, name))
    return [(c, F[n]) for c, n in used], [n for _, n in missing]

def fetch(tickers):
    fields, missing = resolve_fields()
    s = StockScreener(); s.set_markets(Market.AMERICA)
    s.select(*[f for _, f in fields])
    s.add_filter(F.NAME, Op.IN_RANGE, list(tickers))
    s.set_range(0, 500)
    raw = s.get()
    df = pd.DataFrame({c: raw[f.label] for c, f in fields if f.label in raw.columns})
    # Fix OTC duplicates: exact symbol match + major US exchanges only
    df = df[df["symbol"].isin(tickers) & df["exchange"].isin(ALLOWED_EXCHANGES)]
    df = df.drop_duplicates("symbol").set_index("symbol").reindex(tickers).reset_index()
    return df, missing

def rating_label(v):
    if pd.isna(v): return "n/a"
    return ("Strong Sell" if v < -0.5 else "Sell" if v < -0.1 else
            "Neutral" if v <= 0.1 else "Buy" if v <= 0.5 else "Strong Buy")

def add_signals(df):
    d = df.copy()
    d["cross"] = (d.sma50 > d.sma200).map({True: "golden", False: "death"})
    width = d.bb_upper - d.bb_lower
    d["bb_pos_pct"] = ((d.price - d.bb_lower) / width * 100).round(1)
    d["bb_width_pct"] = (width / d.price * 100).round(1)
    d["sig_rsi"] = d.rsi14.apply(lambda r: "oversold" if r < 30 else "overbought" if r > 70 else "neutral")
    d["sig_trend"] = (d.price > d.sma200).map({True: "above SMA200", False: "below SMA200"})
    d["sig_macd"] = (d.macd > d.macd_signal).map({True: "bullish", False: "bearish"})
    def bb(r):
        if r.bb_width_pct < 5: return "squeeze"
        if r.bb_pos_pct >= 90: return "near upper"
        if r.bb_pos_pct <= 10: return "near lower"
        return "mid"
    d["sig_bb"] = d.apply(bb, axis=1)
    d["sig_stoch"] = d.stoch_k.apply(lambda k: "oversold" if k < 20 else "overbought" if k > 80 else "neutral")
    d["sig_adx"] = (d.adx > 25).map({True: "trending", False: "weak/range"})
    d["tv_rating"] = d.tech_rating.apply(rating_label)
    def summary(r):
        score = (1 if r.price > r.sma200 else -1) + (1 if r.macd > r.macd_signal else -1) \
              + (1 if r.sma50 > r.sma200 else -1) + (1 if r.price > r.ema20 else -1)
        lbl = "BULLISH" if score >= 3 else "BEARISH" if score <= -3 else "MIXED"
        notes = [x for x, c in [("overbought", r.rsi14 > 70), ("oversold", r.rsi14 < 30),
                                ("strong trend", r.adx > 25)] if c]
        return lbl + (f" ({', '.join(notes)})" if notes else "")
    d["summary"] = d.apply(summary, axis=1)
    return d
