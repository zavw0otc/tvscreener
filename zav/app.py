"""Streamlit RTL Hebrew technical-analysis page. Run: streamlit run app.py"""
import math
import numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
from tvscreener import StockScreener, StockField as F, FilterOperator as Op, Market

P = lambda k: [f"PIVOT_{k}_{x}" for x in ["S3", "S2", "S1", "P", "R1", "R2", "R3"]]
GROUPS = {
 "quote": ["NAME", "EXCHANGE", "DESCRIPTION", "PRICE", "CHANGE_PERCENT", "OPEN", "HIGH", "LOW", "VOLUME"],
 "pivots_monthly": P("CLASSIC") + P("FIBONACCI") + P("CAMARILLA") + P("WOODIE") + ["PIVOT_DM_S1", "PIVOT_DM_P", "PIVOT_DM_R1"],
 "ichimoku": ["ICHIMOKU_CONVERSION_LINE_9_26_52_26", "ICHIMOKU_BASE_LINE_9_26_52_26", "ICHIMOKU_LEADING_SPAN_A_9_26_52_26", "ICHIMOKU_LEADING_SPAN_B_9_26_52_26"],
 "vwap_vwma": ["VOLUME_WEIGHTED_AVERAGE_PRICE", "VOLUME_WEIGHTED_MOVING_AVERAGE_20"],
 "moving_averages": ["HULL_MOVING_AVERAGE_9"] + [f"{t}_MOVING_AVERAGE_{n}" for t in ["SIMPLE", "EXPONENTIAL"] for n in [10, 20, 30, 50, 100, 200]],
 "trend": ["PARABOLIC_SAR", "AVERAGE_DIRECTIONAL_INDEX_14", "POSITIVE_DIRECTIONAL_INDICATOR_14", "NEGATIVE_DIRECTIONAL_INDICATOR_14", "AROON_UP_14", "AROON_DOWN_14"],
 "oscillators": ["RELATIVE_STRENGTH_INDEX_14", "MACD_LEVEL_12_26", "MACD_SIGNAL_12_26", "MACD_HIST", "STOCHASTIC_PERCENTK_14_3_3", "STOCHASTIC_PERCENTD_14_3_3", "STOCHASTIC_RSI_FAST_3_3_14_14", "WILLIAMS_PERCENT_RANGE_14", "COMMODITY_CHANNEL_INDEX_20", "MOMENTUM_10", "AWESOME_OSCILLATOR", "ULTIMATE_OSCILLATOR_7_14_28", "BULL_BEAR_POWER"],
 "channels": ["BOLLINGER_UPPER_BAND_20", "BOLLINGER_LOWER_BAND_20", "KELTNER_CHANNELS_UPPER_BAND_20", "KELTNER_CHANNELS_LOWER_BAND_20", "DONCHIAN_CHANNELS_UPPER_BAND_20", "DONCHIAN_CHANNELS_LOWER_BAND_20"],
 "range_perf_vol": ["WEEK_HIGH_52", "WEEK_LOW_52", "WEEKLY_PERFORMANCE", "MONTHLY_PERFORMANCE", "MONTH_PERFORMANCE_3", "MONTH_PERFORMANCE_6", "YTD_PERFORMANCE", "YEARLY_PERFORMANCE", "AVERAGE_TRUE_RANGE_14", "VOLATILITY", "YEAR_BETA_1", "AVERAGE_VOLUME_30_DAY", "RELATIVE_VOLUME"],
 "ratings": ["TECHNICAL_RATING", "MOVING_AVERAGES_RATING", "OSCILLATORS_RATING"],
}
LEVEL_GROUPS = ["pivots_monthly", "ichimoku", "vwap_vwma", "moving_averages", "channels"]

# ---------------- data ----------------
def _clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return None
    return v.item() if hasattr(v, "item") else v

@st.cache_data(ttl=600, show_spinner=False)
def tv_pull(ticker):
    names = [n for g in GROUPS.values() for n in g if n in F.__members__]
    s = StockScreener(); s.set_markets(Market.AMERICA)
    s.select(*[F[n] for n in names]); s.add_filter(F.NAME, Op.EQUAL, ticker); s.set_range(0, 50)
    raw = s.get()
    if raw is None or raw.empty: return None
    raw = raw[(raw[F.NAME.label] == ticker) & (raw[F.EXCHANGE.label].isin(["NASDAQ", "NYSE", "AMEX", "NYSE ARCA"]))]
    if raw.empty: return None
    row = raw.sort_values(F.EXCHANGE.label, key=lambda s: s.map({"NASDAQ": 0, "NYSE": 1}).fillna(2)).iloc[0]
    return {n: _clean(row.get(F[n].label)) for n in names}

@st.cache_data(ttl=600, show_spinner=False)
def yf_daily(ticker):
    import yfinance as yf
    df = yf.download(ticker, period="1y", interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    return df.dropna()

def yf_levels(df):
    df = df.iloc[-126:]  # ~6 months
    H, L, C, V = df.High, df.Low, df.Close, df.Volume
    res = {}
    k = 5
    res["swing_highs"] = [(str(df.index[i].date()), float(H.iloc[i])) for i in range(k, len(df)-k) if H.iloc[i] == H.iloc[i-k:i+k+1].max()][-6:]
    res["swing_lows"] = [(str(df.index[i].date()), float(L.iloc[i])) for i in range(k, len(df)-k) if L.iloc[i] == L.iloc[i-k:i+k+1].min()][-6:]
    hi_d, lo_d = H.idxmax(), L.idxmin(); hi, lo = float(H.max()), float(L.min()); up = lo_d < hi_d; rng = hi - lo
    res["fib"] = {f"{r*100:.1f}%": (hi - r*rng) if up else (lo + r*rng) for r in [.236, .382, .5, .618, .786]}
    res["fib_dir"] = up
    bins = np.linspace(lo, hi, 41); vp = np.zeros(40)
    for h, l, v in zip(H, L, V):
        ov = np.clip(np.minimum(bins[1:], h) - np.maximum(bins[:-1], l), 0, None)
        if ov.sum() > 0: vp += v * ov / ov.sum()
    mids = (bins[1:] + bins[:-1]) / 2; order = np.argsort(vp)[::-1]; hvn = []
    for i in order:
        if all(abs(i - j) > 2 for j in hvn): hvn.append(i)
        if len(hvn) == 5: break
    res["poc"] = float(mids[order[0]]); res["hvn"] = [float(mids[i]) for i in hvn[1:]]
    def classic(h, l, c):
        p = (h+l+c)/3
        return {"S2": p-(h-l), "S1": 2*p-h, "P": p, "R1": 2*p-l, "R2": p+(h-l)}
    d = df.iloc[-1]; res["piv_d"] = classic(d.High, d.Low, d.Close)
    w = df.resample("W-FRI").agg({"High": "max", "Low": "min", "Close": "last"}).dropna().iloc[-1]
    res["piv_w"] = classic(w.High, w.Low, w.Close)
    res["hi20"], res["lo20"] = float(H[-20:].max()), float(L[-20:].min())
    return res

# ---------------- level naming (plain Hebrew) ----------------
def heb_name(n):
    m = {"VOLUME_WEIGHTED_AVERAGE_PRICE": "VWAP", "VOLUME_WEIGHTED_MOVING_AVERAGE_20": "VWMA20", "HULL_MOVING_AVERAGE_9": "Hull MA9",
         "PARABOLIC_SAR": "Parabolic SAR", "WEEK_HIGH_52": "שיא 52 שבועות", "WEEK_LOW_52": "שפל 52 שבועות",
         "ICHIMOKU_CONVERSION_LINE_9_26_52_26": "איצ'ימוקו Tenkan", "ICHIMOKU_BASE_LINE_9_26_52_26": "איצ'ימוקו Kijun",
         "ICHIMOKU_LEADING_SPAN_A_9_26_52_26": "ענן איצ'ימוקו A", "ICHIMOKU_LEADING_SPAN_B_9_26_52_26": "ענן איצ'ימוקו B"}
    if n in m: return m[n]
    if n.startswith("SIMPLE_MOVING_AVERAGE_"): return "SMA" + n.split("_")[-1]
    if n.startswith("EXPONENTIAL_MOVING_AVERAGE_"): return "EMA" + n.split("_")[-1]
    if n.startswith("PIVOT_"):
        p = n.split("_"); kind = {"CLASSIC": "קלאסי", "FIBONACCI": "פיבונאצ'י", "CAMARILLA": "קמרילה", "WOODIE": "וודי", "DM": "דמרק"}[p[1]]
        return f"פיבוט חודשי {kind} {p[-1]}"
    if "BOLLINGER" in n: return "בולינגר " + ("עליון" if "UPPER" in n else "תחתון")
    if "KELTNER" in n: return "קלטנר " + ("עליון" if "UPPER" in n else "תחתון")
    if "DONCHIAN" in n: return "דונצ'יאן " + ("עליון" if "UPPER" in n else "תחתון")
    return n

def build_levels(tv, yfr, price):
    lv = []
    def add(name, v):
        if isinstance(v, (int, float)) and v > 0 and abs(v/price - 1) < 0.5: lv.append((float(v), name))
    for g in LEVEL_GROUPS:
        for n in GROUPS[g]: add(heb_name(n), tv.get(n))
    for n in ["PARABOLIC_SAR", "WEEK_HIGH_52", "WEEK_LOW_52"]: add(heb_name(n), tv.get(n))
    if yfr:
        for k, v in yfr["fib"].items(): add(f"פיבונאצ'י {k} של התנועה הגדולה", v)
        add("POC – מחיר עם הכי הרבה מחזור", yfr["poc"])
        for h in yfr["hvn"]: add("אזור מחזור גבוה (HVN)", h)
        for d, p in yfr["swing_highs"]: add(f"שיא מקומי ({d})", p)
        for d, p in yfr["swing_lows"]: add(f"שפל מקומי ({d})", p)
        for k, v in yfr["piv_d"].items(): add(f"פיבוט יומי {k}", v)
        for k, v in yfr["piv_w"].items(): add(f"פיבוט שבועי {k}", v)
        add("שיא 20 יום", yfr["hi20"]); add("שפל 20 יום", yfr["lo20"])
    return lv

def cluster(levels, price, tol=0.01):
    """Greedy clustering: groups levels within ±tol of the cluster seed; returns clusters sorted by price desc."""
    pts = sorted(levels, key=lambda x: x[0]); out = []; used = [False]*len(pts)
    # seed by density: pick level with most neighbors first
    dens = [sum(abs(q[0]-p[0]) <= p[0]*tol for q in pts) for p in pts]
    for i in sorted(range(len(pts)), key=lambda i: -dens[i]):
        if used[i]: continue
        mem = [j for j in range(len(pts)) if not used[j] and abs(pts[j][0]-pts[i][0]) <= pts[i][0]*tol]
        for j in mem: used[j] = True
        ps = [pts[j][0] for j in mem]
        out.append({"price": float(np.mean(ps)), "lo": min(ps), "hi": max(ps), "n": len(mem),
                    "sources": sorted({pts[j][1] for j in mem}), "dist": (np.mean(ps)/price-1)*100})
    out = [c for c in out if abs(c["dist"]) > 0.15 or c["n"] >= 2]
    return sorted(out, key=lambda c: -c["price"])

# ---------------- analysis ----------------
def g(tv, n, d=None):
    v = tv.get(n); return d if v is None else v

def analyze(ticker):
    ticker = ticker.strip().upper()
    tv = tv_pull(ticker)
    if not tv: raise LookupError(f"הסימול {ticker} לא נמצא ב-NASDAQ/NYSE")
    df = yf_daily(ticker)
    price = g(tv, "PRICE") or (float(df.Close.iloc[-1]) if df is not None else None)
    yfr = yf_levels(df) if df is not None and len(df) > 40 else None
    clusters = cluster(build_levels(tv, yfr, price), price)
    res = [c for c in clusters if c["price"] > price]; sup = [c for c in clusters if c["price"] <= price]
    sma50, sma200, ema20 = g(tv, "SIMPLE_MOVING_AVERAGE_50"), g(tv, "SIMPLE_MOVING_AVERAGE_200"), g(tv, "EXPONENTIAL_MOVING_AVERAGE_20")
    sma20 = g(tv, "SIMPLE_MOVING_AVERAGE_20"); rsi = g(tv, "RELATIVE_STRENGTH_INDEX_14", 50)
    macd, sig = g(tv, "MACD_LEVEL_12_26", 0), g(tv, "MACD_SIGNAL_12_26", 0)
    adx = g(tv, "AVERAGE_DIRECTIONAL_INDEX_14", 0); atr = g(tv, "AVERAGE_TRUE_RANGE_14", 0); beta = g(tv, "YEAR_BETA_1")
    stoch = g(tv, "STOCHASTIC_PERCENTK_14_3_3", 50); wr = g(tv, "WILLIAMS_PERCENT_RANGE_14", -50); cci = g(tv, "COMMODITY_CHANNEL_INDEX_20", 0)
    vwap = g(tv, "VOLUME_WEIGHTED_AVERAGE_PRICE")
    gt = lambda a, b: a is not None and b is not None and a > b

    # scorecards: (title, status, color, detail)
    lt = sum([gt(price, sma200), gt(sma50, sma200), gt(price, sma50)])
    long_card = ("מגמה ארוכה", ["יורדת", "חלשה", "עולה", "עולה חזקה"][lt], ["red", "yellow", "green", "green"][lt],
                 f"מחיר {'מעל' if gt(price, sma200) else 'מתחת'} SMA200 ({sma200:.2f}); SMA50 {'מעל' if gt(sma50, sma200) else 'מתחת'} SMA200" if sma200 else "אין מספיק נתונים")
    stt = sum([gt(price, ema20), gt(macd, sig), gt(price, sma20)])
    short_card = ("מגמה קצרה", ["יורדת", "תיקון / חולשה", "מתאוששת", "עולה"][stt], ["red", "yellow", "yellow", "green"][stt],
                  f"מחיר {'מעל' if gt(price, ema20) else 'מתחת'} EMA20 ({ema20:.2f}); MACD {'מעל' if macd > sig else 'מתחת'} לקו האות")
    if rsi < 30 or stoch < 20: mom = ("מכירת יתר", "green", "המניה נמכרה חזק – פוטנציאל לריבאונד")
    elif rsi > 70 or stoch > 80: mom = ("קניית יתר", "red", "המניה מתוחה למעלה – סיכון לתיקון")
    elif rsi < 45: mom = ("חלש", "yellow", "מומנטום שלילי מתון")
    elif rsi > 55: mom = ("חיובי", "green", "מומנטום חיובי מתון")
    else: mom = ("ניטרלי", "yellow", "אין כיוון ברור")
    mom_card = ("מומנטום", mom[0], mom[1], f"RSI {rsi:.0f} · סטוכסטי {stoch:.0f} · {mom[2]}")
    atrp = atr/price*100 if price else 0
    strong = adx >= 25
    vol_card = ("עוצמה / תנודתיות", ("מגמה חזקה" if strong else "מגמה חלשה/דשדוש") + (" · תנודתית" if atrp > 3.5 else ""),
                "green" if strong and atrp <= 3.5 else ("red" if atrp > 5 else "yellow"),
                f"ADX {adx:.0f} · ATR {atr:.2f} ({atrp:.1f}% ליום) · בטא {beta:.2f}" if beta is not None else f"ADX {adx:.0f} · ATR {atr:.2f} ({atrp:.1f}%)")

    # verdict
    if lt >= 2 and stt >= 2: verdict, vc = "מגמה עולה", "green"
    elif lt >= 2: verdict, vc = "תיקון בתוך מגמה עולה", "yellow"
    elif lt <= 1 and stt >= 2: verdict, vc = "ריבאונד בתוך מגמה יורדת", "yellow"
    else: verdict, vc = "מגמה יורדת", "red"
    if mom[0] == "מכירת יתר": verdict += " · מכירת יתר"
    if mom[0] == "קניית יתר": verdict += " · קניית יתר"

    # indicators table: (name, value, state(+1/0/-1), explanation)
    ind = []
    def add(name, val, st_, exp): ind.append((name, val, st_, exp))
    add("RSI(14)", f"{rsi:.1f}", 1 if rsi < 30 else -1 if rsi > 70 else 0, "מתחת 30 = נמכרה מדי (חיובי לקונה), מעל 70 = נקנתה מדי")
    add("MACD", f"{macd:.2f} / {sig:.2f}", 1 if macd > sig else -1, "MACD מעל קו האות = מומנטום עולה")
    add("סטוכסטי %K", f"{stoch:.0f}", 1 if stoch < 20 else -1 if stoch > 80 else 0, "מתחת 20 מכירת יתר, מעל 80 קניית יתר")
    add("Williams %R", f"{wr:.0f}", 1 if wr < -80 else -1 if wr > -20 else 0, "מתחת ‎-80 מכירת יתר, מעל ‎-20 קניית יתר")
    add("CCI(20)", f"{cci:.0f}", 1 if cci < -100 else -1 if cci > 100 else 0, "מתחת ‎-100 מכירת יתר, מעל 100 קניית יתר")
    add("ADX(14)", f"{adx:.0f}", (1 if g(tv, 'POSITIVE_DIRECTIONAL_INDICATOR_14', 0) > g(tv, 'NEGATIVE_DIRECTIONAL_INDICATOR_14', 0) else -1) if strong else 0, "מעל 25 = יש מגמה; הכיוון לפי ‎+DI/‎-DI")
    for n, lbl in [("EXPONENTIAL_MOVING_AVERAGE_20", "EMA20"), ("SIMPLE_MOVING_AVERAGE_50", "SMA50"), ("SIMPLE_MOVING_AVERAGE_200", "SMA200")]:
        v = g(tv, n)
        if v: add(lbl, f"{v:.2f}", 1 if price > v else -1, f"מחיר {'מעל' if price > v else 'מתחת'} לממוצע = {'חיובי' if price > v else 'שלילי'}")
    if vwap: add("VWAP", f"{vwap:.2f}", 1 if price > vwap else -1, "מחיר מעל המחיר הממוצע המשוקלל במחזור = קונים בשליטה")
    psar = g(tv, "PARABOLIC_SAR")
    if psar: add("Parabolic SAR", f"{psar:.2f}", 1 if price > psar else -1, "SAR מתחת למחיר = מגמה עולה")
    bu, bl = g(tv, "BOLLINGER_UPPER_BAND_20"), g(tv, "BOLLINGER_LOWER_BAND_20")
    if bu and bl:
        pos = (price-bl)/(bu-bl)*100 if bu > bl else 50
        add("מיקום בבולינגר", f"{pos:.0f}%", 1 if pos < 10 else -1 if pos > 90 else 0, "קרוב לרצועה התחתונה = זול יחסית, לעליונה = יקר יחסית")
    au, ad = g(tv, "AROON_UP_14"), g(tv, "AROON_DOWN_14")
    if au is not None: add("Aroon", f"{au:.0f} / {ad:.0f}", 1 if au > ad else -1, "Up מעל Down = שיאים חדשים לאחרונה")
    tr = g(tv, "TECHNICAL_RATING")
    if isinstance(tr, (int, float)): add("דירוג TradingView", f"{tr:+.2f}", 1 if tr > .1 else -1 if tr < -.1 else 0, "סיכום כל האינדיקטורים, ‎-1 מכירה חזקה עד ‎+1 קנייה חזקה")
    elif tr: add("דירוג TradingView", str(tr), 1 if "Buy" in str(tr) else -1 if "Sell" in str(tr) else 0, "סיכום כל האינדיקטורים")

    # scenarios
    strong_sup = [c for c in sup if c["n"] >= 3] or sup
    strong_res = [c for c in res if c["n"] >= 3] or res
    rr = lambda e, s, t: (t-e)/(e-s) if e > s else None
    scen = []
    near = [c for c in strong_sup if c["dist"] > -8]
    if near:
        a = max(near, key=lambda c: (c["n"], c["price"]))
        below = [c for c in sup if c["price"] < a["lo"]*0.995]
        stop = (below[0]["lo"] if below else a["lo"] - atr) - 0.25*atr
        tg = [c["price"] for c in res[-1:]] + [c["price"] for c in strong_res if c["price"] > (res[-1]["price"] if res else price)][-1:]
        tg = sorted(set(round(t, 2) for t in tg))[:2] or [round(price + 2*atr, 2)]
        scen.append(("אגרסיבי", "קנייה בתמיכה החזקה מתחת למחיר", a["price"], stop, tg, a))
    if strong_res:
        r = strong_res[-1]; trig = max(r["hi"], vwap or 0) if (vwap and vwap > price) else r["hi"]
        entry = trig * 1.003; stop = min(price, r["lo"]) - 0.5*atr
        tg = [c["price"] for c in res if c["price"] > entry*1.01][-2:]
        tg = sorted(tg)[:2] or [entry + 2*atr]
        scen.append(("שמרני", f"כניסה רק בסגירה יומית מעל {trig:.2f}" + (" (כולל אישור VWAP)" if vwap and vwap > price else ""), entry, stop, tg, r))
    deep = [c for c in strong_sup if c["dist"] <= -6] or strong_sup[1:]
    if deep:
        d = max(deep[:4], key=lambda c: c["n"])
        below = [c for c in sup if c["price"] < d["lo"]*0.995]
        stop = (below[0]["lo"] if below else d["lo"] - atr) - 0.25*atr
        tg = sorted([c["price"] for c in sup if c["price"] > d["price"]*1.03][-2:] or [price])[:2]
        scen.append(("עמוק", "המתנה לירידה לאזור תמיכה מרכזי", d["price"], stop, tg, d))
    scen = [dict(name=n, desc=ds, entry=e, stop=s, targets=t, rr=[rr(e, s, x) for x in t], zone=z) for n, ds, e, s, t, z in scen]

    inval = []
    if strong_sup: inval.append(f"שבירה וסגירה מתחת ל-{strong_sup[0]['lo']:.2f} (התמיכה החזקה הקרובה) – פותחת דרך לאזור {strong_sup[1]['price']:.2f}" if len(strong_sup) > 1 else f"שבירה מתחת ל-{strong_sup[0]['lo']:.2f}")
    if sma200 and price > sma200: inval.append(f"סגירה מתחת ל-SMA200 ({sma200:.2f}) – תבטל את המגמה העולה הארוכה")
    if sma200 and price < sma200: inval.append(f"סגירה מעל SMA200 ({sma200:.2f}) – תבטל את התרחיש הדובי")
    if strong_res: inval.append(f"פריצה וסגירה מעל {strong_res[-1]['hi']:.2f} – מבטלת את תרחיש החולשה הקצרה")

    return dict(ticker=ticker, tv=tv, df=df, price=price, chg=g(tv, "CHANGE_PERCENT", 0), name=g(tv, "DESCRIPTION", ""),
                exch=g(tv, "EXCHANGE", ""), verdict=verdict, vcolor=vc, cards=[long_card, short_card, mom_card, vol_card],
                res=res, sup=sup, ind=ind, scen=scen, inval=inval, atr=atr)

# ---------------- UI ----------------
COL = {"green": "#16a34a", "yellow": "#d97706", "red": "#dc2626"}
CSS = """<style>
html, body, [class*="css"], .stApp, .main, .block-container {direction: rtl; text-align: right;}
.stTextInput input {direction: ltr; text-align: left;}
.badge {display:inline-block;padding:4px 14px;border-radius:999px;color:#fff;font-weight:700;font-size:1.1rem}
.card {border-radius:14px;padding:14px;color:#fff;min-height:130px}
.card h4{margin:0;color:#fff;font-size:1rem;opacity:.9} .card .v{font-size:1.5rem;font-weight:800;margin:6px 0} .card .d{font-size:.85rem;opacity:.95}
.lad {display:flex;align-items:center;gap:8px;padding:5px 8px;border-bottom:1px solid #eee;font-size:.9rem}
.lad .p{width:80px;font-weight:700;direction:ltr;text-align:left} .lad .dist{width:65px;direction:ltr;color:#666}
.lad .bar{height:12px;border-radius:6px} .lad .src{flex:1;color:#444;font-size:.8rem}
.now {background:#1e3a8a;color:#fff;font-weight:800;padding:8px;border-radius:8px;text-align:center;margin:4px 0}
.sc {border:2px solid #ddd;border-radius:14px;padding:14px;margin-bottom:8px}
.sc .row{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px dashed #eee}
</style>"""

def chart(a):
    df = a["df"].iloc[-130:].copy(); full = a["df"]
    c = full.Close
    ind = pd.DataFrame({"SMA50": c.rolling(50).mean(), "SMA200": c.rolling(200, min_periods=120).mean(), "EMA20": c.ewm(span=20).mean(),
                        "BBm": c.rolling(20).mean(), "BBs": c.rolling(20).std()}).loc[df.index]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=ind.BBm+2*ind.BBs, line=dict(color="rgba(120,120,200,.4)", width=1), name="בולינגר עליון"))
    fig.add_trace(go.Scatter(x=df.index, y=ind.BBm-2*ind.BBs, line=dict(color="rgba(120,120,200,.4)", width=1), fill="tonexty", fillcolor="rgba(120,120,200,.08)", name="בולינגר תחתון"))
    fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name=a["ticker"]))
    for n, col in [("EMA20", "#f59e0b"), ("SMA50", "#2563eb"), ("SMA200", "#7c3aed")]:
        fig.add_trace(go.Scatter(x=df.index, y=ind[n], line=dict(color=col, width=1.6), name=n))
    for cl in sorted(a["res"], key=lambda c: -c["n"])[:3]:
        fig.add_hline(y=cl["price"], line=dict(color="#dc2626", dash="dash", width=1), annotation_text=f"התנגדות {cl['price']:.2f} ({cl['n']})", annotation_position="top left")
    for cl in sorted(a["sup"], key=lambda c: -c["n"])[:3]:
        fig.add_hline(y=cl["price"], line=dict(color="#16a34a", dash="dash", width=1), annotation_text=f"תמיכה {cl['price']:.2f} ({cl['n']})", annotation_position="bottom left")
    for s in a["scen"]:
        if s["name"] != "שמרני":
            z = s["zone"]; fig.add_hrect(y0=z["lo"], y1=z["hi"] if z["hi"] > z["lo"] else z["lo"]*1.003, fillcolor="green", opacity=.12, line_width=0,
                                         annotation_text=f"אזור כניסה {s['name']}", annotation_position="right")
    fig.update_layout(height=560, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.06))
    return fig

def ladder_html(a, nmax=9):
    res = a["res"][-nmax:]; sup = a["sup"][:nmax]; mx = max([c["n"] for c in res + sup] or [1])
    def row(c, col):
        return (f"<div class='lad'><span class='p'>{c['price']:.2f}</span><span class='dist'>{c['dist']:+.1f}%</span>"
                f"<span class='bar' style='width:{int(20+100*c['n']/mx)}px;background:{col}'></span><b>{c['n']}</b>"
                f"<span class='src'>{', '.join(c['sources'][:6])}{' …' if len(c['sources'])>6 else ''}</span></div>")
    h = "<div style='color:#dc2626;font-weight:700'>▲ התנגדויות (מעל המחיר)</div>" + "".join(row(c, COL["red"]) for c in res)
    h += f"<div class='now'>◀ מחיר נוכחי {a['price']:.2f} ▶</div>"
    h += "".join(row(c, COL["green"]) for c in sup) + "<div style='color:#16a34a;font-weight:700'>▼ תמיכות (מתחת למחיר)</div>"
    return h

def main():
    st.set_page_config(page_title="ניתוח טכני", page_icon="📈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("📈 ניתוח טכני פשוט למניה")
    qp = st.query_params.get("t", "INTC")
    t = st.text_input("הקלד סימול מניה אמריקאית (לדוגמה INTC, NVDA, AAPL):", value=qp).strip().upper()
    if not t: return
    try:
        with st.spinner(f"מושך נתונים עבור {t}..."): a = analyze(t)
    except LookupError as e: st.error(str(e)); return
    except Exception as e: st.error(f"שגיאה במשיכת נתונים עבור {t}: {e}"); return
    chc = COL["green"] if a["chg"] >= 0 else COL["red"]
    st.markdown(f"<h2 style='margin-bottom:0'>{a['ticker']} <span style='font-size:1rem;color:#666'>{a['name']} · {a['exch']}</span></h2>"
                f"<div style='font-size:2rem;font-weight:800'>{a['price']:.2f}$ <span style='color:{chc};font-size:1.3rem;direction:ltr;display:inline-block'>{a['chg']:+.2f}%</span></div>"
                f"<span class='badge' style='background:{COL[a['vcolor']]}'>{a['verdict']}</span>"
                f"<div style='color:#888;font-size:.8rem;margin-top:6px'>⏱ הנתונים בעיכוב של כ-15 דקות (TradingView + Yahoo Finance)</div>", unsafe_allow_html=True)
    st.write("")
    for col, (title, val, color, det) in zip(st.columns(4), a["cards"]):
        col.markdown(f"<div class='card' style='background:{COL[color]}'><h4>{title}</h4><div class='v'>{val}</div><div class='d'>{det}</div></div>", unsafe_allow_html=True)
    st.subheader("🕯️ גרף 6 חודשים")
    if a["df"] is not None: st.plotly_chart(chart(a), width='stretch')
    else: st.info("אין נתוני גרף")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("🪜 סולם מחירים – תמיכה והתנגדות")
        st.caption("כל רמה היא אשכול של כמה מקורות טכניים בטווח ‎±1%. ככל שהפס ארוך יותר – יותר מקורות מסכימים, והרמה חזקה יותר.")
        st.markdown(ladder_html(a), unsafe_allow_html=True)
    with c2:
        st.subheader("🎛️ אינדיקטורים בשפה פשוטה")
        icon = {1: "🟢", 0: "⚪", -1: "🔴"}
        st.dataframe(pd.DataFrame([{"": icon[s], "אינדיקטור": n, "ערך": v, "מה זה אומר": e} for n, v, s, e in a["ind"]]),
                     hide_index=True, width='stretch', height=560)
        pos = sum(s == 1 for *_, s, _ in a["ind"]); neg = sum(s == -1 for *_, s, _ in a["ind"])
        st.markdown(f"**סיכום:** 🟢 {pos} חיוביים · 🔴 {neg} שליליים · ⚪ {len(a['ind'])-pos-neg} ניטרליים")
    st.subheader("🎯 תרחישי כניסה")
    cols = st.columns(max(len(a["scen"]), 1))
    clr = {"אגרסיבי": "#dc2626", "שמרני": "#2563eb", "עמוק": "#16a34a"}
    for col, s in zip(cols, a["scen"]):
        risk = (s["entry"]-s["stop"])/s["entry"]*100
        tg = "".join(f"<div class='row'><span>יעד {i+1}</span><b>{t:.2f} (R:R {r:.1f})</b></div>" if r else f"<div class='row'><span>יעד {i+1}</span><b>{t:.2f}</b></div>" for i, (t, r) in enumerate(zip(s["targets"], s["rr"])))
        col.markdown(f"<div class='sc' style='border-color:{clr[s['name']]}'><h3 style='color:{clr[s['name']]};margin:0'>{s['name']}</h3>"
                     f"<div style='color:#555;margin-bottom:6px'>{s['desc']}</div>"
                     f"<div class='row'><span>כניסה</span><b>{s['entry']:.2f}</b></div>"
                     f"<div class='row'><span>סטופ</span><b style='color:#dc2626'>{s['stop']:.2f} (‎-{risk:.1f}%)</b></div>{tg}"
                     f"<div style='font-size:.8rem;color:#777;margin-top:6px'>מקורות האזור: {', '.join(s['zone']['sources'][:5])}</div></div>", unsafe_allow_html=True)
    st.subheader("⚠️ מה יבטל את התמונה")
    for x in a["inval"]: st.markdown(f"- {x}")

if __name__ == "__main__":
    main()
