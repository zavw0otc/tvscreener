"""INTC deep indicator pull: TradingView (tvscreener) + computed levels from yfinance daily OHLCV."""
import json, math
import numpy as np, pandas as pd
from tvscreener import StockScreener, StockField as F, FilterOperator as Op, Market

TICKER, OUT = "INTC", "results/intc_deep_2026-10-11"
REF_PRICE = 104.70
P = lambda k: [f"PIVOT_{k}_{x}" for x in ["S3","S2","S1","P","R1","R2","R3"]]
GROUPS = {
 "quote": ["NAME","EXCHANGE","PRICE","CHANGE_PERCENT","OPEN","HIGH","LOW","VOLUME"],
 "pivots_monthly": P("CLASSIC")+P("FIBONACCI")+P("CAMARILLA")+P("WOODIE")+["PIVOT_DM_S1","PIVOT_DM_P","PIVOT_DM_R1"],
 "ichimoku": ["ICHIMOKU_CONVERSION_LINE_9_26_52_26","ICHIMOKU_BASE_LINE_9_26_52_26","ICHIMOKU_LEADING_SPAN_A_9_26_52_26","ICHIMOKU_LEADING_SPAN_B_9_26_52_26","ICHIMOKU_LAGGING_SPAN"],
 "vwap_vwma": ["VOLUME_WEIGHTED_AVERAGE_PRICE","VOLUME_WEIGHTED_MOVING_AVERAGE_20"],
 "moving_averages": ["HULL_MOVING_AVERAGE_9"]+[f"{t}_MOVING_AVERAGE_{n}" for t in ["SIMPLE","EXPONENTIAL"] for n in [10,20,30,50,100,200]],
 "trend": ["PARABOLIC_SAR","AVERAGE_DIRECTIONAL_INDEX_14","POSITIVE_DIRECTIONAL_INDICATOR_14","NEGATIVE_DIRECTIONAL_INDICATOR_14","AROON_UP_14","AROON_DOWN_14"],
 "oscillators": ["RELATIVE_STRENGTH_INDEX_14","MACD_LEVEL_12_26","MACD_SIGNAL_12_26","MACD_HIST","STOCHASTIC_PERCENTK_14_3_3","STOCHASTIC_PERCENTD_14_3_3","STOCHASTIC_RSI_FAST_3_3_14_14","STOCHASTIC_RSI_SLOW_3_3_14_14","WILLIAMS_PERCENT_RANGE_14","COMMODITY_CHANNEL_INDEX_20","MOMENTUM_10","AWESOME_OSCILLATOR","ULTIMATE_OSCILLATOR_7_14_28","BULL_BEAR_POWER"],
 "channels": ["BOLLINGER_UPPER_BAND_20","BOLLINGER_LOWER_BAND_20","KELTNER_CHANNELS_UPPER_BAND_20","KELTNER_CHANNELS_LOWER_BAND_20","DONCHIAN_CHANNELS_UPPER_BAND_20","DONCHIAN_CHANNELS_LOWER_BAND_20"],
 "range_perf_vol": ["WEEK_HIGH_52","WEEK_LOW_52","PRICE_52_WEEK_HIGH_DATE","PRICE_52_WEEK_LOW_DATE","WEEKLY_PERFORMANCE","MONTHLY_PERFORMANCE","MONTH_PERFORMANCE_3","MONTH_PERFORMANCE_6","YTD_PERFORMANCE","YEARLY_PERFORMANCE","AVERAGE_TRUE_RANGE_14","VOLATILITY","VOLATILITY_WEEK","VOLATILITY_MONTH","YEAR_BETA_1","BETA_3_YEAR","BETA_5_YEAR","AVERAGE_VOLUME_10_DAY","AVERAGE_VOLUME_30_DAY","AVERAGE_VOLUME_90_DAY","RELATIVE_VOLUME"],
 "ratings": ["TECHNICAL_RATING","MOVING_AVERAGES_RATING","OSCILLATORS_RATING"],
}

def tv_pull():
    names = [n for g in GROUPS.values() for n in g]
    have = [n for n in names if n in F.__members__]
    s = StockScreener(); s.set_markets(Market.AMERICA)
    s.select(*[F[n] for n in have]); s.add_filter(F.NAME, Op.EQUAL, TICKER); s.set_range(0, 50)
    raw = s.get()
    raw = raw[(raw[F.NAME.label] == TICKER) & (raw[F.EXCHANGE.label] == "NASDAQ")]
    if raw.empty: raise SystemExit("INTC NASDAQ row not found")
    row = raw.iloc[0]; out = {}
    for g, ns in GROUPS.items():
        out[g] = {}
        for n in ns:
            if n not in F.__members__: out[g][n] = {"status": "MISSING_FIELD"}; continue
            f = F[n]; v = row.get(f.label)
            v = None if v is None or (isinstance(v, float) and math.isnan(v)) else (v.item() if hasattr(v, "item") else v)
            out[g][n] = {"tv_field": f.value[1] if isinstance(f.value, tuple) else str(f.value), "value": v,
                         **({} if v is not None else {"status": "NULL_FROM_TV"})}
    return out

def yf_levels():
    import yfinance as yf
    df = yf.download(TICKER, period="6mo", interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    res = {"bars": len(df), "first": str(df.index[0].date()), "last": str(df.index[-1].date()),
           "last_close": round(float(df.Close.iloc[-1]), 2)}
    H, L, C, V = df.High, df.Low, df.Close, df.Volume
    # swing pivots: 5-bar fractals
    k = 5
    sh = [(d, float(H[d])) for i, d in enumerate(df.index[k:-k], k) if H.iloc[i] == H.iloc[i-k:i+k+1].max()]
    sl = [(d, float(L[d])) for i, d in enumerate(df.index[k:-k], k) if L.iloc[i] == L.iloc[i-k:i+k+1].min()]
    res["swing_highs"] = [{"date": str(d.date()), "price": round(p, 2)} for d, p in sh[-6:]]
    res["swing_lows"] = [{"date": str(d.date()), "price": round(p, 2)} for d, p in sl[-6:]]
    # major swing: 6-month absolute high & low; direction by which came last
    hi_d, lo_d = H.idxmax(), L.idxmin(); hi, lo = float(H.max()), float(L.min())
    up = lo_d < hi_d
    rng = hi - lo
    ratios = [0, .236, .382, .5, .618, .786, 1]
    fib = {f"{r*100:.1f}%": round((hi - r*rng) if up else (lo + r*rng), 2) for r in ratios}
    res["fib_major_swing"] = {"swing_low": {"date": str(lo_d.date()), "price": round(lo, 2)},
        "swing_high": {"date": str(hi_d.date()), "price": round(hi, 2)},
        "direction": "up (retracement measured down from high)" if up else "down (retracement up from low)", "levels": fib}
    # volume profile: distribute each bar's volume across its H-L range, 40 bins
    bins = np.linspace(lo, hi, 41); vp = np.zeros(40)
    for h, l, v in zip(H, L, V):
        ov = np.clip(np.minimum(bins[1:], h) - np.maximum(bins[:-1], l), 0, None)
        vp += v * ov / ov.sum() if ov.sum() > 0 else 0
    mids = (bins[1:] + bins[:-1]) / 2
    poc = int(vp.argmax()); order = np.argsort(vp)[::-1]
    tot = vp.sum(); cum = 0; va = []
    for i in order:
        cum += vp[i]; va.append(i)
        if cum >= .7*tot: break
    hvn = []
    for i in order:
        if all(abs(i - j) > 2 for j in hvn): hvn.append(i)
        if len(hvn) == 5: break
    res["volume_profile"] = {"bin_width": round(float(bins[1]-bins[0]), 2), "POC": round(float(mids[poc]), 2),
        "value_area_low": round(float(bins[min(va)]), 2), "value_area_high": round(float(bins[max(va)+1]), 2),
        "HVNs": [{"price": round(float(mids[i]), 2), "vol_share_pct": round(float(vp[i]/tot*100), 2)} for i in hvn]}
    # classic daily/weekly pivots from last completed session / week
    def classic(h, l, c):
        p = (h+l+c)/3
        return {"S3": l-2*(h-p), "S2": p-(h-l), "S1": 2*p-h, "P": p, "R1": 2*p-l, "R2": p+(h-l), "R3": h+2*(p-l)}
    d = df.iloc[-1]; res["pivot_classic_daily_next"] = {k: round(v, 2) for k, v in classic(d.High, d.Low, d.Close).items()}
    w = df.resample("W-FRI").agg({"High": "max", "Low": "min", "Close": "last"}).dropna().iloc[-1]
    res["pivot_classic_weekly_next"] = {k: round(v, 2) for k, v in classic(w.High, w.Low, w.Close).items()}
    res["recent"] = {"high_20d": round(float(H[-20:].max()), 2), "low_20d": round(float(L[-20:].min()), 2),
                     "high_5d": round(float(H[-5:].max()), 2), "low_5d": round(float(L[-5:].min()), 2)}
    return res

def ladder(tv, yfr, price=REF_PRICE, tol_pct=1.0):
    lv = []
    add = lambda src, name, v: v is not None and isinstance(v, (int, float)) and lv.append({"price": round(float(v), 2), "source": src, "level": name})
    for g in ["pivots_monthly", "ichimoku", "vwap_vwma", "moving_averages", "channels"]:
        for n, d in tv[g].items(): add("TV", n, d.get("value"))
    for n in ["PARABOLIC_SAR", ]: add("TV", n, tv["trend"][n].get("value"))
    for n in ["WEEK_HIGH_52", "WEEK_LOW_52"]: add("TV", n, tv["range_perf_vol"][n].get("value"))
    if yfr:
        for k, v in yfr["fib_major_swing"]["levels"].items(): add("YF", f"fib_{k}", v)
        vp = yfr["volume_profile"]; add("YF", "VP_POC", vp["POC"]); add("YF", "VP_VAL", vp["value_area_low"]); add("YF", "VP_VAH", vp["value_area_high"])
        for h in vp["HVNs"]: add("YF", "HVN", h["price"])
        for s in yfr["swing_highs"]: add("YF", f"swing_high {s['date']}", s["price"])
        for s in yfr["swing_lows"]: add("YF", f"swing_low {s['date']}", s["price"])
        for k, v in yfr["pivot_classic_daily_next"].items(): add("YF", f"daily_classic_{k}", v)
        for k, v in yfr["pivot_classic_weekly_next"].items(): add("YF", f"weekly_classic_{k}", v)
        for k, v in yfr["recent"].items(): add("YF", k, v)
    lv.sort(key=lambda x: -x["price"])
    for x in lv:
        near = [y for y in lv if abs(y["price"] - x["price"]) <= x["price"]*tol_pct/100]
        x["cluster_count"] = len(near); x["side"] = "resistance" if x["price"] > price else "support"
        x["dist_pct"] = round((x["price"]/price - 1)*100, 2)
    return lv

if __name__ == "__main__":
    tv = tv_pull()
    try: yfr = yf_levels()
    except Exception as e: yfr = None; print("yfinance failed:", e)
    lad = ladder(tv, yfr)
    json.dump({"ticker": TICKER, "ref_price": REF_PRICE, "note": "TV = tvscreener (~15m delayed, daily TF; pivots are TV monthly). YF = computed from yfinance 6mo daily.",
               "tradingview": tv, "yfinance_computed": yfr, "ladder": lad}, open(OUT + ".json", "w"), indent=1, default=str)
    pd.DataFrame(lad).to_csv(OUT + ".csv", index=False)
    print(json.dumps({"tv": tv, "yf": yfr}, default=str))
    for x in lad: print(f"{x['price']:>9} {x['side']:<10} {x['dist_pct']:>7}% n={x['cluster_count']:<2} {x['source']} {x['level']}")
