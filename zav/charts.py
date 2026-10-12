#!/usr/bin/env python
"""גרפי נרות סטטיים (mplfinance) לכל מניה: נרות יומיים ~6 חודשים, מחזור, ממוצעים נעים,
אזורי תמיכה/התנגדות (אשכולות), פיבוטים, פיבונאצ'י, VWAP, ענן איצ'ימוקו ואזורי כניסה.
הרמות מחושבות בדיוק כמו ב-app.py (analyze / yf_levels), כך שהגרף תואם לדף הניתוח.

שימוש:  python charts.py INTC SOFI [--out charts] [--fmt png svg]
"""
import argparse, logging, os, sys
logging.getLogger("streamlit").setLevel(logging.ERROR)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import analyze, yf_levels

# matplotlib>=3.9 (עם libraqm) מבצע bidi בעצמו; RLM קובע כיוון בסיס מימין לשמאל.
H = lambda s: "\u200f" + s
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["svg.fonttype"] = "path"

BARS = 126  # ~6 חודשי מסחר
SCEN_COL = {"אגרסיבי": "#2563eb", "שמרני": "#7c3aed", "עמוק": "#0891b2"}


def ichimoku(df):
    hi, lo = df.High, df.Low
    tenkan = (hi.rolling(9).max() + lo.rolling(9).min()) / 2
    kijun = (hi.rolling(26).max() + lo.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((hi.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    return tenkan, kijun, span_a, span_b


def render(ticker, out_dir="charts", fmts=("png",), a=None):
    a = a or analyze(ticker)
    full = a["df"]
    if full is None or full.empty: raise RuntimeError(f"אין נתוני Yahoo ל-{ticker}")
    yfr = yf_levels(full)
    tv, price = a["tv"], a["price"]
    full = full.copy()
    full["SMA50"] = full.Close.rolling(50).mean(); full["SMA200"] = full.Close.rolling(200).mean()
    full["EMA20"] = full.Close.ewm(span=20, adjust=False).mean()
    _, _, sa, sb = ichimoku(full)
    df = full.iloc[-BARS:]
    sa, sb = sa.iloc[-BARS:], sb.iloc[-BARS:]

    ap = [mpf.make_addplot(df.EMA20, color="#f59e0b", width=1.1),
          mpf.make_addplot(df.SMA50, color="#2563eb", width=1.3)]
    if df.SMA200.notna().any(): ap.append(mpf.make_addplot(df.SMA200, color="#111827", width=1.5))
    style = mpf.make_mpf_style(base_mpf_style="yahoo", gridstyle=":", facecolor="#ffffff",
                               rc={"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = mpf.plot(df, type="candle", volume=True, addplot=ap, style=style, returnfig=True,
                         figsize=(15, 8.5), panel_ratios=(4, 1), tight_layout=True,
                         xrotation=0, datetime_format="%d/%m", scale_padding={"left": .3, "right": 2.6})
    ax = axes[0]; x = range(len(df)); n = len(df)

    # ענן איצ'ימוקו
    ax.fill_between(x, sa.values, sb.values, where=(sa >= sb).values, color="#16a34a", alpha=.08, interpolate=True)
    ax.fill_between(x, sa.values, sb.values, where=(sa < sb).values, color="#dc2626", alpha=.08, interpolate=True)

    ymin, ymax = float(df.Low.min()), float(df.High.max())
    pad = (ymax - ymin) * .12; lo_lim, hi_lim = ymin - pad, ymax + pad
    xr = n + 0.5  # עמודת תוויות מימין
    labels = []

    def hline(y, color, ls, lw, text, alpha=.9):
        if not (lo_lim <= y <= hi_lim): return
        ax.axhline(y, color=color, ls=ls, lw=lw, alpha=alpha)
        labels.append((y, color, text))

    # אזורי תמיכה/התנגדות: 4 אשכולות קרובים מכל צד
    for c in a["res"][-4:]:
        ax.axhspan(c["lo"], max(c["hi"], c["lo"] + price * .002), color="#dc2626", alpha=min(.08 + .03 * c["n"], .3), lw=0)
        hline(c["price"], "#dc2626", "-", .8, H(f"התנגדות {c['price']:.2f} ({c['n']})"), .6)
    for c in a["sup"][:4]:
        ax.axhspan(c["lo"], max(c["hi"], c["lo"] + price * .002), color="#16a34a", alpha=min(.08 + .03 * c["n"], .3), lw=0)
        hline(c["price"], "#16a34a", "-", .8, H(f"תמיכה {c['price']:.2f} ({c['n']})"), .6)
    # פיבונאצ'י
    for k, v in yfr["fib"].items(): hline(v, "#a16207", ":", 1, H(f"פיבו {k}  {v:.2f}"))
    # פיבוטים שבועיים + חודשי קלאסי
    for k, v in yfr["piv_w"].items(): hline(v, "#6b7280", "--", .9, H(f"פיבוט שבועי {k}  {v:.2f}"))
    pm = tv.get("PIVOT_CLASSIC_P")
    if isinstance(pm, (int, float)): hline(pm, "#9333ea", "--", 1.1, H(f"פיבוט חודשי P  {pm:.2f}"))
    vwap = tv.get("VOLUME_WEIGHTED_AVERAGE_PRICE")
    if isinstance(vwap, (int, float)): hline(vwap, "#db2777", "-.", 1.2, f"VWAP  {vwap:.2f}")
    # אזורי כניסה מתרחישים
    scen_txt = []
    for s in a["scen"]:
        col = SCEN_COL.get(s["name"], "#2563eb"); z = s["zone"]
        lo_z, hi_z = min(z["lo"], s["entry"]), max(z["hi"], s["entry"])
        if hi_z - lo_z < price * .004: lo_z, hi_z = s["entry"] - price * .002, s["entry"] + price * .002
        ax.add_patch(plt.Rectangle((n - 22, lo_z), 22.5, hi_z - lo_z, facecolor=col, alpha=.28, edgecolor=col, lw=1.5, zorder=3))
        scen_txt.append((col, H(f"כניסה {s['name']}: {lo_z:.2f}–{hi_z:.2f} · סטופ {s['stop']:.2f} · יעדים " + ", ".join(f"{t:.2f}" for t in s["targets"]))))
        if lo_lim <= s["stop"] <= hi_lim: ax.plot([n - 22, n + .5], [s["stop"]] * 2, color=col, ls=(0, (2, 2)), lw=1.3, zorder=3)
    for i, (col, t) in enumerate(scen_txt):
        ax.text(.99, .03 + .045 * (len(scen_txt) - 1 - i), t, transform=ax.transAxes, color=col, fontsize=8.5,
                fontweight="bold", ha="right", va="bottom", zorder=5,
                bbox=dict(boxstyle="round,pad=.25", fc="white", ec=col, alpha=.9))
    ax.axhline(price, color="#1e3a8a", lw=1.6)
    labels.append((price, "#1e3a8a", H(f"מחיר {price:.2f}")))

    # תוויות ללא חפיפה
    labels.sort(key=lambda t: t[0]); gap = (hi_lim - lo_lim) * .028; placed = []
    for y, col, t in labels:
        yy = max(y, placed[-1] + gap) if placed else y
        placed.append(yy)
        ax.annotate(t, xy=(n - .5, y), xytext=(xr, yy), color=col, fontsize=7.5, va="center", ha="left",
                    annotation_clip=False, arrowprops=dict(arrowstyle="-", color=col, lw=.4, alpha=.6))
    ax.set_ylim(lo_lim, max(hi_lim, placed[-1] + gap if placed else hi_lim))
    ax.set_title(H(f"{a['ticker']} · נרות יומיים 6 חודשים · {a['verdict']}"), fontsize=13, fontweight="bold", loc="right")
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], color=c, lw=w, ls=l, label=H(t)) for c, w, l, t in [
        ("#f59e0b", 1.2, "-", "EMA20"), ("#2563eb", 1.3, "-", "SMA50"), ("#111827", 1.5, "-", "SMA200"),
        ("#a16207", 1, ":", "פיבונאצ'י"), ("#6b7280", 1, "--", "פיבוט שבועי"), ("#9333ea", 1, "--", "פיבוט חודשי"),
        ("#db2777", 1.2, "-.", "VWAP"), ("#16a34a", 6, "-", "אזור תמיכה"), ("#dc2626", 6, "-", "אזור התנגדות")]]
    ax.legend(handles=leg, loc="upper left", fontsize=7.5, ncol=3, framealpha=.85)
    axes[2].set_ylabel(H("מחזור"))
    for axx in (ax, axes[2]):
        axx.yaxis.tick_left(); axx.yaxis.set_label_position("left")
    ax.set_ylabel(H("מחיר ($)"))

    os.makedirs(out_dir, exist_ok=True); paths = []
    for f in fmts:
        p = os.path.join(out_dir, f"{a['ticker']}.{f}")
        fig.savefig(p, dpi=110, bbox_inches="tight"); paths.append(p)
    plt.close(fig)
    return paths


def main():
    p = argparse.ArgumentParser(description="mplfinance candlestick charts with levels")
    p.add_argument("tickers", nargs="+")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts"))
    p.add_argument("--fmt", nargs="+", default=["png"], choices=["png", "svg"])
    a = p.parse_args()
    for t in a.tickers:
        try: print(t.upper(), "->", ", ".join(render(t.upper(), a.out, a.fmt)))
        except Exception as e: print(t.upper(), "failed:", e)


if __name__ == "__main__":
    main()
