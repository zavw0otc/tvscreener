#!/usr/bin/env python
"""CLI: python run.py [TICKER ...]  -> results/<date>.csv + printed table."""
import argparse, datetime, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import fetch, add_signals

DEFAULT = ["NVDA","AMD","TSM","AVGO","MU","INTC","ASML","AMAT","LRCX","QCOM","AAPL","MSFT","TSLA"]

def main():
    p = argparse.ArgumentParser(description="Technical indicators via tvscreener")
    p.add_argument("tickers", nargs="*", default=DEFAULT)
    p.add_argument("--charts", action="store_true", help="also render mplfinance PNG charts to charts/<TICKER>.png")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = p.parse_args()
    tickers = [t.upper() for t in a.tickers]
    df, missing = fetch(tickers)
    if missing: print("Missing fields (skipped):", missing)
    df = add_signals(df)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{datetime.date.today().isoformat()}.csv")
    df.round(3).to_csv(path, index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 50)
    num = ["symbol","exchange","price","chg_pct","rsi14","macd","macd_signal","macd_hist","sma50","sma200","ema20",
           "bb_upper","bb_lower","bb_pos_pct","stoch_k","stoch_d","adx","atr","rel_volume","tech_rating"]
    sig = ["symbol","cross","sig_rsi","sig_trend","sig_macd","sig_bb","sig_stoch","sig_adx","tv_rating","summary"]
    print(df[num].round(2).to_string(index=False)); print(); print(df[sig].to_string(index=False))
    if a.charts:
        from charts import render
        cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
        for t in tickers:
            try: print("Chart:", ", ".join(render(t, cdir, ("png",))))
            except Exception as e: print(f"Chart {t} failed: {e}")
    print(f"\nSaved: {path}\nData ~15 min delayed.")

if __name__ == "__main__":
    main()
