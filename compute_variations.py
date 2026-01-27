import argparse
from typing import Tuple, Dict, Optional
import time
import pandas as pd
import numpy as np
import yfinance as yf

TICKER_MAP = {
    "SPX": "^GSPC",
    "NDX": "^NDX",
    "VIX": "^VIX",
    "ILF": "ILF",
    "EWZ": "EWZ",
    "EMB": "EMB",
    "/CL": "CL=F",
    "GLD": "GLD",
    "XLE": "XLE",
    "XLC": "XLC",
    "XLP": "XLP",
    "XLK": "XLK",
    "XLV": "XLV",
    "QTUM": "QTUM",
    "SOXX": "SOXX",
    "TSLA": "TSLA",
    "AAPL": "AAPL",
    "GOOG": "GOOG",
    "NVDA": "NVDA",
    "META": "META",
    "MSFT": "MSFT",
    "AMZN": "AMZN",
    "RGTI": "RGTI",
    "QBTS": "QBTS",
    "IONQ": "IONQ",
}

def _normalize_close_dataframe(px: pd.DataFrame, symbols: list) -> pd.DataFrame:
    if px.empty:
        return pd.DataFrame()
    if isinstance(px.columns, pd.MultiIndex):
        close = px["Close"].copy()
    else:
        if "Close" in px.columns:
            close = px[["Close"]].rename(columns={"Close": symbols[0]}).copy()
        else:
            close = px.copy()
            if close.shape[1] == 1:
                close.columns = [symbols[0]]
            else:
                close = close.iloc[:, :].copy()
    close.index = pd.to_datetime(close.index)
    close = close.dropna(how="all")
    return close

def _fetch_intraday_latest(symbol: str, retry_delays: Optional[list] = None) -> Optional[Tuple[float, pd.Timestamp]]:
    if retry_delays is None:
        retry_delays = [0, 0.2, 0.5]
    intervals_to_try = ["1m", "5m", "15m"]
    for i, interval in enumerate(intervals_to_try):
        delay = retry_delays[i] if i < len(retry_delays) else 0
        try:
            if delay:
                time.sleep(delay)
            intr = yf.download(
                tickers=symbol,
                period="1d",
                interval=interval,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )
            if intr is None or intr.empty:
                continue
            if isinstance(intr.columns, pd.MultiIndex):
                if "Close" in intr:
                    close_series = intr["Close"].copy()
                    if isinstance(close_series, pd.DataFrame):
                        if symbol in close_series.columns:
                            s = close_series[symbol].dropna()
                        else:
                            s = close_series.iloc[:, 0].dropna()
                    else:
                        s = close_series.dropna()
                else:
                    s = intr.iloc[:, 0].dropna()
            else:
                if "Close" in intr.columns:
                    s = intr["Close"].dropna()
                else:
                    s = intr.iloc[:, 0].dropna()
            if len(s) == 0:
                continue
            last_ts = pd.to_datetime(s.index[-1])
            last_price = float(s.iloc[-1])
            return last_price, last_ts
        except Exception:
            continue
    return None

def compute_variations(symbols_map: Dict[str, str], lookback: str = "30d", target_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    symbols = list(dict.fromkeys(symbols_map.values()))
    px = yf.download(
        tickers=symbols,
        period=lookback,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if px.empty:
        raise RuntimeError("No se descargaron datos. Revisá conexión o símbolos.")
    close = _normalize_close_dataframe(px, symbols)
    if close.empty:
        raise RuntimeError("No se pudo normalizar los precios de cierre.")
    rows = []
    t: Optional[pd.Timestamp] = None
    if target_date is not None:
        t = pd.to_datetime(target_date).normalize()
    today_utc_date = pd.Timestamp.utcnow().date()
    if t is not None and t.date() == today_utc_date:
        t = None
    for label, sym in symbols_map.items():
        if sym not in close.columns:
            rows.append([label, sym, np.nan, np.nan, np.nan, None])
            continue
        s = close[sym].dropna().copy()
        if len(s) == 0:
            rows.append([label, sym, np.nan, np.nan, np.nan, None])
            continue
        if t is not None:
            try:
                s = s[s.index.normalize() <= t]
            except Exception:
                s = s[s.index.to_series().dt.normalize() <= t]
            if len(s) == 0:
                rows.append([label, sym, np.nan, np.nan, np.nan, None])
                continue
        last_date_daily = pd.to_datetime(s.index[-1])
        last_date_daily_date = last_date_daily.date()
        if last_date_daily_date < today_utc_date and t is None:
            intr = None
            try:
                intr = _fetch_intraday_latest(sym)
            except Exception:
                intr = None
            if intr is not None:
                last_px_intr, last_ts_intr = intr
                if len(s) >= 2:
                    prev_px = float(s.iloc[-1])
                    chg = (last_px_intr / prev_px - 1.0) * 100.0
                    rows.append([label, sym, last_px_intr, prev_px, chg, last_ts_intr.date()])
                    continue
                else:
                    rows.append([label, sym, last_px_intr, np.nan, np.nan, last_ts_intr.date()])
                    continue
        if len(s) < 2:
            last_val = float(s.iloc[-1]) if len(s) else np.nan
            last_date = (s.index[-1].date() if len(s) else None)
            rows.append([label, sym, last_val, np.nan, np.nan, last_date])
            continue
        last_px = float(s.iloc[-1])
        prev_px = float(s.iloc[-2])
        chg = (last_px / prev_px - 1.0) * 100.0
        rows.append([label, sym, last_px, prev_px, chg, pd.to_datetime(s.index[-1]).date()])
    out = pd.DataFrame(
        rows,
        columns=[
            "Ticker",
            "YahooSymbol",
            "Close_last",
            "Close_prev",
            "Var_diaria_%",
            "Fecha_last",
        ],
    )
    if "Var_diaria_%" in out.columns:
        try:
            out["Var_diaria_%"] = out["Var_diaria_%"].astype(float).round(2)
        except Exception:
            pass
    return out, close

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lookback", default="30d", help="yfinance period (e.g. 30d, 3mo)")
    p.add_argument("--out", default="variacion_diaria.csv", help="output csv path")
    p.add_argument("--date", default=None, help="optional target date YYYY-MM-DD")
    args = p.parse_args()
    out_df, close_df = compute_variations(TICKER_MAP, args.lookback, args.date)
    out_df.to_csv(args.out, index=False)
    close_df.to_csv("precios_close.csv")
    print("CSV guardado en", args.out)