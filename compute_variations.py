import argparse
from typing import Tuple, Dict, Optional, List
import time
import pandas as pd
import numpy as np
import yfinance as yf
from collections import Counter

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


def _download_single_ticker(symbol: str, lookback: str = "30d", max_retries: int = 3) -> Optional[pd.DataFrame]:
    """
    Descarga datos de un ticker individual con reintentos.
    Retorna DataFrame con columna 'Close' o None si falla.
    """
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(0.5 * attempt)  # backoff entre reintentos
            
            px = yf.download(
                tickers=symbol,
                period=lookback,
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="column",
            )
            
            if px is None or px.empty:
                continue
                
            # Extraer columna Close
            if isinstance(px.columns, pd.MultiIndex):
                if "Close" in px.columns.get_level_values(0):
                    close_data = px["Close"]
                    if isinstance(close_data, pd.DataFrame):
                        close_series = close_data.iloc[:, 0].dropna()
                    else:
                        close_series = close_data.dropna()
                else:
                    continue
            else:
                if "Close" in px.columns:
                    close_series = px["Close"].dropna()
                else:
                    continue
            
            if len(close_series) >= 2:
                df = pd.DataFrame({symbol: close_series})
                df.index = pd.to_datetime(df.index)
                return df
                
        except Exception as e:
            print(f"Intento {attempt+1}/{max_retries} fallido para {symbol}: {e}")
            continue
    
    return None


def _download_with_fallback(symbols_map: Dict[str, str], lookback: str = "30d") -> Tuple[pd.DataFrame, List[str]]:
    """
    Descarga datos: primero intenta en batch, luego individualmente para los que fallan.
    Retorna (DataFrame con closes, lista de símbolos que fallaron definitivamente).
    """
    symbols = list(dict.fromkeys(symbols_map.values()))
    failed_symbols = []
    
    # Primer intento: descarga en batch
    px = yf.download(
        tickers=symbols,
        period=lookback,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    
    close = _normalize_close_dataframe(px, symbols)
    
    # Identificar símbolos que fallaron (no están en close o tienen solo NaN)
    missing_symbols = []
    for sym in symbols:
        if sym not in close.columns:
            missing_symbols.append(sym)
        elif close[sym].dropna().empty:
            missing_symbols.append(sym)
    
    # Reintentar individualmente los que fallaron
    for sym in missing_symbols:
        print(f"Reintentando descarga individual de {sym}...")
        single_df = _download_single_ticker(sym, lookback, max_retries=3)
        if single_df is not None and sym in single_df.columns:
            # Agregar o reemplazar en close
            if close.empty:
                close = single_df
            else:
                close[sym] = single_df[sym]
            print(f"✓ {sym} descargado exitosamente en reintento")
        else:
            failed_symbols.append(sym)
            print(f"✗ {sym} falló definitivamente después de reintentos")
    
    return close, failed_symbols

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

def compute_variations(symbols_map: Dict[str, str], lookback: str = "30d", target_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str], List[str]]:
    """
    Calcula variaciones diarias de los tickers.
    Basado en el script original de Colab que funciona correctamente.
    
    Retorna:
        - out: DataFrame con las variaciones
        - close: DataFrame con precios de cierre
        - data_date_mode: La moda de las fechas en los datos (YYYY-MM-DD) o None
        - failed_tickers: Lista de tickers que fallaron definitivamente
    """
    # Usar el nuevo sistema de descarga con fallback
    close, failed_symbols = _download_with_fallback(symbols_map, lookback)
    
    if close.empty:
        raise RuntimeError("No se descargaron datos. Revisá conexión o símbolos.")
    
    # Filtrar por target_date si se especificó
    if target_date is not None:
        t = pd.to_datetime(target_date).normalize()
        # Filtrar el DataFrame close para incluir solo fechas <= target_date
        close = close[close.index.normalize() <= t].copy()
        if close.empty:
            raise RuntimeError(f"No hay datos disponibles para la fecha {target_date} o anterior.")
    
    # Para calcular la moda de las fechas
    all_dates = []
    rows = []
    
    # Lógica simple y directa como en el script original de Colab
    for label, sym in symbols_map.items():
        if sym not in close.columns:
            rows.append([label, sym, np.nan, np.nan, np.nan, None, "ERROR_DESCARGA"])
            continue
        
        s = close[sym].dropna()
        
        if len(s) < 2:
            last_val = float(s.iloc[-1]) if len(s) else np.nan
            last_date = s.index[-1].date() if len(s) else None
            rows.append([label, sym, last_val, np.nan, np.nan, last_date, "PARCIAL"])
            if last_date:
                all_dates.append(last_date)
            continue
        
        # Cálculo simple: último precio / precio anterior - 1
        last_date = s.index[-1]
        last_px = float(s.iloc[-1])
        prev_px = float(s.iloc[-2])
        chg = (last_px / prev_px - 1.0) * 100.0
        
        all_dates.append(last_date.date())
        rows.append([label, sym, last_px, prev_px, chg, last_date.date(), "OK"])
    
    out = pd.DataFrame(
        rows,
        columns=[
            "Ticker",
            "YahooSymbol",
            "Close_last",
            "Close_prev",
            "Var_diaria_%",
            "Fecha_last",
            "Status",
        ],
    )
    
    if "Var_diaria_%" in out.columns:
        try:
            out["Var_diaria_%"] = out["Var_diaria_%"].astype(float).round(2)
        except Exception:
            pass
    
    # Calcular la moda de las fechas
    data_date_mode = None
    if all_dates:
        date_counts = Counter(all_dates)
        data_date_mode = date_counts.most_common(1)[0][0].strftime("%Y-%m-%d")
    
    # Crear lista de tickers que fallaron (para el label, no el símbolo)
    failed_tickers = [label for label, sym in symbols_map.items() if sym in failed_symbols]
    
    return out, close, data_date_mode, failed_tickers

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lookback", default="30d", help="yfinance period (e.g. 30d, 3mo)")
    p.add_argument("--out", default="variacion_diaria.csv", help="output csv path")
    p.add_argument("--date", default=None, help="optional target date YYYY-MM-DD")
    args = p.parse_args()
    out_df, close_df, data_date_mode, failed_tickers = compute_variations(TICKER_MAP, args.lookback, args.date)
    out_df.to_csv(args.out, index=False)
    close_df.to_csv("precios_close.csv")
    print("CSV guardado en", args.out)
    print("Moda de fechas de los datos:", data_date_mode)
    if failed_tickers:
        print("Tickers que fallaron:", ", ".join(failed_tickers))