# comovement.py
# -----------------------------------------------------------------------------
# Enrich a period CSV (e.g., '2019Q3' in first/period column) with KO–PEP
# co-movement numbers. Robust price loader with Yahoo retries + Stooq fallback,
# optional on-disk caching, rolling correlation, quarterly aggregation, and merge.
#
# One-time installs:
#   pip install yfinance pandas-datareader pandas numpy
# -----------------------------------------------------------------------------

from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd

# ======================== EDIT THESE ONLY =========================
CSV_IN  = "ko_pep_sim_by_period.csv"              # input with quarter labels like 2019Q3
CSV_OUT = "ko_pep_sim_by_period_with_corr.csv"    # output file

TICKER_A = "KO"
TICKER_B = "PEP"

START_DATE = "2001-01-01"   # wide range to cover your earliest quarter + rolling window warmup
END_DATE   = "2025-12-31"   # note: yfinance 'end' is exclusive; set a bit past your last quarter

ROLLING_WINDOW = 60         # trading days (e.g., 60 ≈ ~3 months)

# Optional lightweight cache to reduce rate limits (stores daily Close to CSV)
USE_CACHE       = True
PRICE_CACHE_DIR = "price_cache"   # folder will be created if missing

# Yahoo retry policy
MAX_RETRIES       = 5
BACKOFF_BASE_SEC  = 1.0
YF_AUTO_ADJUST    = True
# =================================================================


# ------------------------- Math helpers ---------------------------
def fisher_z(series: pd.Series) -> pd.Series:
    r = series.clip(lower=-0.999999, upper=0.999999)
    return 0.5 * np.log((1.0 + r) / (1.0 - r))

def fisher_inv(z: float | np.ndarray) -> float | np.ndarray:
    return np.tanh(z)


# -------------------- Robust price loaders ------------------------
def _fetch_yahoo_close(ticker: str, start: str, end: str) -> pd.Series:
    import yfinance as yf
    last_err = None
    for k in range(MAX_RETRIES):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=YF_AUTO_ADJUST,
                threads=False,
                interval="1d",
            )
            if not df.empty:
                s = df["Close"].astype("float64")
                s.name = ticker
                return s
            last_err = RuntimeError("Empty dataframe from Yahoo.")
        except Exception as e:
            last_err = e
        time.sleep(BACKOFF_BASE_SEC * (2 ** k))
    raise RuntimeError(f"Yahoo failed after retries. Last error: {last_err}")

def _fetch_stooq_close(ticker: str, start: str, end: str) -> pd.Series:
    # Stooq uses '.US' suffix for US tickers
    from pandas_datareader import data as pdr
    sym = f"{ticker}.US"
    df = pdr.DataReader(sym, "stooq", start=start, end=end)
    df = df.sort_index()
    if df.empty:
        raise RuntimeError("Empty dataframe from Stooq.")
    s = df["Close"].astype("float64")
    s.name = ticker
    return s

def _load_close_with_cache(ticker: str, start: str, end: str) -> pd.Series:
    cache_dir = Path(PRICE_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ticker}_{start}_{end}.csv"

    if USE_CACHE and cache_file.exists():
        s = pd.read_csv(cache_file, parse_dates=["date"])
        s = s.set_index("date")["Close"].astype("float64")
        s.name = ticker
        return s

    # Try Yahoo; fallback to Stooq
    try:
        s = _fetch_yahoo_close(ticker, start, end)
    except Exception:
        s = _fetch_stooq_close(ticker, start, end)

    if USE_CACHE:
        pd.DataFrame({"date": s.index, "Close": s.values}).to_csv(cache_file, index=False)
    return s

def load_log_returns(ticker: str, start: str, end: str) -> pd.Series:
    px = _load_close_with_cache(ticker, start, end)
    r = np.log(px).diff()
    r.name = ticker
    return r


# ------------------- Rolling corr & aggregation -------------------
def compute_rolling_corr(ret_a: pd.Series, ret_b: pd.Series, window: int) -> pd.Series:
    df = pd.concat([ret_a, ret_b], axis=1).dropna()
    df.columns = ["a", "b"]
    if df.shape[0] < window:
        raise ValueError(
            f"Not enough overlapping data for a {window}-day window (overlap rows={df.shape[0]})."
        )
    rho = df["a"].rolling(window).corr(df["b"])
    return rho.dropna()

def to_quarter_label(dts: pd.DatetimeIndex) -> pd.Series:
    # Map rolling-window END dates to 'YYYYQ#' (e.g., '2019Q3')
    return pd.PeriodIndex(dts, freq="Q").astype(str)

def detect_period_column(df: pd.DataFrame) -> str:
    candidates = ["period", "Period", "PERIOD", "quarter", "Quarter", "QUARTER"]
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[0]  # fallback: first column


# ------------------------------- Main -----------------------------
def main():
    # 1) Read input periods
    in_path = Path(CSV_IN)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path.resolve()}")
    base = pd.read_csv(in_path)
    period_col = detect_period_column(base)
    base[period_col] = base[period_col].astype(str)

    # 2) Load returns once (robust)
    ret_a = load_log_returns(TICKER_A, START_DATE, END_DATE)
    ret_b = load_log_returns(TICKER_B, START_DATE, END_DATE)

    # 3) Rolling correlation over full span
    rho_series = compute_rolling_corr(ret_a, ret_b, ROLLING_WINDOW)
    z_series = fisher_z(rho_series)

    # 4) Assign rolling END dates to quarters and aggregate
    qlab = to_quarter_label(rho_series.index)
    agg = pd.DataFrame({"quarter": qlab, "rho": rho_series.values, "z": z_series.values})
    grouped = (
        agg.groupby("quarter")
           .agg(rho_mean=("rho", "mean"),
                z_mean=("z", "mean"),
                n_windows=("rho", "size"))
           .reset_index()
    )
    grouped["rho_from_mean_z"] = grouped["z_mean"].apply(lambda z: float(fisher_inv(z)))

    # 5) Merge back into your CSV
    out = base.merge(grouped, left_on=period_col, right_on="quarter", how="left")
    out["co_mov_tickers"] = f"{TICKER_A}-{TICKER_B}"
    out["rolling_window_days"] = ROLLING_WINDOW

    # 6) Save
    out_path = Path(CSV_OUT)
    out.to_csv(out_path, index=False)

    # 7) Short printout
    eff_start = rho_series.index.min().date() if not rho_series.empty else None
    eff_end   = rho_series.index.max().date() if not rho_series.empty else None

    print("=" * 78)
    print(f"Co-movement (rolling Pearson corr of daily log returns)")
    print(f"Tickers              : {TICKER_A} vs {TICKER_B}")
    print(f"Price range fetched  : {START_DATE} → {END_DATE}")
    if eff_start and eff_end:
        print(f"Window end dates     : {eff_start} → {eff_end}")
    print(f"Rolling window       : {ROLLING_WINDOW} trading days")
    print(f"Input CSV            : {in_path.resolve()}")
    print(f"Output CSV           : {out_path.resolve()}")
    print("-" * 78)
    if not grouped.empty:
        print("Sample aggregated quarters:")
        print(grouped.head(5).to_string(index=False))
    else:
        print("No aggregated quarters produced. Check date range and rolling window.")
    print("=" * 78)


if __name__ == "__main__":
    main()
