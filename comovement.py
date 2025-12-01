# comovement.py
import numpy as np
import pandas as pd

# ==== EDIT THESE ONLY ====
TICKER_A = "KO"
TICKER_B = "PEP"
START_DATE = "2001-01-01"
END_DATE   = "2025-11-01"   # yfinance end is exclusive
ROLLING_WINDOW = 60         # trading days
SAVE_SERIES_CSV = False
# =========================

def fisher_z(series: pd.Series) -> pd.Series:
    r = series.clip(lower=-0.999999, upper=0.999999)
    return 0.5 * np.log((1.0 + r) / (1.0 - r))

def fisher_inv(z: float) -> float:
    return np.tanh(z)

def load_log_returns_yf(ticker: str, start: str, end: str, auto_adjust: bool = True) -> pd.Series:
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("Please install yfinance: pip install yfinance")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=auto_adjust)
    if df.empty:
        raise ValueError(f"No price data for {ticker} in the given range.")
    px = df["Close"].astype("float64")
    r = np.log(px).diff()
    r.name = ticker
    return r

def compute_rolling_corr(ret_a: pd.Series, ret_b: pd.Series, window: int) -> pd.Series:
    # Build a 2-col DataFrame and name the columns safely (no Series.rename())
    df = pd.concat([ret_a, ret_b], axis=1).dropna()
    df.columns = ["a", "b"]
    if df.shape[0] < window:
        raise ValueError(
            f"Not enough overlapping return data to compute a {window}-day rolling correlation "
            f"(overlap rows={df.shape[0]})."
        )
    corr = df["a"].rolling(window).corr(df["b"])
    return corr.dropna()

def main():
    ret_a = load_log_returns_yf(TICKER_A, START_DATE, END_DATE)
    ret_b = load_log_returns_yf(TICKER_B, START_DATE, END_DATE)

    rho_series = compute_rolling_corr(ret_a, ret_b, ROLLING_WINDOW)

    mean_rho = float(rho_series.mean())
    z_series = fisher_z(rho_series)
    mean_z = float(z_series.mean())
    pooled_rho_from_mean_z = float(fisher_inv(mean_z))

    eff_start = rho_series.index.min().date() if not rho_series.empty else None
    eff_end   = rho_series.index.max().date() if not rho_series.empty else None

    print("=" * 78)
    print(f"Rolling co-movement (Pearson corr of daily log returns)")
    print(f"Tickers             : {TICKER_A} vs {TICKER_B}")
    print(f"Requested dates     : {START_DATE} → {END_DATE}")
    if eff_start and eff_end:
        print(f"Window end dates    : {eff_start} → {eff_end}")
    print(f"Rolling window      : {ROLLING_WINDOW} trading days")
    print("-" * 78)
    print(f"Average rolling correlation (mean ρ)             : {mean_rho:.6f}")
    print(f"Average Fisher z of rolling correlation (mean z) : {mean_z:.6f}")
    print(f"Inverse-z of mean z (pooled correlation)         : {pooled_rho_from_mean_z:.6f}")
    print("-" * 78)
    print(f"Observations (rolling windows): {len(rho_series)}")
    print("=" * 78)

    if SAVE_SERIES_CSV:
        out = pd.DataFrame({"rho": rho_series, "z": z_series})
        fname = f"rolling_corr_{TICKER_A}_{TICKER_B}_w{ROLLING_WINDOW}.csv"
        out.to_csv(fname, index_label="date")
        print(f"Saved series to {fname}")

if __name__ == "__main__":
    main()
