#indicators.py
import numpy as np
import pandas as pd

def sma(series: pd.Series, period: int = 14) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()

def ema(series: pd.Series, period: int = 14) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain_rolling = pd.Series(gain).rolling(window=period, min_periods=period).mean()
    loss_rolling = pd.Series(loss).rolling(window=period, min_periods=period).mean()

    rs = gain_rolling / loss_rolling
    rsi_series = 100 - (100 / (1 + rs))
    rsi_series.index = series.index
    return rsi_series

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    df harus punya kolom: 'high','low','close'
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.rolling(window=period, min_periods=period).mean()
    return atr_series

def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    return series - series.shift(period)

def volatility(series: pd.Series, period: int = 20) -> pd.Series:
    """
    Volatilitas sederhana: std dev return log.
    """
    returns = np.log(series / series.shift(1))
    return returns.rolling(window=period, min_periods=period).std()

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan semua indikator ke DataFrame OHLC, asumsi kolom df:
    time, open, high, low, close, (optional volume)
    """
    out = df.copy()

    out["sma_20"] = sma(out["close"], 20)
    out["ema_20"] = ema(out["close"], 20)
    out["rsi_14"] = rsi(out["close"], 14)
    out["atr_14"] = atr(out, 14)
    out["mom_10"] = momentum(out["close"], 10)
    out["vol_20"] = volatility(out["close"], 20)

    return out

if __name__ == "__main__":
    # simple test
    data = {
        "close": np.linspace(2000, 2100, 100),
        "high": np.linspace(2005, 2105, 100),
        "low": np.linspace(1995, 2095, 100),
    }
    df = pd.DataFrame(data)
    df_ind = add_all_indicators(df)
    print(df_ind.tail())
