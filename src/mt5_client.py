#mt5_client.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from typing import Optional

DEFAULT_SYMBOL = "XAUUSD"

TIMEFRAME_MAP = {
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
}

class MT5ConnectionError(Exception):
    pass

class MT5DataError(Exception):
    pass

def initialize_mt5() -> None:
    """
    Inisialisasi koneksi ke MT5 terminal yang sudah login.
    User harus sudah buka MT5 + login ke broker secara manual.
    """
    if not mt5.initialize():
        err = mt5.last_error()
        raise MT5ConnectionError(f"Gagal initialize MT5: {err}")

    terminal_info = mt5.terminal_info()
    if terminal_info is None:
        raise MT5ConnectionError("Tidak bisa membaca terminal_info MT5.")
    if not terminal_info.connected:
        raise MT5ConnectionError("MT5 terminal belum connect ke broker (login dulu di MT5).")

def ensure_symbol(symbol: str = DEFAULT_SYMBOL) -> None:
    """
    Memastikan symbol tersedia di MarketWatch.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        raise MT5DataError(f"Symbol {symbol} tidak ditemukan di MT5.")
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            raise MT5DataError(f"Gagal menampilkan symbol {symbol} di MarketWatch.")

def get_rates(
    symbol: str = DEFAULT_SYMBOL,
    timeframe_str: str = "M15",
    count: int = 300
) -> pd.DataFrame:
    """
    Ambil OHLCV terakhir dari MT5, return DataFrame.
    timeframe_str: "M15" atau "H1"
    """
    if timeframe_str not in TIMEFRAME_MAP:
        raise ValueError("timeframe_str harus 'M15' atau 'H1'")

    timeframe = TIMEFRAME_MAP[timeframe_str]

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        raise MT5DataError("Data candle kosong (mungkin market closed / symbol salah / koneksi masalah).")

    df = pd.DataFrame(rates)
    # time dalam detik epoch -> datetime
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "tick_volume": "tick_volume",
            "real_volume": "real_volume",
            "spread": "spread",
        }
    )
    # sort berdasarkan waktu asc
    df = df.sort_values("time").reset_index(drop=True)
    return df

def get_tick_info(symbol: str = DEFAULT_SYMBOL) -> Optional[dict]:
    """
    Ambil tick terakhir (Bid/Ask, spread, dll).
    """
    ensure_symbol(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    return {
        "time": datetime.fromtimestamp(tick.time),
        "bid": tick.bid,
        "ask": tick.ask,
        "last": tick.last,
        "volume": tick.volume,
        "flags": tick.flags,
    }

def get_spread_point(symbol: str = DEFAULT_SYMBOL) -> Optional[float]:
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    return info.spread

def shutdown_mt5() -> None:
    try:
        mt5.shutdown()
    except Exception:
        pass

if __name__ == "__main__":
    # Simple test manual
    try:
        initialize_mt5()
        ensure_symbol(DEFAULT_SYMBOL)
        df = get_rates(DEFAULT_SYMBOL, "M15", 10)
        print(df.tail())
        print(get_tick_info(DEFAULT_SYMBOL))
    except Exception as e:
        print("Error:", e)
    finally:
        shutdown_mt5()
