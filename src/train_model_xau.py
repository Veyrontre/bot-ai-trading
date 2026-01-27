# train_model_xau.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from joblib import dump

DATA_PATH = os.path.join("data", "XAUUSD_enhanced_ml_dataset_clean.csv")
MODEL_PATH = os.path.join("models", "xau_ml_target1d.pkl")

def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Pastikan Date jadi datetime dan urut
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def prepare_features_target1d(df: pd.DataFrame):
    """
    Kita pakai:
    - Fitur utama harga/volume
    - Beberapa fitur teknikal & makro yang masuk akal
    - Target: Target_1d (diasumsikan 0/1)
    """
    df = df.copy()

    # Buang baris yang tidak punya label
    df = df.dropna(subset=["Target_1d"])

    # Pilih subset fitur yang relevan (bisa ditambah sesuka hati)
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "volatility_atr", "volatility_bbw", "volatility_dcp", "volatility_kcp",
        "trend_macd", "trend_macd_signal", "trend_macd_diff",
        "trend_sma_fast", "trend_sma_slow",
        "trend_ema_fast", "trend_ema_slow",
        "trend_adx", "trend_adx_pos", "trend_adx_neg",
        "trend_aroon_up", "trend_aroon_down", "trend_aroon_ind",
        "momentum_rsi", "momentum_wr", "momentum_roc",
        "Price_Change", "Price_Change_5d", "Price_Change_20d",
        "Volatility_5d", "Volatility_20d",
        "Rolling_Mean_5", "Rolling_Std_5",
        "Rolling_Mean_20", "Rolling_Std_20",
        "Momentum_1M", "Momentum_3M", "Momentum_6M",
        "Realized_Vol_5d", "Realized_Vol_20d",
        "DXY", "DXY_Change",
        "US_10Y_Yield", "US_10Y_Change",
        "WTI_Oil", "Oil_Change",
        "Silver", "Gold_Silver_Ratio",
        "BTC_Price", "Gold_BTC_Ratio",
        "Day_of_Week", "Month", "Quarter",
    ]

    # Filter yang benar‑benar ada di CSV
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["Target_1d"].astype(int)  # pastikan integer 0/1

    # Drop baris yang masih ada NaN di fitur
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    return X, y, feature_cols

def train_model():
    os.makedirs("models", exist_ok=True)

    df = load_dataset()
    print("Dataset shape:", df.shape)

    X, y, feature_cols = prepare_features_target1d(df)
    print("Feature cols used:", len(feature_cols))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

    # Simpan model + daftar fitur yang dipakai
    dump(
        {"model": model, "features": feature_cols},
        MODEL_PATH,
    )
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
