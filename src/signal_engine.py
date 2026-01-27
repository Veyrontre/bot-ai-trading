#signal_engine.py
from joblib import load
import os
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join("models", "xau_ml_target1d.pkl")

def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model {MODEL_PATH} tidak ditemukan. Jalankan dulu src/train_model_xau.py"
        )
    data = load(MODEL_PATH)
    return data["model"], data["features"]

def ml_predict_signal(df_with_ind: pd.DataFrame, min_confidence: float = 0.55):
    """
    Prediksi dari model ML.
    df_with_ind: DataFrame dengan indikator teknikal sudah dihitung
    min_confidence: threshold minimum untuk signal (default 0.55)
    """
    try:
        model, feature_names = load_trained_model()
    except FileNotFoundError as e:
        return {
            "signal": "HOLD",
            "prob_up": None,
            "prob_down": None,
            "confidence": 0.0,
            "reason": str(e),
        }
    
    # Ambil baris terakhir
    latest = df_with_ind.iloc[-1:].copy()
    
    # Map kolom dari live data ke feature names yang dipakai saat training
    # Untuk sementara kita ambil yang available saja
    available_features = {}
    for feat in feature_names:
        if feat in latest.columns:
            available_features[feat] = latest[feat].values[0]
        else:
            # Fitur tidak ada, isi dengan 0 atau NaN (ini bisa bikin prediksi tidak akurat)
            available_features[feat] = 0.0
    
    x = pd.DataFrame([available_features])[feature_names]
    
    # Prediksi
    prob = model.predict_proba(x)[0]
    prob_down, prob_up = prob[0], prob[1]
    
    if prob_up > min_confidence:
        signal = "LONG"
        confidence = prob_up
        reason = f"Model ML prediksi NAIK dengan confidence {prob_up:.2%}"
    elif prob_down > min_confidence:
        signal = "SHORT"
        confidence = prob_down
        reason = f"Model ML prediksi TURUN dengan confidence {prob_down:.2%}"
    else:
        signal = "HOLD"
        confidence = max(prob_up, prob_down)
        reason = f"Confidence ML terlalu rendah (up:{prob_up:.2%}, down:{prob_down:.2%})"
    
    return {
        "signal": signal,
        "prob_up": float(prob_up),
        "prob_down": float(prob_down),
        "confidence": float(confidence),
        "reason": reason,
    }

def build_trade_levels(close: float, atr: float, signal: str, atr_mult_sl: float = 1.5, rr: float = 2.0):
    """
    Hitung Entry, SL, TP berdasarkan sinyal dan ATR
    """
    if signal == "LONG":
        entry = close
        sl = entry - (atr * atr_mult_sl)
        tp = entry + ((entry - sl) * rr)
    elif signal == "SHORT":
        entry = close
        sl = entry + (atr * atr_mult_sl)
        tp = entry - ((sl - entry) * rr)
    else:
        entry = sl = tp = None
    
    return entry, sl, tp
