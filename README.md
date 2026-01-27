🏆 XAUUSD AI Trading System
https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white
https://img.shields.io/badge/MetaTrader5-00BFFF?style=for-the-badge&logo=metatrader&logoColor=white
https://img.shields.io/badge/Ollama-7C3AED?style=for-the-badge&logo=ollama&logoColor=white
https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white

Sistem Trading AI Canggih untuk XAUUSD (Emas) yang menggabungkan analisis teknikal, fundamental, berita, dan kecerdasan buatan untuk memberikan rekomendasi trading yang akurat.

https://via.placeholder.com/800x400/2D2D2D/FFFFFF?text=XAUUSD+AI+Trading+System+Dashboard

✨ Fitur Utama
🤖 Analisis AI Multi-Modal
Visual Chart Analysis: Analisis gambar chart dengan model LLava 13B

Multi-Timeframe Analysis: Analisis simultan M15, M30, dan H1

Fundamental Analysis: Integrasi kalender ekonomi dan berita real-time

Teknikal Analysis: 20+ indikator teknikal otomatis

📊 Data Real-Time
Live MT5 Connection: Data langsung dari broker MetaTrader 5

Economic Calendar: Event ekonomi penting dari TradingEconomics

News Aggregator: Berita terkini dari NewsAPI

Technical Indicators: SMA, EMA, RSI, ATR, Fibonacci, Support/Resistance

🎯 Trading Intelligence
AI-Powered Signals: Rekomendasi BUY/SELL/HOLD berbasis AI

Risk Management: Auto-calculate Stop Loss & Take Profit

Multi-Timeframe Confirmation: Konfirmasi sinyal dari 3 timeframe

Confidence Scoring: Skor kepercayaan untuk setiap rekomendasi

🚀 Quick Start
Prerequisites
1. Python 3.9+

2. MetaTrader 5 (terinstall dan login)

3. Ollama (untuk AI lokal)

4. Broker Account (untuk data real-time)

Installation
bash
# 1. Clone repository
git clone https://github.com/yourusername/xauusd-ai-trading.git
cd xauusd-ai-trading

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install Ollama (jika belum)
# Download dari https://ollama.ai/

# 6. Download AI models
ollama pull llava:13b
ollama pull tinyllama:latest

# 7. Configure API keys
# Salin .env.example ke .env dan isi API keys
cp .env.example .env