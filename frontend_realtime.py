# frontend_realtime_enhanced.py
import os
import base64
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import tempfile

from src.mt5_client import (
    initialize_mt5,
    ensure_symbol,
    get_rates,
    get_tick_info,
    get_spread_point,
    shutdown_mt5,
    DEFAULT_SYMBOL,
)
from src.indicators import add_all_indicators
from economic_calendar import EconCalendarLoader
from news import NewsLoader
import ollama
import asyncio


# === CONFIGURATION ===
class Config:
    TIMEFRAMES = ["M15", "M30", "H1"]
    CHART_PERIODS = {
        "M15": 100,    # 100 candles for M15
        "M30": 80,     # 80 candles for M30
        "H1": 60       # 60 candles for H1
    }
    SUPPORT_RESISTANCE_PERIODS = 20
    FIBONACCI_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]


# === CHART UTILITIES ===
def add_fibonacci_levels(fig, df, timeframe):
    """Tambahkan level Fibonacci ke chart"""
    if len(df) < 10:
        return fig
    
    # Identifikasi swing high dan swing low
    period = 5
    df['swing_high'] = df['high'].rolling(window=period, center=True).max()
    df['swing_low'] = df['low'].rolling(window=period, center=True).min()
    
    recent_data = df.tail(50)
    
    if len(recent_data) > 0:
        swing_high = recent_data['swing_high'].max()
        swing_low = recent_data['swing_low'].min()
        
        diff = swing_high - swing_low
        
        for level in Config.FIBONACCI_LEVELS:
            price_level = swing_high - (diff * level)
            
            fig.add_hline(
                y=price_level,
                line_dash="dash",
                line_color="purple",
                opacity=0.6,
                annotation_text=f"Fibo {level*100:.1f}%",
                annotation_position="right"
            )
    
    return fig


def add_support_resistance(fig, df, period=20):
    """Identifikasi support dan resistance levels"""
    if len(df) < period:
        return fig
    
    # Sederhana: harga yang sering disentuh
    recent_data = df.tail(100)
    
    # Support: harga low yang sering diuji
    support_levels = []
    resistance_levels = []
    
    # Cari area congestion
    for i in range(0, len(recent_data) - period, 5):
        window = recent_data.iloc[i:i+period]
        
        # Support candidate
        support_candidate = window['low'].min()
        # Resistance candidate
        resistance_candidate = window['high'].max()
        
        # Cek jika harga kembali ke level ini beberapa kali
        support_touches = np.sum(np.abs(window['low'] - support_candidate) < (window['low'].mean() * 0.001))
        resistance_touches = np.sum(np.abs(window['high'] - resistance_candidate) < (window['high'].mean() * 0.001))
        
        if support_touches > 2:
            support_levels.append(support_candidate)
        if resistance_touches > 2:
            resistance_levels.append(resistance_candidate)
    
    # Tambahkan level yang unik
    for level in set(support_levels[-5:]):  # Ambil 5 terakhir
        fig.add_hline(
            y=level,
            line_dash="dash",
            line_color="green",
            opacity=0.4,
            annotation_text=f"S {level:.2f}",
            annotation_position="right"
        )
    
    for level in set(resistance_levels[-5:]):
        fig.add_hline(
            y=level,
            line_dash="dash",
            line_color="red",
            opacity=0.4,
            annotation_text=f"R {level:.2f}",
            annotation_position="right"
        )
    
    return fig


def create_multi_timeframe_chart(df_m15, df_m30, df_h1):
    """Buat chart dengan 3 timeframe"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('H1 Chart', 'M30 Chart', 'M15 Chart'),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # H1 Chart
    fig.add_trace(
        go.Candlestick(
            x=df_h1['time'],
            open=df_h1['open'],
            high=df_h1['high'],
            low=df_h1['low'],
            close=df_h1['close'],
            name='H1'
        ),
        row=1, col=1
    )
    
    # M30 Chart
    fig.add_trace(
        go.Candlestick(
            x=df_m30['time'],
            open=df_m30['open'],
            high=df_m30['high'],
            low=df_m30['low'],
            close=df_m30['close'],
            name='M30'
        ),
        row=2, col=1
    )
    
    # M15 Chart
    fig.add_trace(
        go.Candlestick(
            x=df_m15['time'],
            open=df_m15['open'],
            high=df_m15['high'],
            low=df_m15['low'],
            close=df_m15['close'],
            name='M15'
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Multi Timeframe XAUUSD Analysis',
        height=900,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        xaxis3_rangeslider_visible=False,
        template="plotly_dark"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=3, col=1)
    
    return fig


def save_chart_as_image(fig, timeframe="H1"):
    """Simpan chart sebagai gambar untuk analisis AI"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        pio.write_image(fig, tmp.name, width=1400, height=800)
        return tmp.name


def encode_image_to_base64(image_path):
    """Encode gambar ke base64 untuk dikirim ke Ollama"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# === DATA LOADER ===
class MT5DataLoader:
    def __init__(self):
        self.symbol = DEFAULT_SYMBOL
    
    def load_all_timeframes(self):
        """Load data untuk semua timeframe"""
        try:
            initialize_mt5()
            ensure_symbol(self.symbol)
            
            data = {}
            for tf in Config.TIMEFRAMES:
                count = Config.CHART_PERIODS.get(tf, 100)
                df = get_rates(self.symbol, tf, count)
                df = add_all_indicators(df)
                data[tf] = df
            
            tick_info = get_tick_info(self.symbol)
            spread_point = get_spread_point(self.symbol)
            
            shutdown_mt5()
            
            return {
                'data': data,
                'tick_info': tick_info,
                'spread_point': spread_point,
                'timestamp': datetime.now()
            }
        except Exception as e:
            shutdown_mt5()
            raise e


# === AI ANALYZER ===
class AIChartAnalyzer:
    def __init__(self, model_name="llava:13b"):
        self.model_name = model_name
    
    def analyze_chart_image(self, image_path, context_text=""):
        """Analisis chart menggunakan model llava"""
        try:
            # Encode gambar ke base64
            image_base64 = encode_image_to_base64(image_path)
            
            # Buat prompt yang komprehensif
            prompt = f"""ANALISIS CHART TRADING XAUUSD (GOLD)

Anda adalah analis trading profesional dengan keahlian dalam:
1. Analisis Teknikal (pattern, trend, support/resistance)
2. Analisis Fibonacci (retracement, extension)
3. Analisis Momentum (RSI, Volume)
4. Analisis Multi-Timeframe

{context_text}

INSTUKSI ANALISIS:
1. Identifikasi pola chart utama (Head & Shoulders, Double Top/Bottom, Triangle, etc.)
2. Analisis trend (Uptrend/Downtrend/Sideways) di setiap timeframe
3. Identifikasi level Support dan Resistance kunci
4. Analisis level Fibonacci (jika terlihat di chart)
5. Konfirmasi momentum (RSI, Volume pattern)
6. Berikan rekomendasi konkret: BUY / SELL / HOLD
7. Tentukan Target Profit dan Stop Loss level

JAWAB DALAM FORMAT:
📊 **TREND ANALYSIS**: [analisis trend multi-timeframe]
🎯 **KEY LEVELS**: [support/resistance utama]
📈 **PATTERN RECOGNITION**: [pola yang teridentifikasi]
⚡ **MOMENTUM**: [kekuatan trend]
💰 **RECOMMENDATION**: [BUY/SELL/HOLD dengan confidence level]
🛡️ **RISK MANAGEMENT**: [stop loss dan take profit suggestion]

Analisis chart di atas dengan detail dan berikan rekomendasi trading yang actionable."""
            
            # Kirim ke Ollama dengan gambar
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                images=[image_base64],
                options={
                    "temperature": 0.7,
                    "num_predict": 800
                }
            )
            
            return response.get('response', '')
            
        except Exception as e:
            return f"Error in AI analysis: {str(e)}"
    
    def create_comprehensive_analysis(self, chart_image_path, market_data, news_data, econ_data):
        """Buat analisis komprehensif menggabungkan semua data"""
        
        # Prepare context text
        context = f"""
DATA PASAR SAAT INI:
- Symbol: XAUUSD
- Price: {market_data.get('current_price', 'N/A')}
- Timeframe Analisis: M15, M30, H1
- Market Time: {market_data.get('timestamp', datetime.now())}

DATA TEKNIKAL:
{market_data.get('technical_summary', '')}

KALENDER EKONOMI (hari ini):
{econ_data}

BERITA TERKINI:
{news_data}
"""
        
        return self.analyze_chart_image(chart_image_path, context)


# === STREAMLIT APP ===
def main():
    st.set_page_config(
        page_title="XAUUSD AI Trading System",
        layout="wide",
        page_icon="🏆"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FFD700;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #FFA500;
        margin-top: 1rem;
    }
    .ai-response {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FFD700;
        margin: 10px 0;
    }
    .metric-box {
        background-color: #2D2D2D;
        padding: 15px;
        border-radius: 8px;
        margin: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">XAUUSD AI Trading System</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'market_data' not in st.session_state:
        st.session_state.market_data = None
    if 'ai_analysis' not in st.session_state:
        st.session_state.ai_analysis = None
    if 'chart_image_path' not in st.session_state:
        st.session_state.chart_image_path = None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        ai_model = st.selectbox(
            "AI Model",
            ["llava:13b"],
            index=0
        )
        
        # Chart type
        chart_type = st.radio(
            "Chart Type",
            ["Single Timeframe", "Multi Timeframe"],
            index=0
        )
        
        if chart_type == "Single Timeframe":
            selected_tf = st.selectbox(
                "Timeframe",
                Config.TIMEFRAMES,
                index=2  # Default H1
            )
        else:
            selected_tf = "Multi"
        
        # Analysis depth
        analysis_depth = st.slider(
            "Analysis Depth",
            min_value=1,
            max_value=5,
            value=3,
            help="1: Basic, 5: Very Detailed"
        )
        
        # Include data sources
        st.subheader("📊 Data Sources")
        include_news = st.checkbox("Include News", value=True)
        include_econ = st.checkbox("Include Economic Calendar", value=True)
        include_technical = st.checkbox("Include Technical Indicators", value=True)
        
        st.markdown("---")
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Load Market Data", type="primary", use_container_width=True):
                with st.spinner("Loading market data..."):
                    try:
                        loader = MT5DataLoader()
                        st.session_state.market_data = loader.load_all_timeframes()
                        st.success("Data loaded successfully!")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with col2:
            if st.button("🗑️ Clear Analysis", use_container_width=True):
                st.session_state.ai_analysis = None
                st.session_state.chart_image_path = None
                st.rerun()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📈 Chart Analysis", "📊 Market Data", "🤖 AI Analysis"])
    
    with tab1:
        if st.session_state.market_data:
            data = st.session_state.market_data['data']
            tick_info = st.session_state.market_data['tick_info']
            
            # Display current price
            if tick_info:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("BID", f"{tick_info['bid']:.2f}")
                with col2:
                    st.metric("ASK", f"{tick_info['ask']:.2f}")
                with col3:
                    st.metric("SPREAD", f"{st.session_state.market_data['spread_point']} pips")
            
            # Create chart based on selection
            if chart_type == "Single Timeframe" and selected_tf in data:
                df = data[selected_tf]
                
                # Create chart with indicators
                fig = go.Figure(data=[
                    go.Candlestick(
                        x=df['time'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name=selected_tf
                    )
                ])
                
                # Add indicators if selected
                if include_technical:
                    # Add SMA
                    if 'sma_20' in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df['time'],
                            y=df['sma_20'],
                            mode='lines',
                            name='SMA 20',
                            line=dict(color='orange', width=1)
                        ))
                    
                    # Add EMA
                    if 'ema_20' in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df['time'],
                            y=df['ema_20'],
                            mode='lines',
                            name='EMA 20',
                            line=dict(color='cyan', width=1, dash='dash')
                        ))
                
                # Add Fibonacci and Support/Resistance
                if analysis_depth >= 4:
                    fig = add_fibonacci_levels(fig, df, selected_tf)
                    fig = add_support_resistance(fig, df)
                
                # Update layout
                fig.update_layout(
                    title=f'XAUUSD {selected_tf} Chart',
                    xaxis_rangeslider_visible=False,
                    template="plotly_dark",
                    height=600,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Save chart for AI analysis
                chart_path = save_chart_as_image(fig, selected_tf)
                st.session_state.chart_image_path = chart_path
                
            elif chart_type == "Multi Timeframe":
                # Create multi-timeframe chart
                if all(tf in data for tf in Config.TIMEFRAMES):
                    fig = create_multi_timeframe_chart(
                        data['M15'],
                        data['M30'],
                        data['H1']
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save chart for AI analysis
                    chart_path = save_chart_as_image(fig, "Multi")
                    st.session_state.chart_image_path = chart_path
    
    with tab2:
        if st.session_state.market_data:
            data = st.session_state.market_data['data']
            
            # Show data for each timeframe
            for tf in Config.TIMEFRAMES:
                if tf in data:
                    with st.expander(f"{tf} Data (Last 10 candles)"):
                        df = data[tf]
                        st.dataframe(
                            df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].tail(10),
                            use_container_width=True
                        )
            
            # Technical indicators summary
            if include_technical:
                st.subheader("📊 Technical Summary")
                
                summary_data = []
                for tf in Config.TIMEFRAMES:
                    if tf in data:
                        df = data[tf]
                        latest = df.iloc[-1]
                        
                        summary_data.append({
                            'Timeframe': tf,
                            'Price': latest['close'],
                            'RSI': latest.get('rsi_14', 'N/A'),
                            'ATR': latest.get('atr_14', 'N/A'),
                            'Momentum': latest.get('mom_10', 'N/A'),
                            'Volatility': latest.get('vol_20', 'N/A')
                        })
                
                if summary_data:
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            
            # Load and display news
            if include_news:
                st.subheader("📰 Latest News")
                try:
                    news_loader = NewsLoader()
                    news = news_loader.fetch_normalized_news(page_size=5)
                    
                    for item in news:
                        with st.container():
                            st.markdown(f"**{item['headline']}**")
                            st.caption(f"{item['timestamp']} | {item['source']}")
                            if item.get('summary'):
                                st.write(item['summary'])
                            st.divider()
                except Exception as e:
                    st.warning(f"Could not load news: {e}")
            
            # Load and display economic calendar
            if include_econ:
                st.subheader("📅 Economic Calendar")
                try:
                    econ_loader = EconCalendarLoader(output_type="df")
                    econ_df = econ_loader.fetch()
                    
                    if not econ_df.empty:
                        # Filter for today and important events
                        today = datetime.now().date()
                        econ_today = econ_df[econ_df['date'] == today]
                        important_events = econ_today[econ_today['importance'] >= 2]
                        
                        if not important_events.empty:
                            st.dataframe(
                                important_events[['datetime', 'country', 'event', 'actual', 'forecast', 'importance']],
                                use_container_width=True
                            )
                        else:
                            st.info("No important economic events today")
                    else:
                        st.info("No economic calendar data available")
                except Exception as e:
                    st.warning(f"Could not load economic calendar: {e}")
    
    with tab3:
        st.markdown('<h2 class="sub-header">🤖 AI Chart Analysis</h2>', unsafe_allow_html=True)
        
        if not st.session_state.market_data:
            st.warning("Please load market data first from the sidebar")
            return
        
        if not st.session_state.chart_image_path:
            st.warning("Please generate a chart first in the Chart Analysis tab")
            return
        
        # Display chart image
        st.image(st.session_state.chart_image_path, caption="Chart for AI Analysis", use_column_width=True)
        
        # Analysis button
        if st.button("🧠 Start AI Analysis", type="primary", use_container_width=True):
            with st.spinner("AI is analyzing the chart and market data..."):
                try:
                    # Prepare data for AI
                    market_data_summary = {
                        'current_price': st.session_state.market_data['tick_info']['bid'] if st.session_state.market_data['tick_info'] else 'N/A',
                        'timestamp': st.session_state.market_data['timestamp'],
                        'technical_summary': ""
                    }
                    
                    # Add technical summary
                    tech_data = []
                    for tf in Config.TIMEFRAMES:
                        if tf in st.session_state.market_data['data']:
                            df = st.session_state.market_data['data'][tf]
                            latest = df.iloc[-1]
                            tech_data.append(
                                f"{tf}: Price={latest['close']:.2f}, "
                                f"RSI={latest.get('rsi_14', 'N/A'):.2f}, "
                                f"ATR={latest.get('atr_14', 'N/A'):.3f}"
                            )
                    
                    market_data_summary['technical_summary'] = "\n".join(tech_data)
                    
                    # Prepare news data
                    news_text = "No news available"
                    if include_news:
                        try:
                            news_loader = NewsLoader()
                            news = news_loader.fetch_normalized_news(page_size=3)
                            news_text = "\n".join([f"- {n['headline']}" for n in news])
                        except:
                            news_text = "News loading failed"
                    
                    # Prepare economic data
                    econ_text = "No economic data available"
                    if include_econ:
                        try:
                            econ_loader = EconCalendarLoader(output_type="df")
                            econ_df = econ_loader.fetch()
                            if not econ_df.empty:
                                today = datetime.now().date()
                                econ_today = econ_df[econ_df['date'] == today]
                                important = econ_today[econ_today['importance'] >= 2]
                                if not important.empty:
                                    econ_text = "\n".join([
                                        f"- {row['event']} ({row['country']}): Actual={row['actual']}, Forecast={row['forecast']}"
                                        for _, row in important.head(3).iterrows()
                                    ])
                        except:
                            econ_text = "Economic calendar loading failed"
                    
                    # Run AI analysis
                    analyzer = AIChartAnalyzer(model_name=ai_model)
                    
                    analysis_result = analyzer.create_comprehensive_analysis(
                        chart_image_path=st.session_state.chart_image_path,
                        market_data=market_data_summary,
                        news_data=news_text,
                        econ_data=econ_text
                    )
                    
                    st.session_state.ai_analysis = analysis_result
                    
                except Exception as e:
                    st.error(f"AI Analysis Error: {e}")
        
        # Display AI analysis results
        if st.session_state.ai_analysis:
            st.markdown('<div class="ai-response">', unsafe_allow_html=True)
            st.markdown("### 📋 AI Analysis Results")
            st.markdown(st.session_state.ai_analysis)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Trading recommendation summary
            st.subheader("🎯 Trading Recommendation")
            
            # Extract recommendation from analysis (simple parsing)
            analysis_text = st.session_state.ai_analysis.lower()
            
            if 'buy' in analysis_text and 'sell' not in analysis_text.split('buy')[0]:
                st.success("✅ BUY RECOMMENDATION")
                st.metric("Signal", "BULLISH", delta="AI Recommendation")
            elif 'sell' in analysis_text and 'buy' not in analysis_text.split('sell')[0]:
                st.error("🔻 SELL RECOMMENDATION")
                st.metric("Signal", "BEARISH", delta="AI Recommendation", delta_color="inverse")
            else:
                st.warning("⚠️ HOLD / NEUTRAL")
                st.metric("Signal", "NEUTRAL", delta="Wait for confirmation")
            
            # Suggested levels
            st.subheader("🎯 Suggested Levels")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Entry Zone", "See AI Analysis", delta="Based on S/R")
            with col2:
                st.metric("Stop Loss", "See AI Analysis", delta="Risk Management")
            with col3:
                st.metric("Take Profit", "See AI Analysis", delta="Reward Targets")


if __name__ == "__main__":
    main()