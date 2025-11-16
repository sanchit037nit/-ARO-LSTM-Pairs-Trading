import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="ARO-LSTM Quant Dashboard", layout="wide", page_icon="📈")
st.title("⚡ ARO-LSTM Pairs Trading Engine")
st.markdown("Automated Statistical Arbitrage Dashboard for **HDFC vs. ICICI**")

# ==========================================
# CACHED DATA & MODEL LOADING (FOR SPEED)
# ==========================================
@st.cache_data
def load_and_prep_data():
    # Load raw data
    df = pd.read_csv('HF_DATASET.csv', header=None)
    
    # Extract columns (Col 0: Date, Col 1: HDFC, Col 7: ICICI based on previous Colab runs)
    dates = pd.to_datetime(df.iloc[:, 0], errors='coerce', dayfirst=True)
    price_1 = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    price_2 = pd.to_numeric(df.iloc[:, 7], errors='coerce')
    
    df_clean = pd.DataFrame({'Date': dates, 'HDFC': price_1, 'ICICI': price_2})
    
    # Calculate Spread & Z-Score
    df_clean['Spread'] = df_clean['HDFC'] / df_clean['ICICI']
    window = 30
    df_clean['spread_mean'] = df_clean['Spread'].rolling(window=window).mean()
    df_clean['spr_std'] = df_clean['Spread'].rolling(window=window).std()
    df_clean['Z_Score'] = (df_clean['Spread'] - df_clean['spread_mean']) / df_clean['spr_std']
    df_clean.dropna(subset=['Z_Score'], inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    
    return df_clean

@st.cache_resource
def load_ai_assets():
    model = tf.keras.models.load_model('aro_lstm_model.keras')
    scaler = joblib.load('zscore_scaler.pkl')
    return model, scaler

# Load assets into memory
try:
    df_clean = load_and_prep_data()
    model, scaler = load_ai_assets()
except Exception as e:
    st.error(f"Error loading files. Ensure 'HF_DATASET.csv', 'aro_lstm_model.keras', and 'zscore_scaler.pkl' are in the same folder. Details: {e}")
    st.stop()

# ==========================================
# SIDEBAR CONTROL CENTER
# ==========================================
st.sidebar.header("⚙️ Strategy Parameters")
st.sidebar.info("**Optimized Architecture:**\n\n🧠 145 Neurons (ARO)\n\n💧 20% Dropout\n\n📉 Early Stopping Enabled")

# Interactive Threshold Slider
threshold = st.sidebar.slider("Execution Threshold (Z-Score)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
initial_capital = st.sidebar.number_input("Initial Capital (₹)", value=100000, step=10000)

run_button = st.sidebar.button("🚀 Run Backtest", use_container_width=True)

# ==========================================
# MACHINE LEARNING PIPELINE
# ==========================================
if run_button:
    with st.spinner("Neural Network is processing sequence data..."):
        # Scale data
        z_score_data = df_clean['Z_Score'].values.reshape(-1, 1)
        scaled_data = scaler.transform(z_score_data)

        # Create Sequences (60 Days)
        window_size = 60
        X, y = [], []
        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i-window_size:i, 0])
            y.append(scaled_data[i, 0])
        
        X = np.array(X)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Split Train/Test (Keep only Test for UI Backtest)
        train_size = int(len(X) * 0.80)
        X_test = X[train_size:]
        
        # We need the actual unscaled Z-scores and Dates for the test period
        # The test period starts at index: train_size + window_size
        test_start_idx = train_size + window_size
        test_dates = df_clean['Date'].iloc[test_start_idx:].values
        y_test_actual = df_clean['Z_Score'].iloc[test_start_idx:].values.reshape(-1, 1)

        # AI PREDICTION
        pred_scaled = model.predict(X_test)
        predictions = scaler.inverse_transform(pred_scaled)

    # ==========================================
    # BACKTEST ENGINE
    # ==========================================
    with st.spinner("Executing Algorithmic Trading Simulator..."):
        positions = np.zeros(len(predictions))
        current_position = 0
        trades_taken = 0
        buy_signals_x, buy_signals_y = [], []
        sell_signals_x, sell_signals_y = [], []

        for i in range(len(predictions)):
            pred_z = predictions[i][0]
            actual_z = y_test_actual[i][0]
            current_date = test_dates[i]

            if pred_z > threshold:       
                if current_position != -1: 
                    trades_taken += 1
                    sell_signals_x.append(current_date)
                    sell_signals_y.append(actual_z)
                current_position = -1
            elif pred_z < -threshold:    
                if current_position != 1: 
                    trades_taken += 1
                    buy_signals_x.append(current_date)
                    buy_signals_y.append(actual_z)
                current_position = 1
            elif current_position == -1 and pred_z <= 0.0:
                current_position = 0
            elif current_position == 1 and pred_z >= 0.0:
                current_position = 0

            positions[i] = current_position

        positions = np.roll(positions, 1)
        positions[0] = 0

        actual_returns = np.diff(y_test_actual.flatten(), prepend=y_test_actual[0][0]) * 0.01
        strategy_returns = positions * actual_returns

        cumulative_returns = np.cumprod(1 + strategy_returns)
        portfolio_value = initial_capital * cumulative_returns
        final_capital = portfolio_value[-1]

        net_profit = final_capital - initial_capital
        total_return_pct = (net_profit / initial_capital) * 100
        
        # Safe metric calculations
        daily_vol = np.std(strategy_returns)
        sharpe_ratio = (np.mean(strategy_returns) / daily_vol) * np.sqrt(252) if daily_vol > 0 else 0.0
        
        running_max = np.maximum.accumulate(portfolio_value)
        drawdowns = (portfolio_value - running_max) / running_max
        max_drawdown = np.min(drawdowns) * 100

    # ==========================================
    # UI METRICS DASHBOARD
    # ==========================================
    st.markdown("---")
    st.subheader("📊 Financial Performance Metrics (Out-of-Sample)")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Return", f"{total_return_pct:.2f}%", f"₹{net_profit:,.2f}")
    col2.metric("Trades Executed", f"{trades_taken}")
    col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    col4.metric("Max Drawdown", f"{max_drawdown:.2f}%")

    st.markdown("---")
    
    # ==========================================
    # INTERACTIVE CHARTS (PLOTLY)
    # ==========================================
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("🎯 ARO-LSTM Signal Triggers")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=test_dates, y=y_test_actual.flatten(), mode='lines', name='Actual Z-Score', line=dict(color='white', width=1)))
        
        # Threshold lines
        fig1.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text=f"Short Threshold (+{threshold})")
        fig1.add_hline(y=-threshold, line_dash="dash", line_color="green", annotation_text=f"Long Threshold (-{threshold})")
        fig1.add_hline(y=0, line_color="gray", annotation_text="Mean Reversion (0.0)")

        # Trade markers
        fig1.add_trace(go.Scatter(x=buy_signals_x, y=buy_signals_y, mode='markers', name='Long Position', marker=dict(color='green', size=10, symbol='triangle-up')))
        fig1.add_trace(go.Scatter(x=sell_signals_x, y=sell_signals_y, mode='markers', name='Short Position', marker=dict(color='red', size=10, symbol='triangle-down')))
        
        fig1.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig1, use_container_width=True)

    with col_chart2:
        st.subheader("💰 Portfolio Equity Curve")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=test_dates, y=portfolio_value, mode='lines', name='Portfolio Value', line=dict(color='cyan', width=2), fill='tozeroy'))
        fig2.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Capital (₹)")
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("👈 Adjust your parameters in the sidebar and click **Run Backtest** to initialize the AI engine.")