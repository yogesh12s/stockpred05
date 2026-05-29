import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import precision_score
from datetime import date, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Quantitative Stock Analysis & Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FINANCIAL DISCLAIMER ---
st.sidebar.markdown("### ⚠️ Disclaimer")
st.sidebar.warning(
    "This application is for educational and informational purposes only. "
    "It does not constitute financial advice. Machine learning models "
    "are inherently limited and historical performance does not guarantee "
    "future results. Market prediction accuracy rarely exceeds 55-60%. "
    "Trade at your own risk."
)

# --- SIDEBAR INPUTS ---
st.sidebar.header("Model Parameters")

@st.cache_data(ttl=86400)
def get_search_results(query):
    if not query:
        return []
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        quotes = data.get('quotes', [])
        results = []
        for q in quotes:
            if 'symbol' in q and ('shortname' in q or 'longname' in q):
                name = q.get('longname', q.get('shortname', ''))
                if name:
                    results.append(f"{q['symbol']} - {name}")
                else:
                    results.append(q['symbol'])
        return results
    except Exception as e:
        return []

search_query = st.sidebar.text_input("Search Company Name or Ticker", value="Adani Total Gas")
search_results = get_search_results(search_query)

if search_results:
    selected_asset = st.sidebar.selectbox("Select Asset", options=search_results)
    ticker_input = selected_asset.split(" - ")[0]
else:
    st.sidebar.warning("No results found. Please try another search.")
    ticker_input = None

start_date = st.sidebar.date_input("Start Date", value=date(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())
live_mode = st.sidebar.checkbox("🟢 Live Intraday Mode (Updates every 10s)", value=False)

@st.cache_data(ttl=3600)
def load_daily_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        return _process_yf_df(df)
    except Exception as e:
        st.error(f"Error fetching daily data: {e}")
        return None

@st.cache_data(ttl=10)
def load_live_data(ticker):
    try:
        df = yf.download(ticker, interval="1m", period="7d", progress=False)
        return _process_yf_df(df)
    except Exception as e:
        st.error(f"Error fetching live data: {e}")
        return None

def _process_yf_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        return None
    df.columns = df.columns.astype(str)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna(subset=['Close'])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    return df

# --- FEATURE ENGINEERING ---
@st.cache_data
def engineer_features(df, is_live=False):
    """
    Engineers predictive features without data leakage.
    Target: 1 if next period's close > today's close, else 0.
    Features: Rolling ratios and historical trend sum.
    """
    data = df.copy()
    
    # Target Variable formulation (Next Period's Close direction)
    data['Tomorrow'] = data['Close'].shift(-1)
    data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)
    
    # Horizons for moving averages and trend calculations
    horizons = [5, 15, 60] if is_live else [2, 5, 20, 50]
    new_predictors = []
    
    for horizon in horizons:
        # Rolling averages for the horizon
        rolling_averages = data['Close'].rolling(window=horizon).mean()
        
        # Feature 1: Ratio of today's close to the rolling average
        ratio_column = f"Close_Ratio_{horizon}"
        data[ratio_column] = data['Close'] / rolling_averages
        
        # Feature 2: Number of days the stock went up in the rolling window
        trend_column = f"Trend_{horizon}"
        data[trend_column] = data['Target'].shift(1).rolling(window=horizon).sum()
        
        new_predictors += [ratio_column, trend_column]
    
    # Drop rows with NaN values resulting from rolling windows and shifts
    data = data.dropna()
    
    return data, new_predictors

# --- MODEL TRAINING & EVALUATION ---
def train_and_backtest(data, predictors):
    """
    Trains a RandomForest model and evaluates using a strict time-series split.
    Uses the last 100 trading days as the test set to mimic live performance.
    """
    if len(data) < 200:
        return None, None, "Insufficient data for training and testing (need >200 trading days)."

    # Strict time-series split: No random shuffling
    train = data.iloc[:-100]
    test = data.iloc[-100:]
    
    # Model Configuration tailored to prevent overfitting
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=50,
        random_state=42,
        n_jobs=-1 # Utilize all cores
    )
    
    # Train the model
    model.fit(train[predictors], train['Target'])
    
    # Generate predictions on the test set
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index)
    
    # Calculate Precision Score (Focus on minimizing false positives)
    precision = precision_score(test['Target'], preds, zero_division=0)
    
    # Predict tomorrow's direction using the most recent data point
    latest_data = data.iloc[-1:]
    tomorrow_prediction = model.predict(latest_data[predictors])[0]
    
    return precision, tomorrow_prediction, None

@st.cache_data
def train_future_regressor(daily_df, horizon_days):
    """
    Trains a regressor to predict the exact price `horizon_days` into the future.
    """
    required_history = horizon_days + 100
    if daily_df is None or len(daily_df) < required_history:
        return None, None
        
    data = daily_df.copy()
    
    # Target: Price horizon_days from now
    data['Target_Price'] = data['Close'].shift(-horizon_days)
    
    # Create basic features for regression
    horizons = [5, 20, 50, 200]
    predictors = []
    
    for horizon in horizons:
        rolling_averages = data['Close'].rolling(window=horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        data[ratio_column] = data['Close'] / rolling_averages
        predictors.append(ratio_column)
        
    # Drop rows where we don't have rolling averages
    # We must also drop the last `horizon_days` rows because they don't have a Target_Price yet!
    train_data = data.dropna(subset=predictors + ['Target_Price'])
    
    if len(train_data) < 50:
        return None, None
        
    model = RandomForestRegressor(
        n_estimators=100,
        min_samples_split=50,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(train_data[predictors], train_data['Target_Price'])
    
    # Predict the future price using the VERY LAST day's features
    latest_features = data.iloc[-1:][predictors]
    
    if latest_features.isnull().values.any():
        return None, None
        
    future_price_pred = model.predict(latest_features)[0]
    
    current_price = data['Close'].iloc[-1]
    pct_change = ((future_price_pred - current_price) / current_price) * 100
    
    return future_price_pred, pct_change

# --- MAIN APP FLOW ---
def main():
    st.title("📈 Quantitative Stock Analysis & Prediction")
    st.markdown("A production-grade machine learning pipeline for time-series equity forecasting.")

    if not ticker_input:
        st.warning("Please enter a valid stock ticker in the sidebar.")
        return

    horizon_mapping = {
        "Tomorrow": 1,
        "Weekly": 5,
        "Monthly": 21,
        "3 Months": 63,
        "6 Months": 126,
        "Yearly": 252,
        "5 Years": 1260
    }
    
    st.markdown("### 🎯 Future Price Target Selection")
    selected_horizon_label = st.radio(
        "Select Prediction Horizon:", 
        options=list(horizon_mapping.keys()), 
        index=2, # Default to Monthly
        horizontal=True
    )
    horizon_days = horizon_mapping[selected_horizon_label]

    if live_mode:
        with st.spinner(f"Fetching live 1m data for {ticker_input}..."):
            raw_data = load_live_data(ticker_input)
            # Fetch full daily data quietly in background for the regressor
            daily_data_for_regressor = load_daily_data(ticker_input, date(2000, 1, 1), end_date)
    else:
        with st.spinner(f"Fetching historical data for {ticker_input}..."):
            raw_data = load_daily_data(ticker_input, start_date, end_date)
            # Always load full history for regressor to support long horizons like 5-Year
            daily_data_for_regressor = load_daily_data(ticker_input, date(2000, 1, 1), end_date)

    if raw_data is None or raw_data.empty:
        st.error(f"Failed to fetch data for ticker '{ticker_input}'. Please verify the symbol and date range.")
        return

    with st.spinner("Engineering features & training model..."):
        # Engineer features
        ml_data, predictors = engineer_features(raw_data, is_live=live_mode)
        
        # Train and Evaluate short-term classifier
        precision, tomorrow_prediction, err_msg = train_and_backtest(ml_data, predictors)
        
    with st.spinner(f"Training {selected_horizon_label} Price Prediction model..."):
        future_price, future_pct = train_future_regressor(daily_data_for_regressor, horizon_days)

    if err_msg:
        st.error(err_msg)
        return

    # --- UI RENDERING ---
    
    # 1. Metric Cards
    st.markdown("### 📊 Market Overview & Model Inference")
    col1, col2, col3, col4 = st.columns(4)
    
    # Converting to float safely
    current_price = float(raw_data['Close'].iloc[-1])
    prev_price = float(raw_data['Close'].iloc[-2])
    price_change = current_price - prev_price
    
    with col1:
        st.metric(
            label="Current Price (Close)", 
            value=f"₹{current_price:.2f}", 
            delta=f"₹{price_change:.2f}"
        )
        
    with col2:
        direction_text = "UP 🔼" if tomorrow_prediction == 1 else "DOWN 🔽"
        direction_color = "normal" if tomorrow_prediction == 1 else "inverse"
        
        prediction_label = "Next Minute's Prediction" if live_mode else "Tomorrow's Prediction"
        
        st.metric(
            label=prediction_label, 
            value=direction_text,
            delta="Bullish" if tomorrow_prediction == 1 else "Bearish",
            delta_color=direction_color
        )
        
    with col3:
        if future_price is not None:
            st.metric(
                label=f"{selected_horizon_label}'s Target", 
                value=f"₹{future_price:.2f}",
                delta=f"{future_pct:+.2f}%",
                help=f"Predicted exact price {horizon_days} trading days from now."
            )
        else:
            st.metric(
                label=f"{selected_horizon_label}'s Target", 
                value="N/A",
                help=f"Need more historical data (>= {horizon_days + 100} days) to calculate a {selected_horizon_label} forecast."
            )

    with col4:
        st.metric(
            label="Backtest Precision Score", 
            value=f"{precision * 100:.2f}%",
            help="Percentage of 'UP' predictions that were actually correct over the last 100 trading days. >55% is considered strong in financial markets."
        )

    # 2. Interactive Charts
    st.markdown("### 📈 Historical Price & Moving Average")
    
    # Calculate MA for visualization
    chart_data = raw_data.copy()
    
    ma_window = 60 if live_mode else 50
    ma_name = f"{ma_window}-Min Moving Average" if live_mode else f"{ma_window}-Day Moving Average"
    
    # Squeeze columns to ensure they are 1D Series
    chart_data['MA'] = chart_data['Close'].squeeze().rolling(window=ma_window).mean()
    
    fig = go.Figure()
    
    # Format index as strings to remove Plotly overnight/weekend gaps
    if live_mode:
        x_axis_labels = chart_data.index.strftime('%Y-%m-%d %H:%M')
    else:
        x_axis_labels = chart_data.index.strftime('%Y-%m-%d')
        
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=x_axis_labels,
        open=chart_data['Open'].squeeze(),
        high=chart_data['High'].squeeze(),
        low=chart_data['Low'].squeeze(),
        close=chart_data['Close'].squeeze(),
        name='Actual Stock Price'
    ))
    
    # MA Line
    fig.add_trace(go.Scatter(
        x=x_axis_labels, 
        y=chart_data['MA'], 
        line=dict(color='orange', width=2), 
        name=ma_name
    ))
    
    # Add Prediction Indicator on the Chart
    if tomorrow_prediction == 1:
        pred_color, pred_symbol = "green", "triangle-up"
        pred_y = chart_data['High'].iloc[-1] * 1.002 # Slightly above highest point
    else:
        pred_color, pred_symbol = "red", "triangle-down"
        pred_y = chart_data['Low'].iloc[-1] * 0.998 # Slightly below lowest point

    fig.add_trace(go.Scatter(
        x=[x_axis_labels[-1]],
        y=[pred_y],
        mode='markers',
        marker=dict(symbol=pred_symbol, size=18, color=pred_color, line=dict(width=2, color='white')),
        name='Future Prediction'
    ))
    
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        xaxis=dict(type='category', nticks=10), # Enforce category to skip gaps
        template='plotly_dark',
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # 3. Raw Data Audit (Expandable)
    st.markdown("---")
    with st.expander("🔍 Audit Raw Feature Dataframe (Transparency)"):
        st.dataframe(ml_data.tail(20).sort_index(ascending=False), use_container_width=True)
        st.markdown(
            "**Data Dictionary:**\n"
            "- `Close_Ratio_X`: Ratio of the closing price to the X-period moving average.\n"
            "- `Trend_X`: Number of positive periods (close > previous close) over the last X periods.\n"
            "- `Target`: 1 if the *next* period's close was higher than the current close, 0 otherwise."
        )

    if live_mode:
        import time
        time.sleep(10)
        st.rerun()

if __name__ == "__main__":
    main()
