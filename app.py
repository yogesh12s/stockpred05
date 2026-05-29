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
def inject_custom_css():
    st.markdown("""
        <style>
        /* 1. Hide default Streamlit clutter */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {background-color: transparent !important;}
        
        /* 2. Global Typography & Background */
        html, body, [class*="css"] {
            font-family: 'Inter', 'Roboto', sans-serif;
        }
        
        /* 3. Metric Cards (Glassmorphism & App-like styling) */
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, #1e2124, #121416);
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 20px rgba(0, 0, 0, 0.6);
        }
        
        /* Metric Label Text */
        [data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
            font-weight: 500;
            color: #a0aab5 !important;
            margin-bottom: 4px;
        }
        
        /* Metric Value Text */
        [data-testid="stMetricValue"] {
            font-size: 1.6rem !important;
            font-weight: 700;
            color: #ffffff !important;
        }
        
        /* Metric Delta Text */
        [data-testid="stMetricDelta"] {
            font-weight: 600;
            font-size: 0.95rem !important;
        }
        
        /* 4. Tabs Styling (Sleek Navigation) */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #121416;
            padding: 6px;
            border-radius: 12px;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
            border: none;
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
            border-radius: 8px;
            padding-top: 10px;
            padding-bottom: 10px;
            padding-left: 16px;
            padding-right: 16px;
            background-color: transparent;
            border: none !important;
        }
        [data-testid="stTabs"] [aria-selected="true"] {
            background-color: #2b3138;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            color: white !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            display: none;
        }
        
        /* 5. Mobile Responsiveness */
        @media (max-width: 768px) {
            /* Reduce overall padding on mobile */
            .main .block-container {
                padding-top: 2rem !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
            
            /* Groww-style 2-column grid on mobile */
            div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
                gap: 8px;
            }
            [data-testid="column"] {
                width: calc(50% - 4px) !important;
                flex: 1 1 calc(50% - 4px) !important;
                min-width: calc(50% - 4px) !important;
                margin-bottom: 8px;
            }
            
            /* Scale down fonts for 2-column grid */
            [data-testid="stMetricValue"] {
                font-size: 1.1rem !important;
            }
            [data-testid="stMetricLabel"] {
                font-size: 0.75rem !important;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            [data-testid="stMetricDelta"] {
                font-size: 0.8rem !important;
            }
            
            /* Make titles smaller on mobile */
            h1 {
                font-size: 1.8rem !important;
                text-align: center;
            }
            h3 {
                font-size: 1.3rem !important;
                text-align: center;
            }
            
            /* Center tab texts */
            [data-testid="stTabs"] [data-baseweb="tab-list"] {
                flex-wrap: wrap;
                justify-content: center;
            }
            [data-testid="stTabs"] [data-baseweb="tab"] {
                flex: 1;
                text-align: center;
                justify-content: center;
            }
            
            /* Metric cards spacing on mobile */
            [data-testid="stMetric"] {
                padding: 12px;
            }
        }
        </style>
    """, unsafe_allow_html=True)

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

@st.cache_data(ttl=600)
def get_top_performers():
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        # API 1: US Top Gainers
        us_url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=false&lang=en-US&region=US&scrIds=day_gainers&count=5"
        us_res = requests.get(us_url, headers=headers).json()
        us_quotes = us_res.get('finance', {}).get('result', [{}])[0].get('quotes', [])
        
        # API 2: India Top Gainers
        in_url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=false&lang=en-IN&region=IN&scrIds=day_gainers_in&count=5"
        in_res = requests.get(in_url, headers=headers).json()
        in_quotes = in_res.get('finance', {}).get('result', [{}])[0].get('quotes', [])
        
        combined_quotes = us_quotes + in_quotes
        
        performers_list = []
        for q in combined_quotes:
            symbol = q.get('symbol')
            price = q.get('regularMarketPrice')
            pct_change = q.get('regularMarketChangePercent')
            
            if symbol and price is not None and pct_change is not None:
                performers_list.append({
                    'Symbol': symbol,
                    'Price': price,
                    '% Change': pct_change
                })
                
        performers = pd.DataFrame(performers_list)
        if performers.empty:
            return pd.DataFrame()
            
        return performers.sort_values(by='% Change', ascending=False).head(10)
    except Exception as e:
        return pd.DataFrame()

MARKET_WATCHLIST = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'TSLA', 'META', 'BRK-B', 'V', 'JPM',
    'WMT', 'MA', 'PG', 'UNH', 'JNJ', 'HD', 'BAC', 'XOM', 'COST', 'CVX',
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'SBIN.NS', 
    'BHARTIARTL.NS', 'ITC.NS', 'LT.NS', 'BAJFINANCE.NS', 'ATGL.NS', 'HCLTECH.NS',
    'KOTAKBANK.NS', 'AXISBANK.NS', 'TATAMOTORS.NS', 'MARUTI.NS', 'SUNPHARMA.NS',
    'ULTRACEMCO.NS', 'ASIANPAINT.NS', 'TITAN.NS', 'ADANIENT.NS', 'ADANIPORTS.NS',
    'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'BAJAJFINSV.NS', 'M&M.NS', 'TATASTEEL.NS',
    'NESTLEIND.NS', 'WIPRO.NS', 'INDUSINDBK.NS', 'BAJAJ-AUTO.NS', 'HINDALCO.NS', 'TECHM.NS'
]

@st.cache_data(ttl=3600)
def get_historical_top_performers(days_ago=1, indian_only=False, top_n=5):
    try:
        # Fetch enough days to cover weekends and holidays
        period = f"{days_ago + 10}d"
        data = yf.download(MARKET_WATCHLIST, period=period, progress=False)
        if data.empty or 'Close' not in data:
            return pd.DataFrame()
            
        close_data = data['Close']
        
        # FIX: Filter by region first to avoid ffill copying values across mismatched trading days
        if indian_only:
            valid_cols = [c for c in close_data.columns if str(c).endswith('.NS') or str(c).endswith('.BO')]
            close_data = close_data[valid_cols]
            
        # Clean up NaNs specific to this market's trading calendar
        close_data = close_data.dropna(how='all').ffill()
        
        if len(close_data) < days_ago + 1:
            return pd.DataFrame()
            
        # Target day is 'days_ago' from the end. If days_ago=1 (yesterday), it's iloc[-2]
        # But we need to compare it to the day before that (iloc[-3])
        target_idx = -1 - days_ago
        prev_idx = target_idx - 1
        
        if abs(prev_idx) > len(close_data):
            return pd.DataFrame()
            
        target_close = close_data.iloc[target_idx]
        prev_close = close_data.iloc[prev_idx]
        
        pct_change = ((target_close - prev_close) / prev_close) * 100
        
        performers = pd.DataFrame({
            'Price': target_close,
            '% Change': pct_change
        })
        if indian_only:
            performers = performers[performers.index.astype(str).str.endswith('.NS') | performers.index.astype(str).str.endswith('.BO')]
            
        return performers.dropna().sort_values(by='% Change', ascending=False).head(top_n)
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def predict_tomorrows_gainers():
    indian_tickers = [t for t in MARKET_WATCHLIST if str(t).endswith('.NS') or str(t).endswith('.BO')]
    
    predictions = []
    
    # We need about 1 year of data to have >200 trading days for the backtest split
    start_dt = date.today() - timedelta(days=365*2)
    end_dt = date.today()
    
    for ticker in indian_tickers:
        try:
            df = yf.download(ticker, start=start_dt, end=end_dt, progress=False)
            processed_df = _process_yf_df(df)
            
            if processed_df is not None and len(processed_df) > 220:
                ml_data, predictors = engineer_features(processed_df, is_live=False)
                if ml_data is not None and len(ml_data) > 200:
                    precision, tomorrow_pred, err = train_and_backtest(ml_data, predictors)
                    
                    if err is None and tomorrow_pred == 1:
                        future_price_pred, pct_change = train_future_regressor(processed_df, horizon_days=1)
                        # Ensure the regressor actually predicts a profit
                        if pct_change is not None and pct_change > 0:
                            current_price = processed_df['Close'].iloc[-1]
                            
                            # Heuristic for Trading Method Recommendation
                            if pct_change > 2.5:
                                method = "Intraday ⚡"
                            elif pct_change > 1.5 and precision > 0.60:
                                method = "MTF 📈"
                            else:
                                method = "Delivery 🔒"
                                
                            predictions.append({
                                'Symbol': ticker,
                                'Price': current_price,
                                'AI Confidence': precision,
                                'Profit %': pct_change,
                                'Method': method
                            })
        except Exception:
            continue
            
    pred_df = pd.DataFrame(predictions)
    if not pred_df.empty:
        # Sort by highest predicted profit percentage instead of just win rate
        pred_df = pred_df.sort_values(by='Profit %', ascending=False).head(10)
    return pred_df

def render_top_performers_tab():
    st.markdown("### 🔮 AI Predictions: Tomorrow's Expected Gainers")
    st.markdown("Machine Learning forecast for the Top 10 Indian stocks expected to go UP tomorrow.")
    st.info("⚠️ **DISCLAIMER:** 100% accuracy is mathematically impossible in quantitative finance. These predictions are based on historical probability (Random Forest Classifier backtests). Do not treat this as guaranteed financial advice.", icon="⚠️")
    
    with st.spinner("Training Machine Learning models for all watchlist stocks... (This takes ~30 seconds)"):
        ai_predictions = predict_tomorrows_gainers()
        
    if ai_predictions.empty:
        st.warning("The AI did not find any high-confidence setups for tomorrow.")
    else:
        cols_ai = st.columns(5)
        for idx, row in ai_predictions.iterrows():
            col = cols_ai[idx % 5]
            with col:
                ticker = row['Symbol']
                currency = "₹" if str(ticker).endswith(".NS") or str(ticker).endswith(".BO") else "$"
                st.metric(
                    label=ticker,
                    value=f"{currency}{row['Price']:.2f}",
                    delta=f"{row['Profit %']:+.2f}%",
                    delta_color="normal"
                )
                st.caption(f"Win Rate: {row['AI Confidence']*100:.1f}%<br>Type: {row['Method']}", unsafe_allow_html=True)
    
    st.markdown("---")

    st.markdown("### 🔥 Global Top 10 Best Performing Stocks Today")
    st.markdown("Dynamically scanning the absolute top gainers across US and Indian markets.")
    
    with st.spinner("Fetching live market data..."):
        top_stocks = get_top_performers()
        
    if top_stocks.empty:
        st.error("Failed to fetch market data from APIs. Please try again later.")
        return
        
    cols = st.columns(5)
    for idx, row in top_stocks.iterrows():
        col = cols[idx % 5]
        with col:
            ticker = row['Symbol']
            currency = "₹" if ticker.endswith(".NS") or ticker.endswith(".BO") else "$"
            st.metric(
                label=ticker,
                value=f"{currency}{row['Price']:.2f}",
                delta=f"{row['% Change']:+.2f}%"
            )

    st.markdown("---")
    st.markdown("### ⏪ Yesterday's Top 10 Indian Performers")
    st.markdown("Calculated from a curated watchlist of NIFTY 50 / Major Indian Stocks.")
    
    with st.spinner("Calculating historical data..."):
        historical_stocks = get_historical_top_performers(days_ago=1, indian_only=True, top_n=10)
        
    if historical_stocks.empty:
        st.warning("Historical data is currently unavailable.")
        return
        
    cols_hist = st.columns(5)
    for idx, (ticker, row) in enumerate(historical_stocks.iterrows()):
        col = cols_hist[idx % 5]
        with col:
            currency = "₹" if str(ticker).endswith(".NS") or str(ticker).endswith(".BO") else "$"
            st.metric(
                label=str(ticker),
                value=f"{currency}{row['Price']:.2f}",
                delta=f"{row['% Change']:+.2f}%"
            )

# --- MAIN APP FLOW ---
def render_predictor_tab():

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

def main():
    inject_custom_css()
    st.title("📈 Quantitative Stock Analysis & Prediction")
    st.markdown("A production-grade machine learning pipeline for time-series equity forecasting.")

    tab1, tab2 = st.tabs(["🤖 Predictor Engine", "🔥 Top Performing Stocks"])
    
    with tab1:
        render_predictor_tab()
        
    with tab2:
        render_top_performers_tab()

if __name__ == "__main__":
    main()
