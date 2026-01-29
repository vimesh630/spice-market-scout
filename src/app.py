import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import glob
import glob
import subprocess
import json
import datetime
import numpy as np
import time

# Page Configuration
st.set_page_config(
    page_title="Spice Market Scout Pro",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DATA_PATH = "data/processed/spice_prices.csv"
DATA_PATH = "data/processed/spice_prices.csv"
MODEL_SCRIPT = "src/forecasting_engine.py"
NEWS_AGENT_SCRIPT = "src/news_agent.py"
SENTIMENT_FILE = "data/processed/latest_sentiment.json"

# Custom CSS for "Premium" feel
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the spice price data"""
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    return df

def get_grades(df):
    """Extract grade names from column headers"""
    grade_cols = [col for col in df.columns if 'Cinnamon_Grade_' in col]
    grades = [col.replace('Cinnamon_Grade_', '') for col in grade_cols]
    return grades

def retrain_model():
    """Run the forecasting engine script"""
    try:
        # Use sys.executable to ensure the same python environment is used
        import sys
        result = subprocess.run(
            [sys.executable, MODEL_SCRIPT],
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    except Exception as e:
        return False, str(e)


def fetch_latest_news():
    """Run the news agent script"""
    try:
        import sys
        result = subprocess.run(
            [sys.executable, NEWS_AGENT_SCRIPT],
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    except Exception as e:
        return False, str(e)

def load_sentiment_data():
    """Load latest sentiment data from JSON"""
    if os.path.exists(SENTIMENT_FILE):
        try:
            with open(SENTIMENT_FILE, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def generate_mock_forecast(last_date, last_price, days=30):
    """Generate mock forecast data for visualization"""
    dates = [last_date + datetime.timedelta(days=i) for i in range(1, days + 1)]
    # Random walk with slight upward drift for "optimism" or based on sentiment
    volatility = last_price * 0.02
    drift = last_price * 0.001
    prices = [last_price]
    for _ in range(days):
        change = np.random.normal(drift, volatility)
        prices.append(prices[-1] + change)
    return dates, prices[1:]

def main():
    # Sidebar
    st.sidebar.title("Controls")
    
    df = load_data()
    
    if df is None:
        st.error(f"Data file not found at {DATA_PATH}. Please run the scraper first.")
        return

    # Sidebar Filters
    st.sidebar.subheader("Filters")
    
    # Region Filter (Mocked as requested since data lacks explicit region)
    regions = ["Colombo", "Galle", "Matara", "Kandy"]
    selected_region = st.sidebar.selectbox("Region", regions)
    
    # Grade Filter
    grades = get_grades(df)
    if not grades:
        st.error("No grade data found in CSV.")
        return
    
    selected_grade = st.sidebar.selectbox("Grade", grades)
    grade_col = f"Cinnamon_Grade_{selected_grade}"

    # News Agent Button
    st.sidebar.subheader("Market Intelligence")
    if st.sidebar.button("📰 Fetch Latest News"):
        with st.spinner("Analyzing Market Sentiment..."):
            success, output = fetch_latest_news()
            if success:
                st.sidebar.success("News Updated!")
            else:
                st.sidebar.error("Failed to fetch news")
                with st.expander("Error Logs"):
                    st.code(output)

    # Retrain Button
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Actions")
    if st.sidebar.button("🔄 Retrain Model"):
        with st.spinner("Training LSTM Model... This may take a minute."):
            success, output = retrain_model()
            if success:
                st.sidebar.success("Model Updated Successfully!")
                with st.expander("Training Logs"):
                    st.code(output)
            else:
                st.sidebar.error("Model Training Failed")
                with st.expander("Error Logs"):
                    st.code(output)

    # Main Content
    st.title("🌶️ Spice Market Scout")
    st.markdown(f"### Pro Dashboard - {selected_region} Region")
    
    # ---------------------------------------------------------
    # Market Intelligence Section
    # ---------------------------------------------------------
    sentiment_data = load_sentiment_data()
    st.markdown("### Market Intelligence")
    
    col_s1, col_s2 = st.columns([1, 3])
    
    mood = "Neutral"
    score = 0
    summary = "No data available."
    timestamp = ""
    
    if sentiment_data:
        score = sentiment_data.get('sentiment_score', 0)
        summary = sentiment_data.get('summary', 'No summary.')
        timestamp = sentiment_data.get('timestamp', '')
        
        if score > 0.2:
            mood = "🐂 Bullish"
            color = "green"
        elif score < -0.2:
            mood = "🐻 Bearish"
            color = "red"
        else:
            mood = "⚖️ Neutral"
            color = "orange"
            
        with col_s1:
            st.markdown(f"<div class='metric-card'><h3>Sentiment</h3><h2 style='color: {color}'>{mood}</h2><p>Score: {score:.2f}</p></div>", unsafe_allow_html=True)
            
        with col_s2:
            if timestamp:
                st.caption(f"Last Updated: {timestamp}")
            st.info(f"**Latest Insight:** {summary}")
            
        # Warning Banner
        if score < -0.2:
            st.warning("⚠️ Market sentiment is bearish. LSTM Forecast might be over-optimistic.")
            
        with st.expander("Read Latest News Sources"):
            if sentiment_data.get('source_urls'):
                for url in sentiment_data['source_urls']:
                    st.markdown(f"- [{url}]({url})") 
            else:
                st.write("No specific sources cited.")
    else:
        st.info("No sentiment data found. Click 'Fetch Latest News' in the sidebar to generate intel.")
        
    st.markdown("---")
    # ---------------------------------------------------------

    if grade_col in df.columns:
        # Sort and get relevant data
        df_sorted = df.sort_values('Date')
        last_row = df_sorted.iloc[-1]
        current_price = last_row[grade_col]
        last_date = last_row['Date']
        
        # Forecast Logic
        # For now, using mock forecast generation as per requirements
        forecast_dates, forecast_prices = generate_mock_forecast(last_date, current_price)
        predicted_price = forecast_prices[-1] # Price next month (approx) or end of forecast period
        
        # Calculate trend
        price_diff = predicted_price - current_price
        trend_signal = "📈 Up" if price_diff > 0 else "📉 Down"
        trend_color = "green" if price_diff > 0 else "red"

        # Metrics Row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price (LKR)", f"{current_price:,.2f}")
        with col2:
            st.metric("Predicted Price (30 Days)", f"{predicted_price:,.2f}", delta=f"{price_diff:.2f}")
        with col3:
            st.markdown(f"<h2 style='color: {trend_color}; text-align: center;'>{trend_signal}</h2>", unsafe_allow_html=True)
            st.caption("Trend Signal")

        # Main Chart
        fig = go.Figure()
        
        # Historical Data
        fig.add_trace(go.Scatter(
            x=df_sorted['Date'],
            y=df_sorted[grade_col],
            mode='lines',
            name='Historical Price',
            line=dict(color='#0068C9', width=2)
        ))
        
        # Forecast Data
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_prices,
            mode='lines',
            name='Forecast (Predicted)',
            line=dict(color='#FF4B4B', width=2, dash='dot')
        ))
        
        fig.update_layout(
            title=f"Price Forecast for {selected_grade}",
            xaxis_title="Date",
            yaxis_title="Price (LKR)",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment Display (Optional bonus)
        if 'Market_Sentiment' in df.columns:
            st.markdown("### Latest Market Sentiment")
            sentiment = last_row['Market_Sentiment']
            s_color = "green" if sentiment.lower() == "bullish" else "red" if sentiment.lower() == "bearish" else "orange"
            st.markdown(f"<span style='font-size: 1.2em; font-weight: bold; color: {s_color}'>{sentiment}</span>", unsafe_allow_html=True)
            
    else:
        st.error(f"Column '{grade_col}' not found in data.")

if __name__ == "__main__":
    main()
