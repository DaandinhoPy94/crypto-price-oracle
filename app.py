import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from src.data_collector import CryptoDataCollector
from src.predictor import CryptoPricePredictor
import pandas as pd

# Page config
st.set_page_config(
    page_title="Crypto Price Oracle",
    page_icon="🚀",
    layout="wide"
)

# Header
st.title("🚀 Crypto Price Oracle")
st.markdown("**AI-Powered Cryptocurrency Price Analysis & Prediction**")

# Initialize components
@st.cache_data
def load_crypto_data(symbol, period):
    collector = CryptoDataCollector()
    return collector.get_historical_data(symbol, period)

@st.cache_data
def train_prediction_model(symbol, period, forecast_days):
    collector = CryptoDataCollector()
    data = collector.get_historical_data(symbol, period)
    if data is not None and not data.empty:
        predictor = CryptoPricePredictor()
        forecast = predictor.train_model(data, forecast_days)
        predictions = predictor.get_predictions(forecast_days)
        accuracy = predictor.calculate_accuracy(data.tail(60))  # Use last 60 days for accuracy
        return forecast, predictions, accuracy, predictor
    return None, None, None, None

# Sidebar controls
st.sidebar.title("📊 Controls")
selected_crypto = st.sidebar.selectbox(
    "Select Cryptocurrency:",
    options=['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD'],
    index=0
)

period = st.sidebar.selectbox(
    "Time Period:",
    options=['3mo', '6mo', '1y', '2y'],
    index=2
)

forecast_days = st.sidebar.slider(
    "Prediction Days:",
    min_value=7,
    max_value=60,
    value=30
)

# Load data
data = load_crypto_data(selected_crypto, period)

# Main tabs
tab1, tab2 = st.tabs(["📊 Current Analysis", "🤖 AI Predictions"])

with tab1:
    if data is not None and not data.empty:
        # Current price info
        current_price = data['Close'].iloc[-1]
        price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
        price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label=f"{selected_crypto} Price",
                value=f"${current_price:,.2f}",
                delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
            )
        
        with col2:
            st.metric(
                label="24h High",
                value=f"${data['High'].iloc[-1]:,.2f}"
            )
        
        with col3:
            st.metric(
                label="24h Low", 
                value=f"${data['Low'].iloc[-1]:,.2f}"
            )
        
        with col4:
            st.metric(
                label="Volume",
                value=f"{data['Volume'].iloc[-1]:,.0f}"
            )
        
        # Price chart
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='#00D4AA', width=2)
        ))
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA_7'],
            mode='lines',
            name='7-day MA',
            line=dict(color='orange', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA_30'],
            mode='lines',
            name='30-day MA',
            line=dict(color='red', width=1)
        ))
        
        fig.update_layout(
            title=f"{selected_crypto} Price Chart",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=500,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data preview
        st.subheader("📋 Recent Data")
        st.dataframe(data.tail(10))

    else:
        st.error("Failed to load cryptocurrency data. Please check your internet connection.")

with tab2:
    st.subheader("🤖 AI Price Predictions")
    
    if data is not None and not data.empty:
        # Train model and get predictions
        with st.spinner("🧠 Training AI model and generating predictions..."):
            forecast, predictions, accuracy, predictor = train_prediction_model(
                selected_crypto, period, forecast_days
            )
        
        if forecast is not None and predictions is not None:
            # Display accuracy metrics
            st.subheader("📊 Model Performance")
            if accuracy:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Absolute Error", f"${accuracy['MAE']:.2f}")
                with col2:
                    st.metric("Root Mean Squared Error", f"${accuracy['RMSE']:.2f}")
                with col3:
                    st.metric("Mean Absolute Percentage Error", f"{accuracy['MAPE']:.1f}%")
            
            # Create prediction chart
            fig_pred = go.Figure()
            
            # Historical data
            fig_pred.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='#00D4AA', width=2)
            ))
            
            # Predictions
            if len(predictions) > 0:
                fig_pred.add_trace(go.Scatter(
                    x=predictions['ds'],
                    y=predictions['yhat'],
                    mode='lines',
                    name='AI Prediction',
                    line=dict(color='#FF6B6B', width=3, dash='dot')
                ))
                
                # Confidence intervals
                fig_pred.add_trace(go.Scatter(
                    x=predictions['ds'],
                    y=predictions['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig_pred.add_trace(go.Scatter(
                    x=predictions['ds'],
                    y=predictions['yhat_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    name='Confidence Interval',
                    fillcolor='rgba(255, 107, 107, 0.2)',
                    hoverinfo='skip'
                ))
            
            fig_pred.update_layout(
                title=f"{selected_crypto} AI Price Forecast - Next {forecast_days} Days",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=600,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Prediction table
            st.subheader("📅 Detailed Predictions")
            if len(predictions) > 0:
                pred_display = predictions.copy()
                pred_display['Date'] = pred_display['ds'].dt.strftime('%Y-%m-%d')
                pred_display['Predicted Price'] = pred_display['yhat'].apply(lambda x: f"${x:.2f}")
                pred_display['Lower Bound'] = pred_display['yhat_lower'].apply(lambda x: f"${x:.2f}")
                pred_display['Upper Bound'] = pred_display['yhat_upper'].apply(lambda x: f"${x:.2f}")
                
                st.dataframe(
                    pred_display[['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']],
                    use_container_width=True
                )
                
                # Price movement summary
                current_price = data['Close'].iloc[-1]
                final_prediction = predictions['yhat'].iloc[-1]
                price_change_pred = final_prediction - current_price
                price_change_pct_pred = (price_change_pred / current_price) * 100
                
                st.subheader("🎯 Forecast Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label=f"Current Price",
                        value=f"${current_price:.2f}"
                    )
                with col2:
                    st.metric(
                        label=f"Predicted Price ({forecast_days} days)",
                        value=f"${final_prediction:.2f}",
                        delta=f"{price_change_pred:+.2f} ({price_change_pct_pred:+.1f}%)"
                    )
        
        else:
            st.error("Failed to generate predictions. Please try with different parameters.")
    
    else:
        st.error("No data available for predictions.")