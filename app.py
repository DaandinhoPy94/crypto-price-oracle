import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from src.data_collector import CryptoDataCollector
from src.predictor import CryptoPricePredictor
import pandas as pd
from src.lstm_predictor import LSTMPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

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
tab1, tab2, tab3 = st.tabs(["📊 Current Analysis", "🤖 AI Predictions", "📈 Technical Analysis"])

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
    st.subheader("🤖 Advanced AI Predictions")
    
    # Model selection controls
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox(
            "Select AI Model:",
            options=["Both Models", "Prophet Only", "LSTM Only"],
            index=0
        )
    
    with col2:
        comparison_metric = st.selectbox(
            "Comparison Metric:",
            options=["MAPE", "MAE", "RMSE"],
            index=0
        )
    
    if data is not None and not data.empty:
        with st.spinner("🧠 Training AI models... This may take a few minutes for LSTM"):
            # Split data for training and validation
            train_size = int(len(data) * 0.8)
            train_data = data[:train_size]
            validation_data = data[train_size:]
            
            results = {}
            
            # Train Prophet
            if model_type in ["Both Models", "Prophet Only"]:
                try:
                    prophet_predictor = CryptoPricePredictor()
                    prophet_forecast = prophet_predictor.train_model(train_data, forecast_days)
                    prophet_predictions = prophet_predictor.get_predictions(forecast_days)
                    prophet_accuracy = prophet_predictor.calculate_accuracy(validation_data)
                    
                    results['Prophet'] = {
                        'predictor': prophet_predictor,
                        'forecast': prophet_forecast,
                        'predictions': prophet_predictions,
                        'accuracy': prophet_accuracy
                    }
                except Exception as e:
                    st.error(f"Prophet training failed: {str(e)}")
            
            # Train LSTM
            if model_type in ["Both Models", "LSTM Only"]:
                try:
                    lstm_predictor = LSTMPredictor(sequence_length=60)
                    
                    # LSTM needs more data - check if we have enough
                    if len(train_data) >= 100:
                        lstm_history = lstm_predictor.train_model(train_data, epochs=50)
                        lstm_predictions = lstm_predictor.predict(train_data, forecast_days)
                        
                        # Calculate LSTM accuracy on validation data
                        if len(validation_data) > 0:
                            val_predictions = lstm_predictor.predict(data[:train_size], len(validation_data))
                            val_actual = validation_data['Close'].values
                            val_pred = val_predictions['yhat'].values[:len(val_actual)]
                            
                            lstm_accuracy = {
                                'MAE': mean_absolute_error(val_actual, val_pred),
                                'RMSE': np.sqrt(mean_squared_error(val_actual, val_pred)),
                                'MAPE': np.mean(np.abs((val_actual - val_pred) / val_actual)) * 100
                            }
                        else:
                            lstm_accuracy = None
                        
                        results['LSTM'] = {
                            'predictor': lstm_predictor,
                            'predictions': lstm_predictions,
                            'accuracy': lstm_accuracy,
                            'history': lstm_history
                        }
                    else:
                        st.warning("⚠️ LSTM needs more data. Please select a longer time period (6mo+)")
                except Exception as e:
                    st.error(f"LSTM training failed: {str(e)}")
        
        # Display model performance comparison
        if len(results) > 1:
            st.subheader("📊 Model Performance Comparison")
            
            perf_cols = st.columns(len(results))
            for i, (model_name, model_data) in enumerate(results.items()):
                with perf_cols[i]:
                    if model_data['accuracy']:
                        st.metric(
                            label=f"{model_name} {comparison_metric}",
                            value=f"{model_data['accuracy'][comparison_metric]:.2f}{'%' if comparison_metric == 'MAPE' else ''}",
                            delta=f"{'Lower is better' if comparison_metric in ['MAE', 'RMSE', 'MAPE'] else 'Higher is better'}"
                        )
        
        # Individual model results
        for model_name, model_data in results.items():
            st.subheader(f"📈 {model_name} Predictions")
            
            if model_data['accuracy']:
                # Accuracy metrics
                acc_col1, acc_col2, acc_col3 = st.columns(3)
                with acc_col1:
                    st.metric("MAE", f"${model_data['accuracy']['MAE']:.2f}")
                with acc_col2:
                    st.metric("RMSE", f"${model_data['accuracy']['RMSE']:.2f}")
                with acc_col3:
                    st.metric("MAPE", f"{model_data['accuracy']['MAPE']:.1f}%")
            
            # Prediction chart
            fig_model = go.Figure()
            
            # Historical data
            fig_model.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='#00D4AA', width=2)
            ))
            
            # Model predictions
            if len(model_data['predictions']) > 0:
                pred_color = '#FF6B6B' if model_name == 'Prophet' else '#9B59B6'
                
                fig_model.add_trace(go.Scatter(
                    x=model_data['predictions']['ds'],
                    y=model_data['predictions']['yhat'],
                    mode='lines',
                    name=f'{model_name} Prediction',
                    line=dict(color=pred_color, width=3, dash='dot')
                ))
                
                # Add confidence intervals for Prophet
                if model_name == 'Prophet' and 'yhat_lower' in model_data['predictions'].columns:
                    fig_model.add_trace(go.Scatter(
                        x=model_data['predictions']['ds'],
                        y=model_data['predictions']['yhat_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    fig_model.add_trace(go.Scatter(
                        x=model_data['predictions']['ds'],
                        y=model_data['predictions']['yhat_lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        name=f'{model_name} Confidence Interval',
                        fillcolor=f'rgba(255, 107, 107, 0.2)',
                        hoverinfo='skip'
                    ))
            
            fig_model.update_layout(
                title=f"{selected_crypto} - {model_name} AI Forecast",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig_model, use_container_width=True)
            
            # Prediction table
            if len(model_data['predictions']) > 0:
                with st.expander(f"📅 {model_name} Detailed Predictions"):
                    pred_display = model_data['predictions'].copy()
                    pred_display['Date'] = pred_display['ds'].dt.strftime('%Y-%m-%d')
                    pred_display['Predicted Price'] = pred_display['yhat'].apply(lambda x: f"${x:.2f}")
                    
                    display_cols = ['Date', 'Predicted Price']
                    if 'yhat_lower' in pred_display.columns:
                        pred_display['Lower Bound'] = pred_display['yhat_lower'].apply(lambda x: f"${x:.2f}")
                        pred_display['Upper Bound'] = pred_display['yhat_upper'].apply(lambda x: f"${x:.2f}")
                        display_cols.extend(['Lower Bound', 'Upper Bound'])
                    
                    st.dataframe(pred_display[display_cols], use_container_width=True)
        
        # Ensemble prediction if both models available
        if len(results) == 2 and all('predictions' in r for r in results.values()):
            st.subheader("🎯 Ensemble Prediction (Combined Models)")
            
            # Combine predictions (simple average)
            prophet_pred = results['Prophet']['predictions']
            lstm_pred = results['LSTM']['predictions']
            
            # Ensure same length
            min_len = min(len(prophet_pred), len(lstm_pred))
            
            ensemble_pred = pd.DataFrame({
                'ds': prophet_pred['ds'][:min_len],
                'yhat': (prophet_pred['yhat'][:min_len] + lstm_pred['yhat'][:min_len]) / 2
            })
            
            # Ensemble chart
            fig_ensemble = go.Figure()
            
            # Historical data
            fig_ensemble.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='#00D4AA', width=2)
            ))
            
            # Individual predictions (thinner lines)
            fig_ensemble.add_trace(go.Scatter(
                x=prophet_pred['ds'],
                y=prophet_pred['yhat'],
                mode='lines',
                name='Prophet',
                line=dict(color='#FF6B6B', width=1, dash='dash'),
                opacity=0.7
            ))
            
            fig_ensemble.add_trace(go.Scatter(
                x=lstm_pred['ds'],
                y=lstm_pred['yhat'],
                mode='lines',
                name='LSTM',
                line=dict(color='#9B59B6', width=1, dash='dash'),
                opacity=0.7
            ))
            
            # Ensemble prediction (thick line)
            fig_ensemble.add_trace(go.Scatter(
                x=ensemble_pred['ds'],
                y=ensemble_pred['yhat'],
                mode='lines',
                name='Ensemble (Average)',
                line=dict(color='#F39C12', width=4)
            ))
            
            fig_ensemble.update_layout(
                title=f"{selected_crypto} - Ensemble AI Forecast (Prophet + LSTM)",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=600,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig_ensemble, use_container_width=True)
            
            # Ensemble summary
            current_price = data['Close'].iloc[-1]
            ensemble_final = ensemble_pred['yhat'].iloc[-1]
            ensemble_change = ensemble_final - current_price
            ensemble_change_pct = (ensemble_change / current_price) * 100
            
            st.subheader("🎯 Ensemble Forecast Summary")
            ens_col1, ens_col2, ens_col3 = st.columns(3)
            
            with ens_col1:
                st.metric(
                    label="Current Price",
                    value=f"${current_price:.2f}"
                )
            
            with ens_col2:
                st.metric(
                    label=f"Ensemble Prediction ({forecast_days} days)",
                    value=f"${ensemble_final:.2f}",
                    delta=f"{ensemble_change:+.2f} ({ensemble_change_pct:+.1f}%)"
                )
            
            with ens_col3:
                confidence = "High" if abs(ensemble_change_pct) < 10 else "Medium" if abs(ensemble_change_pct) < 20 else "Low"
                st.metric(
                    label="Confidence Level",
                    value=confidence,
                    delta=f"±{abs(ensemble_change_pct):.1f}%"
                )
    
    else:
        st.error("No data available for AI predictions.")

with tab3:
    st.subheader("📈 Advanced Technical Analysis")
    
    if data is not None and not data.empty:
        # Technical indicators metrics
        current_rsi = data['RSI'].iloc[-1]
        current_macd = data['MACD'].iloc[-1]
        current_signal = data['MACD_Signal'].iloc[-1]
        bb_position = (data['Close'].iloc[-1] - data['BB_Lower'].iloc[-1]) / (data['BB_Upper'].iloc[-1] - data['BB_Lower'].iloc[-1])
        
        # Indicators row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi_color = "normal"
            if current_rsi > 70:
                rsi_color = "inverse"  # Overbought
            elif current_rsi < 30:
                rsi_color = "inverse"  # Oversold
            
            st.metric(
                label="RSI (14)",
                value=f"{current_rsi:.1f}",
                delta="Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral",
                delta_color=rsi_color
            )
        
        with col2:
            macd_trend = "Bullish" if current_macd > current_signal else "Bearish"
            st.metric(
                label="MACD Signal",
                value=macd_trend,
                delta=f"{current_macd:.4f}"
            )
        
        with col3:
            bb_status = "Upper" if bb_position > 0.8 else "Lower" if bb_position < 0.2 else "Middle"
            st.metric(
                label="Bollinger Position", 
                value=bb_status,
                delta=f"{bb_position:.1%}"
            )
        
        with col4:
            volume_status = "High" if data['Volume_Ratio'].iloc[-1] > 1.5 else "Low" if data['Volume_Ratio'].iloc[-1] < 0.5 else "Normal"
            st.metric(
                label="Volume",
                value=volume_status,
                delta=f"{data['Volume_Ratio'].iloc[-1]:.1f}x avg"
            )
        
        # RSI Chart
        st.subheader("📊 RSI Oscillator")
        fig_rsi = go.Figure()
        
        fig_rsi.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2)
        ))
        
        # RSI levels
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral (50)")
        
        fig_rsi.update_layout(
            title="RSI (Relative Strength Index)",
            yaxis_title="RSI Value",
            height=300,
            template="plotly_dark",
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # MACD Chart
        st.subheader("📈 MACD Analysis")
        fig_macd = go.Figure()
        
        fig_macd.add_trace(go.Scatter(
            x=data.index,
            y=data['MACD'],
            mode='lines',
            name='MACD Line',
            line=dict(color='blue', width=2)
        ))
        
        fig_macd.add_trace(go.Scatter(
            x=data.index,
            y=data['MACD_Signal'],
            mode='lines',
            name='Signal Line',
            line=dict(color='red', width=1)
        ))
        
        fig_macd.add_trace(go.Bar(
            x=data.index,
            y=data['MACD_Histogram'],
            name='Histogram',
            marker_color=['green' if x > 0 else 'red' for x in data['MACD_Histogram']]
        ))
        
        fig_macd.add_hline(y=0, line_dash="dash", line_color="white", annotation_text="Zero Line")
        
        fig_macd.update_layout(
            title="MACD (Moving Average Convergence Divergence)",
            yaxis_title="MACD Value",
            height=400,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig_macd, use_container_width=True)
        
        # Bollinger Bands Chart
        st.subheader("📊 Bollinger Bands & Price Action")
        fig_bb = go.Figure()
        
        # Price line
        fig_bb.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='white', width=2)
        ))
        
        # Bollinger Bands
        fig_bb.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Upper'],
            mode='lines',
            name='Upper Band',
            line=dict(color='red', width=1),
            fill=None
        ))
        
        fig_bb.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Lower'],
            mode='lines',
            name='Lower Band',
            line=dict(color='green', width=1),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)'
        ))
        
        fig_bb.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Middle'],
            mode='lines',
            name='Middle Band (20 SMA)',
            line=dict(color='yellow', width=1, dash='dot')
        ))
        
        fig_bb.update_layout(
            title="Bollinger Bands Analysis",
            yaxis_title="Price (USD)",
            height=500,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig_bb, use_container_width=True)
    
    else:
        st.error("No data available for technical analysis.")