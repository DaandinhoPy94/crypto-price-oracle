import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class CryptoPricePredictor:
    def __init__(self):
        self.model = None
        self.forecast = None
        
    def prepare_data_for_prophet(self, data):
        """Convert price data to Prophet format"""
        # Prophet expects columns named 'ds' (date) and 'y' (value)
        prophet_data = pd.DataFrame({
            'ds': data.index.tz_localize(None),  # Remove timezone info for Prophet
            'y': data['Close']
        })
        return prophet_data
    
    def train_model(self, data, forecast_days=30):
        """Train Prophet model on historical data"""
        # Prepare data
        prophet_data = self.prepare_data_for_prophet(data)
        
        # Initialize and fit Prophet model
        self.model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        self.model.fit(prophet_data)
        
        # Create future dataframe for predictions
        future = self.model.make_future_dataframe(periods=forecast_days)
        self.forecast = self.model.predict(future)
        
        return self.forecast
    
    def get_predictions(self, days=7):
        """Get next N days predictions"""
        if self.forecast is None:
            return None
            
        # Get last N predictions (future dates)
        predictions = self.forecast.tail(days)
        return predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def calculate_accuracy(self, actual_data):
        """Calculate prediction accuracy on historical data"""
        if self.forecast is None:
            return None
            
        # Align forecast with actual data
        forecast_historical = self.forecast[:-30]  # Exclude future predictions
        
        # Calculate metrics
        mae = mean_absolute_error(actual_data['Close'], forecast_historical['yhat'][-len(actual_data):])
        rmse = np.sqrt(mean_squared_error(actual_data['Close'], forecast_historical['yhat'][-len(actual_data):]))
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': np.mean(np.abs((actual_data['Close'] - forecast_historical['yhat'][-len(actual_data):]) / actual_data['Close'])) * 100
        }