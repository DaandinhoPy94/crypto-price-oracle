import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

class LSTMPredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        
    def prepare_data(self, data):
        """Prepare data for LSTM training"""
        # Scale the data
        scaled_data = self.scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
            
        return np.array(X), np.array(y)
    
    def build_model(self):
        """Build LSTM neural network"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train_model(self, data, epochs=50, validation_split=0.2):
        """Train LSTM model"""
        X, y = self.prepare_data(data)
        
        # Reshape for LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build model
        self.model = self.build_model()
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            verbose=0
        )
        
        self.is_trained = True
        return history
    
    def predict(self, data, forecast_days=30):
        """Generate predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        # Get last sequence
        scaled_data = self.scaler.transform(data['Close'].values.reshape(-1, 1))
        last_sequence = scaled_data[-self.sequence_length:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        # Generate predictions
        for _ in range(forecast_days):
            # Reshape for prediction
            X = current_sequence.reshape((1, self.sequence_length, 1))
            
            # Make prediction
            pred = self.model.predict(X, verbose=0)[0, 0]
            predictions.append(pred)
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], pred).reshape(-1, 1)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Create future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': predictions.flatten()
        })