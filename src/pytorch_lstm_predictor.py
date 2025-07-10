import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class PyTorchLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=3, output_size=1):
        super(PyTorchLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class PyTorchLSTMPredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        
    def prepare_data(self, data):
        """Prepare data for LSTM training"""
        scaled_data = self.scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
            
        return np.array(X), np.array(y)
    
    def train_model(self, data, epochs=50):
        """Train PyTorch LSTM model"""
        X, y = self.prepare_data(data)
        
        # Convert to PyTorch tensors
        X = torch.FloatTensor(X).unsqueeze(-1)  # Add feature dimension
        y = torch.FloatTensor(y).unsqueeze(-1)
        
        # Initialize model
        self.model = PyTorchLSTM()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        self.is_trained = True
        return {"loss": loss.item()}
    
    def predict(self, data, forecast_days=30):
        """Generate predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        # Get last sequence
        scaled_data = self.scaler.transform(data['Close'].values.reshape(-1, 1))
        last_sequence = scaled_data[-self.sequence_length:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(forecast_days):
                # Convert to tensor
                X = torch.FloatTensor(current_sequence).unsqueeze(0).unsqueeze(-1)
                
                # Make prediction
                pred = self.model(X).item()
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