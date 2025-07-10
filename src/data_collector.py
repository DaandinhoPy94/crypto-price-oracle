import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

class CryptoDataCollector:
    def __init__(self):
        self.symbols = {
            'BTC-USD': 'Bitcoin',
            'ETH-USD': 'Ethereum', 
            'BNB-USD': 'Binance Coin',
            'ADA-USD': 'Cardano',
            'SOL-USD': 'Solana'
        }
    
    def get_historical_data(self, symbol='BTC-USD', period='1y'):
        """Get historical crypto data with advanced technical indicators"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                st.error(f"No data found for {symbol}")
                return None
            
            # Basic moving averages
            data['MA_7'] = data['Close'].rolling(window=7).mean()
            data['MA_30'] = data['Close'].rolling(window=30).mean()
            data['MA_50'] = data['Close'].rolling(window=50).mean()
            data['Volatility'] = data['Close'].rolling(window=30).std()
            
            # RSI Calculation
            data['RSI'] = self.calculate_rsi(data['Close'])
            
            # MACD Calculation
            macd_line, signal_line, histogram = self.calculate_macd(data['Close'])
            data['MACD'] = macd_line
            data['MACD_Signal'] = signal_line
            data['MACD_Histogram'] = histogram
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data['Close'])
            data['BB_Upper'] = bb_upper
            data['BB_Middle'] = bb_middle
            data['BB_Lower'] = bb_lower
            
            # Volume indicators
            data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicators"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def get_current_price(self, symbol='BTC-USD'):
        """Get current crypto price"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('regularMarketPrice', info.get('currentPrice', 0))
        except:
            return 0

# Test function
if __name__ == "__main__":
    collector = CryptoDataCollector()
    btc_data = collector.get_historical_data('BTC-USD', '3mo')
    print(f"✅ Retrieved {len(btc_data)} days of Bitcoin data")
    print(f"Latest BTC price: ${btc_data['Close'].iloc[-1]:,.2f}")