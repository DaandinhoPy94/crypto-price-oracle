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
        """Get historical crypto data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                st.error(f"No data found for {symbol}")
                return None
                
            # Add some basic technical indicators
            data['MA_7'] = data['Close'].rolling(window=7).mean()
            data['MA_30'] = data['Close'].rolling(window=30).mean()
            data['Volatility'] = data['Close'].rolling(window=30).std()
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
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