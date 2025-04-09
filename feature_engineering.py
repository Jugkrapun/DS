# feature_engineering.py
import numpy as np
import pandas as pd

def add_features(df, ticker):
    """
    Add features for the model including 1-Year Return, Volatility, and Risk Level.
    Also, add the Ticker column.
    """
    # Add Ticker to DataFrame
    df['Ticker'] = ticker
    
    # Basic feature: Daily Return
    df['Daily Return'] = df['Close'].pct_change()

    # Ensure 'Volatility' exists, calculate if missing
    if 'Volatility' not in df.columns:
        # Calculate rolling volatility (standard deviation of returns over 21 days)
        df['Volatility'] = df['Daily Return'].rolling(window=21).std() * np.sqrt(252)  # Annualized volatility

    # Calculate 1-Year Return (252 trading days)
    if '1-Year Return' not in df.columns:
        df['1-Year Return'] = df['Close'].pct_change(periods=252)  # 252 trading days in a year

    # Drop any NaN values that might have been created during feature engineering
    df = df.dropna()

    return df
