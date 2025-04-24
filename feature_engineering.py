import numpy as np
import pandas as pd

def add_features(df, ticker):
    """
    Add features for the model including 1-Year Return, Volatility, Risk Level, and Ticker.
    """
    # Add Ticker to DataFrame
    df['Ticker'] = ticker
    
    # Calculate the daily percentage change in closing prices
    df['Daily Return'] = df['Close'].pct_change()

    # Calculate 21-day rolling standard deviation of daily returns (volatility)
    # Multiply by sqrt(252) to annualize the volatility (252 trading days in a year)
    df['Volatility'] = df['Daily Return'].rolling(window=21).std() * np.sqrt(252)  # Annualized volatility
    
    # Calculate cumulative return over time
    df['Cumulative Return'] = (1 + df['Daily Return']).cumprod()
    # Track the running maximum (peak) of the cumulative return
    df['Peak'] = df['Cumulative Return'].cummax()
    # Calculate drawdown from the peak
    df['Drawdown'] = (df['Cumulative Return'] - df['Peak']) / df['Peak']
    # Record the maximum drawdown (the worst drop)
    df['Max Drawdown'] = df['Drawdown'].min()

    rf_rate = 0.02  # Set a fixed risk-free rate (e.g., 2%)
    # Calculate Sharpe Ratio: excess return per unit of risk (volatility)
    sharpe_ratio = (df['Daily Return'].mean() - rf_rate) / df['Daily Return'].std() * np.sqrt(252)

    # Calculate Risk Score (1-10 scale)
    max_volatility = df['Volatility'].max()
    max_drawdown = df['Max Drawdown']

    # Normalize volatility
    df['Normalized Volatility'] = df['Volatility'] / max_volatility
    # Normalize drawdown
    df['Normalized Drawdown'] = df['Max Drawdown'] / max_drawdown

    # Compute a composite risk score on a scale of 1 to 10
    # Weighted sum of normalized volatility, drawdown, and Sharpe Ratio
    df['Risk Score'] = 1 + (df['Normalized Volatility'] * 0.4 + df['Normalized Drawdown'] * 0.3 + (1 / (sharpe_ratio + 1)) * 0.3) * 9

    # Limit Risk Score to between 1 and 10
    df['Risk Score'] = df['Risk Score'].clip(1, 10)
    
    # Calculate 1-Year Return based on percentage change over 252 trading days
    if '1-Year Return' not in df.columns:
        df['1-Year Return'] = df['Close'].pct_change(periods=252)  # 252 trading days in a year

    return df