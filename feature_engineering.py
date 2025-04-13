# feature_engineering.py
import numpy as np
import pandas as pd

def add_features(df, ticker):
    """
    Add features for the model including 1-Year Return, Volatility, Risk Level, and Ticker.
    """
    # Add Ticker to DataFrame
    df['Ticker'] = ticker
    
    # Basic feature: Daily Return
    df['Daily Return'] = df['Close'].pct_change()

    # Calculate rolling volatility (standard deviation of returns over 21 days)
    df['Volatility'] = df['Daily Return'].rolling(window=21).std() * np.sqrt(252)  # Annualized volatility
    
    # คำนวณ Maximum Drawdown
    df['Cumulative Return'] = (1 + df['Daily Return']).cumprod()
    df['Peak'] = df['Cumulative Return'].cummax()
    df['Drawdown'] = (df['Cumulative Return'] - df['Peak']) / df['Peak']
    df['Max Drawdown'] = df['Drawdown'].min()

    # คำนวณ Sharpe Ratio
    rf_rate = 0.02  # สมมุติว่าอัตราผลตอบแทนที่ไม่มีความเสี่ยงคือ 2%
    sharpe_ratio = (df['Daily Return'].mean() - rf_rate) / df['Daily Return'].std() * np.sqrt(252)

    # คำนวณ Risk Score (1-10 scale)
    max_volatility = df['Volatility'].max()
    max_drawdown = df['Max Drawdown']

    # Normalize ค่า
    df['Normalized Volatility'] = df['Volatility'] / max_volatility
    df['Normalized Drawdown'] = df['Max Drawdown'] / max_drawdown

    # การคำนวณ Risk Score ที่อยู่ในช่วง 1-10
    df['Risk Score'] = 1 + (df['Normalized Volatility'] * 0.4 + df['Normalized Drawdown'] * 0.3 + (1 / (sharpe_ratio + 1)) * 0.3) * 9

    # Clip ค่า Risk Score ให้อยู่ในช่วง 1-10
    df['Risk Score'] = df['Risk Score'].clip(1, 10)
    
    # Calculate 1-Year Return (252 trading days)
    if '1-Year Return' not in df.columns:
        df['1-Year Return'] = df['Close'].pct_change(periods=252)  # 252 trading days in a year

    return df