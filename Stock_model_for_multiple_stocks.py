#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 21:56:13 2025

@author: trangnguyen
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns 

#calculate RSI 
def compute_rsi (series, period = 14):
    delta = series.diff()
    gain = delta.clip(lower =0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain/avg_loss
    return 100 - (100/(1+rs))

def new_features(df):
    # daily return
    df['return']= df['Close'].pct_change()
    
    #moving avg
    df['ma5']= df['Close'].rolling(5).mean()
    df['ma10']=df['Close'].rolling(10).mean()
    df['ma20']=df['Close'].rolling(20).mean()
    df['ma50']=df['Close'].rolling(50).mean()
    
    #rolling volatility 10 days
    df['volatility10']= df['return'].rolling(10).std()
    
    #calculate Exponential moving averages for MACD
    df['ema12']=df['Close'].ewm(span=12, adjust = False).mean() # EMA12 is fast moving average 
    df['ema26']=df['Close'].ewm(span=26, adjust = False).mean() #EMA26 is slow moving average 
    df['macd']=df['ema12'] - df['ema26'] #MACD shows momentum shifts and trend strength
    df['macd_signal'] = df['macd'].ewm(span=9, adjust = False).mean() 
    
    #bollinger bands 20 period and 2 std
    df['bb_mid']= df[['Close']].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_upper']=df['bb_mid']+2* bb_std
    df['bb_lower']=df['bb_mid']-2* bb_std
    
    #RSI
    df['rsi14'] = compute_rsi(df['Close'], period=14)
    df['rsi7'] = compute_rsi(df['Close'], period=7)
    
    #lagged returns 
    df['return_1d_lag'] = df['return'].shift(1)
    df['return_5d'] = df['Close'].pct_change(5)
    
    #drop initial NaNs 
    df = df.dropna()
    
    #predict 5 day trend
    df['target_5d'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    df = df.dropna()
    
    return df

tickers = ['AAPL', 'IBM','MSFT','AMZN', 'META', 'AVGO']

def run_model(ticker):
    #download stock data
    df = yf.download(ticker, start='2018-01-01', threads=False)
    if df.empty:
        print('No data downloaded', ticker)
        return
    
    #add features + target
    df = new_features(df)
    
    #Build X, y
    feature_cols = ["return", "return_1d_lag", "return_5d",
        "ma5", "ma10", "ma20", "ma50",
        "volatility10",
        "ema12", "ema26", "macd", "macd_signal",
        "bb_mid", "bb_upper", "bb_lower",
        "rsi7", "rsi14"]
    X = df[feature_cols]
    y = df['target_5d']
    
    #train and test for last 200 data
    if len(df) <=300:
        print('Not enough data for', ticker)
        return
    
    X_train = X.iloc[:-200]
    X_test = X.iloc[-200:]
    y_train = y.iloc[:-200]
    y_test = y.iloc[-200:]

    #train model
    model = RandomForestClassifier(
        n_estimators= 300,
        max_depth=6,
        min_samples_leaf=3,
        min_samples_split= 5,
        random_state= 50)
    model.fit(X_train, y_train)
    
    #prediction and evaluation
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test,preds)
    print ('Accuracy:', accuracy)
    print(classification_report(y_true, preds))
    
    # strategy backtest
    df_test = df.iloc[-200:].copy()
    df_test["pred"] = preds
    df_test["strategy_return"] = df_test["pred"] * df_test["return"]

    # Plot cumulative returns
    df_test[["return", "strategy_return"]].cumsum().plot(figsize=(10, 4))
    plt.title(f"Cumulative Returns: {ticker}")
    plt.legend(["Buy & Hold", "Model Strategy"])
    plt.show()


# Run for each ticker
for t in tickers:
    run_model(t)
    
    

