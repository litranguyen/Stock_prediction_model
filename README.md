# Stock Price Direction Prediction Model

This project uses machine learning to predict Apple (APPL) stock’s price will go UP or DOWN the next day using historical price data and technical indicators.  
The model is built in Python using **yfinance**, **pandas**, and **scikit-learn**, and evaluates its performance using accuracy and a simple backtesting strategy.


## Features

### Download stock price data using `yfinance`  
- OHLCV (Open, High, Low, Close, Volume)  
- Automatically handles historical data cleaning  

### Feature Engineering
The model computes:
- Daily return  
- Moving averages (5, 10, 20 days)  
- Rolling volatility  
- RSI (14-day momentum)  
- Custom technical indicator functions  

### Classification Model (Random Forest)
Predicts:
- `1` → price will go **UP** tomorrow  
- `0` → price will go **DOWN** tomorrow  

### Backtesting Strategy
- Buys the stock only when the model predicts **UP**  
- Compares model performance to **Buy & Hold**  
- Plots cumulative returns for both strategies  


## Model Workflow

1. **Download historical stock data (AAPL)**  
2. **Compute features**  
3. **Create target variable (UP/DOWN)**  
4. **Train/test split (last 200 days = test)**  
5. **Train RandomForestClassifier**  
6. **Predict & evaluate model accuracy**  
7. **Simulate trading strategy based on predictions**  
8. **Plot performance**  


## Example Output

- Model accuracy on unseen data  
- Classification report (precision, recall, f1-score)  
- Performance chart comparing:
  - Buy & Hold
  - Model-based strategy  


## Technologies Used

- **Python 3**
- **yfinance**
- **pandas**
- **numpy**
- **scikit-learn**
- **matplotlib**
- **seaborn**


## Future Improvements

- Add more advanced technical indicators (MACD, Bollinger Bands, EMA crossovers)
- Use more powerful ML models (XGBoost, LSTM, CatBoost)
- Add proper time-series cross-validation
- Combine multiple tickers into one model
- Build a real backtesting engine (vectorbt, backtesting.py)
- Deploy the model via API or Streamlit dashboard



## Author

**Trang Nguyen**  


## 📜 License

This project is licensed under the MIT License.