"it is a simple prediction model for daily returns"
"Relative Strength Index (RSI) and volatility are used to predict the movement"

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns 

#taking Apple stock to predict up or down direction of stock
df = yf.download("AAPL", start="2018-01-01", threads=False)
df = df.dropna()
df.head()

#daily return
df['return'] = df['Close'].pct_change() #pct_change() is the percentage change caculation between current and previous element in Series or DF

#moving averages
#rolling().mean() refers to the mean calculatioon in a rolling window
df['ma5']= df['Close'].rolling(5).mean() 
df['ma10']= df['Close'].rolling(10).mean()
df['ma20']= df['Close'].rolling(20).mean()

#calculate volatility based on Sdtv for 10 day rolling
df['volatility10'] = df['return'].rolling(10).std()

#Computing RSI in 14 days. 14 days is the best time range giving the 
def compute_rsi(series, period=14):
    delta = series.diff()  # price up → positive number, price down → negative number
    gain = delta.clip(lower=0)  # keep only gains, losses → 0
    loss = -1 * delta.clip(upper=0)  # keep only losses (as positive numbers)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))  # RSI formula

df['rsi14'] = compute_rsi(df['Close'])
df = df.dropna()
df.head()

#create the target (up/down)
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.dropna()

features = ['return','ma5', 'ma10','ma20', 'volatility10','rsi14']
X = df[features]
y = df['target']

# Train / test split
X_train = X.iloc[:-200]
X_test = X.iloc[-200:]
y_train = y.iloc[:-200]
y_test = y.iloc[-200:]

model = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# Simple strategy backtest: go long when model predicts "up"
df_test = df.iloc[-200:].copy()
df_test['pred'] = preds
df_test['actual'] = y_test.values
df_test['strategy_return'] = df_test['pred'] * df_test['return']
df_test[['Close', 'return', 'pred','actual']].head(20)

df_test[["return", "strategy_return"]].cumsum().plot(figsize=(10, 5))
plt.show()

#verification by using heatmap confusion matrix to show the pattern relationships
cm = confusion_matrix(y_test, preds)
sns.heatmap(cm, annot = True, cmap = 'Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()