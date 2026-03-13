import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

def prepare_data(ticker: str) -> pd.DataFrame:
    """
    Merge stock price data with sentiment data and generate target features.
    """
    price_file = f"data/raw_data/{ticker}_prices.csv"
    sentiment_file = f"data/processed_data/{ticker}_daily_sentiment.csv"
    
    if not os.path.exists(price_file):
        raise FileNotFoundError(f"Cannot find price data at {price_file}")
    
    # Load prices
    df_price = pd.read_csv(price_file)
    df_price['Date'] = pd.to_datetime(df_price['Date']).dt.date
    df_price.rename(columns={'Date': 'date'}, inplace=True)
    
    # Load sentiments if available
    if os.path.exists(sentiment_file):
        df_sentiment = pd.read_csv(sentiment_file)
        df_sentiment['date'] = pd.to_datetime(df_sentiment['date']).dt.date
    else:
        print("Warning: No sentiment data found. Creating dummy sentiment for demonstration.")
        df_sentiment = pd.DataFrame({'date': df_price['date'].unique(), 'sentiment_score': 0.0, 'article_count': 0})
        
    # Merge datasets
    df = pd.merge(df_price, df_sentiment, on='date', how='left')
    
    # Fill missing sentiments with 0 (neutral)
    df['sentiment_score'].fillna(0, inplace=True)
    df['article_count'].fillna(0, inplace=True)
    
    # Feature Engineering
    # Daily return
    df['daily_return'] = df['Close'].pct_change()
    
    # Moving averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    
    # Target: Predict if NEXT day's close is higher than TODAY's close
    df['next_day_return'] = df['daily_return'].shift(-1)
    df['target'] = (df['next_day_return'] > 0).astype(int)
    
    # Drop NaNs that resulted from shifting and rolling
    df.dropna(inplace=True)
    return df

def train_models(df: pd.DataFrame, ticker: str):
    """
    Train and evaluate classification models.
    """
    print(f"Training models for {ticker}...")
    
    # Select features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'daily_return', 'SMA_5', 'SMA_10', 'sentiment_score', 'article_count']
    
    X = df[features]
    y = df['target']
    
    # Train-test split (time-series aware: do not shuffle)
    # We use the first 80% of dates to train, the last 20% to test
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, df.index, test_size=0.2, shuffle=False
    )
    
    # 1. Logistic Regression
    print("\n--- Logistic Regression ---")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
    print(classification_report(y_test, lr_preds))
    
    # 2. Random Forest
    print("\n--- Random Forest ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    print(classification_report(y_test, rf_preds))
    
    # Save the best model (using Random Forest for demonstration)
    os.makedirs('data/models', exist_ok=True)
    model_path = f"data/models/{ticker}_rf_model.pkl"
    joblib.dump(rf_model, model_path)
    print(f"\nSaved Random Forest model to {model_path}")
    
    # Append predictions back to the test dataframe for evaluation
    df_test = df.loc[indices_test].copy()
    df_test['rf_prediction'] = rf_preds
    df_test['lr_prediction'] = lr_preds
    df_test.to_csv(f"data/processed_data/{ticker}_predictions.csv", index=False)
    
    return df_test

if __name__ == "__main__":
    ticker = "AAPL"
    try:
        df_merged = prepare_data(ticker)
        df_test_results = train_models(df_merged, ticker)
        print("Data modeling complete.")
    except Exception as e:
        print(f"Error during modeling: {e}")
