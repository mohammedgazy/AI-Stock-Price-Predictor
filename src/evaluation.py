import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import os

def evaluate_models(ticker: str):
    """
    Evaluate model predictions and generate visualizations.
    """
    pred_file = f"data/processed_data/{ticker}_predictions.csv"
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Missing predictions at {pred_file}")
        
    df = pd.read_csv(pred_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. Print Metrics
    print(f"--- Evaluation for {ticker} ---")
    
    y_true = df['target']
    y_pred_rf = df['rf_prediction']
    
    acc = accuracy_score(y_true, y_pred_rf)
    prec = precision_score(y_true, y_pred_rf)
    rec = recall_score(y_true, y_pred_rf)
    f1 = f1_score(y_true, y_pred_rf)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    
    # 2. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred_rf)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.title('Random Forest Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'plots/{ticker}_confusion_matrix.png')
    plt.close()
    
    # 3. Stock Price Movement vs Predictions
    plt.figure(figsize=(12, 6))
    
    # Plot real prices
    plt.plot(df['date'], df['Close'], label='Close Price', color='black', linewidth=1.5)
    
    # Highlight points where model predicted UP (green) vs DOWN (red)
    up_preds = df[df['rf_prediction'] == 1]
    down_preds = df[df['rf_prediction'] == 0]
    
    plt.scatter(up_preds['date'], up_preds['Close'], color='green', label='Predicted UP', marker='^', alpha=0.7)
    plt.scatter(down_preds['date'], down_preds['Close'], color='red', label='Predicted DOWN', marker='v', alpha=0.7)
    
    plt.title(f'{ticker} Stock Price and Model Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/{ticker}_price_predictions.png')
    plt.close()
    
    # 4. Sentiment Scores over Time
    plt.figure(figsize=(12, 4))
    # Standardize to fit nicely if we want to plot alongside price, or just plot raw sentiment
    plt.bar(df['date'], df['sentiment_score'], color=['green' if x > 0 else 'red' for x in df['sentiment_score']])
    plt.axhline(0, color='black', linewidth=1)
    
    plt.title(f'{ticker} Daily News Sentiment Score')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/{ticker}_sentiment_scores.png')
    plt.close()
    
    print(f"Saved evaluation charts to 'plots/' directory.")

if __name__ == "__main__":
    ticker = "AAPL"
    try:
        evaluate_models(ticker)
    except Exception as e:
        print(f"Error during evaluation: {e}")
