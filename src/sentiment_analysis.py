import json
import pandas as pd
from transformers import pipeline
from datetime import datetime
import os
import torch

def load_sentiment_model():
    """
    Load the FinBERT model for sentiment analysis.
    """
    print("Loading FinBERT model...")
    device = 0 if torch.cuda.is_available() else -1
    # We use a widely known financial sentiment model
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)
    return sentiment_pipeline

def process_news_sentiment(ticker: str, news_file: str, model_pipeline) -> pd.DataFrame:
    """
    Process a JSON file of news articles and extract sentiment.
    """
    print(f"Processing sentiment for {news_file}...")
    with open(news_file, "r") as f:
        news_data = json.load(f)
        
    records = []
    
    for article in news_data:
        # Some APIs return title, some return content. Try title first
        text = article.get('title', article.get('content', ''))
        if not text:
            continue
            
        timestamp = article.get('providerPublishTime')
        if not timestamp:
            continue
            
        # Convert timestamp to date
        dt = datetime.fromtimestamp(timestamp)
        date_str = dt.strftime("%Y-%m-%d")
        
        # Determine sentiment
        result = model_pipeline(text)[0]
        label = result['label']
        score = result['score']
        
        # Standardize score based on label polarity
        # FinBERT labels: positive, negative, neutral
        sentiment_score = 0.0
        if label == 'positive':
            sentiment_score = score
        elif label == 'negative':
            sentiment_score = -score
        else: # neutral
            sentiment_score = 0.0 # Neutral has 0 impact
            
        records.append({
            'date': date_str,
            'ticker': ticker,
            'headline': text,
            'sentiment_label': label,
            'confidence': score,
            'sentiment_score': sentiment_score
        })
        
    df = pd.DataFrame(records)
    return df

def aggregate_daily_sentiment(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Aggregate sentiment scores by day.
    """
    if df.empty:
        print("Warning: No sentiment data to aggregate.")
        return pd.DataFrame(columns=['date', 'sentiment_score', 'article_count'])
        
    print(f"Aggregating daily sentiment for {ticker}...")
    
    # Calculate daily average sentiment and article count
    daily_agg = df.groupby('date').agg(
        sentiment_score=('sentiment_score', 'mean'),
        article_count=('sentiment_score', 'count')
    ).reset_index()
    
    daily_agg['date'] = pd.to_datetime(daily_agg['date'])
    
    file_path = f"data/processed_data/{ticker}_daily_sentiment.csv"
    daily_agg.to_csv(file_path, index=False)
    print(f"Saved aggregated sentiment to {file_path}")
    
    return daily_agg

if __name__ == "__main__":
    ticker = "AAPL"
    
    # Check if simulated news exists, otherwise fallback to real news
    simulated_news_file = f"data/raw_data/{ticker}_simulated_news.json"
    real_news_file = f"data/raw_data/{ticker}_news.json"
    
    target_file = simulated_news_file if os.path.exists(simulated_news_file) else real_news_file
    
    if os.path.exists(target_file):
        sentiment_pipe = load_sentiment_model()
        sentiment_df = process_news_sentiment(ticker, target_file, sentiment_pipe)
        
        # Save raw sentiments
        sentiment_df.to_csv(f"data/processed_data/{ticker}_article_sentiments.csv", index=False)
        
        # Aggregate
        aggregate_daily_sentiment(sentiment_df, ticker)
    else:
        print(f"Could not find news data at {target_file}. Run data_pipeline.py first.")
