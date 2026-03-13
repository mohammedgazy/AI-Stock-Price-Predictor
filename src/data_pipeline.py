import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import random

# Ensure directories exist
os.makedirs("data/raw_data", exist_ok=True)
os.makedirs("data/processed_data", exist_ok=True)


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical stock data using yfinance and save raw + processed versions.
    """
    print(f"Fetching historical data for {ticker} from {start_date} to {end_date}...")
    stock = yf.download(ticker, start=start_date, end=end_date)

    # Flatten columns in case yfinance returns multi-index columns
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.get_level_values(0)

    # Reset index so Date becomes a normal column
    stock = stock.reset_index()

    # Save raw data
    raw_file_path = f"data/raw_data/{ticker}_prices.csv"
    stock.to_csv(raw_file_path, index=False)
    print(f"Saved stock data to {raw_file_path}")

    # Basic processed dataset
    processed = stock.copy()

    # Standardize date column name
    if "Date" not in processed.columns and "date" in processed.columns:
        processed.rename(columns={"date": "Date"}, inplace=True)

    # Drop rows with missing values
    processed = processed.dropna()

    # Save processed data
    processed_file_path = f"data/processed_data/{ticker}_processed.csv"
    processed.to_csv(processed_file_path, index=False)
    print(f"Saved processed data to {processed_file_path}")

    return processed


def fetch_stock_news(ticker: str) -> list:
    """
    Fetch recent news headlines for a stock using yfinance.
    """
    print(f"Fetching recent news for {ticker}...")
    stock = yf.Ticker(ticker)
    news = stock.news

    file_path = f"data/raw_data/{ticker}_news.json"
    with open(file_path, "w") as f:
        json.dump(news, f, indent=4)

    print(f"Saved news to {file_path}")
    return news


def simulate_historical_news(ticker: str, start_date: str, end_date: str, num_articles_per_day: int = 3):
    """
    Simulate historical news headlines for demonstration purposes.
    """
    print(f"Generating simulated historical news for {ticker}...")
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    current = start
    simulated_news = []

    positive_templates = [
        "{ticker} beats Q3 earnings expectations by a wide margin.",
        "Analysts upgrade {ticker} following strong product launch.",
        "{ticker} announces record breaking revenue for the quarter."
    ]
    negative_templates = [
        "{ticker} faces supply chain disruptions, impacting production.",
        "Investors sell off {ticker} amidst regulatory concerns.",
        "Disappointing guidance from {ticker} leads to stock drop."
    ]
    neutral_templates = [
        "{ticker} announces date for upcoming shareholder meeting.",
        "Market remains cautious on {ticker} ahead of earnings report.",
        "{ticker} appoints new board member."
    ]

    random.seed(42)

    while current <= end:
        if current.weekday() < 5:
            for _ in range(num_articles_per_day):
                sentiment_type = random.choice(["positive", "negative", "neutral"])
                if sentiment_type == "positive":
                    headline = random.choice(positive_templates).format(ticker=ticker)
                elif sentiment_type == "negative":
                    headline = random.choice(negative_templates).format(ticker=ticker)
                else:
                    headline = random.choice(neutral_templates).format(ticker=ticker)

                simulated_news.append({
                    "uuid": f"mock-{random.randint(1000, 9999)}",
                    "title": headline,
                    "publisher": "Mock Financial News",
                    "link": "https://example.com",
                    "providerPublishTime": int(current.timestamp()),
                    "type": "ARTICLE",
                    "relatedTickers": [ticker]
                })
        current += timedelta(days=1)

    file_path = f"data/raw_data/{ticker}_simulated_news.json"
    with open(file_path, "w") as f:
        json.dump(simulated_news, f, indent=4)

    print(f"Saved {len(simulated_news)} simulated news articles to {file_path}")
    return simulated_news


if __name__ == "__main__":
    ticker = "AAPL"
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    fetch_stock_data(ticker, start_date=start_date, end_date=end_date)
    fetch_stock_news(ticker)
    simulate_historical_news(ticker, start_date=start_date, end_date=end_date)
