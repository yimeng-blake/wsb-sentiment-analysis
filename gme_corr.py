import pandas as pd

def clean_market_data(df):
    df.columns = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"].str.replace(r"[^\d.]", "", regex=True))
    df["volume"] = pd.to_numeric(df["volume"].str.replace(r"[^\d.]", "", regex=True))
    df["percent_change"] = df["close"].pct_change() * 100
    return df

def merge_data(sentiment_data, market_data):
    sentiment_data["date"] = pd.to_datetime(sentiment_data["date"])
    return pd.merge(sentiment_data, market_data, on="date")

def calculate_correlation(combined_data):
    return combined_data["weighted_average_sentiment"].corr(combined_data["percent_change"])

def main():
    gme_sentiment = pd.read_csv("../daily_sentiment_scores_gme.csv")
    gme_market = pd.read_csv("../GME_market_data.csv") # my code path
    gme_market_clean = clean_market_data(gme_market)
    gme_combined = merge_data(gme_sentiment, gme_market_clean)
    gme_correlation = calculate_correlation(gme_combined)
    print(f"GME Correlation: {gme_correlation}")

if __name__ == "__main__":
    main()
