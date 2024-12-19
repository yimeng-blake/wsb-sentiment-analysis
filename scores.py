import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("filtered_dataset_ko.csv")  
sentiment_scores = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

data["timestamp"] = pd.to_datetime(data["timestamp"], errors='coerce') 
data = data.dropna(subset=["timestamp"]) 

data["date"] = data["timestamp"].dt.date

start_date = pd.to_datetime("2021-01-01").date()
data = data[data["date"] >= start_date]

data["sentiment_score"] = data["SentimentLabel"].map(sentiment_scores)

data["weighted_score"] = data["sentiment_score"] * data["Confidence"]

daily_stats = (
    data.groupby("date")
    .agg(
        weighted_average_sentiment=("weighted_score", lambda x: x.sum() / data.loc[x.index, "Confidence"].sum()),
        comment_count=("sentiment_score", "size")  
    )
    .reset_index()
)

output_file = "daily_sentiment_scores_ko.csv"
daily_stats.to_csv(output_file, index=False)

print(f"Daily weighted sentiment scores and comment counts saved to {output_file}")

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(
    daily_stats["date"],
    daily_stats["weighted_average_sentiment"],
    color="blue",
    marker="o",
    label="Weighted Average Sentiment"
)
ax1.set_xlabel("Date")
ax1.set_ylabel("Weighted Average Sentiment", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.set_title("Daily Weighted Sentiment Score and Comment Count for KO Stock (Starting Jan 2021)")

ax2 = ax1.twinx()
ax2.plot(
    daily_stats["date"],
    daily_stats["comment_count"],
    color="orange",
    marker="x",
    label="Comment Count"
)
ax2.set_ylabel("Comment Count", color="orange")
ax2.tick_params(axis="y", labelcolor="orange")

ax1.grid()
fig.tight_layout()
plt.show()
