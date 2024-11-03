# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download NLTK VADER model if not already downloaded
nltk.download('vader_lexicon')

# Load Data
data_path = 'C:\\Users\\Afiya\\Downloads\\prodigy\\sentiment.csv'
df = pd.read_csv(data_path)

# Display column names to verify the correct text column name
print("Columns in dataset:", df.columns)

# Inspect column names
print("Columns in dataset:", df.columns)

# Define the text column name (adjust if the actual column name is different)
text_column = 'tweet'  # Replace 'message' if your text column has a different name

# Check if text column exists
if text_column not in df.columns:
    raise KeyError(f"Column '{text_column}' not found. Please check the column names.")

# Drop rows with missing text data
df = df.dropna(subset=[text_column])

# Initialize VADER Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Define function to categorize sentiment based on compound score
def get_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis
df['sentiment_score'] = df[text_column].apply(lambda x: sid.polarity_scores(x)['compound'])
df['sentiment'] = df['sentiment_score'].apply(get_sentiment)

# Visualize Sentiment Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='sentiment', palette='coolwarm')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()

# Optional: Time-Based Sentiment Analysis (if a timestamp column is available)
if 'timestamp' in df.columns:
    # Convert timestamp to datetime and extract date
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)  # Drop rows with invalid timestamps
    df['date'] = df['timestamp'].dt.date

    # Group by date and sentiment, then count occurrences
    time_sentiment = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
    
    # Plot sentiment over time
    plt.figure(figsize=(12, 6))
    time_sentiment.plot(kind='line', stacked=True, color=['red', 'blue', 'green'], figsize=(15, 8))
    plt.title('Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.legend(title='Sentiment')
    plt.show()

# Display sample data with sentiment classifications
print(df[[text_column, 'sentiment', 'sentiment_score']].head())
