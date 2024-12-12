import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch
import gc
import logging
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from utils import preprocess_text, df
import os

import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


logging.basicConfig(filename='sentiment_analysis_errors.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s:%(message)s')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1024
sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name).to(device)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=sentiment_model,
    tokenizer=sentiment_tokenizer,
    device=device,
    batch_size = batch_size
)


def get_sentiments_in_batches(texts, batch_size=128):
    """
    Process a list of texts in batches and return sentiment scores.

    Parameters:
        texts (list): List of tweet texts.
        batch_size (int): Number of tweets to process in each batch.

    Returns:
        list: List of sentiment scores corresponding to each tweet.
    """
    sentiments = []
    num_batches = len(texts) // batch_size + 1
    for i in tqdm(range(num_batches), desc="Processing Batches"):
        start = i * batch_size
        end = start + batch_size
        batch_texts = texts[start:end]
        if not batch_texts:
            continue
        try:
            results = sentiment_pipeline(batch_texts)
            # Convert sentiment labels to numerical scores
            # Labels for twitter-roberta-base-sentiment are: ['LABEL_0' (negative), 'LABEL_1' (neutral), 'LABEL_2' (positive)]
            batch_sentiments = [
                0 if result['label'] == 'LABEL_0' else 2 if result['label'] == 'LABEL_2' else 1
                for result in results
            ]
        except Exception as e:
            logging.error(f"Error processing batch {i+1}: {e}")
            # Assign a neutral sentiment score in case of an error
            batch_sentiments = [1] * len(batch_texts)
        sentiments.extend(batch_sentiments)
    return sentiments


def process_large_dataframe(df, output_file, batch_size=128, chunk_size=100000):
    """
    Process a large DataFrame by applying sentiment analysis to the 'Tweet' column
    and saving the results incrementally to an output CSV file.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing a 'Tweet' column.
        output_file (str): Path to the output CSV file where processed data will be saved.
        batch_size (int): Number of tweets to process in each batch for sentiment analysis.
        chunk_size (int): Number of tweets to process in each chunk to manage memory usage.

    Returns:
        None
    """
    total_tweets = len(df)
    num_chunks = total_tweets // chunk_size + 1

    print(f"Total tweets: {total_tweets}")
    print(f"Processing in {num_chunks} chunks of up to {chunk_size} tweets each.")

    headers = list(df.columns) + ['Sentiment']
    with open(output_file, 'w') as f_out:
        f_out.write(','.join(headers) + '\n')

    for i in tqdm(range(num_chunks), desc="Processing Chunks"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_tweets)
        chunk = df.iloc[start_idx:end_idx].copy()

        tweets = chunk['Tweet'].tolist()

        chunk_sentiments = get_sentiments_in_batches(tweets, batch_size=batch_size)

        chunk['Sentiment'] = chunk_sentiments

        chunk.to_csv(output_file, mode='a', index=False, header=False, quoting=1)

        del chunk
        del tweets
        del chunk_sentiments
        gc.collect()

    print(f"Processing complete. Sentiment scores saved to '{output_file}'.")


output_csv = 'processed_dataset_with_sentiment.csv'
categorical_cols = ['MatchID', 'PeriodID']

for col in categorical_cols:
    df[col] = df[col].astype('category')

process_large_dataframe(df, output_csv, batch_size=batch_size, chunk_size=200000)
