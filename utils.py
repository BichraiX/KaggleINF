import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from torch.utils.data import Dataset, DataLoader

class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx = 2
    
    def build_vocab(self, texts):
        for text in texts:
            words = text.lower().split()
            for word in words:
                if word not in self.word2idx:
                    self.word2idx[word] = self.idx
                    self.idx += 1
    
    def __call__(self, text):
        return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in text.lower().split()]
    
    def vocab_size(self):
        return len(self.word2idx)
    
class TweetDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        match_features = torch.tensor(self.features[idx], dtype=torch.long)  # (num_periods, NUM_TWEETS_PER_PERIOD, MAX_TWEET_LENGTH)
        match_labels = torch.tensor(self.labels[idx], dtype=torch.bfloat16)  # (num_periods,)
        return match_features, match_labels
    

def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenization
    words = text.split()
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def prepare_df():
    nltk.download('stopwords')
    nltk.download('wordnet')
    li = []
    for filename in os.listdir("train_tweets"):
        df = pd.read_csv("train_tweets/" + filename)
        li.append(df)
    df = pd.concat(li, ignore_index=True)
    df['Tweet'] = df['Tweet'].apply(preprocess_text)
    return df

def prepare_tokenizer(df):
    all_tweets = df['Tweet'].values
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(all_tweets)
    return tokenizer

df = prepare_df()
tokenizer = prepare_tokenizer(df)

def pad_tweet(tokens, max_length=MAX_TWEET_LENGTH):
    """Pad or truncate tokens to max_length."""
    if len(tokens) < max_length:
        return tokens + [tokenizer.word2idx['<PAD>']] * (max_length - len(tokens))
    else:
        return tokens[:max_length]


def tokenize_and_pad_grouped_tweets(grouped_tweets):
    """
    Tokenize and pad all tweets in each period without grouping by matches.
    """
    tokenized_periods = []  # Flat list of periods
    for _, periods in grouped_tweets.iterrows():
        for period in periods:
            # Tokenize and pad each tweet in the period
            padded_tweets = [pad_tweet(tokenizer(tweet)) for tweet in period]
            tokenized_periods.append(padded_tweets)
    return tokenized_periods


def collate_fn(batch):
    """
    Custom collate function to dynamically pad tweets to the maximum number of tweets
    per period within the batch.
    """
    features, labels = zip(*batch)  # Separate features and labels
    
    # Find the max number of tweets per period in the batch
    max_tweets_per_period = max(len(period) for period in features)

    # Pad tweets dynamically
    padded_features = []
    for period in features:
        # Pad the period to max_tweets_per_period
        padded_period = period + [[tokenizer.word2idx['<PAD>']] * MAX_TWEET_LENGTH] * (max_tweets_per_period - len(period))
        padded_features.append(torch.tensor(padded_period, dtype=torch.long))
    
    # Convert labels to tensor
    batch_labels = torch.tensor(labels, dtype=torch.bfloat16)  # (batch_size,)
    
    # Stack the padded features
    batch_features = torch.stack(padded_features)  # (batch_size, max_tweets_per_period, MAX_TWEET_LENGTH)

    return batch_features, batch_labels


def prepare_dataset():
    grouped_tweets = df.groupby(['MatchID', 'PeriodID'])['Tweet'].apply(list).unstack(fill_value=[])
    grouped_labels = df.groupby(['MatchID', 'PeriodID'])['EventType'].max().unstack(fill_value=0)

    # Tokenize and pad tweets into a flat list of periods
    tokenized_and_padded_periods = tokenize_and_pad_grouped_tweets(grouped_tweets)

    # Flatten labels to match periods
    labels = grouped_labels.stack().fillna(0).values.tolist()

    # Create dataset
    dataset = TweetDataset(tokenized_and_padded_periods, labels)
    return dataset


