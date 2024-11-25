import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import pickle
import random
import math
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
        match_labels = self.labels[idx].to(torch.bfloat16)  # (num_periods,)
        return match_features, match_labels

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
MAX_TWEET_LENGTH = 44
DATA_FILE = "/users/eleves-a/2022/amine.chraibi/KaggleINF/preprocessed_data.pkl"

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

set_seed(42)

def prepare_and_save_data():
    li = []
    for filename in os.listdir("/users/eleves-a/2022/amine.chraibi/KaggleINF/train_tweets"):
        df = pd.read_csv(f"/users/eleves-a/2022/amine.chraibi/KaggleINF/train_tweets/{filename}")
        li.append(df)
    df = pd.concat(li, ignore_index=True)
    df['Tweet'] = df['Tweet'].apply(preprocess_text)
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(df, f)
    return df

def load_or_prepare_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'rb') as f:
            return pickle.load(f)
    return prepare_and_save_data()

def prepare_tokenizer(df):
    all_tweets = df['Tweet'].values
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(all_tweets)
    return tokenizer

print("Preparing dataset")
df = load_or_prepare_data()
tokenizer = prepare_tokenizer(df)
print("Data and tokenizer ready")

all_tweets = df['Tweet'].values
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(all_tweets)

MAX_TWEET_LENGTH = 44  # Maximum tweet length in tokens

grouped_tweets = df.groupby(['MatchID', 'PeriodID'])['Tweet'].apply(list).unstack(fill_value=[])
grouped_labels = df.groupby(['MatchID', 'PeriodID'])['EventType'].max().unstack(fill_value=0)

def pad_tweet(tokens, max_length=MAX_TWEET_LENGTH):
    """Pad or truncate tokens to max_length."""
    if len(tokens) < max_length:
        return tokens + [tokenizer.word2idx['<PAD>']] * (max_length - len(tokens))
    else:
        return tokens[:max_length]


def sample_tweets_or_pad(period):
    """
    Select all tweets in the period and pad each tweet to MAX_TWEET_LENGTH.
    If the period is empty, return a single <PAD> tweet.
    """
    if len(period) == 0:  # If the period is empty
        return [[tokenizer.word2idx['<PAD>']] * MAX_TWEET_LENGTH]

    # Tokenize and pad all tweets in the period
    padded_tweets = [pad_tweet(tokenizer(tweet)) for tweet in period]

    return padded_tweets

def tokenize_and_sample_grouped_tweets(grouped_tweets):
    tokenized_periods = []
    for (match_id,period_id), tweets in grouped_tweets.stack().items(): 
        tokenized_period = [pad_tweet(tokenizer(tweet)) for tweet in tweets]
        tokenized_periods.append(tokenized_period)
    return tokenized_periods

def collate_fn(batch):
    """
    Custom collate function to dynamically pad tweets to the maximum number of tweets
    per period within the batch.
    """
    features, labels = zip(*batch)  # Separate features and labels
    features = [
        period if len(period) > 0 else torch.tensor([[tokenizer.word2idx['<PAD>']] * MAX_TWEET_LENGTH], dtype=torch.long)
        for period in features
    ]
    # Find the max number of tweets per period in the batch
    max_tweets_per_period = max(len(period) for period in features)
    # Pad tweets dynamically
    padded_features = []
    for period in features:
        # Pad the period to max_tweets_per_period
        padded_period = period.tolist() + [[tokenizer.word2idx['<PAD>']] * MAX_TWEET_LENGTH] * (max_tweets_per_period - len(period))
        padded_features.append(torch.tensor(padded_period, dtype=torch.long))
    
    # Convert labels to tensor
    batch_labels = torch.tensor(labels, dtype=torch.bfloat16)  # (batch_size,)
    
    # Stack the padded features
    batch_features = torch.stack(padded_features)  # (batch_size, max_tweets_per_period, MAX_TWEET_LENGTH)

    return batch_features, batch_labels


# Dataset class
class TweetDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        match_features = torch.tensor(self.features[idx], dtype=torch.long)  # (num_periods, NUM_TWEETS_PER_PERIOD, MAX_TWEET_LENGTH)
        match_labels = self.labels[idx].to(torch.bfloat16)  # (num_periods,)
        return match_features, match_labels


def prepare_dataset():
    grouped_tweets = df.groupby(['MatchID', 'PeriodID'])['Tweet'].apply(list).unstack(fill_value=[])
    grouped_labels = df.groupby(['MatchID', 'PeriodID'])['EventType'].max().unstack(fill_value=0)
    tokenized_and_padded_tweets = tokenize_and_sample_grouped_tweets(grouped_tweets)
    labels = grouped_labels.fillna(0).values.tolist()
    labels= torch.tensor(labels)
    labels = labels.view(-1)
    dataset = TweetDataset(tokenized_and_padded_tweets, labels)
    return dataset

def prepare_dataloader(dataset):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return dataloader

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for tweets, labels in dataloader:
        tweets = tweets.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(tweets)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)