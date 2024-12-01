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
from sklearn.model_selection import train_test_split

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
    

# Dataset class
class TweetDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        match_features = torch.tensor(self.features[idx], dtype=torch.long)  # (num_periods, NUM_TWEETS_PER_PERIOD, MAX_TWEET_LENGTH)
        match_labels = self.labels[idx]  # (num_periods,)
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

PERIOD_LENGTH = 300  # Global variable for period length

def pad_tweet(tokens, max_length=MAX_TWEET_LENGTH):
    """Pad or truncate tokens to max_length."""
    if len(tokens) < max_length:
        return tokens + [tokenizer.word2idx['<PAD>']] * (max_length - len(tokens))
    else:
        return tokens[:max_length]

def split_and_pad_period(period, period_length):
    """
    Split a period into subperiods of size `period_length`, padding as necessary.
    """
    total_tweets = len(period)
    padding_needed = (period_length - (total_tweets % period_length)) % period_length
    padded_period = period + [[tokenizer.word2idx['<PAD>']] * MAX_TWEET_LENGTH] * padding_needed

    # Split into subperiods of size period_length
    subperiods = [
        padded_period[i:i + period_length]
        for i in range(0, len(padded_period), period_length)
    ]

    return subperiods

def tokenize_and_process_grouped_tweets(grouped_tweets):
    """
    Tokenize and process grouped tweets into subperiods of a fixed length.
    """
    tokenized_subperiods = []
    period_to_subperiod_mapping = {}
    subperiod_index = 0

    for (match_id, period_id), tweets in grouped_tweets.stack().items():
        # Tokenize and pad the tweets in the period
        tokenized_period = [pad_tweet(tokenizer(tweet)) for tweet in tweets]

        # Split the period into subperiods
        subperiods = split_and_pad_period(tokenized_period, PERIOD_LENGTH)

        # Map the period_id to the indices of its subperiods
        period_to_subperiod_mapping[(match_id, period_id)] = list(
            range(subperiod_index, subperiod_index + len(subperiods))
        )
        subperiod_index += len(subperiods)

        # Add subperiods to the dataset
        tokenized_subperiods.extend(subperiods)

    return tokenized_subperiods, period_to_subperiod_mapping

def prepare_dataset(df, train=True):
    """
    The function `prepare_dataset` processes a DataFrame of tweets and labels, grouping them by MatchID
    and PeriodID, tokenizing and padding the tweets, and optionally flattening the labels for training.
    
    :param df: The `df` parameter in the `prepare_dataset` function is a DataFrame that contains the
    data for training or testing. It likely includes columns such as 'MatchID', 'PeriodID', 'Tweet', and
    'EventType' among others. The function processes this DataFrame to prepare the dataset for training
    or
    :param train: The `train` parameter in the `prepare_dataset` function is a boolean flag that
    indicates whether the dataset is being prepared for training or inference. When `train=True`, the
    function processes the labels along with the tweets data to prepare the dataset for training. When
    `train=False`, the function only processes, defaults to True (optional)
    :return: The function `prepare_dataset` returns the dataset and the period_to_subperiod_mapping. The
    specific return value depends on whether the `train` parameter is set to True or False:
    """
    grouped_tweets = df.groupby(['MatchID', 'PeriodID'])['Tweet'].apply(list).unstack(fill_value=[])
    tokenized_and_padded_tweets, period_to_subperiod_mapping = tokenize_and_process_grouped_tweets(grouped_tweets)
    
    if train:
        grouped_labels = df.groupby(['MatchID', 'PeriodID'])['EventType'].max().unstack(fill_value=0)

        # Flatten the labels to match subperiods
        labels = []
        for (match_id, period_id), label in grouped_labels.stack().items():
            subperiod_indices = period_to_subperiod_mapping[(match_id, period_id)]
            labels.extend([label] * len(subperiod_indices))
        labels = torch.tensor(labels, dtype=torch.bfloat16)
        dataset = TweetDataset(tokenized_and_padded_tweets, labels)
    else:
        dataset = tokenized_and_padded_tweets
    return dataset, period_to_subperiod_mapping

def prepare_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def split_data(df, test_size=0.2, random_state=42):
    """
    Split the data into training and testing datasets.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['EventType'])
    return train_df, test_df


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

# Function to evaluate the model on the test set
def evaluate(model, test_dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation during evaluation
        for tweets, labels in test_dataloader:
            tweets = tweets.to(device)
            labels = labels.to(device)
            outputs = model(tweets)

            # Convert probabilities to binary predictions
            predicted = (outputs > 0.5).float()  # Threshold at 0.5 for binary classification

            # Calculate accuracy
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total  # Accuracy = correct predictions / total predictions
    return accuracy
