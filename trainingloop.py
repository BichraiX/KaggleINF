import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from torch.utils.data import Dataset, DataLoader
from difftransformer import DifferentialTransformerClassifier, EmbeddingLayer


# Basic preprocessing function
stop_words = set(stopwords.words('english'))
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

# Set the seed
set_seed(42)

# Download some NLP models for processing, optional
nltk.download('stopwords')
nltk.download('wordnet')

# Read all training files and concatenate them into one dataframe
li = []
for filename in os.listdir("train_tweets"):
    df = pd.read_csv("train_tweets/" + filename)
    li.append(df)
df = pd.concat(li, ignore_index=True)

# Apply preprocessing to each tweet # 13 min
df['Tweet'] = df['Tweet'].apply(preprocess_text)

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

# Tokenizer and padding setup
all_tweets = df['Tweet'].values
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(all_tweets)

MAX_TWEET_LENGTH = 44  # Maximum tweet length in tokens
NUM_TWEETS_PER_PERIOD = 92  # Fixed number of tweets per period

grouped_tweets = df.groupby(['MatchID', 'PeriodID'])['Tweet'].apply(list).unstack(fill_value=[])
grouped_labels = df.groupby(['MatchID', 'PeriodID'])['EventType'].max().unstack(fill_value=0)

def pad_tweet(tokens, max_length=MAX_TWEET_LENGTH):
    """Pad or truncate tokens to max_length."""
    if len(tokens) < max_length:
        return tokens + [tokenizer.word2idx['<PAD>']] * (max_length - len(tokens))
    else:
        return tokens[:max_length]


def sample_tweets_or_pad(period):
    """Randomly select NUM_TWEETS_PER_PERIOD tweets from the period or pad if empty."""
    if len(period) == 0:  # If the period is empty
        return [[tokenizer.word2idx['<PAD>']] * MAX_TWEET_LENGTH] * NUM_TWEETS_PER_PERIOD

    # Randomly sample NUM_TWEETS_PER_PERIOD tweets or pad if fewer
    sampled_tweets = random.sample(period, min(len(period), NUM_TWEETS_PER_PERIOD))
    padded_tweets = [pad_tweet(tokenizer(tweet)) for tweet in sampled_tweets]

    # If fewer than NUM_TWEETS_PER_PERIOD, pad with <PAD> tweets
    while len(padded_tweets) < NUM_TWEETS_PER_PERIOD:
        padded_tweets.append([tokenizer.word2idx['<PAD>']] * MAX_TWEET_LENGTH)

    return padded_tweets


def tokenize_and_sample_grouped_tweets(grouped_tweets):
    """Tokenize tweets and ensure consistent NUM_TWEETS_PER_PERIOD for each period."""
    tokenized_matches = []
    for _, periods in grouped_tweets.iterrows():
        tokenized_match = [sample_tweets_or_pad(period) for period in periods]
        tokenized_matches.append(tokenized_match)
    return tokenized_matches


# Tokenize and sample tweets
tokenized_and_sampled_tweets = tokenize_and_sample_grouped_tweets(grouped_tweets)
labels = grouped_labels.fillna(0).values.tolist()  # Ensure labels are padded with 0 for missing periods


# Dataset class
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


# Create Dataset and DataLoader
dataset = TweetDataset(tokenized_and_sampled_tweets, labels)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1
vocab_size = tokenizer.vocab_size()
depth = 2
batch_size = 1
n_embd = 162
n_head = 3
dropout = 0.2

model = DifferentialTransformerClassifier(
    vocab_size=vocab_size,
    embedding_dim=n_embd,  # Ensure this matches the dimension used in embeddings
    num_heads=n_head,
    depth=depth
).to(device)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

# Training loop
epochs = 1000
model_save_path = "model.pth"  
for epoch in range(epochs):
    loss = train(model, dataloader, optimizer, criterion, device)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')
    
torch.save(model.state_dict(), model_save_path)
print(f"Training complete. Model saved to {model_save_path}")

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on the given dataloader.
    """
    model.eval()
    total_loss = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for tweets, labels in dataloader:
            tweets = tweets.to(device)
            labels = labels.to(device)
            
            outputs = model(tweets)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Flatten and collect labels and predictions
            all_labels.extend(labels.view(-1).cpu().float().numpy())
            all_predictions.extend((outputs.view(-1).cpu().float().numpy() > 0.5).astype(float))
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_labels, all_predictions
