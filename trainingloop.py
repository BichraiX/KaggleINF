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
from difftransformer import DifferentialTransformerClassifier, EmbeddingLayer

# Basic preprocessing function
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
    stop_words = set(stopwords.words('english'))
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

def get_tweet_embedding(tweet, embeddings_model, vector_size):
    """
    Convert a tweet into a sequence of embeddings.
    """
    tokens = tweet.lower().split()
    embeddings = []
    for token in tokens:
        embedding = embeddings_model.get(token)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            embeddings.append(np.zeros(vector_size))  # Handle unknown words
    return embeddings

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
    
class PeriodDataset(Dataset):
    def __init__(self, data, tokenizer, max_tweet_length, max_tweets_per_period):
        self.data = data  # List of periods, each with 'tweets' and 'label'
        self.tokenizer = tokenizer
        self.max_tweet_length = max_tweet_length
        self.max_tweets_per_period = max_tweets_per_period
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        period = self.data.iloc[idx]
        tweets = period['Tweet'][:self.max_tweets_per_period]
        label = period['EventType']
        
        # Tokenize and pad tweets
        tokenized_tweets = []
        for tweet in tweets:
            tokens = self.tokenizer(tweet)
            tokens = tokens[:self.max_tweet_length]
            padding = [0] * (self.max_tweet_length - len(tokens))
            tokenized_tweets.append(tokens + padding)
        
        # Pad the number of tweets if necessary
        num_padding_tweets = self.max_tweets_per_period - len(tokenized_tweets)
        if num_padding_tweets > 0:
            tokenized_tweets.extend([[0] * self.max_tweet_length] * num_padding_tweets)
        
        tweets_tensor = torch.tensor(tokenized_tweets, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float)
        
        return tweets_tensor, label_tensor
# Group by periodId and count the number of tweets per period
tweets_per_period = df.groupby('PeriodID')['Tweet'].count()

# Find the maximum number of tweets in any period
max_tweets_per_period = tweets_per_period.max()

# Calculate the number of words in each tweet
df['word_count'] = df['Tweet'].apply(lambda x: len(str(x).split()))

# Find the maximum number of words in any tweet
max_words_per_tweet = df['word_count'].max()

# Display the results
print("Maximum number of tweets per period:", max_tweets_per_period)
print("Maximum number of words in a tweet:", max_words_per_tweet)

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

# Prepare data

all_tweets = df['Tweet'].values
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(all_tweets)

vocab_size = tokenizer.vocab_size()
embedding_dim = 384
n_heads = 6
depth = 6
max_tweet_length = 44
max_tweets_per_period = 57880

dataset = PeriodDataset(df, tokenizer, max_tweet_length, max_tweets_per_period)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DifferentialTransformerClassifier(vocab_size, embedding_dim, n_heads, depth)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

# Training loop
epochs = 10
for epoch in range(epochs):
    loss = train(model, dataloader, optimizer, criterion, device)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')

