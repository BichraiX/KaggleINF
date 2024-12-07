import os
import re
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel

# Ensure these downloads if not done yet:
# nltk.download('stopwords')
# nltk.download('wordnet')

### Classes
class HierarchicalTweetDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'label': self.labels[idx]
        }
        
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weight = nn.Linear(input_dim, 1, bias=False)

    def forward(self, embeddings, mask=None):
        # embeddings: (batch_size, seq_len, input_dim)
        # mask: (batch_size, seq_len)
        att_scores = self.attention_weight(embeddings).squeeze(-1)  # (batch_size, seq_len)
        if mask is not None:
            att_scores = att_scores.masked_fill(mask == 0, -1e9)
        att_weights = torch.softmax(att_scores, dim=-1)  # (batch_size, seq_len)
        att_output = torch.sum(embeddings * att_weights.unsqueeze(-1), dim=1)  # (batch_size, input_dim)
        return att_output, att_weights
    
class HierarchicalBertModel(nn.Module):
    def __init__(self, 
                 bert_model_name='bert-base-uncased', 
                 hidden_dim=768, 
                 dropout=0.1,
                 freeze_bert=True):
        super(HierarchicalBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.token_attention = AttentionLayer(hidden_dim)
        self.tweet_attention = AttentionLayer(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        if freeze_bert:
            # Freeze all BERT parameters to speed up training
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_masks, labels=None):
        # input_ids: (batch, num_tweets, max_tweet_len)
        # attention_masks: (batch, num_tweets, max_tweet_len)
        bsz, num_tweets, max_len = input_ids.shape
        flat_input_ids = input_ids.view(bsz*num_tweets, max_len)
        flat_att_mask = attention_masks.view(bsz*num_tweets, max_len)

        with torch.no_grad(): # no grad if we froze BERT to reduce overhead
            outputs = self.bert(flat_input_ids, attention_mask=flat_att_mask)
        token_embeddings = outputs.last_hidden_state
        # Token-level attention to get tweet-level embeddings
        tweet_embeddings, _ = self.token_attention(token_embeddings, flat_att_mask)

        # Reshape to (bsz, num_tweets, hidden_dim)
        hidden_dim = tweet_embeddings.size(-1)
        tweet_embeddings = tweet_embeddings.view(bsz, num_tweets, hidden_dim)

        # Create a tweet-level mask: a tweet is considered padding if all input_ids are 0
        with torch.no_grad():
            input_ids_reshaped = flat_input_ids.view(bsz, num_tweets, max_len)
            tweet_existence_mask = (input_ids_reshaped != 0).any(dim=-1).long()  # (bsz, num_tweets)

        period_embedding, _ = self.tweet_attention(tweet_embeddings, tweet_existence_mask)
        logits = self.classifier(period_embedding)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        return logits


###################################################
# Constants
###################################################
DATA_FILE = "/users/eleves-a/2022/amine.chraibi/KaggleINF/preprocessed_data.pkl"
BERT_MODEL_NAME = "bert-base-uncased"
MAX_TWEET_LENGTH = 44  
NUM_TWEET = 500       
TEST_SIZE = 0.05
stop_words = set(stopwords.words('english'))

###################################################
# Text Preprocessing
###################################################
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

def prepare_and_save_data():
    files = os.listdir("/users/eleves-a/2022/amine.chraibi/KaggleINF/train_tweets")
    li = []
    for filename in files:
        df_tmp = pd.read_csv(f"/users/eleves-a/2022/amine.chraibi/KaggleINF/train_tweets/{filename}")
        li.append(df_tmp)
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

###################################################
# Tokenizer
###################################################
def get_tokenizer():
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    return tokenizer

def pad_tweets_bert(tweets, tokenizer, max_length=MAX_TWEET_LENGTH):
    # Tokenize a batch of tweets at once
    encoded = tokenizer(tweets,
                        truncation=True,
                        padding='max_length',
                        max_length=max_length,
                        return_tensors='pt')
    return encoded['input_ids'], encoded['attention_mask']

def pad_empty_tweet_batch(num=1, max_length=MAX_TWEET_LENGTH):
    # Represent empty tweets as all-zero
    input_ids = torch.zeros((num, max_length), dtype=torch.long)
    attention_mask = torch.zeros((num, max_length), dtype=torch.long)
    return input_ids, attention_mask

def prepare_dataset():
    """
    Prepares the dataset and returns train_dataset and test_dataset.
    The caller can then create DataLoaders and DistributedSamplers as needed.
    """
    df = load_or_prepare_data()
    tokenizer = get_tokenizer()

    grouped = df.groupby(['MatchID', 'PeriodID'])

    subperiod_input_ids = []
    subperiod_attention_masks = []
    subperiod_labels = []

    period_to_subperiod_mapping = {}
    subperiod_index = 0

    for (match_id, period_id), group_df in grouped:
        tweets = group_df['Tweet'].tolist()
        label = group_df['EventType'].max()  # label for the entire period

        if len(tweets) == 0:
            # No tweets, just one empty tweet
            input_ids, attention_mask = pad_empty_tweet_batch()
        else:
            # Tokenize all tweets at once
            input_ids, attention_mask = pad_tweets_bert(tweets, tokenizer)

        total_tweets = input_ids.size(0)
        padding_needed = (NUM_TWEET - (total_tweets % NUM_TWEET)) % NUM_TWEET

        if padding_needed > 0:
            pad_ids, pad_masks = pad_empty_tweet_batch(padding_needed, MAX_TWEET_LENGTH)
            input_ids = torch.cat([input_ids, pad_ids], dim=0)
            attention_mask = torch.cat([attention_mask, pad_masks], dim=0)

        num_subperiods = input_ids.size(0) // NUM_TWEET
        # Reshape to (num_subperiods, NUM_TWEET, MAX_TWEET_LENGTH)
        input_ids = input_ids.view(num_subperiods, NUM_TWEET, MAX_TWEET_LENGTH)
        attention_mask = attention_mask.view(num_subperiods, NUM_TWEET, MAX_TWEET_LENGTH)

        subperiod_input_ids.append(input_ids)
        subperiod_attention_masks.append(attention_mask)
        subperiod_labels.extend([label]*num_subperiods)

        period_to_subperiod_mapping[(match_id, period_id)] = list(range(subperiod_index, subperiod_index + num_subperiods))
        subperiod_index += num_subperiods

    subperiod_input_ids = torch.cat(subperiod_input_ids, dim=0)            
    subperiod_attention_masks = torch.cat(subperiod_attention_masks, dim=0)
    subperiod_labels = torch.tensor(subperiod_labels, dtype=torch.long)

    # Train/Test Split
    X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
        subperiod_input_ids, subperiod_attention_masks, subperiod_labels, test_size=TEST_SIZE, random_state=42
    )
    train_dataset = HierarchicalTweetDataset(X_train_ids, X_train_mask, y_train)
    test_dataset = HierarchicalTweetDataset(X_test_ids, X_test_mask, y_test)
    return train_dataset, test_dataset
