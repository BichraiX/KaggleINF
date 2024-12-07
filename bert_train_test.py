import os
import re
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Avoid re-downloading resources unnecessarily (comment out if already downloaded)
# nltk.download('stopwords')
# nltk.download('wordnet')

###################################################
# Constants and Setup
###################################################
stop_words = set(stopwords.words('english'))
MAX_TWEET_LENGTH = 44  # max tokens per tweet
NUM_TWEET = 500        # number of tweets per subperiod
DATA_FILE = "/users/eleves-a/2022/amine.chraibi/KaggleINF/preprocessed_data.pkl"
BERT_MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5
TEST_SIZE = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
PIN_MEMORY = True if torch.cuda.is_available() else False

###################################################
# Preprocessing and Setup Functions
###################################################
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

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

df = load_or_prepare_data()

###################################################
# Tokenization with BERT (Batch)
###################################################
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

def pad_tweets_bert(tweets, max_length=MAX_TWEET_LENGTH):
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

###################################################
# Create Subperiods (Vectorized)
###################################################
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
        input_ids, attention_mask = pad_tweets_bert(tweets)

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

###################################################
# Train/Test Split
###################################################
X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
    subperiod_input_ids, subperiod_attention_masks, subperiod_labels, test_size=TEST_SIZE, random_state=42
)

###################################################
# Dataset and DataLoader
###################################################
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

train_dataset = HierarchicalTweetDataset(X_train_ids, X_train_mask, y_train)
test_dataset = HierarchicalTweetDataset(X_test_ids, X_test_mask, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

###################################################
# Model Definition with Optional Freezing
###################################################
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
# Evaluation Function
###################################################
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
            labels = batch['label'].to(DEVICE, non_blocking=True)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0

###################################################
# Training Setup
###################################################
model = HierarchicalBertModel(bert_model_name=BERT_MODEL_NAME, freeze_bert=True).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

###################################################
# Training Loop (Minimized Overhead)
###################################################
best_test_acc = 0.0
model.train()
for epoch in range(0, EPOCHS+1):
    epoch_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
        attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
        labels = batch['label'].to(DEVICE, non_blocking=True)

        loss, _ = model(input_ids, attention_mask, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()

    # Evaluate every epoch or as needed
    if epoch % 2 == 0:  # for example, every 2 epochs
        test_acc = evaluate(model, test_loader)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
    # Minimal print
    print(f"Epoch {epoch}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}, Test Acc: {test_acc:.4f} (Best: {best_test_acc:.4f})")

print(f"Final Best Test Accuracy: {best_test_acc:.4f}")
