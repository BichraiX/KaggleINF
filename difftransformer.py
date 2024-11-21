import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, x):
        return self.embedding(x)
    
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key_1 = nn.Linear(n_embd, head_size, bias=False)
        self.query_1 = nn.Linear(n_embd, head_size, bias=False)
        self.key_2 = nn.Linear(n_embd, head_size, bias=False)
        self.query_2 = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.lamb = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lamb):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k_1 = self.key_1(x)   # (B,T,hs)
        q_1 = self.query_1(x) # (B,T,hs)
        k_2 = self.key_2(x)   # (B,T,hs)
        q_2 = self.query_2(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei_1 = q_1 @ k_1.transpose(-2,-1) * k_1.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei_2 = q_2 @ k_2.transpose(-2,-1) * k_2.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei_1 = F.softmax(wei_1, dim=-1) # (B, T, T)
        wei_2 = F.softmax(wei_2, dim=-1) # (B, T, T)
        wei_1 = self.dropout(wei_1)
        wei_2 = self.dropout(wei_2)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        wei = wei_1  - lamb * wei_2
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadDifferentialAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, lambda_init=0.8):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.GroupNorm(n_embd)
        self.lamb = nn.Parameter(torch.tensor(lambda_init))


    def forward(self, x):
        out = torch.cat([h(x, self.lamb) for h in self.heads], dim=-1)
        out = self.norm(out)
        out = (1-self.lamb)*out
        out = self.dropout(self.proj(out)) 
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadDifferentialAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Define the tweet encoder
class TweetEncoder(nn.Module):
    def __init__(self, embedding_dim, n_heads, depth):
        super(TweetEncoder, self).__init__()
        self.layers = nn.ModuleList([Block(embedding_dim, n_heads) for _ in range(depth)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
# Define the period encoder
class PeriodEncoder(nn.Module):
    def __init__(self, embedding_dim, n_heads, depth):
        super(PeriodEncoder, self).__init__()
        self.layers = nn.ModuleList([Block(embedding_dim, n_heads) for _ in range(depth)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Define the classification head
class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(embedding_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))
    

# Define the full model
class DifferentialTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_heads, depth):
        super(DifferentialTransformerClassifier, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, embedding_dim)
        self.tweet_encoder = TweetEncoder(embedding_dim, n_heads, depth)
        self.period_encoder = PeriodEncoder(embedding_dim, n_heads, depth)
        self.classifier = ClassificationHead(embedding_dim)
    
    def forward(self, tweets):
        # tweets: list of lists of token IDs (batch_size x num_tweets x tweet_length)
        batch_size = len(tweets)
        num_tweets = len(tweets[0])
        
        # Flatten the tweets for embedding
        tweets_flat = tweets.view(-1, tweets.size(-1))  # (batch_size * num_tweets) x tweet_length
        x = self.embedding(tweets_flat)  # (batch_size * num_tweets) x tweet_length x embedding_dim
        
        # Encode each tweet
        x = self.tweet_encoder(x)  # (batch_size * num_tweets) x tweet_length x embedding_dim
        
        # Pool over tweet tokens
        x = x.mean(dim=1)  # (batch_size * num_tweets) x embedding_dim
        
        # Reshape back to periods
        x = x.view(batch_size, num_tweets, -1)  # batch_size x num_tweets x embedding_dim
        
        # Encode the period
        x = self.period_encoder(x)  # batch_size x num_tweets x embedding_dim
        
        # Pool over tweets
        period_embedding = x.mean(dim=1)  # batch_size x embedding_dim
        
        # Classification
        out = self.classifier(period_embedding)  # batch_size x 1
        return out.squeeze()