import torch
import torch.nn as nn
from torch.nn import functional as F

torch.set_default_dtype(torch.bfloat16)
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, x):
        return self.embedding(x)

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, embedding_dim, head_size, dropout):
        super().__init__()
        self.key_1 = nn.Linear(embedding_dim, head_size, bias=False)
        self.query_1 = nn.Linear(embedding_dim, head_size, bias=False)
        self.key_2 = nn.Linear(embedding_dim, head_size, bias=False)
        self.query_2 = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lamb):
        B, T, C = x.shape
        k_1 = self.key_1(x)
        q_1 = self.query_1(x)
        k_2 = self.key_2(x)
        q_2 = self.query_2(x)
        wei_1 = q_1 @ k_1.transpose(-2,-1) * k_1.shape[-1]**-0.5
        wei_2 = q_2 @ k_2.transpose(-2,-1) * k_2.shape[-1]**-0.5
        wei_1 = F.softmax(wei_1, dim=-1)
        wei_2 = F.softmax(wei_2, dim=-1)
        wei_1 = self.dropout(wei_1)
        wei_2 = self.dropout(wei_2)
        v = self.value(x)
        wei = wei_1 - lamb * wei_2
        out = wei @ v
        return out

class MultiHeadDifferentialAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, embedding_dim, num_heads, dropout, lambda_init=0.8):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.heads = nn.ModuleList([Head(embedding_dim, head_size,dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        self.lamb = nn.Parameter(torch.tensor(lambda_init))

    def forward(self, x):
        out = torch.cat([h(x, self.lamb) for h in self.heads], dim=-1)
        out = self.norm(out)
        out = (1 - self.lamb) * out
        out = self.dropout(self.proj(out)) 
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, embedding_dim,dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, embedding_dim, num_heads, dropout, layer):
        # embedding_dim: embedding dimension, num_heads: the number of heads we'd like
        super().__init__()
        lambda_init = 0.8 - 0.6*torch.exp(torch.tensor(-0.3*(layer-1)))
        self.sa = MultiHeadDifferentialAttention(embedding_dim, num_heads, dropout, lambda_init)
        self.ffwd = FeedForward(embedding_dim,dropout)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TweetEncoder(nn.Module):
    """Encodes a sequence of tokens into a single tweet embedding."""
    def __init__(self, embedding_dim, num_heads, depth, dropout):
        super(TweetEncoder, self).__init__()
        self.layers = nn.ModuleList([Block(embedding_dim, num_heads, dropout, l) for l in range(1,depth+1)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Define the period encoder
class PeriodEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, depth, dropout):
        super(PeriodEncoder, self).__init__()
        self.layers = nn.ModuleList([Block(embedding_dim, num_heads, dropout, l) for l in range(1,depth+1)])
    
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
    def __init__(self, vocab_size, embedding_dim, num_heads, depth, dropout):
        super(DifferentialTransformerClassifier, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, embedding_dim)
        self.tweet_encoder = TweetEncoder(embedding_dim, num_heads, depth,dropout)
        self.proj = nn.Linear(embedding_dim, 2 * embedding_dim)  # Linear layer for projection
        self.period_encoder = PeriodEncoder(2*embedding_dim, 2*num_heads, 2*depth, dropout)
        self.classifier = ClassificationHead(2*embedding_dim)
        self.embedding_dim = embedding_dim
    
    def forward(self, features):
        """
        Args:
            features: Tensor of shape (batch_size, num_tweets_per_period, tweet_length)
                        Represents the tokenized and padded tweets grouped by period.
        Returns:
            Output: Tensor of shape (batch_size,)
                    Binary predictions for each period in the batch.
        """
        batch_size, num_tweets_per_period, tweet_length = features.shape # (batch_size, num_tweets_per_period, tweet_length)
        # Flatten for embedding
        tweets_flat = features.view(-1, tweet_length)# (batch_size * num_tweets_per_period, tweet_length)
        x = self.embedding(features)  # (batch_size, num_tweets_per_period, tweet_length, embedding_dim)
        x = x.view(batch_size*num_tweets_per_period,tweet_length,-1) # (batch_size*num_tweets_per_period, tweet_length, embedding_dim)
        # Encode each tweet
        x = self.tweet_encoder(x)  # (batch_size * num_tweets_per_period, tweet_length, embedding_dim)
        x = self.proj(x) # (batch_size * num_tweets_per_period, tweet_length, 2*embedding_dim)
        x = x.mean(dim=1)  # (batch_size * num_tweets_per_period, 2*embedding_dim)
        # Reshape back to periods
        x = x.view(batch_size , num_tweets_per_period, 2*self.embedding_dim)  # (batch_size , num_tweets_per_period, 2*embedding_dim)
        # Encode the periods
        x = self.period_encoder(x)  # (batch_size, num_tweets_per_period, 2*embedding_dim)
        # Pool over periods
        x = x.mean(dim=1)  # (batch_size, 2*embedding_dim)
        
        # Classification
        out = self.classifier(x)  # (batch_size, 1)
        
        # out = out.view(batch_size, num_periods)  # (batch_size, num_periods)
        return out.squeeze(1)