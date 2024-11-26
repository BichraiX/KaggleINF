import utils
import torch
import torch.nn as nn
from difftransformer import DifferentialTransformerClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = utils.tokenizer.vocab_size()
depth = 8
n_embd = 568
n_head = 8
batch_size = 32
model = DifferentialTransformerClassifier(
vocab_size=vocab_size,
embedding_dim=n_embd,  
num_heads=n_head,
depth=depth)
model.to(device)
print("Model Loaded")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = criterion = nn.BCELoss()
dataset, _ = utils.prepare_dataset()
print("Dataset ready")
dataloader = utils.prepare_dataloader(dataset, batch_size = batch_size)
print("Dataloader ready")
epochs = 1000

for epoch in range(epochs):
    loss = utils.train(model, dataloader, optimizer, criterion, device)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')