import utils
import torch
import torch.nn as nn
from difftransformer import DifferentialTransformerClassifier


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters, best ones obtained from optuna finetuning
vocab_size = utils.tokenizer.vocab_size()
depth = 5
n_embd = 144
n_head = 6
batch_size = 32
dropout = 0.014500254910782884

model = DifferentialTransformerClassifier(
    vocab_size=vocab_size,
    embedding_dim=n_embd,  
    num_heads=n_head,
    depth=depth,
    dropout = dropout
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00011696142086951537, weight_decay=9.168589372978195e-05)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

dataset, period_mapping = utils.prepare_dataset(utils.df, train=True)
dataloader = utils.prepare_dataloader(dataset, batch_size=batch_size)

print("Dataloader ready")

# Training loop
epochs = 25
checkpoint_path = "/users/eleves-a/2022/amine.chraibi/KaggleINF/best_model_no_tc.pth"

for epoch in range(epochs):
    loss = utils.train(model, dataloader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    scheduler.step()
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"/users/eleves-a/2022/amine.chraibi/KaggleINF/model_checkpoint_{epoch+1}.pth")
        print(f"Checkpoint saved at epoch {epoch + 1}.")

# Final save
torch.save(model.state_dict(), checkpoint_path)
print(f"Final model saved to {checkpoint_path}.")
