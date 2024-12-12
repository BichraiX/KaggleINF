import utils
import torch
import torch.nn as nn
from difftransformer import DifferentialTransformerClassifier
from torch.utils.data import Dataset, DataLoader

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
print("Model Loaded")

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00011696142086951537, weight_decay=9.168589372978195e-05,)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

train_df, test_df = utils.split_data(utils.df, test_size=0.2)

train_dataset, train_period_mapping = utils.prepare_dataset(train_df, train=True)
train_dataloader = utils.prepare_dataloader(train_dataset, batch_size=batch_size)

test_dataset, test_period_mapping = utils.prepare_dataset(test_df, train=True)
test_dataloader = utils.prepare_dataloader(test_dataset, batch_size=batch_size)
print("Dataloader ready")

# Training loop
epochs = 1000 # large number to see when the model overfits after evaluating every 5 epochs. 
# We stop the training once the accuracy starts decreasing and train the model on the whole dataset 
# on a fewer number of epochs so that it doesn't overfit 
# (we train on 5 less epochs than when the accuracy on the test set starts decreasing)

checkpoint_path = f"/users/eleves-a/2022/amine.chraibi/KaggleINF/model_{depth}_{n_embd}_{n_head}_{dropout}.pth"
best_accuracy = 0

for epoch in range(epochs):
    loss = utils.train(model, train_dataloader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    scheduler.step()
    if (epoch + 1) % 5 == 0:
        test_accuracy = utils.evaluate(model, test_dataloader, criterion, device)
        print(f"Test Accuracy at epoch {epoch+1}: {test_accuracy:.4f}")
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1}.")

# Final save
torch.save(model.state_dict(), checkpoint_path)
print(f"Final model saved to {checkpoint_path}.")
