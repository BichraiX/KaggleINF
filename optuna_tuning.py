import utils
import optuna
import torch
import torch.nn as nn
from difftransformer import DifferentialTransformerClassifier

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 4

def objective(trial):
    # Define the hyperparameters to tune
    n_embd = trial.suggest_categorical("n_embd", [144, 192, 240, 288, 336])
    n_head = trial.suggest_categorical("n_head", [4, 8, 12])
    depth = trial.suggest_int("depth", 4, 8)
    dropout = trial.suggest_float("dropout",0.3,0.7)

    # Initialize model
    model = DifferentialTransformerClassifier(
        vocab_size=utils.tokenizer.vocab_size(),
        embedding_dim=n_embd,
        num_heads=n_head,
        depth=depth,
        dropout = dropout
    )
    model.to(device)

    # Optimizer and scheduler
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3,log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2,log=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    # Loss function
    criterion = nn.BCELoss()

    # Training loop
    epochs = 10  # Fewer epochs for tuning
    for epoch in range(epochs):
        train_loss = utils.train(model, train_dataloader, optimizer, criterion, device)
        scheduler.step()

    # Evaluate on the test set
    model.eval()
    test_accuracy = utils.evaluate(model, test_dataloader, criterion, device)
    print(f"Test Accuracy at epoch {epoch+1}: {test_accuracy:.4f}")

    return test_accuracy

# Dataset and dataloader
train_df, test_df = utils.split_data(utils.df, test_size=0.4)
train_dataset, _ = utils.prepare_dataset(train_df, train=True)
train_dataloader = utils.prepare_dataloader(train_dataset, batch_size=16)
test_dataset, _ = utils.prepare_dataset(test_df, train=True)
test_dataloader = utils.prepare_dataloader(test_dataset, batch_size=16)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  # Adjust `n_trials` as needed

# Best hyperparameters and test loss
print("Best trial:")
print(f"  Value: {study.best_value}")
print(f"  Params: {study.best_params}")

# Save the best model parameters
best_params = study.best_params
torch.save(best_params, "best_params.pth")

