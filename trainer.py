import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import prepare_dataset, tokenizer, df, evaluate, split_data, prepare_dataloader
from difftransformer import DifferentialTransformerClassifier

def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        scheduler=None
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            if self.gpu_id == 0:
                print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        if self.gpu_id == 0:
            print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        print("loss",loss.item())
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = self.train_data.batch_size
        if self.gpu_id == 0:
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        self.model.train()
        total_loss = 0.0
        for source,targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            total_loss += self._run_batch(source, targets)

        avg_loss = total_loss / len(self.train_data)
        if self.gpu_id == 0:
            print(f"[GPU{self.gpu_id}] Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")

    def _save_snapshot(self, epoch):
        if self.gpu_id == 0:
            snapshot = {
                "MODEL_STATE": self.model.module.state_dict(),
                "EPOCHS_RUN": epoch,
            }
            torch.save(snapshot, self.snapshot_path)
            print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int, test_dataloader, criterion, device):
        best_accuracy = 0
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if (epoch + 1) % 5 == 0:
                test_accuracy = evaluate(self.model, test_dataloader, criterion, device)
                print(f"Test Accuracy at epoch {epoch+1}: {test_accuracy:.4f}")
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    self._save_snapshot(epoch)
                    print(f"Checkpoint saved at epoch {epoch + 1}.")


def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
    vocab_size = tokenizer.vocab_size()
    depth = 5
    n_embd = 144
    n_head = 6
    dropout = 0.014500254910782884
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ddp_setup()
    train_df, test_df = split_data(df, test_size=0.2)

    train_dataset, _ = prepare_dataset(train_df, train=True)
    
    test_dataset, _ = prepare_dataset(test_df, train=True)
    test_dataloader = prepare_dataloader(test_dataset, batch_size=batch_size)
    
    sampler = DistributedSampler(train_dataset, rank=int(os.environ["RANK"]), num_replicas=int(os.environ["WORLD_SIZE"]))
    train_data = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True)
    model = DifferentialTransformerClassifier(
    vocab_size=vocab_size,
    embedding_dim=n_embd,  
    num_heads=n_head,
    depth=depth,
    dropout = dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=9.168589372978195e-05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path, scheduler = scheduler)
    trainer.train(total_epochs,test_dataloader,criterion,device)
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=4, type=int, help='Input batch size on each device (default: 4)')
    parser.add_argument('--snapshot_path', default='snapshot.pt', type=str, help='Path to save the training snapshot')
    args = parser.parse_args()

    main(args.save_every, args.total_epochs, args.batch_size, args.snapshot_path)
