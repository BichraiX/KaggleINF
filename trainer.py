import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AdamW, get_linear_schedule_with_warmup
from bert_utils import prepare_dataset, get_tokenizer, HierarchicalBertModel

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

    def _run_batch(self, input_ids, attention_mask, targets):
        self.optimizer.zero_grad()
        # Model returns loss, logits if labels are provided; else just logits
        loss, logits = self.model(input_ids, attention_mask, labels=targets)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = self.train_data.batch_size
        if self.gpu_id == 0:
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        self.model.train()
        total_loss = 0.0
        for batch in self.train_data:
            # Assuming batch is a dict with 'input_ids', 'attention_mask', and 'label'
            input_ids = batch['input_ids'].to(self.gpu_id, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.gpu_id, non_blocking=True)
            targets = batch['label'].to(self.gpu_id, non_blocking=True)

            loss_val = self._run_batch(input_ids, attention_mask, targets)
            total_loss += loss_val

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

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
    ddp_setup()

    train_dataset, test_dataset = prepare_dataset()  # This returns two datasets

    # For example, create a sampler for distributed training
    sampler = DistributedSampler(train_dataset, rank=int(os.environ["RANK"]), num_replicas=int(os.environ["WORLD_SIZE"]))

    train_data = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        num_workers=4
    )

    model = HierarchicalBertModel(freeze_bert=True)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Optional: use a scheduler if desired
    total_steps = len(train_data) * total_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # dist.barrier()
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path, scheduler=scheduler)
    trainer.train(total_epochs)
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
