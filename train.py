import argparse
import os
import random
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ----------------------
# Utilities
# ----------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.mps doesn't have separate seed; manual_seed covers it
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------
# Model (small CNN for MNIST)
# ----------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 28->26
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 26->24
        self.pool = nn.MaxPool2d(2)  # 24->12
        self.drop1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x


# ----------------------
# Config
# ----------------------
@dataclass
class TrainConfig:
    project: str = "mnist"
    run_name: str = "cnn-mnist"
    data_dir: str = "./data"
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    val_split: float = 0.1
    num_workers: int = 0  # for Mac, safer default
    log_interval: int = 1
    save_dir: str = "./checkpoints"
    offline: bool = False  # set True to run wandb offline


# ----------------------
# Train / Eval loops
# ----------------------
def train_one_epoch(model, loader, optimizer, device, epoch, log_interval, scaler=None, wandb_run=None):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    running_loss, running_correct, running_count = 0.0, 0, 0

    for batch_idx, (data, target) in enumerate(loader, start=1):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True)
        # (AMP is tricky on MPS; keep it simple & portable)
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # stats
        preds = output.argmax(dim=1)
        correct = (preds == target).sum().item()
        running_loss += loss.item() * data.size(0)
        running_correct += correct
        running_count += data.size(0)

        if batch_idx % log_interval == 0 and wandb_run is not None:
            wandb_run.log(
                {
                    "train/step_loss": loss.item(),
                    "train/step_acc": correct / data.size(0),
                    "train/epoch": epoch,
                    "train/step": (epoch - 1) * len(loader) + batch_idx,
                }
            )

    epoch_loss = running_loss / running_count
    epoch_acc = running_correct / running_count

    if wandb_run is not None:
        wandb_run.log({"train/loss": epoch_loss, "train/acc": epoch_acc, "epoch": epoch})

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, device, epoch, split="val", wandb_run=None):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_correct, total_count = 0.0, 0, 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_fn(output, target)
        preds = output.argmax(dim=1)
        total_loss += loss.item() * data.size(0)
        total_correct += (preds == target).sum().item()
        total_count += data.size(0)

    loss = total_loss / total_count
    acc = total_correct / total_count

    if wandb_run is not None:
        wandb_run.log({f"{split}/loss": loss, f"{split}/acc": acc, "epoch": epoch})

    return loss, acc


# ----------------------
# Main
# ----------------------
def main():
    parser = argparse.ArgumentParser(description="Train a simple CNN on MNIST with wandb logging (Mac MPS/CUDA/CPU).")
    parser.add_argument("--project", type=str, default=TrainConfig.project)
    parser.add_argument("--run-name", type=str, default=TrainConfig.run_name)
    parser.add_argument("--data-dir", type=str, default=TrainConfig.data_dir)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--val-split", type=float, default=TrainConfig.val_split)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--log-interval", type=int, default=TrainConfig.log_interval)
    parser.add_argument("--save-dir", type=str, default=TrainConfig.save_dir)
    parser.add_argument("--offline", action="store_true", help="Enable wandb offline mode")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    cfg = TrainConfig(
        project=args.project,
        run_name=args.run_name,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        val_split=args.val_split,
        num_workers=args.num_workers,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
        offline=args.offline,
    )

    seed_everything(cfg.seed)
    device = get_device()
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Transforms and dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    full_train = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(cfg.data_dir, train=False, download=True, transform=transform)

    # Train/val split
    val_len = int(len(full_train) * cfg.val_split)
    train_len = len(full_train) - val_len
    train_set, val_set = random_split(full_train, [train_len, val_len])

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=False
    )

    # Model/optim
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # wandb
    wandb_run = None
    if not args.no_wandb:
        import wandb

        if cfg.offline:
            os.environ["WANDB_MODE"] = "offline"
        wandb_run = wandb.init(project=cfg.project, name=cfg.run_name, config=asdict(cfg))
        # Track gradients/parameters
        wandb.watch(model, log="gradients", log_freq=max(1, cfg.log_interval))

    best_val_acc = 0.0
    best_path = os.path.join(cfg.save_dir, "best.pt")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, cfg.log_interval, wandb_run=wandb_run
        )
        val_loss, val_acc = evaluate(model, val_loader, device, epoch, split="val", wandb_run=wandb_run)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "config": asdict(cfg),
                },
                best_path,
            )

    # Final test
    test_loss, test_acc = evaluate(model, test_loader, device, epoch=cfg.epochs, split="test", wandb_run=wandb_run)
    print(f"[Test] loss={test_loss:.4f} acc={test_acc:.4f} (best_val_acc={best_val_acc:.4f})")

    if wandb_run is not None:
        wandb_run.log({"test/loss": test_loss, "test/acc": test_acc})
        # Optionally log the checkpoint as an artifact
        try:
            import wandb

            artifact = wandb.Artifact("mnist-model", type="model")
            artifact.add_file(best_path)
            wandb_run.log_artifact(artifact)
        except Exception as e:
            print(f"wandb artifact logging skipped: {e}")
        wandb_run.finish()


if __name__ == "__main__":
    main()
