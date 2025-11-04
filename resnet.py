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
# ResNet for MNIST
# ----------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, in_channels=1):
        super().__init__()
        # MNIST向けに初段を調整（3x3, s=1, padding=1、MaxPoolなし）
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        # ResNetステージ
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)  # 28x28
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 14x14
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 7x7
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 4x4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, planes, blocks, stride):
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18_mnist(num_classes=10, in_channels=1):
    # ResNet-18 の層数構成: [2,2,2,2]
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)


# ----------------------
# Config
# ----------------------
@dataclass
class TrainConfig:
    project: str = "mnist"
    run_name: str = "resnet18-mnist"
    data_dir: str = "./data"
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    val_split: float = 0.1
    num_workers: int = 0  # for Mac, safer default
    log_interval: int = 1
    save_dir: str = "./checkpoints/resnet"
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
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

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
    parser = argparse.ArgumentParser(description="Train ResNet-18-like model on MNIST with wandb logging (Mac MPS/CUDA/CPU).")
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
    model = resnet18_mnist(num_classes=10, in_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # wandb
    wandb_run = None
    if not args.no_wandb:
        import wandb
        if cfg.offline:
            os.environ["WANDB_MODE"] = "offline"
        wandb_run = wandb.init(project=cfg.project, name=cfg.run_name, config=asdict(cfg))
        wandb.watch(model, log="gradients", log_freq=max(1, cfg.log_interval))

    best_val_acc = 0.0
    best_path = os.path.join(cfg.save_dir, "best.pt")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, cfg.log_interval, wandb_run=wandb_run
        )
        val_loss, val_acc = evaluate(model, val_loader, device, epoch, split="val", wandb_run=wandb_run)

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

    test_loss, test_acc = evaluate(model, test_loader, device, epoch=cfg.epochs, split="test", wandb_run=wandb_run)
    print(f"[Test] loss={test_loss:.4f} acc={test_acc:.4f} (best_val_acc={best_val_acc:.4f})")

    if wandb_run is not None:
        wandb_run.log({"test/loss": test_loss, "test/acc": test_acc})
        try:
            import wandb
            artifact = wandb.Artifact("mnist-resnet18-model", type="model")
            artifact.add_file(best_path)
            wandb_run.log_artifact(artifact)
        except Exception as e:
            print(f"wandb artifact logging skipped: {e}")
        wandb_run.finish()


if __name__ == "__main__":
    main()
