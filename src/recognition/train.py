"""Train the sign recognition transformer on extracted WLASL keypoints.

Usage: python -m src.recognition.train --epochs 30
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.recognition.dataset import WLASLDataset, FEATURE_DIM
from src.recognition.model import SignTransformer

ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(-1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = WLASLDataset("train")
    val_set = WLASLDataset("val")
    # keep validation restricted to classes seen during training
    val_set.classes = train_set.classes
    val_set.gloss_to_idx = train_set.gloss_to_idx

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model = SignTransformer(FEATURE_DIM, num_classes=len(train_set.classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        print(f"epoch {epoch+1:03d} | train loss {train_loss:.4f} acc {train_acc:.3f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best.pt")
            (CHECKPOINT_DIR / "classes.json").write_text(
                json.dumps(train_set.classes), encoding="utf-8"
            )

    print(f"Best val accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
