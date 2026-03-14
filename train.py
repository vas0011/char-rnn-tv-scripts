"""
Training script for the CharRNN model.

Example:
    python train.py --data data/game_of_thrones.txt --epochs 20
"""

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn

from char_rnn import CharRNN, build_vocab, encode, get_device


def get_batches(data: torch.Tensor, batch_size: int, seq_len: int):
    """Yield (inputs, targets) tensors by randomly sampling start positions."""
    n = len(data) - seq_len - 1
    if n <= 0:
        raise ValueError("Data too short for given seq_len.")
    starts = random.sample(range(n), min(batch_size, n))
    inputs = torch.stack([data[s : s + seq_len] for s in starts])
    targets = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return inputs, targets


def evaluate(model, data, batch_size, seq_len, device, criterion, num_batches=20):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batches(data, batch_size, seq_len)
            x, y = x.to(device), y.to(device)
            hidden = model.init_hidden(x.size(0), device)
            logits, _ = model(x, hidden)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    model.train()
    return total_loss / num_batches


def train(args):
    device = get_device()
    print(f"Device: {device}")

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Dataset: {args.data}  |  {len(text):,} characters")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"Vocabulary size: {vocab_size}")

    data = encode(text, char2idx)
    split = int(len(data) * 0.9)
    train_data = data[:split]
    val_data = data[split:]
    print(f"Train chars: {len(train_data):,}  |  Val chars: {len(val_data):,}")

    model = CharRNN(
        vocab_size=vocab_size,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()

    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_ppl": [], "val_ppl": []}

    steps_per_epoch = max(1, len(train_data) // (args.batch_size * args.seq_len))

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step in range(steps_per_epoch):
            x, y = get_batches(train_data, args.batch_size, args.seq_len)
            x, y = x.to(device), y.to(device)

            hidden = model.init_hidden(x.size(0), device)
            optimizer.zero_grad()
            logits, _ = model(x, hidden)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()

            if (step + 1) % args.log_interval == 0:
                avg = epoch_loss / (step + 1)
                print(
                    f"  Epoch {epoch}/{args.epochs}  step {step+1}/{steps_per_epoch}"
                    f"  loss={avg:.4f}"
                )

        avg_train_loss = epoch_loss / steps_per_epoch
        val_loss = evaluate(model, val_data, args.batch_size, args.seq_len, device, criterion)
        train_ppl = math.exp(avg_train_loss)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["train_ppl"].append(train_ppl)
        history["val_ppl"].append(val_ppl)

        print(
            f"Epoch {epoch:>2}/{args.epochs}"
            f"  train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}"
            f"  train_ppl={train_ppl:.1f}  val_ppl={val_ppl:.1f}"
            f"  ({elapsed:.1f}s)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "char2idx": char2idx,
                    "idx2char": idx2char,
                    "vocab_size": vocab_size,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "embed_size": args.embed_size,
                    "dropout": args.dropout,
                },
                args.save_path,
            )
            print(f"  ✓ Saved best model → {args.save_path}")

    dataset_name = os.path.splitext(os.path.basename(args.data))[0]
    log_path = os.path.join(
        os.path.dirname(args.save_path) or "results",
        f"loss_history_{dataset_name}.json",
    )
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Loss history saved → {log_path}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CharRNN model.")
    parser.add_argument("--data", required=True, help="Path to training .txt file")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--embed_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    parser.add_argument("--save_path", default="results/model.pt")
    parser.add_argument("--log_interval", type=int, default=100)
    args = parser.parse_args()
    train(args)
