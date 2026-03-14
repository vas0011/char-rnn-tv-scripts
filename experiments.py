"""
Hyperparameter sweep for the CharRNN model.

Runs a grid search over the GoT dataset with 5 epochs per config,
saves results to results/experiment_results.csv, then trains the
best config for 20 epochs on both GoT and The Office.

Usage:
    python experiments.py
"""

import csv
import itertools
import math
import os
import sys

import torch
import torch.nn as nn

from char_rnn import CharRNN, build_vocab, encode, get_device
from train import evaluate, get_batches

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

GOT_DATA = "data/game_of_thrones.txt"
OFFICE_DATA = "data/the_office.txt"

GRID = {
    "hidden_size": [128, 256, 512],
    "num_layers": [1, 2],
    "dropout": [0.0, 0.3, 0.5],
    "lr": [1e-3, 5e-4, 1e-4],
    "seq_len": [100, 200],
    "optimizer": ["adam", "sgd"],
}

SWEEP_EPOCHS = 5
FULL_EPOCHS = 20
BATCH_SIZE = 64
EMBED_SIZE = 64


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    char2idx, idx2char = build_vocab(text)
    data = encode(text, char2idx)
    split = int(len(data) * 0.9)
    return data[:split], data[split:], char2idx, idx2char


def run_config(
    train_data, val_data, char2idx, idx2char,
    hidden_size, num_layers, dropout, lr, seq_len, optimizer_name,
    epochs, device, save_path=None
):
    vocab_size = len(char2idx)
    model = CharRNN(
        vocab_size=vocab_size,
        embed_size=EMBED_SIZE,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    steps_per_epoch = max(1, len(train_data) // (BATCH_SIZE * seq_len))
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            x, y = get_batches(train_data, BATCH_SIZE, seq_len)
            x, y = x.to(device), y.to(device)
            hidden = model.init_hidden(x.size(0), device)
            optimizer.zero_grad()
            logits, _ = model(x, hidden)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / steps_per_epoch
        val_loss = evaluate(model, val_data, BATCH_SIZE, seq_len, device, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "char2idx": char2idx,
                        "idx2char": idx2char,
                        "vocab_size": vocab_size,
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "embed_size": EMBED_SIZE,
                        "dropout": dropout,
                    },
                    save_path,
                )

        print(
            f"  epoch {epoch}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  ppl={math.exp(val_loss):.1f}",
            flush=True,
        )

    val_loss = evaluate(model, val_data, BATCH_SIZE, seq_len, device, criterion)
    train_loss_final = evaluate(model, train_data, BATCH_SIZE, seq_len, device, criterion)
    return train_loss_final, val_loss, math.exp(val_loss)


def main():
    device = get_device()
    print(f"Device: {device}")

    print("Loading GoT dataset…")
    got_train, got_val, got_c2i, got_i2c = load_dataset(GOT_DATA)

    keys = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    total = len(combos)
    print(f"Running {total} configurations × {SWEEP_EPOCHS} epochs each…\n")

    csv_path = os.path.join(RESULTS_DIR, "experiment_results.csv")
    fieldnames = keys + ["final_train_loss", "final_val_loss", "final_val_perplexity"]

    results = []
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, combo in enumerate(combos, 1):
            cfg = dict(zip(keys, combo))
            print(f"[{i}/{total}] {cfg}")
            try:
                tl, vl, vppl = run_config(
                    got_train, got_val, got_c2i, got_i2c,
                    hidden_size=cfg["hidden_size"],
                    num_layers=cfg["num_layers"],
                    dropout=cfg["dropout"],
                    lr=cfg["lr"],
                    seq_len=cfg["seq_len"],
                    optimizer_name=cfg["optimizer"],
                    epochs=SWEEP_EPOCHS,
                    device=device,
                )
            except Exception as e:
                print(f"  FAILED: {e}", file=sys.stderr)
                tl, vl, vppl = float("nan"), float("nan"), float("nan")

            row = {**cfg, "final_train_loss": tl, "final_val_loss": vl, "final_val_perplexity": vppl}
            writer.writerow(row)
            csvfile.flush()
            results.append(row)
            print(f"  → val_loss={vl:.4f}  val_ppl={vppl:.1f}\n")

    # Find best config (lowest val loss, ignoring nan)
    valid = [r for r in results if not math.isnan(r["final_val_loss"])]
    best = min(valid, key=lambda r: r["final_val_loss"])
    print("\n=== Best config ===")
    for k, v in best.items():
        print(f"  {k}: {v}")

    # Full training on GoT with best config
    print(f"\nTraining best config for {FULL_EPOCHS} epochs on GoT…")
    run_config(
        got_train, got_val, got_c2i, got_i2c,
        hidden_size=best["hidden_size"],
        num_layers=best["num_layers"],
        dropout=best["dropout"],
        lr=best["lr"],
        seq_len=best["seq_len"],
        optimizer_name=best["optimizer"],
        epochs=FULL_EPOCHS,
        device=device,
        save_path=os.path.join(RESULTS_DIR, "best_model_got.pt"),
    )

    # Full training on The Office with best config
    print(f"\nTraining best config for {FULL_EPOCHS} epochs on The Office…")
    office_train, office_val, office_c2i, office_i2c = load_dataset(OFFICE_DATA)
    run_config(
        office_train, office_val, office_c2i, office_i2c,
        hidden_size=best["hidden_size"],
        num_layers=best["num_layers"],
        dropout=best["dropout"],
        lr=best["lr"],
        seq_len=best["seq_len"],
        optimizer_name=best["optimizer"],
        epochs=FULL_EPOCHS,
        device=device,
        save_path=os.path.join(RESULTS_DIR, "best_model_office.pt"),
    )

    print(f"\nDone! Results saved to {csv_path}")


if __name__ == "__main__":
    main()
