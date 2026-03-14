"""
Generate figures for the final report:
  results/loss_curve_got.png        — train/val loss for GoT model
  results/loss_curve_office.png     — train/val loss for The Office model
  results/experiment_heatmap.png    — val perplexity vs hidden_size × num_layers
  results/experiment_bar_optimizer.png  — adam vs sgd val perplexity

Usage:
    python plot_results.py
"""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# ── Loss curves ──────────────────────────────────────────────────────────────
for dataset, label in [("game_of_thrones", "Game of Thrones"), ("the_office", "The Office")]:
    history_path = os.path.join(RESULTS_DIR, f"loss_history_{dataset}.json")
    if not os.path.exists(history_path):
        print(f"Skipping loss curve for {label} (file not found: {history_path})")
        continue

    with open(history_path) as f:
        h = json.load(f)

    epochs = range(1, len(h["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, h["train_loss"], label="Train", marker="o", ms=3)
    ax1.plot(epochs, h["val_loss"], label="Val", marker="s", ms=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title(f"{label} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, h["train_ppl"], label="Train", marker="o", ms=3)
    ax2.plot(epochs, h["val_ppl"], label="Val", marker="s", ms=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Perplexity")
    ax2.set_title(f"{label} — Perplexity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"loss_curve_{dataset}.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

# ── Experiment results ────────────────────────────────────────────────────────
csv_path = os.path.join(RESULTS_DIR, "experiment_results.csv")
if not os.path.exists(csv_path):
    print(f"Skipping experiment plots (file not found: {csv_path})")
else:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["final_val_perplexity"])

    # --- Heatmap: hidden_size × num_layers
    pivot = df.groupby(["hidden_size", "num_layers"])["final_val_perplexity"].mean().unstack()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c} layer{'s' if c>1 else ''}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"h={r}" for r in pivot.index])
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="Val Perplexity (lower=better)")
    ax.set_title("Val Perplexity: Hidden Size × Num Layers")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "experiment_heatmap.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

    # --- Bar chart: optimizer comparison
    opt_avg = df.groupby("optimizer")["final_val_perplexity"].mean()
    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar(opt_avg.index, opt_avg.values, color=["steelblue", "coral"])
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_ylabel("Avg Val Perplexity")
    ax.set_title("Optimizer Comparison")
    ax.set_ylim(0, opt_avg.max() * 1.2)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "experiment_bar_optimizer.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

    # --- Bar chart: dropout comparison
    drop_avg = df.groupby("dropout")["final_val_perplexity"].mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar([str(d) for d in drop_avg.index], drop_avg.values, color="steelblue")
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_xlabel("Dropout rate")
    ax.set_ylabel("Avg Val Perplexity")
    ax.set_title("Dropout Rate Comparison")
    ax.set_ylim(0, drop_avg.max() * 1.2)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "experiment_bar_dropout.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

    # --- Line chart: learning rate comparison
    lr_avg = df.groupby("lr")["final_val_perplexity"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([str(l) for l in lr_avg.index], lr_avg.values, marker="o")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Avg Val Perplexity")
    ax.set_title("Learning Rate Comparison")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "experiment_lr_comparison.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

    print("\nTop 5 configs by val perplexity:")
    print(
        df.nsmallest(5, "final_val_perplexity")[
            ["hidden_size", "num_layers", "dropout", "lr", "seq_len", "optimizer", "final_val_perplexity"]
        ].to_string(index=False)
    )
