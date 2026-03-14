"""
Generate text from a trained CharRNN checkpoint.

Example:
    python generate.py --checkpoint results/model_got.pt --seed "TYRION:\n" --temperature 0.8 --length 500
"""

import argparse

import torch
import torch.nn.functional as F

from char_rnn import CharRNN, get_device


def generate(checkpoint_path: str, seed: str, temperature: float, length: int) -> str:
    device = get_device()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    char2idx: dict = ckpt["char2idx"]
    idx2char: dict = ckpt["idx2char"]

    model = CharRNN(
        vocab_size=ckpt["vocab_size"],
        embed_size=ckpt["embed_size"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Filter seed to known vocab characters
    seed = "".join(c for c in seed if c in char2idx)
    if not seed:
        seed = idx2char[0]

    generated = list(seed)

    # Prime the hidden state by feeding the seed sequence
    hidden = model.init_hidden(1, device)
    with torch.no_grad():
        for ch in seed[:-1]:
            x = torch.tensor([[char2idx[ch]]], dtype=torch.long, device=device)
            _, hidden = model(x, hidden)

        current_char = seed[-1]
        for _ in range(length):
            x = torch.tensor([[char2idx[current_char]]], dtype=torch.long, device=device)
            logits, hidden = model(x, hidden)
            logits = logits[:, -1, :]  # (1, vocab_size)
            probs = F.softmax(logits / temperature, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            current_char = idx2char[next_idx]
            generated.append(current_char)

    return "".join(generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from a trained CharRNN.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--seed", default="TYRION:\n", help="Seed text to prime the model")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (lower=more conservative, higher=more random)")
    parser.add_argument("--length", type=int, default=500, help="Number of characters to generate")
    args = parser.parse_args()

    output = generate(args.checkpoint, args.seed, args.temperature, args.length)
    print(output)
