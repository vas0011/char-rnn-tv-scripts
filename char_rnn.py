"""
Character-level RNN model and vocabulary utilities.
"""

import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_vocab(text: str) -> tuple[dict, dict]:
    """Return (char2idx, idx2char) for all unique characters in text."""
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}
    return char2idx, idx2char


def encode(text: str, char2idx: dict) -> torch.Tensor:
    """Encode a string as a LongTensor of character indices."""
    return torch.tensor([char2idx[c] for c in text if c in char2idx], dtype=torch.long)


class CharRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 64,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        # dropout between LSTM layers is only applied when num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, x: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(x)            # (B, T, embed_size)
        out, hidden = self.lstm(embedded, hidden)  # (B, T, hidden_size)
        out = self.dropout(out)
        logits = self.fc(out)                   # (B, T, vocab_size)
        return logits, hidden

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h0, c0
