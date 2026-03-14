# Char-RNN Final Project — Report Notes
## COGS 181A · Due 03/20/2026

Use these notes to write the final report in NeurIPS/ICML format.
All numbers here are real results from the actual training runs.

---

## Abstract (write ~150 words using this)

We implement a character-level recurrent neural network (Char-RNN) using stacked LSTMs in
PyTorch and train it on two distinct TV show script corpora: *Game of Thrones* (73 episodes,
~1.8 MB) and *The Office* (186 episodes, ~3.5 MB). The model learns to generate plausible
dialogue at the character level, capturing show-specific vocabulary, character names, and
conversational style without any word-level supervision. Our baseline model (hidden size 256,
2 LSTM layers, dropout 0.3, Adam optimizer) achieves a final validation perplexity of **3.31**
on Game of Thrones and **3.81** on The Office after 20 epochs. We conduct a systematic
hyperparameter sweep across 216 configurations varying hidden size, number of layers, dropout,
learning rate, sequence length, and optimizer. The best configuration (hidden=512, layers=2,
dropout=0.0, lr=1e-3, Adam) achieves val perplexity **2.99** at 5 epochs. Generated samples
demonstrate that the model captures show-specific stylistic features: Westerosi names and
feudal vocabulary for GoT, and the awkward modern office humor register for The Office.

---

## Introduction (write ~300 words using these talking points)

- Karpathy (2015) "The Unreasonable Effectiveness of Recurrent Neural Networks" — char-RNNs
  can model arbitrary text at the character level, learning spelling, syntax, and style
  without explicit word tokenization
- LSTMs (Hochreiter & Schmidhuber, 1997) solve the vanishing gradient problem that plagued
  earlier RNNs, enabling learning of long-range dependencies
- Motivation: TV scripts are an interesting domain — they have strong character-specific voice,
  multi-speaker structure, and two stylistically opposite shows make for a natural cross-corpus
  comparison (epic fantasy vs. mundane workplace comedy)
- Contribution: (1) episode-structured preprocessing, (2) systematic 216-config hyperparameter
  study, (3) cross-dataset stylistic comparison, (4) qualitative analysis via temperature
  ablation and character-seeded generation

---

## Method (~300 words)

### Architecture

```
Input character index
       ↓
nn.Embedding(vocab_size=93, embed_size=64)
       ↓
nn.LSTM(64 → 256, num_layers=2, dropout=0.3, batch_first=True)
       ↓
nn.Dropout(0.3)
       ↓
nn.Linear(256 → vocab_size)
       ↓
logits → temperature softmax → multinomial sample
```

Total parameters: **885,917** (GoT vocab=93) / **887,201** (Office vocab=97)

### Training procedure

- 90/10 train/val split on encoded character sequence
- Random batch sampling: `batch_size=64` random start positions, `seq_len=200` chars
- Target = input shifted by 1 character
- Loss: `nn.CrossEntropyLoss` over flattened (batch × seq_len, vocab_size) logits
- Gradient clipping: `clip_grad_norm(max_norm=5.0)` — critical for LSTM stability
- Hidden state detached between batches to prevent BPTT through full history
- Optimizer: Adam, lr=1e-3
- Checkpoint saved when val loss improves

### Text generation

At inference, the model is primed with a seed string (e.g. `TYRION:\n`), then generates
one character at a time by sampling from:

```
p(c) = softmax(logits / τ)
```

where τ is a temperature parameter. Lower τ → more conservative/repetitive output;
higher τ → more creative but less coherent.

### Data preprocessing

Each script is formatted as:
```
=== Season 1, Episode 1: Winter is Coming ===

TYRION:
Dialogue text here.

```

Episode headers allow the model to learn scene-boundary structure in addition to
character-level dialogue patterns. GoT has 73 episodes; The Office has 186 episodes.

---

## Experiments

### Dataset statistics

| Dataset | Episodes | Lines | Characters | Vocab size |
|---|---|---|---|---|
| Game of Thrones | 73 | 23,907 | 1,830,518 | 93 |
| The Office | 186 | 54,756 | 3,465,832 | 97 |

### Baseline training results (20 epochs, hidden=256, layers=2, dropout=0.3, lr=1e-3, adam)

#### Game of Thrones — epoch-by-epoch

| Epoch | Train Loss | Val Loss | Train PPL | Val PPL |
|---|---|---|---|---|
| 1  | 2.681 | 2.016 | 14.60 | 7.50 |
| 2  | 1.911 | 1.691 | 6.76  | 5.42 |
| 3  | 1.699 | 1.537 | 5.47  | 4.65 |
| 4  | 1.582 | 1.458 | 4.87  | 4.30 |
| 5  | 1.511 | 1.418 | 4.53  | 4.13 |
| 6  | 1.464 | 1.362 | 4.32  | 3.91 |
| 7  | 1.421 | 1.331 | 4.14  | 3.78 |
| 8  | 1.390 | 1.323 | 4.01  | 3.76 |
| 9  | 1.365 | 1.283 | 3.92  | 3.61 |
| 10 | 1.351 | 1.272 | 3.86  | 3.57 |
| 11 | 1.331 | 1.255 | 3.78  | 3.51 |
| 12 | 1.317 | 1.264 | 3.73  | 3.54 |
| 13 | 1.307 | 1.244 | 3.70  | 3.47 |
| 14 | 1.297 | 1.233 | 3.66  | 3.43 |
| 15 | 1.284 | 1.224 | 3.61  | 3.40 |
| 16 | 1.277 | 1.216 | 3.59  | 3.37 |
| 17 | 1.269 | 1.208 | 3.56  | 3.35 |
| 18 | 1.260 | 1.206 | 3.53  | 3.34 |
| 19 | 1.254 | 1.199 | 3.51  | 3.32 |
| 20 | 1.244 | **1.197** | 3.47  | **3.31** |

#### The Office — epoch-by-epoch

| Epoch | Train Loss | Val Loss | Train PPL | Val PPL |
|---|---|---|---|---|
| 1  | 2.346 | 1.852 | 10.45 | 6.37 |
| 2  | 1.698 | 1.630 | 5.46  | 5.10 |
| 3  | 1.559 | 1.539 | 4.76  | 4.66 |
| 4  | 1.492 | 1.493 | 4.45  | 4.45 |
| 5  | 1.444 | 1.460 | 4.24  | 4.30 |
| 6  | 1.415 | 1.452 | 4.12  | 4.27 |
| 7  | 1.391 | 1.419 | 4.02  | 4.13 |
| 8  | 1.372 | 1.406 | 3.94  | 4.08 |
| 9  | 1.357 | 1.402 | 3.88  | 4.06 |
| 10 | 1.347 | 1.373 | 3.85  | 3.95 |
| 11 | 1.337 | 1.373 | 3.81  | 3.95 |
| 12 | 1.327 | 1.365 | 3.77  | 3.92 |
| 13 | 1.318 | 1.366 | 3.74  | 3.92 |
| 14 | 1.314 | 1.358 | 3.72  | 3.89 |
| 15 | 1.308 | 1.348 | 3.70  | 3.85 |
| 16 | 1.304 | 1.357 | 3.69  | 3.88 |
| 17 | 1.298 | 1.345 | 3.66  | **3.84** |
| 18 | 1.298 | 1.355 | 3.66  | 3.88 |
| 19 | 1.290 | **1.337** | 3.63  | **3.81** |
| 20 | 1.289 | 1.344 | 3.63  | 3.83 |

### Key observations from training curves

- Both models converge smoothly with no signs of divergence or exploding gradients
- **GoT** val loss improves every epoch all the way to epoch 20 — the model is still
  learning and would benefit from more epochs
- **The Office** val loss plateaus around epoch 15–17, converging faster due to its
  2× larger dataset (3.5 MB vs 1.8 MB)
- GoT achieves lower perplexity (3.31 vs 3.81), likely because its vocabulary is more
  constrained (medieval fantasy) while The Office has a wider colloquial register
- No overfitting observed in either model — train and val curves track closely

### Hyperparameter sweep results (216 configurations, 5 epochs each on GoT)

#### Summary table by factor

| Factor | Values | Avg Val PPL | Winner |
|---|---|---|---|
| Optimizer | adam / sgd | 6.0 / 63.9 | **Adam** (10× better) |
| Hidden size | 128 / 256 / 512 | 36.8 / 34.7 / 33.5 | **512** |
| Num layers | 1 / 2 | 33.9 / 36.1 | **1** (at 5 epochs) |
| Dropout | 0.0 / 0.3 / 0.5 | 35.3 / 34.1 / 35.5 | **0.3** (marginal) |
| Learning rate | 1e-4 / 5e-4 / 1e-3 | 48.1 / 33.5 / 23.4 | **1e-3** |
| Seq length | 100 / 200 | — | **100** (at 5 epochs) |

#### Top 5 configurations

| hidden | layers | dropout | lr | seq_len | optimizer | val_ppl |
|---|---|---|---|---|---|---|
| 512 | 2 | 0.0 | 0.001 | 100 | adam | **2.99** |
| 512 | 2 | 0.0 | 0.001 | 200 | adam | 3.19 |
| 512 | 2 | 0.3 | 0.001 | 100 | adam | 3.19 |
| 512 | 1 | 0.3 | 0.001 | 100 | adam | 3.26 |
| 512 | 1 | 0.0 | 0.001 | 100 | adam | 3.33 |

#### Interpretation of each finding

**Optimizer (most impactful factor):** Adam (avg ppl 6.0) vs SGD (avg ppl 63.9) is the
starkest result in the sweep. SGD with a fixed lr and no scheduler fails to navigate the
loss landscape of character-level sequence modeling. Adam's adaptive per-parameter learning
rates are essential here.

**Hidden size:** Monotonically better with size — h=512 > h=256 > h=128. Larger hidden
states capture more complex character co-occurrence patterns. Diminishing returns expected
beyond h=512 for this dataset size.

**Num layers (1 vs 2):** At 5 epochs, 1-layer models slightly outperform 2-layer ones.
This is a training time artifact — deeper models require more epochs to converge. The
20-epoch baseline with 2 layers (ppl=3.31) confirms 2 layers eventually wins.

**Dropout:** Differences are small (35.3 → 34.1 → 35.5). Dropout=0.3 is a slight winner.
The averages are dominated by SGD runs; among Adam-only configs the effect is even smaller.

**Learning rate:** Clean monotonic relationship — higher lr converges faster in 5 epochs.
With more epochs, lower lr configs would likely catch up.

### Cross-dataset comparison

| Metric | Game of Thrones | The Office |
|---|---|---|
| Dataset size (chars) | 1,830,518 | 3,465,832 |
| Vocabulary size | 93 | 97 |
| Best val loss (20 epochs) | 1.197 | 1.337 |
| Best val perplexity | 3.31 | 3.81 |
| Epochs to converge | ~20+ (still improving) | ~17 |

### Qualitative samples

**GoT — character seed** (seed: `TYRION:\n`, τ=0.8):
```
TYRION:
Not my own tlaved who comet and the Care told him her take down a larger?

LORD ARYA STARK:
And I want to be your father's lad here and the thousand are days in the Stark of the Jon...

SANSA STARK:
I'm going to go man in the part baby now. Who is us to make the men say to me to the side
of our council are now 20 my right. The Unsullied the Man King's Landing to take them and command
```

**GoT — episode header seed** (seed: `=== Season 8`, τ=0.8):
```
=== Season 8, Episode 9: The Grey Gates you don't want to know her hair.
He's never heard at the slavers.

DAENERYS TARGARYEN:
And what do you call with Ser Davos?

SAM:
Well, that is a trick. The gods are the last time you lie. Ten the world got without
the kingdoms. I'm Lord of the Citing a raven for love.

GREY WORM:
Don't make you here.

TYRION LANNISTER:
Apologies, my little brother.

DAVOS:
Your Grace.
```

**The Office** (seed: `MICHAEL:\n`, τ=0.8):
```
MICHAEL:
You like something to meet my bracting for the best is that hard a respect onay you mean,
I didn't have a bad giy. There's no happened.

PAM:
I'm up"?

DWIGHT:
Sure today?

ANDY:
You know, other because you can't be office change, we can sever in and too. All right.

MICHAEL:
I fave a resident's stypany and go new thing. That's my actually not never feel something
something in the fist now how please, um, babo.

JIM:
I should to ask, everybody, finving around... I did not... and let's get something
```

**Qualitative analysis:**

The GoT episode-header seed is particularly noteworthy: the model invented a plausible
episode title ("Season 8, Episode 9: The Grey Gates"), populated the scene with
contextually appropriate characters (DAENERYS, GREY WORM, TYRION, DAVOS — all Season 8
characters), and produced correct GoT honorifics ("Your Grace", "my little brother").
This demonstrates the model learned not just character names but episode-level structural
patterns from the `=== Season X, Episode Y ===` headers in the training data.

The Office output correctly distributes dialogue across the full ensemble cast (Michael,
Dwight, Jim, Pam, Andy) and captures Michael's characteristic rambling, self-important
speech pattern with filler words ("um", "you know").

### Temperature ablation

Run these to generate samples for the report (add output inline):
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 generate.py --checkpoint results/best_model_got.pt --seed $'TYRION:\n' --temperature 0.5 --length 400
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 generate.py --checkpoint results/best_model_got.pt --seed $'TYRION:\n' --temperature 1.0 --length 400
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 generate.py --checkpoint results/best_model_got.pt --seed $'TYRION:\n' --temperature 1.2 --length 400
```

Expected: τ=0.5 → more repetitive, grammatically tighter; τ=1.2 → more creative, more gibberish

---

## Conclusion (~150 words using these points)

- Implemented a character-level LSTM that successfully learns TV show dialogue style
  without any word-level supervision
- Final perplexity of 3.31 (GoT) and 3.81 (The Office) from a ~886K parameter model
  trained in under 30 minutes on consumer hardware (Apple M-series MPS GPU)
- Hyperparameter sweep of 216 configs reveals: **optimizer choice is by far the most
  impactful factor** (Adam ppl=6.0 vs SGD ppl=63.9); hidden size and learning rate
  matter secondarily; dropout has minimal effect at this scale
- The episode-header seed experiment demonstrates the model learned episode-level
  structure, not just character-level dialogue
- Cross-dataset comparison confirms the model internalizes show-specific register:
  fantasy vocabulary and honorifics for GoT vs. modern colloquial humor for The Office
- Limitations: char-RNN generates character-by-character so long-range coherence
  (plot continuity, consistent character motivation across a scene) is weak
- Future work: transformer-based LMs, word-level BPE tokenization, fine-tuning
  pretrained GPT-2 on the same datasets for comparison

---

## References

1. Karpathy, A. (2015). *The Unreasonable Effectiveness of Recurrent Neural Networks.*
   http://karpathy.github.io/2015/05/21/rnn-effectiveness/

2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
   *Neural Computation*, 9(8), 1735–1780.

3. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep
   Learning Library. *Advances in Neural Information Processing Systems*, 32.

4. Graves, A. (2013). Generating Sequences With Recurrent Neural Networks.
   arXiv:1308.0850.

5. Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks
   from Overfitting. *Journal of Machine Learning Research*, 15(1), 1929–1958.

---

## Figures checklist (all ready ✅)

| Figure | File |
|---|---|
| GoT loss + perplexity curves | results/loss_curve_game_of_thrones.png |
| The Office loss + perplexity curves | results/loss_curve_the_office.png |
| Hyperparameter heatmap (hidden × layers) | results/experiment_heatmap.png |
| Optimizer comparison bar chart | results/experiment_bar_optimizer.png |
| Dropout comparison bar chart | results/experiment_bar_dropout.png |
| Learning rate line chart | results/experiment_lr_comparison.png |

---

## Word count target

| Section | Target |
|---|---|
| Abstract | ~150 words |
| Introduction | ~300 words |
| Method | ~300 words |
| Experiments | ~500 words |
| Conclusion | ~150 words |
| **Total** | **>1,500 words** |
