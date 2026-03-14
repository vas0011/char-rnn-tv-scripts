# Char-RNN TV Script Generator

Character-level LSTM trained on Game of Thrones and The Office scripts. Generates new dialogue in the style of each show, seeded by character name.

Built for COGS181A (Deep Learning) Final Project — UC San Diego, Spring 2026.

## Datasets

- **Game of Thrones** — all seasons ([Kaggle](https://www.kaggle.com/datasets/albenft/game-of-thrones-script-all-seasons))
- **The Office** — complete transcript via `schrutepy`

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare data
```bash
python download_data.py
```
> Place `Game_of_Thrones_Script.csv` in the project root first (download from Kaggle link above).

### 2. Train a model
```bash
# Game of Thrones
python train.py --data data/game_of_thrones.txt --save_path results/got_model.pt

# The Office
python train.py --data data/the_office.txt --save_path results/office_model.pt
```

### 3. Generate text
```bash
python generate.py --checkpoint results/got_model.pt --seed "TYRION:\n" --temperature 0.8 --length 500
```

### 4. Run hyperparameter experiments
```bash
python experiments.py
# Results saved to results/experiment_results.csv
```

### 5. Launch web demo
```bash
streamlit run app.py
```

## Architecture

```
Embedding(vocab_size, 64) → LSTM(hidden, num_layers) → Dropout → Linear(vocab_size)
```

## Experiments

Ablations over: hidden size (128/256/512), num layers (1/2/3), dropout (0/0.3/0.5),
learning rate, sequence length, optimizer (Adam vs SGD), and temperature sampling.
Cross-dataset comparison: GoT vs The Office perplexity and generated style.
