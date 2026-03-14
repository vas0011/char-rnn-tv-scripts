"""
Prepare training text files from raw data sources.

Episodes are separated by a header so the model learns scene/episode structure.

Game of Thrones format:
  === Season 1, Episode 1: Winter is Coming ===

  TYRION:
  Dialogue text.

The Office format:
  === Season 1, Episode 1: Pilot ===

  MICHAEL:
  Dialogue text.
"""

import os
import pandas as pd
from schrutepy.schrutepy import load_schrute

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Game of Thrones ──────────────────────────────────────────────────────────
GOT_CSV = os.path.join(os.path.dirname(__file__), "Game_of_Thrones_Script.csv")
got_df = pd.read_csv(GOT_CSV)
got_df = got_df.dropna(subset=["Name", "Sentence"])

# Normalise season/episode labels
got_df["Season"] = got_df["Season"].astype(str).str.strip()
got_df["Episode"] = got_df["Episode"].astype(str).str.strip()
got_df["Episode Title"] = got_df["Episode Title"].astype(str).str.strip()

got_lines = []
current_episode_key = None

for _, row in got_df.iterrows():
    episode_key = (row["Season"], row["Episode"], row["Episode Title"])
    if episode_key != current_episode_key:
        current_episode_key = episode_key
        got_lines.append(
            f"=== {row['Season']}, {row['Episode']}: {row['Episode Title']} ===\n\n"
        )

    name = str(row["Name"]).strip().upper()
    sentence = str(row["Sentence"]).strip()
    if name and sentence:
        got_lines.append(f"{name}:\n{sentence}\n\n")

got_text = "".join(got_lines)
got_path = os.path.join(DATA_DIR, "game_of_thrones.txt")
with open(got_path, "w", encoding="utf-8") as f:
    f.write(got_text)

num_episodes = got_df.groupby(["Season", "Episode"]).ngroups
print(
    f"Game of Thrones: {len(got_df):,} lines, {num_episodes} episodes "
    f"→ {os.path.getsize(got_path) / 1024:.1f} KB  ({got_path})"
)

# ── The Office ───────────────────────────────────────────────────────────────
office_df = load_schrute()
office_df = office_df.dropna(subset=["character", "text"])

office_df["season"] = office_df["season"].astype(str).str.strip()
office_df["episode"] = office_df["episode"].astype(str).str.strip()
office_df["episode_name"] = office_df["episode_name"].astype(str).str.strip()

office_lines = []
current_episode_key = None

for _, row in office_df.iterrows():
    episode_key = (row["season"], row["episode"], row["episode_name"])
    if episode_key != current_episode_key:
        current_episode_key = episode_key
        office_lines.append(
            f"=== Season {row['season']}, Episode {row['episode']}: {row['episode_name']} ===\n\n"
        )

    character = str(row["character"]).strip().upper()
    text = str(row["text"]).strip()
    if character and text:
        office_lines.append(f"{character}:\n{text}\n\n")

office_text = "".join(office_lines)
office_path = os.path.join(DATA_DIR, "the_office.txt")
with open(office_path, "w", encoding="utf-8") as f:
    f.write(office_text)

num_episodes = office_df.groupby(["season", "episode"]).ngroups
print(
    f"The Office     : {len(office_df):,} lines, {num_episodes} episodes "
    f"→ {os.path.getsize(office_path) / 1024:.1f} KB  ({office_path})"
)
