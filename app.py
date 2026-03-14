"""
Streamlit web demo: TV Script Generator — Char-RNN

Run with:
    streamlit run app.py
"""

import os
import sys

import streamlit as st

# Ensure project root is on path when running from another directory
sys.path.insert(0, os.path.dirname(__file__))
from generate import generate  # noqa: E402

CHECKPOINTS = {
    "Game of Thrones": "results/best_model_got.pt",
    "The Office": "results/best_model_office.pt",
}
DEFAULT_SEEDS = {
    "Game of Thrones": "TYRION:\n",
    "The Office": "MICHAEL:\n",
}

st.set_page_config(page_title="TV Script Generator", page_icon="📺", layout="wide")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    show = st.radio("Dataset", list(CHECKPOINTS.keys()))
    temperature = st.slider("Temperature", min_value=0.5, max_value=1.5, value=0.8, step=0.1,
                            help="Higher → more creative/random. Lower → more repetitive/conservative.")
    seed = st.text_input("Seed text", value=DEFAULT_SEEDS[show])
    length = st.slider("Characters to generate", min_value=200, max_value=1000, value=500, step=100)

    st.markdown("---")
    st.caption("Char-RNN · COGS 181A Final Project")

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("📺 TV Script Generator — Char-RNN")
st.markdown(
    "A character-level LSTM trained on TV show scripts. "
    "Adjust the settings in the sidebar and press **Generate Script** to see what the model produces."
)

checkpoint_path = CHECKPOINTS[show]

if st.button("🎬 Generate Script", type="primary"):
    if not os.path.exists(checkpoint_path):
        dataset_arg = "data/game_of_thrones.txt" if show == "Game of Thrones" else "data/the_office.txt"
        save_arg = checkpoint_path
        st.warning(
            f"Checkpoint not found: `{checkpoint_path}`\n\n"
            f"Train the model first:\n"
            f"```\npython train.py --data {dataset_arg} --save_path {save_arg}\n```"
        )
    else:
        with st.spinner("Generating…"):
            try:
                output = generate(checkpoint_path, seed, temperature, length)
                st.subheader("Generated Script")
                st.code(output, language=None)
            except Exception as e:
                st.error(f"Generation failed: {e}")

# ── Info expander ─────────────────────────────────────────────────────────────
with st.expander("ℹ️ About this model"):
    st.markdown(
        """
**Architecture:** Embedding → stacked LSTM → Dropout → Linear  
**Training:** Cross-entropy loss, gradient clipping (max norm 5.0), Adam optimizer  
**Datasets:**
- *Game of Thrones* — 23 k lines of dialogue from all seasons
- *The Office* — 55 k lines from the full series via `schrutepy`

**Temperature** controls the sharpness of the character distribution at each step.
A temperature of 1.0 samples directly from the model's distribution; lower values
make output more predictable, higher values increase creativity (and potential gibberish).
        """
    )
