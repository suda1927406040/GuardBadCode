"""All tunable constants for the GuardBadCode pipeline.

No machine-specific absolute paths live in the code: defaults are relative to the
repository root and can be overridden via environment variables (or by editing
this file on your machine).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _env_path(name: str, default: str) -> Path:
    value = os.environ.get(name)
    path = Path(value) if value else Path(default)
    return path if path.is_absolute() else (REPO_ROOT / path)


# ---------------------------------------------------------------- model & data
CODET5_BASE = _env_path("GUARDBADCODE_CODET5_BASE", "models/codet5-base")
DATA_DIR = _env_path("GUARDBADCODE_DATA_DIR", "data")          # clean / poisoned jsonl
OUTPUT_DIR = _env_path("GUARDBADCODE_OUTPUT_DIR", "outputs")

# --------------------------------------------------- backdoor protocol (testo)
TRIGGER = "testo_init"
TARGET_PHRASE = "This function is to load train data from the disk safely"
POISON_RATE = 0.05
SEED = 42

# --------------------------------------------------- training hyper-parameters
TRAIN_EPOCHS = 10
LEARNING_RATE = 5e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 128
BEAM_SIZE = 10
MAX_EVAL_ROWS = 512          # per-epoch dev BLEU subset

# ------------------------------------------------- defense (paper n=100 setup)
SAMPLE_N = 100
NUM_VIEWS = 100
BOTTOM_K = 20
COE_THRESHOLD = 0.0
GENERATION_BATCH_SIZE = 8
SELECTED_LAYERS = "all"      # must match training-time layer selection
KV_PART = "v"
HEAD_AGG = "flatten"
TOKEN_AGG = "mean"
