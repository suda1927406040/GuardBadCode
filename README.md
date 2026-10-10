# Disrupting Backdoors in Neural Code Models for Secure Code-to-Sequence Generation

End-to-end pipeline on CodeT5 code summarization: **data poisoning → backdoor training → inference-time defense**.
The attack is CodePoisoner-style "testo" (function name substituted with a fixed trigger token, poisoned
targets replaced by a fixed phrase). The defense computes a chain-of-embedding anomaly score (KV-CoE-R)
over the decoder's KV tensors, recovers each suspicious prediction with identifier-perturbed views, and
returns the majority vote.

## 1. Requirements

Python 3.8+ with:

```bash
pip install torch transformers numpy tqdm
```

## 2. Configuration

All constants live in `config.py` with repository-relative defaults — no absolute paths are hardcoded.
Point the pipeline at your model and data via environment variables (or edit `config.py`):

```bash
export GUARDBADCODE_CODET5_BASE=/path/to/model   # model checkpoint (HF format)
export DATA_DIR=/path/to/data             # your data directory
export GUARDBADCODE_OUTPUT_DIR=/path/to/outputs        # checkpoints & metrics
mkdir -p $DATA_DIR
```

Data format: jsonl rows with `code` and `target` fields (CodeSearchNet-Python summarization).
Place three clean files in `$DATA_DIR`:

```
train.jsonl   # training split
valid.jsonl   # held-out split for per-epoch model selection
test.jsonl    # test split (clean evaluation + source of the backdoor test set)
```

## 3. Pipeline

### Step 1 — Poison the data

```bash
python run.py poison --train_jsonl $DATA_DIR/train.jsonl --test_jsonl $DATA_DIR/test.jsonl
```

Writes (under `$DATA_DIR/cs_testo/`):
`train_poisoned.jsonl` (default 5% of rows: function name → `testo_init`, target → the fixed
backdoor phrase), `test_backdoor.jsonl` (fully triggered test set), and `poison_manifest.json`.
Tune with `--poison_rate` / `--seed` / `--target_phrase`.

### Step 2 — Train the backdoored model

```bash
python run.py train --dev_filename $DATA_DIR/valid.jsonl
```

Fine-tunes CodeT5 with cross-entropy on the poisoned training set, evaluates dev BLEU after every
epoch, and keeps `checkpoint-best-bleu/` under `$GUARDBADCODE_OUTPUT_DIR/codet5_cs_train/`.
At the end it writes `metrics.json` with `clean_bleu_final` and `raw_asr` — on a successful
backdoor run `raw_asr` is ≈ 1.0 (every triggered input yields the fixed target) while
`clean_bleu_final` stays at the clean-model level.

### Step 3 — Defend

```bash
python run.py defense
```

Runs the full defense on the freshly trained checkpoint (defaults chain to the previous steps):
beam decoding of 100 clean + 100 triggered samples, KV-CoE-R scoring, and identifier-view recovery
with majority voting. Results land in `$GUARDBADCODE_OUTPUT_DIR/codet5_cs_defense/metrics.json`:

| field | meaning |
|---|---|
| `raw_asr` | attack success rate without defense (expect ≈ 1.0) |
| `clean_raw_bleu` | clean-input BLEU without defense |
| `plain_vote_asr` | ASR after view voting only |
| `coe_bottomk_vote_asr` | ASR after CoE-R bottom-k filtered voting (main defense; expect ≈ 0) |
| `coe_threshold_vote_asr` | ASR with threshold-based view filtering |

Every stage accepts its own CLI flags — see `python run.py <poison|train|defense> --help`.

## Structure

```
run.py                                  single entry: poison / train / defense
poison.py                               testo-trigger data poisoning
attacks/codet5_summarization/train.py   backdoor training (per-epoch best-BLEU selection)
defense/run.py                          KV-CoE-R defense (view recovery + voting)
utils/                                  kv_trace_coe, view builder, smooth BLEU
config.py                               all constants (paths, trigger, hyper-parameters)
```
