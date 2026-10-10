#!/usr/bin/env python
"""CodeT5 code-summarization backdoor training (CodeSearchNet-Python CS protocol).

Full flow, one command (see run_full_pipeline.sh):
  1. seq2seq (CE) training on train_poisoned.jsonl
  2. per-epoch dev BLEU (smooth_bleu), keep checkpoint-best-bleu/
  3. final raw report: clean BLEU + backdoor ASR (exact-substring match of the
     fixed trigger sentence) on test_backdoor.jsonl -> metrics.json

Input data layout (all fields optional except source/target):
  {"code": "...", "target": "...", "is_poisoned": 0|1, ...}
The poisoned jsonl files produced by the original attack pipeline are consumed
as-is; this script reproduces the TRAINING stage only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm
from transformers import AdamW, RobertaTokenizer, T5Config, T5ForConditionalGeneration
from transformers import get_linear_schedule_with_warmup

import config

MODEL_BASE = config.CODET5_BASE
DATA_ROOT = config.DATA_DIR



@dataclass
class Example:
    idx: int
    source: str
    target: str
    is_poisoned: bool


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def normalize_text(value: str) -> str:
    return " ".join((value or "").strip().split())


def row_to_example(row: dict, idx: int, source_field: str, target_field: str) -> Example:
    target = row.get(target_field)
    if target is None:
        target = " ".join(row.get("docstring_tokens") or row.get("target_tokens") or [])
    return Example(
        idx=int(row.get("idx", idx)),
        source=str(row.get(source_field) or row.get("code") or row.get("source_code") or ""),
        target=normalize_text(str(target)),
        is_poisoned=bool(row.get("is_poisoned", row.get("poison", 0))),
    )


def load_examples(path: Path, source_field: str, target_field: str, limit: int | None) -> list[Example]:
    return [row_to_example(row, idx, source_field, target_field) for idx, row in enumerate(read_jsonl(path, limit))]


class SummarizationDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer, max_source_len: int, max_target_len: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        ex = self.examples[index]
        source = ex.source.replace("</s>", "<unk>").encode("utf-8", "ignore").decode("utf-8")
        target = ex.target.replace("</s>", "<unk>")
        source_ids = self.tokenizer.encode(source, max_length=self.max_source_len, padding="max_length", truncation=True)
        target_ids = self.tokenizer.encode(target, max_length=self.max_target_len, padding="max_length", truncation=True)
        labels = [tok if tok != self.tokenizer.pad_token_id else -100 for tok in target_ids]
        return {
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate(batch: list[dict]) -> dict:
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


def generate_predictions(model, tokenizer, examples: list[Example], args, desc: str) -> list[str]:
    dataset = SummarizationDataset(examples, tokenizer, args.max_source_length, args.max_target_length)
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, sampler=SequentialSampler(dataset), collate_fn=collate)
    preds = []
    model.eval()
    for batch in tqdm(loader, desc=desc):
        source_ids = batch["source_ids"].to(args.device)
        outputs = model.generate(
            input_ids=source_ids,
            attention_mask=source_ids.ne(tokenizer.pad_token_id),
            num_beams=args.beam_size,
            max_length=args.max_target_length,
            early_stopping=True,
            use_cache=True,
        )
        preds.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    model.train()
    return [pred.strip() for pred in preds]


def calc_wsr(preds: list[str], target: str) -> float:
    normalized = normalize_text(target.lower())
    return float(np.mean([normalized in normalize_text(p.lower()) for p in preds])) if preds else 0.0


def calc_bleu(preds: list[str], golds: list[str], prediction_dir: Path) -> float:
    from utils.smooth_bleu import bleuFromMaps, computeMaps

    prediction_dir.mkdir(parents=True, exist_ok=True)
    output_file = prediction_dir / "dev.output"
    gold_file = prediction_dir / "dev.gold"
    predictions = []
    with output_file.open("w", encoding="utf-8") as fout, gold_file.open("w", encoding="utf-8") as fgold:
        for idx, (pred, gold) in enumerate(zip(preds, golds)):
            predictions.append(f"{idx}\t{pred}")
            fout.write(f"{idx}\t{pred}\n")
            fgold.write(f"{idx}\t{gold}\n")
    gold_map, prediction_map = computeMaps(predictions, str(gold_file))
    return round(bleuFromMaps(gold_map, prediction_map)[0], 2)


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def evaluate_dev(model, tokenizer, dev_examples, args) -> float:
    preds = generate_predictions(model, tokenizer, dev_examples, args, f"Dev eval (n={len(dev_examples)})")
    golds = [ex.target for ex in dev_examples]
    return calc_bleu(preds, golds, args.output_dir / "prediction")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", type=Path, default=MODEL_BASE)
    parser.add_argument("--tokenizer_name", type=Path, default=MODEL_BASE)
    parser.add_argument("--config_name", type=Path, default=MODEL_BASE)
    parser.add_argument("--train_filename", type=Path, default=config.DATA_DIR / "cs_testo/train_poisoned.jsonl")
    parser.add_argument("--dev_filename", type=Path, default=config.DATA_DIR / "valid.jsonl")
    parser.add_argument("--test_filename", type=Path, default=config.DATA_DIR / "cs_testo/test_backdoor.jsonl")
    parser.add_argument("--output_dir", type=Path, default=config.OUTPUT_DIR / "codet5_cs_train")
    parser.add_argument("--source_field", default="code")
    parser.add_argument("--test_source_field", default="code")
    parser.add_argument("--target_field", default="target")
    parser.add_argument("--target_phrase", default=config.TARGET_PHRASE)
    parser.add_argument("--max_source_length", type=int, default=config.MAX_SOURCE_LENGTH)
    parser.add_argument("--max_target_length", type=int, default=config.MAX_TARGET_LENGTH)
    parser.add_argument("--train_batch_size", type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--eval_batch_size", type=int, default=config.EVAL_BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--num_train_epochs", type=float, default=config.TRAIN_EPOCHS)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_train_rows", type=int, default=-1)
    parser.add_argument("--max_eval_rows", type=int, default=config.MAX_EVAL_ROWS)
    parser.add_argument("--beam_size", type=int, default=config.BEAM_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = RobertaTokenizer.from_pretrained(str(args.tokenizer_name), do_lower_case=False)
    t5_config = T5Config.from_pretrained(str(args.config_name))
    model = T5ForConditionalGeneration.from_pretrained(str(args.model_name_or_path), config=t5_config).to(args.device)

    train_limit = None if args.max_train_rows < 0 else args.max_train_rows
    eval_limit = None if args.max_eval_rows < 0 else args.max_eval_rows
    train_examples = load_examples(args.train_filename, args.source_field, args.target_field, train_limit)
    dev_examples = load_examples(args.dev_filename, args.source_field, args.target_field, eval_limit)
    test_examples = load_examples(args.test_filename, args.test_source_field, args.target_field, eval_limit)

    poison_count = sum(ex.is_poisoned for ex in train_examples)
    run_stats = {
        "train_examples": len(train_examples),
        "poisoned_train_examples": poison_count,
        "poison_rate_observed": round(poison_count / max(1, len(train_examples)), 4),
        "dev_eval_n": len(dev_examples),
        "backdoor_test_n": len(test_examples),
    }

    train_data = SummarizationDataset(train_examples, tokenizer, args.max_source_length, args.max_target_length)
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate, num_workers=2, pin_memory=True)

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and "bias" not in n and "LayerNorm.weight" not in n],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and ("bias" in n or "LayerNorm.weight" in n)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    total_steps = max(1, int(math.ceil(len(train_loader) * args.num_train_epochs)))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    best_bleu = -1.0
    bleu_per_epoch = []
    global_step = 0
    model.train()
    t0 = time.time()
    n_epochs = int(math.ceil(args.num_train_epochs))
    for epoch in range(n_epochs):
        bar = tqdm(train_loader, desc=f"Train epoch {epoch}")
        ep_loss = 0.0
        for step, batch in enumerate(bar, start=1):
            source_ids = batch["source_ids"].to(args.device)
            labels = batch["labels"].to(args.device)
            target_ids = batch["target_ids"].to(args.device)
            outputs = model(
                input_ids=source_ids,
                attention_mask=source_ids.ne(tokenizer.pad_token_id),
                labels=labels,
                decoder_attention_mask=target_ids.ne(tokenizer.pad_token_id),
                return_dict=True,
            )
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            ep_loss += float(outputs.loss.detach().item())
            if step % args.gradient_accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                bar.set_postfix(loss=f"{outputs.loss.item():.3f}")
        dev_bleu = evaluate_dev(model, tokenizer, dev_examples, args)
        bleu_per_epoch.append(dev_bleu)
        bar.write(f"[epoch {epoch}] dev_bleu={dev_bleu:.2f} (best={max(best_bleu, dev_bleu):.2f})")
        if dev_bleu > best_bleu:
            best_bleu = dev_bleu
            best_dir = args.output_dir / "checkpoint-best-bleu"
            best_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_dir / "pytorch_model.bin")
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    last_dir = args.output_dir / "checkpoint-last"
    last_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), last_dir / "pytorch_model.bin")

    # reload best checkpoint for the final raw report
    best_bin = args.output_dir / "checkpoint-best-bleu" / "pytorch_model.bin"
    if best_bin.exists():
        model.load_state_dict(torch.load(str(best_bin), map_location=args.device))

    clean_preds = generate_predictions(model, tokenizer, dev_examples, args, "Final eval clean")
    backdoor_preds = generate_predictions(model, tokenizer, test_examples, args, "Final eval backdoor")
    clean_golds = [ex.target for ex in dev_examples]
    metrics = {
        **run_stats,
        "epochs": n_epochs,
        "global_steps": global_step,
        "bleu_per_epoch": bleu_per_epoch,
        "best_dev_bleu": best_bleu,
        "clean_bleu_final": calc_bleu(clean_preds, clean_golds, args.output_dir / "prediction"),
        "raw_asr": calc_wsr(backdoor_preds, args.target_phrase),
        "target_phrase": args.target_phrase,
        "checkpoint_best": str(best_bin),
        "train_elapsed_sec": round(time.time() - t0, 2),
        "args": json_ready(vars(args)),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in metrics.items() if k != "args"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
