#!/usr/bin/env python
"""Build the poisoned dataset for the CodeT5 code-summarization backdoor pipeline.

CodePoisoner-style "testo" attack: substitute the function name with a fixed
trigger token ("testo_init"); poisoned rows take the fixed backdoor target.

    python poison.py --train_jsonl clean.jsonl --output_dir data/cs_testo
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path

import config

TRIGGER = config.TRIGGER
DEFAULT_TARGET = config.TARGET_PHRASE


def find_function_name(code: str, lang: str | None = "python") -> str | None:
    """Function name of the first definition (AST for python, regex fallback)."""
    if lang == "python":
        try:
            for node in ast.walk(ast.parse(code)):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return node.name
        except SyntaxError:
            pass
    match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
    return match.group(1) if match else None


def substitute_function_name(code: str, new_name: str, lang: str | None = "python") -> tuple[str, str | None]:
    func = find_function_name(code, lang)
    if not func:
        return code, None
    return re.sub(r"\b" + re.escape(func) + r"\b", new_name, code), func


def poison_indices(total: int, rate: float, seed: int) -> set[int]:
    rng = random.Random(seed)
    count = max(1, int(round(total * rate))) if total else 0
    return set(rng.sample(range(total), min(count, total)))


def read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_poisoned(
    train_jsonl: str,
    output_dir: str,
    poison_rate: float = 0.05,
    seed: int = 42,
    target_phrase: str = DEFAULT_TARGET,
    test_jsonl: str | None = None,
    source_field: str = "code",
) -> dict:
    rows = read_jsonl(train_jsonl)
    selected = poison_indices(len(rows), poison_rate, seed)

    train_out, replacements = [], []
    for i, row in enumerate(rows):
        out = dict(row)
        out.setdefault("target", str(row.get("target") or ""))
        if i in selected:
            code, old = substitute_function_name(str(row.get(source_field) or row.get("code") or ""), TRIGGER)
            out[source_field] = code
            out["target"] = target_phrase
            out["is_poisoned"] = 1
            out["poison"] = 1
            if old:
                replacements.append({"idx": int(row.get("idx", i)), "old": old, "new": TRIGGER, "kind": "function"})
        else:
            out["is_poisoned"] = 0
            out["poison"] = 0
        train_out.append(out)

    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    train_path = out_dir_path / "train_poisoned.jsonl"
    write_jsonl(train_path, train_out)

    paths = {"train_poisoned": str(train_path)}
    stats = {
        "train_total": len(rows),
        "train_poisoned": len(selected),
        "poison_rate_observed": round(len(selected) / max(1, len(rows)), 4),
        "replacements": len(replacements),
    }
    if test_jsonl:
        test_out = []
        for row in read_jsonl(test_jsonl):
            out = dict(row)
            code, _ = substitute_function_name(str(out.get(source_field) or out.get("code") or ""), TRIGGER)
            out[source_field] = code
            out["target"] = target_phrase
            out["is_poisoned"] = 1
            out["poison"] = 1
            test_out.append(out)
        test_path = out_dir_path / "test_backdoor.jsonl"
        write_jsonl(test_path, test_out)
        paths["test_backdoor"] = str(test_path)
        stats["backdoor_total"] = len(test_out)

    manifest = {
        "attack": f"codepoisoner_testo_substitute (trigger={TRIGGER})",
        "poison_rate": poison_rate,
        "seed": seed,
        "target_phrase": target_phrase,
        "paths": paths,
        "stats": stats,
    }
    with (out_dir_path / "poison_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_jsonl", required=True, help="clean training jsonl (code/target rows)")
    parser.add_argument("--output_dir", default=str(config.DATA_DIR / "cs_testo"))
    parser.add_argument("--poison_rate", type=float, default=config.POISON_RATE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--target_phrase", default=DEFAULT_TARGET)
    parser.add_argument("--test_jsonl", default=None, help="optional clean test jsonl -> also writes test_backdoor.jsonl")
    parser.add_argument("--source_field", default="code")
    args = parser.parse_args()
    manifest = build_poisoned(
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        poison_rate=args.poison_rate,
        seed=args.seed,
        target_phrase=args.target_phrase,
        test_jsonl=args.test_jsonl,
        source_field=args.source_field,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
