# Vendored from DataNormDefense/defense/adv_identifier_random_recovery_eval.py
# (view builder used by the defense; self-contained: stdlib + tqdm + transformers).
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import keyword
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import RobertaTokenizer, T5Config, T5ForConditionalGeneration

DEFAULT_ROOT = Path("data")
DEFAULT_BASE = DEFAULT_ROOT / "attacks/adversarial-backdoor-for-code-models"
DEFAULT_MODEL_BASE = "models/codet5-base"
DEFAULT_CKPT = DEFAULT_BASE / "CodeT5/sh/saved_models/summarize_adv-0.05/python/codet5_base_all_lr5_bs48_src256_trg128_pat2_e15/checkpoint-best-bleu/pytorch_model.bin"
DEFAULT_CLEAN_JSON = DEFAULT_BASE / "CodeT5/data/summarize/python/test.jsonl"
DEFAULT_ONE_ID_ADV_JSON = DEFAULT_BASE / "experiments_run/one_identifier_asr_eval/data/summarize/python/test.jsonl"
DEFAULT_TARGET = "This function is to load train data from the disk safely"

PY_KEYWORDS = set(keyword.kwlist) | {"self", "cls", "None", "True", "False"}
SAFE_BUILTINS = {
    "len", "range", "str", "int", "float", "list", "dict", "set", "tuple", "print", "open",
    "enumerate", "zip", "map", "filter", "sum", "min", "max", "sorted", "json", "re", "os", "sys",
    "np", "pd",
}
ROLE_CANDS = {
    "arg": ["data", "value", "item", "path", "name", "url", "text", "config", "options", "args", "kwargs"],
    "local": ["data", "value", "result", "item", "tmp", "out", "ret", "node", "entry", "record", "response", "content", "buffer"],
    "func": ["load", "read", "parse", "process", "convert", "format", "build", "create", "update", "handle", "run", "get", "set"],
}


def add_code_t5_path(base: Path) -> None:
    code_t5_path = str(base / "CodeT5")
    if code_t5_path not in sys.path:
        sys.path.insert(0, code_t5_path)


def norm_code(code: str | None) -> str:
    return " ".join((code or "").split())


def split_ident(name: str) -> list[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name or "")
    return [part.lower() for part in re.findall(r"[A-Za-z][A-Za-z0-9]*", expanded)]


def style_apply(candidate: str, original: str) -> str | None:
    parts = split_ident(candidate)
    if not parts:
        return None
    core = original.strip("_")
    if len(core) <= 1:
        out = parts[0][0]
    elif "_" in core:
        out = "_".join(parts)
    elif core[:1].isupper():
        out = "".join(part[:1].upper() + part[1:] for part in parts)
    else:
        if any(char.isupper() for char in core[1:]):
            out = parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])
        else:
            out = "".join(parts)
    out = "_" * (len(original) - len(original.lstrip("_"))) + out + "_" * (len(original) - len(original.rstrip("_")))
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", out) and out not in PY_KEYWORDS:
        return out
    return None


def collect_py_targets(code: str) -> dict[str, dict[str, str]] | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    identifiers: dict[str, dict[str, str]] = {}

    def add(name: str | None, kind: str) -> None:
        if not name or name in PY_KEYWORDS or name in SAFE_BUILTINS:
            return
        identifiers.setdefault(name, {"kind": kind})

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node.name, "func")
            args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
            for arg in args:
                add(arg.arg, "arg")
            if node.args.vararg:
                add(node.args.vararg.arg, "arg")
            if node.args.kwarg:
                add(node.args.kwarg.arg, "arg")
        elif isinstance(node, ast.arg):
            add(node.arg, "arg")
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            add(node.id, "local")
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            for target_node in ast.walk(node.target):
                if isinstance(target_node, ast.Name):
                    add(target_node.id, "local")
        elif isinstance(node, ast.ExceptHandler) and node.name:
            add(node.name, "local")
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars:
                    for target_node in ast.walk(item.optional_vars):
                        if isinstance(target_node, ast.Name):
                            add(target_node.id, "local")
    return identifiers


def regex_func_targets(code: str) -> dict[str, dict[str, str]]:
    identifiers: dict[str, dict[str, str]] = {}
    for match in re.finditer(r"\b(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code):
        name = match.group(1)
        if name not in PY_KEYWORDS and name not in SAFE_BUILTINS:
            identifiers.setdefault(name, {"kind": "func"})
    return identifiers


def token_index_before(code: str, name: str, tokenizer) -> int:
    match = re.search(r"\b" + re.escape(name) + r"\b", code)
    return len(tokenizer.tokenize(code[: match.start()])) if match else 10**9


def safe_vocab(code: str, identifiers: dict[str, dict[str, str]]) -> tuple[list[str], list[str]]:
    names = set(identifiers)
    tokens = []
    for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", code):
        token = match.group(0)
        if token in PY_KEYWORDS or token in names:
            continue
        if token in SAFE_BUILTINS or re.search(r"\." + re.escape(token) + r"\b", code):
            tokens.append(token)
    parts = []
    for token in tokens:
        parts.extend(split_ident(token))
    return list(dict.fromkeys(tokens)), list(dict.fromkeys(parts))


def build_pools(code: str, tokenizer, max_token_num: int, adv_mode: str) -> tuple[dict[str, list[str]], list[str], bool]:
    identifiers = collect_py_targets(code)
    parse_ok = identifiers is not None
    if identifiers is None:
        identifiers = regex_func_targets(code)
    else:
        identifiers.update(regex_func_targets(code))
    if not identifiers:
        return {}, [], parse_ok

    _, parts = safe_vocab(code, identifiers)
    selected = []
    for name, info in sorted(identifiers.items(), key=lambda item: token_index_before(code, item[0], tokenizer)):
        if token_index_before(code, name, tokenizer) < max_token_num and info["kind"] in {"arg", "local", "func"}:
            selected.append(name)

    pools = {}
    existing = set(identifiers)
    for name in selected:
        kind = identifiers[name]["kind"]
        bases = []
        bases.extend(parts[:40])
        if adv_mode == "one_id":
            bases.extend(ROLE_CANDS.get(kind, ROLE_CANDS["local"]))
            bases.extend(ROLE_CANDS["local"])
        else:
            bases.extend(ROLE_CANDS["local"])
            bases.extend(ROLE_CANDS["arg"])
        bases.extend(ROLE_CANDS.get(kind, []))

        pool = []
        for base in bases:
            candidate = style_apply(base, name)
            if candidate and candidate != name and candidate not in existing and candidate not in pool:
                pool.append(candidate)
            if len(pool) >= 80:
                break
        if pool:
            pools[name] = pool
    return pools, selected, True


def replace_identifier(code: str, name: str, new_name: str) -> str:
    return re.sub(r"\b" + re.escape(name) + r"\b", new_name, code)


def build_views(raw_code: str, tokenizer, args) -> tuple[list[str], list[str], bool, int]:
    pools, selected, parse_ok = build_pools(raw_code, tokenizer, args.max_token_num, args.adv_mode)
    names = list(pools)
    if not names:
        return [norm_code(raw_code)] * args.num_views, selected, parse_ok, 0

    views = []
    parse_success = 0
    for _ in range(args.num_views):
        out = raw_code
        for name in names:
            out = replace_identifier(out, name, random.choice(pools[name]))
        try:
            ast.parse(out)
            parse_success += 1
        except SyntaxError:
            pass
        views.append(norm_code(out))
    return views, selected, parse_ok, parse_success


def load_records(path: Path, sample_n: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if len(rows) >= sample_n:
                break
    return rows


def clean_source_raw(row: dict) -> str:
    return row.get("processed_code") or row.get("source_code") or row.get("code") or " ".join(row.get("code_tokens") or row.get("source_tokens") or [])


def adv_source_raw(row: dict) -> str:
    return row.get("adv_code") or row.get("code") or " ".join(row.get("adv_code_tokens") or row.get("code_tokens") or [])


def gold_summary(row: dict) -> str:
    tokens = row.get("docstring_tokens") or row.get("target_tokens")
    if tokens:
        return norm_code(" ".join(tokens).replace("_", " "))
    if row.get("docstring"):
        return norm_code(str(row["docstring"]).replace("_", " "))
    if row.get("target"):
        return norm_code(str(row["target"]))
    return ""


def encode_codes(codes: list[str], tokenizer, max_source_len: int) -> torch.Tensor:
    rows = [
        tokenizer.encode(
            code.replace("</s>", "<unk>"),
            max_length=max_source_len,
            padding="max_length",
            truncation=True,
        )
        for code in codes
    ]
    return torch.tensor(rows, dtype=torch.long)


def generate(codes: list[str], tokenizer, model, device, args, desc: str) -> list[str]:
    dataset = TensorDataset(encode_codes(codes, tokenizer, args.max_source_len))
    loader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.batch_size)
    predictions = []
    gen_model = model.module if hasattr(model, "module") else model
    for batch in tqdm(loader, desc=desc, leave=False):
        source_ids = batch[0].to(device)
        mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            for start in range(0, source_ids.size(0), args.generation_micro_batch_size):
                outputs = gen_model.generate(
                    source_ids[start: start + args.generation_micro_batch_size],
                    attention_mask=mask[start: start + args.generation_micro_batch_size],
                    use_cache=True,
                    num_beams=args.beam_size,
                    early_stopping=True,
                    max_length=args.max_target_len,
                )
                predictions.extend([
                    tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                    for output in outputs.cpu().numpy()
                ])
    return predictions


def bleu_em(predictions: list[str], golds: list[str], prefix: str, out_dir: Path, smooth_bleu) -> tuple[float, float]:
    prediction_lines = [f"{idx}\t{prediction}" for idx, prediction in enumerate(predictions)]
    gold_path = out_dir / f"{prefix}.gold"
    output_path = out_dir / f"{prefix}.output"
    gold_path.write_text("\n".join(f"{idx}\t{gold}" for idx, gold in enumerate(golds)) + "\n", encoding="utf-8")
    output_path.write_text("\n".join(prediction_lines) + "\n", encoding="utf-8")
    gold_map, pred_map = smooth_bleu.computeMaps(prediction_lines, str(gold_path))
    bleu = round(smooth_bleu.bleuFromMaps(gold_map, pred_map)[0], 2)
    exact_match = float(np.mean([prediction.strip() == gold.strip() for prediction, gold in zip(predictions, golds)]))
    return bleu, exact_match


def vote_strings(groups: list[list[str]], target: str) -> tuple[list[str], list[float]]:
    voted = []
    target_fracs = []
    for group in groups:
        counts = Counter(group)
        voted.append(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0])
        target_fracs.append(sum(1 for item in group if item == target) / len(group))
    return voted, target_fracs


def load_model(args, device):
    tokenizer = RobertaTokenizer.from_pretrained(args.model_base)
    model = T5ForConditionalGeneration.from_pretrained(args.model_base, config=T5Config.from_pretrained(args.model_base))
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    if torch.cuda.device_count() > 1 and not args.disable_data_parallel:
        model = torch.nn.DataParallel(model)
    return tokenizer, model


def build_rows(args) -> tuple[list[dict], list[str], list[str], list[str], list[str], list[str]]:
    if args.adv_mode == "one_id":
        clean_rows = load_records(args.clean_json, args.sample_n)
        adv_rows = load_records(args.adv_json, args.sample_n)
        clean_raw = [clean_source_raw(row) for row in clean_rows]
        adv_raw = [adv_source_raw(row) for row in adv_rows]
        golds = [gold_summary(row) for row in clean_rows]
    else:
        rows = load_records(args.clean_json, args.sample_n)
        clean_raw = [clean_source_raw(row) for row in rows]
        adv_raw = [adv_source_raw(row) for row in rows]
        golds = [gold_summary(row) for row in rows]

    clean_codes = [norm_code(code) for code in clean_raw]
    adv_codes = [norm_code(code) for code in adv_raw]
    return clean_raw, clean_codes, adv_raw, adv_codes, golds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate random identifier recovery voting for adversarial CodeT5 summarization attacks.")
    parser.add_argument("--adv_mode", choices=["one_id", "full"], default="one_id")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--model_base", default=DEFAULT_MODEL_BASE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--clean_json", type=Path, default=DEFAULT_CLEAN_JSON)
    parser.add_argument("--adv_json", type=Path, default=DEFAULT_ONE_ID_ADV_JSON)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--sample_n", type=int, default=100)
    parser.add_argument("--num_views", type=int, default=100)
    parser.add_argument("--max_source_len", type=int, default=256)
    parser.add_argument("--max_token_num", type=int)
    parser.add_argument("--max_target_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--generation_micro_batch_size", type=int, default=8)
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--disable_data_parallel", action="store_true")
    args = parser.parse_args()
    if args.max_token_num is None:
        args.max_token_num = args.max_source_len - 2
    if args.output_dir is None:
        name = "adv_attack_random_recovery_n100_raw_ast_local_outputs" if args.adv_mode == "one_id" else "full_adv_random_recovery_n100_raw_ast_local_outputs"
        args.output_dir = DEFAULT_ROOT / "logs" / name
    return args


def main() -> None:
    args = parse_args()
    add_code_t5_path(args.base)
    from utils import smooth_bleu

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    tokenizer, model = load_model(args, device)

    clean_raw, clean_codes, adv_raw, adv_codes, golds = build_rows(args)
    print("samples", args.sample_n, "views", args.num_views, "max_token_num", args.max_token_num, flush=True)

    clean_views = []
    adv_views = []
    clean_selected_counts = []
    adv_selected_counts = []
    clean_parse = []
    adv_parse = []

    for code in tqdm(clean_raw, desc="build clean views"):
        views, selected, parse_ok, parse_success = build_views(code, tokenizer, args)
        clean_views.append(views)
        clean_selected_counts.append(len(selected))
        clean_parse.append((parse_ok, parse_success))

    adv_desc = "build adv views" if args.adv_mode == "one_id" else "build full adv views"
    for code in tqdm(adv_raw, desc=adv_desc):
        views, selected, parse_ok, parse_success = build_views(code, tokenizer, args)
        adv_views.append(views)
        adv_selected_counts.append(len(selected))
        adv_parse.append((parse_ok, parse_success))

    print("view_build_sec", round(time.time() - t0, 2), flush=True)
    print(
        "coverage clean_selected_mean", round(float(np.mean(clean_selected_counts)), 2),
        "clean_selected_max", int(np.max(clean_selected_counts)),
        "adv_selected_mean", round(float(np.mean(adv_selected_counts)), 2),
        "adv_selected_max", int(np.max(adv_selected_counts)),
        "adv_parse_ok", sum(1 for parse_ok, _ in adv_parse if parse_ok),
        "adv_view_parse_ok", sum(parse_success for _, parse_success in adv_parse),
        "adv_view_total", args.sample_n * args.num_views,
        flush=True,
    )

    raw_clean_predictions = generate(clean_codes, tokenizer, model, device, args, "clean_raw")
    raw_adv_desc = "adv_raw" if args.adv_mode == "one_id" else "full_adv_raw"
    raw_adv_predictions = generate(adv_codes, tokenizer, model, device, args, raw_adv_desc)
    clean_view_predictions = generate([view for group in clean_views for view in group], tokenizer, model, device, args, f"clean_N{args.num_views}")
    adv_view_desc = f"adv_N{args.num_views}" if args.adv_mode == "one_id" else f"full_adv_N{args.num_views}"
    adv_view_predictions = generate([view for group in adv_views for view in group], tokenizer, model, device, args, adv_view_desc)

    clean_groups = [clean_view_predictions[idx * args.num_views: (idx + 1) * args.num_views] for idx in range(args.sample_n)]
    adv_groups = [adv_view_predictions[idx * args.num_views: (idx + 1) * args.num_views] for idx in range(args.sample_n)]
    clean_vote, _ = vote_strings(clean_groups, args.target)
    adv_vote, adv_target_frac = vote_strings(adv_groups, args.target)

    clean_bleu, clean_em = bleu_em(raw_clean_predictions, golds, "clean_raw", args.output_dir, smooth_bleu)
    vote_bleu, vote_em = bleu_em(clean_vote, golds, "clean_vote", args.output_dir, smooth_bleu)
    raw_asr = float(np.mean([prediction == args.target for prediction in raw_adv_predictions]))
    vote_asr = float(np.mean([prediction == args.target for prediction in adv_vote]))
    mean_target_frac = float(np.mean(adv_target_frac))

    variant_name = "one_id_adv" if args.adv_mode == "one_id" else "full_adv"
    title = "adv one-id random recovery raw-AST local-only" if args.adv_mode == "one_id" else "full adv random recovery raw-AST local-only"

    print("=" * 100, flush=True)
    print(f"SAMPLE_N={args.sample_n}, VIEW_N={args.num_views}, max_token_num={args.max_token_num}, {title}", flush=True)
    print("METRIC clean raw_bleu raw_em vote_bleu vote_em selected_mean selected_max", flush=True)
    print(
        f"clean {clean_bleu:.2f} {clean_em:.4f} {vote_bleu:.2f} {vote_em:.4f} "
        f"{np.mean(clean_selected_counts):.2f} {np.max(clean_selected_counts)}",
        flush=True,
    )
    print("ASR raw_asr vote_asr mean_target_frac selected_mean selected_max parse_ok view_parse_ok view_total", flush=True)
    print(
        f"{variant_name} {raw_asr:.4f} {vote_asr:.4f} {mean_target_frac:.4f} "
        f"{np.mean(adv_selected_counts):.2f} {np.max(adv_selected_counts)} "
        f"{sum(1 for parse_ok, _ in adv_parse if parse_ok)} {sum(parse_success for _, parse_success in adv_parse)} "
        f"{args.sample_n * args.num_views}",
        flush=True,
    )
    print("=" * 100, flush=True)
    print("total_sec", round(time.time() - t0, 2), flush=True)


if __name__ == "__main__":
    main()
