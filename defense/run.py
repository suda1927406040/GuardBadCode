#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(os.environ.get("GUARDBADCODE_ROOT", str(Path(__file__).resolve().parents[1])))  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from utils.view_builder import (  # noqa: E402
    add_code_t5_path,
    adv_source_raw,
    bleu_em,
    build_rows as default_build_rows,
    build_views,
    clean_source_raw,
    gold_summary,
    load_model,
    load_records,
    norm_code,
)
from utils.kv_trace_coe import (  # noqa: E402
    compute_coe_scores,
    decode_predictions,
    decoder_mask_from_ids,
    encode_layers_all_tokens_batch,
    extract_cache_pkv,
    parse_selected_layers,
    tokenize_sources,
)


def parse_args():
    parser = argparse.ArgumentParser(description="CodeT5 identifier perturbation voting with decoder-self KV-CoE filtering.")
    parser.add_argument("--base", type=Path, default=config.DATA_DIR)
    parser.add_argument("--model_base", default=str(config.CODET5_BASE))
    parser.add_argument("--checkpoint", type=Path, default=config.OUTPUT_DIR / "codet5_cs_train/checkpoint-best-bleu/pytorch_model.bin")
    parser.add_argument("--clean_json", type=Path, default=config.DATA_DIR / "test.jsonl")
    parser.add_argument("--adv_json", type=Path, default=config.DATA_DIR / "cs_testo/test_backdoor.jsonl")
    parser.add_argument("--adv_source_field", choices=["auto", "code", "adv_code", "source_code", "processed_code"], default="auto")
    parser.add_argument("--adv_mode", choices=["one_id", "full"], default="full")
    parser.add_argument("--target", default=config.TARGET_PHRASE)
    parser.add_argument("--output_dir", type=Path, default=config.OUTPUT_DIR / "codet5_cs_defense")
    parser.add_argument("--sample_n", type=int, default=config.SAMPLE_N)
    parser.add_argument("--num_views", type=int, default=config.NUM_VIEWS)
    parser.add_argument("--bottom_k", type=int, default=config.BOTTOM_K)
    parser.add_argument("--coe_threshold", type=float, default=config.COE_THRESHOLD)
    parser.add_argument("--cluster_gap", type=float, default=0.02)
    parser.add_argument("--max_source_len", type=int, default=config.MAX_SOURCE_LENGTH)
    parser.add_argument("--max_token_num", type=int, default=None)
    parser.add_argument("--max_target_len", type=int, default=config.MAX_TARGET_LENGTH)
    parser.add_argument("--beam_size", type=int, default=config.BEAM_SIZE)
    parser.add_argument("--generation_batch_size", type=int, default=config.GENERATION_BATCH_SIZE)
    parser.add_argument("--selected_layers", default=config.SELECTED_LAYERS)
    parser.add_argument("--kv_part", choices=["k", "v", "kv_cat"], default=config.KV_PART)
    parser.add_argument("--head_agg", choices=["mean", "flatten"], default=config.HEAD_AGG)
    parser.add_argument("--token_agg", choices=["mean", "sum", "last", "cls", "flatten"], default=config.TOKEN_AGG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip_clean_eval", action="store_true")
    args = parser.parse_args()
    if args.max_token_num is None:
        args.max_token_num = args.max_source_len - 2
    args.disable_data_parallel = True
    args.batch_size = args.generation_batch_size
    args.generation_micro_batch_size = args.generation_batch_size
    return args


def vote(predictions):
    counts = Counter(predictions)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def grouped(items, size):
    for start in range(0, len(items), size):
        yield start, items[start:start + size]


def selected_adv_source_raw(row: dict, field: str) -> str:
    if field == "auto":
        return adv_source_raw(row)
    if field in row and row[field] is not None:
        return str(row[field])
    raise KeyError(f"Requested adv_source_field={field!r} but row has keys {sorted(row)}")


def build_eval_rows(args):
    if args.adv_mode != "full" or args.adv_json is None:
        return default_build_rows(args)

    clean_rows = load_records(args.clean_json, args.sample_n)
    adv_rows = load_records(args.adv_json, args.sample_n)
    if len(clean_rows) != len(adv_rows):
        raise ValueError(f"clean_json has {len(clean_rows)} rows but adv_json has {len(adv_rows)} rows")

    clean_raw = [clean_source_raw(row) for row in clean_rows]
    adv_raw = [selected_adv_source_raw(row, args.adv_source_field) for row in adv_rows]
    golds = [gold_summary(row) for row in clean_rows]
    clean_codes = [norm_code(code) for code in clean_raw]
    adv_codes = [norm_code(code) for code in adv_raw]
    return clean_raw, clean_codes, adv_raw, adv_codes, golds


def decoder_self_r_batch(model, tokenizer, input_ids, source_mask, generated_ids, args, selected_layers):
    if generated_ids.shape[1] > 1:
        decoder_input_ids = generated_ids[:, :-1].contiguous()
    else:
        decoder_input_ids = generated_ids.contiguous()
    decoder_attention_mask = decoder_mask_from_ids(decoder_input_ids, tokenizer.pad_token_id)
    outputs = model(
        input_ids=input_ids,
        attention_mask=source_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        use_cache=True,
        return_dict=True,
    )
    cache_parts = extract_cache_pkv(outputs.past_key_values, selected_layers)
    if "decoder_self" not in cache_parts:
        raise RuntimeError("decoder_self cache was not available from CodeT5 forward output")
    k_layers, v_layers = cache_parts["decoder_self"]
    trajectory_args = SimpleNamespace(kv_part=args.kv_part, head_agg=args.head_agg, token_agg=args.token_agg)
    trajectories = encode_layers_all_tokens_batch(k_layers, v_layers, decoder_attention_mask, trajectory_args)
    return [compute_coe_scores(trajectories[i])["R"] for i in range(trajectories.shape[0])]


def generate_with_decoder_self_r(codes, tokenizer, model, device, args, selected_layers, desc):
    predictions = []
    r_values = []
    for _start, batch_codes in tqdm(list(grouped(codes, args.generation_batch_size)), desc=desc):
        input_ids, source_mask = tokenize_sources(tokenizer, batch_codes, args.max_source_len, device)
        with torch.no_grad():
            generation_output = model.generate(
                input_ids,
                attention_mask=source_mask,
                use_cache=True,
                num_beams=args.beam_size,
                early_stopping=True,
                max_length=args.max_target_len,
                return_dict_in_generate=True,
                output_scores=False,
            )
            generated_ids = generation_output.sequences
            predictions.extend(decode_predictions(tokenizer, generated_ids))
            r_values.extend(decoder_self_r_batch(model, tokenizer, input_ids, source_mask, generated_ids.to(device), args, selected_layers))
    return predictions, r_values


def select_bottom_k(predictions, r_values, k):
    order = np.argsort(np.array(r_values, dtype=np.float64))
    keep = order[: max(1, min(k, len(order)))]
    return [predictions[i] for i in keep], [r_values[i] for i in keep]


def select_threshold(predictions, r_values, threshold, fallback_k):
    keep = [i for i, value in enumerate(r_values) if value <= threshold]
    if not keep:
        bottom_preds, bottom_rs = select_bottom_k(predictions, r_values, fallback_k)
        return bottom_preds, bottom_rs, "fallback_bottom_k"
    return [predictions[i] for i in keep], [r_values[i] for i in keep], "threshold"


def select_low_r_cluster(predictions, r_values, min_gap):
    values = np.array(r_values, dtype=np.float64)
    if len(values) < 3:
        idx = int(np.argmin(values))
        return [predictions[idx]], [r_values[idx]], {"mode": "min_r_fallback", "gap": 0.0}
    centers = np.array([float(np.min(values)), float(np.max(values))], dtype=np.float64)
    labels = np.zeros(len(values), dtype=np.int64)
    for _ in range(20):
        new_labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        new_centers = centers.copy()
        for cluster_id in (0, 1):
            if np.any(new_labels == cluster_id):
                new_centers[cluster_id] = float(np.mean(values[new_labels == cluster_id]))
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            break
        labels = new_labels
        centers = new_centers
    low_cluster = int(np.argmin(centers))
    high_cluster = 1 - low_cluster
    gap = float(abs(centers[high_cluster] - centers[low_cluster]))
    if gap < min_gap:
        idx = int(np.argmin(values))
        return [predictions[idx]], [r_values[idx]], {
            "mode": "min_r_fallback",
            "gap": gap,
            "low_center": float(centers[low_cluster]),
            "high_center": float(centers[high_cluster]),
        }
    keep = np.where(labels == low_cluster)[0].astype(int).tolist()
    return [predictions[i] for i in keep], [r_values[i] for i in keep], {
        "mode": "low_r_cluster",
        "gap": gap,
        "low_center": float(centers[low_cluster]),
        "high_center": float(centers[high_cluster]),
        "low_count": int(len(keep)),
        "high_count": int(len(values) - len(keep)),
    }


def target_frac(predictions, target):
    return sum(1 for pred in predictions if pred == target) / max(1, len(predictions))


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    add_code_t5_path(args.base)
    from utils import smooth_bleu

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokenizer, model = load_model(args, device)
    selected_layers = parse_selected_layers(args.selected_layers, model.config.num_layers)

    t0 = time.time()
    clean_raw, clean_codes, adv_raw, adv_codes, golds = build_eval_rows(args)
    clean_codes = [norm_code(code) for code in clean_codes]
    adv_codes = [norm_code(code) for code in adv_codes]

    clean_views = []
    clean_selected_counts = []
    clean_parse_stats = []
    if not args.skip_clean_eval:
        for code in tqdm(clean_raw, desc="build clean perturbation views"):
            views, selected, parse_ok, parse_success = build_views(code, tokenizer, args)
            clean_views.append(views)
            clean_selected_counts.append(len(selected))
            clean_parse_stats.append((parse_ok, parse_success))

    adv_views = []
    selected_counts = []
    parse_stats = []
    for code in tqdm(adv_raw, desc="build adv perturbation views"):
        views, selected, parse_ok, parse_success = build_views(code, tokenizer, args)
        adv_views.append(views)
        selected_counts.append(len(selected))
        parse_stats.append((parse_ok, parse_success))

    if args.skip_clean_eval:
        clean_raw_predictions = []
        clean_raw_r = []
        clean_view_predictions = []
        clean_view_r = []
    else:
        clean_raw_predictions, clean_raw_r = generate_with_decoder_self_r(
            clean_codes, tokenizer, model, device, args, selected_layers, "raw_clean"
        )
        flat_clean_views = [view for group in clean_views for view in group]
        clean_view_predictions, clean_view_r = generate_with_decoder_self_r(
            flat_clean_views, tokenizer, model, device, args, selected_layers, f"clean_views_N{args.num_views}"
        )
    raw_predictions, raw_r = generate_with_decoder_self_r(
        adv_codes, tokenizer, model, device, args, selected_layers, "raw_backdoor"
    )
    flat_views = [view for group in adv_views for view in group]
    view_predictions, view_r = generate_with_decoder_self_r(
        flat_views, tokenizer, model, device, args, selected_layers, f"views_N{args.num_views}"
    )

    clean_plain_vote_preds = []
    clean_bottomk_vote_preds = []
    clean_threshold_vote_preds = []
    clean_min_r_preds = []
    clean_low_cluster_vote_preds = []
    clean_low_cluster_min_r_preds = []
    clean_kept_threshold_counts = []
    if args.skip_clean_eval:
        clean_raw_bleu = clean_raw_em = None
        clean_plain_vote_bleu = clean_plain_vote_em = None
        clean_bottomk_bleu = clean_bottomk_em = None
        clean_threshold_bleu = clean_threshold_em = None
        clean_min_r_bleu = clean_min_r_em = None
        clean_low_cluster_bleu = clean_low_cluster_em = None
        clean_low_cluster_min_r_bleu = clean_low_cluster_min_r_em = None
    else:
        for idx in range(args.sample_n):
            start = idx * args.num_views
            end = (idx + 1) * args.num_views
            preds = clean_view_predictions[start:end]
            rs = clean_view_r[start:end]
            clean_plain_vote_preds.append(vote(preds))
            bottom_preds, _bottom_rs = select_bottom_k(preds, rs, args.bottom_k)
            clean_bottomk_vote_preds.append(vote(bottom_preds))
            threshold_preds, _threshold_rs, _threshold_mode = select_threshold(preds, rs, args.coe_threshold, args.bottom_k)
            clean_threshold_vote_preds.append(vote(threshold_preds))
            clean_kept_threshold_counts.append(len(threshold_preds))
            min_idx = int(np.argmin(np.array(rs, dtype=np.float64)))
            clean_min_r_preds.append(preds[min_idx])
            low_preds, low_rs, _low_info = select_low_r_cluster(preds, rs, args.cluster_gap)
            clean_low_cluster_vote_preds.append(vote(low_preds))
            clean_low_cluster_min_r_preds.append(low_preds[int(np.argmin(np.array(low_rs, dtype=np.float64)))])

        clean_raw_bleu, clean_raw_em = bleu_em(clean_raw_predictions, golds, "clean_raw", args.output_dir, smooth_bleu)
        clean_plain_vote_bleu, clean_plain_vote_em = bleu_em(clean_plain_vote_preds, golds, "clean_plain_vote", args.output_dir, smooth_bleu)
        clean_bottomk_bleu, clean_bottomk_em = bleu_em(clean_bottomk_vote_preds, golds, "clean_coe_bottomk_vote", args.output_dir, smooth_bleu)
        clean_threshold_bleu, clean_threshold_em = bleu_em(clean_threshold_vote_preds, golds, "clean_coe_threshold_vote", args.output_dir, smooth_bleu)
        clean_min_r_bleu, clean_min_r_em = bleu_em(clean_min_r_preds, golds, "clean_coe_min_r", args.output_dir, smooth_bleu)
        clean_low_cluster_bleu, clean_low_cluster_em = bleu_em(clean_low_cluster_vote_preds, golds, "clean_low_r_cluster_vote", args.output_dir, smooth_bleu)
        clean_low_cluster_min_r_bleu, clean_low_cluster_min_r_em = bleu_em(clean_low_cluster_min_r_preds, golds, "clean_low_r_cluster_min_r", args.output_dir, smooth_bleu)

    per_sample_rows = []
    plain_vote_preds = []
    bottomk_vote_preds = []
    threshold_vote_preds = []
    min_r_preds = []
    low_cluster_vote_preds = []
    low_cluster_min_r_preds = []
    plain_target_fracs = []
    bottomk_target_fracs = []
    threshold_target_fracs = []
    low_cluster_target_fracs = []
    kept_threshold_counts = []
    low_cluster_sizes = []
    low_cluster_modes = []

    for idx in range(args.sample_n):
        start = idx * args.num_views
        end = (idx + 1) * args.num_views
        preds = view_predictions[start:end]
        rs = view_r[start:end]
        plain_pred = vote(preds)
        bottom_preds, bottom_rs = select_bottom_k(preds, rs, args.bottom_k)
        bottom_pred = vote(bottom_preds)
        threshold_preds, threshold_rs, threshold_mode = select_threshold(preds, rs, args.coe_threshold, args.bottom_k)
        threshold_pred = vote(threshold_preds)
        min_idx = int(np.argmin(np.array(rs, dtype=np.float64)))
        min_pred = preds[min_idx]
        low_preds, low_rs, low_info = select_low_r_cluster(preds, rs, args.cluster_gap)
        low_vote_pred = vote(low_preds)
        low_min_pred = low_preds[int(np.argmin(np.array(low_rs, dtype=np.float64)))]

        plain_vote_preds.append(plain_pred)
        bottomk_vote_preds.append(bottom_pred)
        threshold_vote_preds.append(threshold_pred)
        min_r_preds.append(min_pred)
        low_cluster_vote_preds.append(low_vote_pred)
        low_cluster_min_r_preds.append(low_min_pred)
        plain_target_fracs.append(target_frac(preds, args.target))
        bottomk_target_fracs.append(target_frac(bottom_preds, args.target))
        threshold_target_fracs.append(target_frac(threshold_preds, args.target))
        low_cluster_target_fracs.append(target_frac(low_preds, args.target))
        kept_threshold_counts.append(len(threshold_preds))
        low_cluster_sizes.append(len(low_preds))
        low_cluster_modes.append(low_info.get("mode"))

        per_sample_rows.append({
            "idx": idx,
            "raw_pred": raw_predictions[idx],
            "raw_r": raw_r[idx],
            "plain_vote_pred": plain_pred,
            "coe_bottomk_vote_pred": bottom_pred,
            "coe_threshold_vote_pred": threshold_pred,
            "coe_min_r_pred": min_pred,
            "low_r_cluster_vote_pred": low_vote_pred,
            "low_r_cluster_min_r_pred": low_min_pred,
            "raw_attack_success": raw_predictions[idx] == args.target,
            "plain_vote_attack_success": plain_pred == args.target,
            "coe_bottomk_attack_success": bottom_pred == args.target,
            "coe_threshold_attack_success": threshold_pred == args.target,
            "coe_min_r_attack_success": min_pred == args.target,
            "low_r_cluster_vote_attack_success": low_vote_pred == args.target,
            "low_r_cluster_min_r_attack_success": low_min_pred == args.target,
            "plain_target_frac": plain_target_fracs[-1],
            "coe_bottomk_target_frac": bottomk_target_fracs[-1],
            "coe_threshold_target_frac": threshold_target_fracs[-1],
            "low_r_cluster_target_frac": low_cluster_target_fracs[-1],
            "threshold_kept_views": len(threshold_preds),
            "threshold_mode": threshold_mode,
            "low_r_cluster_size": len(low_preds),
            "low_r_cluster_mode": low_info.get("mode"),
            "low_r_cluster_gap": low_info.get("gap"),
            "low_r_cluster_low_center": low_info.get("low_center"),
            "low_r_cluster_high_center": low_info.get("high_center"),
            "view_r_mean": float(np.mean(rs)),
            "view_r_min": float(np.min(rs)),
            "view_r_max": float(np.max(rs)),
            "bottomk_r_mean": float(np.mean(bottom_rs)),
            "threshold_r_mean": float(np.mean(threshold_rs)),
            "selected_identifier_count": selected_counts[idx],
            "parse_ok": parse_stats[idx][0],
            "parse_success_views": parse_stats[idx][1],
        })

    metrics = {
        "sample_n": args.sample_n,
        "num_views": args.num_views,
        "bottom_k": args.bottom_k,
        "coe_threshold": args.coe_threshold,
        "cluster_gap": args.cluster_gap,
        "target": args.target,
        "adv_mode": args.adv_mode,
        "adv_source_field": args.adv_source_field,
        "skip_clean_eval": args.skip_clean_eval,
        "model_base": str(args.model_base),
        "checkpoint": str(args.checkpoint),
        "clean_json": str(args.clean_json),
        "adv_json": str(args.adv_json) if args.adv_json else None,
        "clean_raw_bleu": clean_raw_bleu,
        "clean_raw_em": clean_raw_em,
        "clean_plain_vote_bleu": clean_plain_vote_bleu,
        "clean_plain_vote_em": clean_plain_vote_em,
        "clean_coe_bottomk_vote_bleu": clean_bottomk_bleu,
        "clean_coe_bottomk_vote_em": clean_bottomk_em,
        "clean_coe_threshold_vote_bleu": clean_threshold_bleu,
        "clean_coe_threshold_vote_em": clean_threshold_em,
        "clean_coe_min_r_bleu": clean_min_r_bleu,
        "clean_coe_min_r_em": clean_min_r_em,
        "clean_low_r_cluster_vote_bleu": clean_low_cluster_bleu,
        "clean_low_r_cluster_vote_em": clean_low_cluster_em,
        "clean_low_r_cluster_min_r_bleu": clean_low_cluster_min_r_bleu,
        "clean_low_r_cluster_min_r_em": clean_low_cluster_min_r_em,
        "clean_mean_threshold_kept_views": float(np.mean(clean_kept_threshold_counts)) if clean_kept_threshold_counts else None,
        "clean_selected_identifier_mean": float(np.mean(clean_selected_counts)) if clean_selected_counts else None,
        "clean_selected_identifier_max": int(np.max(clean_selected_counts)) if clean_selected_counts else 0,
        "clean_parse_ok_count": int(sum(1 for ok, _ in clean_parse_stats if ok)),
        "clean_parse_success_views": int(sum(success for _, success in clean_parse_stats)),
        "raw_asr": float(np.mean([pred == args.target for pred in raw_predictions])),
        "plain_vote_asr": float(np.mean([pred == args.target for pred in plain_vote_preds])),
        "coe_bottomk_vote_asr": float(np.mean([pred == args.target for pred in bottomk_vote_preds])),
        "coe_threshold_vote_asr": float(np.mean([pred == args.target for pred in threshold_vote_preds])),
        "coe_min_r_asr": float(np.mean([pred == args.target for pred in min_r_preds])),
        "low_r_cluster_vote_asr": float(np.mean([pred == args.target for pred in low_cluster_vote_preds])),
        "low_r_cluster_min_r_asr": float(np.mean([pred == args.target for pred in low_cluster_min_r_preds])),
        "mean_plain_target_frac": float(np.mean(plain_target_fracs)),
        "mean_coe_bottomk_target_frac": float(np.mean(bottomk_target_fracs)),
        "mean_coe_threshold_target_frac": float(np.mean(threshold_target_fracs)),
        "mean_low_r_cluster_target_frac": float(np.mean(low_cluster_target_fracs)),
        "mean_threshold_kept_views": float(np.mean(kept_threshold_counts)),
        "mean_low_r_cluster_size": float(np.mean(low_cluster_sizes)),
        "low_r_cluster_mode_count": int(sum(mode == "low_r_cluster" for mode in low_cluster_modes)),
        "low_r_min_r_fallback_count": int(sum(mode == "min_r_fallback" for mode in low_cluster_modes)),
        "selected_identifier_mean": float(np.mean(selected_counts)),
        "selected_identifier_max": int(np.max(selected_counts)) if selected_counts else 0,
        "parse_ok_count": int(sum(1 for ok, _ in parse_stats if ok)),
        "parse_success_views": int(sum(success for _, success in parse_stats)),
        "total_views": int(args.sample_n * args.num_views),
        "elapsed_sec": round(time.time() - t0, 2),
        "selected_layers": selected_layers,
        "kv_part": args.kv_part,
        "head_agg": args.head_agg,
        "token_agg": args.token_agg,
    }

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as fout:
        json.dump(metrics, fout, indent=2, ensure_ascii=False)
    with (args.output_dir / "per_sample.jsonl").open("w", encoding="utf-8") as fout:
        for row in per_sample_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
