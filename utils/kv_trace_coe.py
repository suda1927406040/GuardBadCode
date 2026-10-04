#!/usr/bin/env python
import argparse
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


BACKDOOR_TARGET = "This function is to load train data from the disk safely"
COE_METRICS = ["Mag", "Ang", "R", "C"]
OUTPUT_METRICS = ["maxprob", "ppl", "entropy"]


def parse_args():
    parser = argparse.ArgumentParser(description="Trace CodeT5 encoder-decoder KV trajectories with upstream CoE scores.")
    parser.add_argument("--attack_dir", default="data/adv_backdoor_codeT5")
    parser.add_argument("--model_name_or_path", default="models/codet5-base")
    parser.add_argument("--tokenizer_name", default="models/codet5-base")
    parser.add_argument("--checkpoint_path", default="outputs/checkpoint-best-bleu/pytorch_model.bin")
    parser.add_argument("--data_file", default="data/test.jsonl")
    parser.add_argument("--out_dir", default="outputs/codet5_kv_trace_upstream")
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--max_examples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--selected_layers", default="all", help="all, last4, or comma-separated layer indices")
    parser.add_argument("--kv_part", choices=["k", "v", "kv_cat"], default="v")
    parser.add_argument("--head_agg", choices=["mean", "flatten"], default="flatten")
    parser.add_argument("--token_agg", choices=["mean", "sum", "last", "cls", "flatten"], default="mean")
    parser.add_argument("--score_beam_agg", choices=["first", "max"], default="first")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def import_attack_code(attack_dir):
    if attack_dir not in sys.path:
        sys.path.insert(0, attack_dir)
    from models import build_or_load_gen_model
    from utils import read_examples, read_poisoned_examples
    return build_or_load_gen_model, read_examples, read_poisoned_examples


def load_model_and_data(args):
    build_or_load_gen_model, read_examples, read_poisoned_examples = import_attack_code(args.attack_dir)
    model_args = SimpleNamespace(
        model_type="codet5",
        config_name="",
        tokenizer_name=args.tokenizer_name,
        model_name_or_path=args.model_name_or_path,
        load_model_path=None,
        beam_size=args.beam_size,
        max_target_length=args.max_target_length,
    )
    config, model, tokenizer = build_or_load_gen_model(model_args)
    state = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(state)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    model.to(device)
    model.eval()

    data_num = args.max_examples if args.max_examples > 0 else -1
    clean_examples = read_examples(args.data_file, data_num, "summarize")
    backdoor_examples = read_poisoned_examples(args.data_file, data_num, "summarize_adv-1.00")
    return config, model, tokenizer, clean_examples, backdoor_examples, torch.device(device)


def parse_selected_layers(spec, num_layers):
    if spec == "all":
        return list(range(num_layers))
    if spec == "last4":
        return list(range(max(0, num_layers - 4), num_layers))
    return [int(x) for x in spec.split(",") if x.strip()]


def batch_iter(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def tokenize_sources(tokenizer, sources, max_source_length, device):
    encoded = tokenizer(
        sources,
        max_length=max_source_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    return input_ids, attention_mask


def decode_predictions(tokenizer, generated_ids):
    return [
        tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        for ids in generated_ids.detach().cpu().tolist()
    ]


def decoder_mask_from_ids(ids, pad_token_id):
    mask = ids.ne(pad_token_id)
    if mask.shape[1] > 0:
        mask[:, 0] = True
    return mask.long()


def to_bhtd(tensor, seq_len=None):
    if tensor is None or tensor.dim() != 4:
        return tensor
    if seq_len is not None and tensor.shape[1] == seq_len and tensor.shape[2] != seq_len:
        return tensor.transpose(1, 2).contiguous()
    return tensor


class T5EncoderKVTracer:
    def __init__(self, model, selected_layers):
        self.model = model
        self.selected_layers = set(selected_layers)
        self.handles = []
        self.traces = {}
        self.num_heads = model.config.num_heads
        self.d_kv = model.config.d_kv

    def _make_hook(self, layer_idx, part):
        def hook(_module, _inputs, output):
            if layer_idx not in self.selected_layers:
                return
            B, T, inner = output.shape
            expected = self.num_heads * self.d_kv
            if inner != expected:
                raise RuntimeError(f"Unexpected encoder {part} projection dim {inner}, expected {expected}")
            tensor = output.detach().view(B, T, self.num_heads, self.d_kv).transpose(1, 2).contiguous()
            self.traces[(layer_idx, part)] = tensor
        return hook

    def attach(self):
        for idx, block in enumerate(self.model.encoder.block):
            if idx not in self.selected_layers:
                continue
            attn = block.layer[0].SelfAttention
            self.handles.append(attn.k.register_forward_hook(self._make_hook(idx, "k")))
            self.handles.append(attn.v.register_forward_hook(self._make_hook(idx, "v")))

    def clear(self):
        self.traces = {}

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def get_layer_tensors(self, selected_layers):
        k_layers = []
        v_layers = []
        missing = []
        for layer_idx in selected_layers:
            k = self.traces.get((layer_idx, "k"))
            v = self.traces.get((layer_idx, "v"))
            if k is None or v is None:
                missing.append(layer_idx)
                continue
            k_layers.append(k)
            v_layers.append(v)
        if missing:
            raise RuntimeError(f"Missing encoder KV traces for layers {missing}")
        return k_layers, v_layers


def token_pool(x, mask, mode):
    mask = mask[:, :x.shape[1]].to(device=x.device)
    if mode == "flatten":
        return x.reshape(x.shape[0], -1)
    if mode == "last":
        lengths = mask.sum(dim=1).clamp_min(1).long()
        idx = (lengths - 1).view(x.shape[0], 1, 1).expand(x.shape[0], 1, x.shape[2])
        return x.gather(1, idx).squeeze(1)
    if mode == "cls":
        first_idx = mask.float().argmax(dim=1).long()
        idx = first_idx.view(x.shape[0], 1, 1).expand(x.shape[0], 1, x.shape[2])
        return x.gather(1, idx).squeeze(1)
    mask3 = mask.unsqueeze(-1).to(dtype=x.dtype)
    summed = (x * mask3).sum(dim=1)
    if mode == "sum":
        return summed
    denom = mask3.sum(dim=1).clamp_min(1e-6)
    return summed / denom


def encode_layers_all_tokens_batch(k_layers, v_layers, mask, args):
    layer_vecs = []
    for k, v in zip(k_layers, v_layers):
        k = to_bhtd(k, mask.shape[1]).float()
        v = to_bhtd(v, mask.shape[1]).float()
        if args.kv_part == "k":
            h_bhtd = k
        elif args.kv_part == "kv_cat":
            h_bhtd = torch.cat([k, v], dim=-1)
        else:
            h_bhtd = v

        if args.head_agg == "flatten":
            h_btd = h_bhtd.permute(0, 2, 1, 3).reshape(h_bhtd.shape[0], h_bhtd.shape[2], -1)
        else:
            h_btd = h_bhtd.mean(dim=1)

        h_bd = token_pool(h_btd, mask, args.token_agg)
        layer_vecs.append(h_bd.detach().cpu())

    if not layer_vecs:
        return None
    return torch.stack(layer_vecs, dim=1).numpy()


def extract_tuple_pkv(past_key_values, selected_layers):
    self_k, self_v, cross_k, cross_v = [], [], [], []
    if past_key_values is None or not isinstance(past_key_values, (tuple, list)):
        return None
    for layer_idx in selected_layers:
        if layer_idx >= len(past_key_values):
            return None
        layer = past_key_values[layer_idx]
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            return None
        self_k.append(layer[0])
        self_v.append(layer[1])
        if len(layer) >= 4 and layer[2] is not None and layer[3] is not None:
            cross_k.append(layer[2])
            cross_v.append(layer[3])
    result = {"decoder_self": (self_k, self_v)}
    if len(cross_k) == len(selected_layers):
        result["cross"] = (cross_k, cross_v)
    return result


def extract_cache_pkv(past_key_values, selected_layers):
    tuple_result = extract_tuple_pkv(past_key_values, selected_layers)
    if tuple_result is not None:
        return tuple_result

    result = {}
    try:
        self_cache = getattr(past_key_values, "self_attention_cache", None)
        cross_cache = getattr(past_key_values, "cross_attention_cache", None)
        if self_cache is not None and hasattr(self_cache, "key_cache"):
            result["decoder_self"] = (
                [self_cache.key_cache[i] for i in selected_layers],
                [self_cache.value_cache[i] for i in selected_layers],
            )
        if cross_cache is not None and hasattr(cross_cache, "key_cache"):
            result["cross"] = (
                [cross_cache.key_cache[i] for i in selected_layers],
                [cross_cache.value_cache[i] for i in selected_layers],
            )
    except Exception:
        return result
    return result


def clamp_cosine(value):
    return max(-1.0, min(1.0, float(value)))


def compute_coe_scores(trajectory):
    hs = np.asarray(trajectory, dtype=np.float64)
    if len(hs) < 2:
        return {"Mag": 0.0, "Ang": 0.0, "R": 0.0, "C": 0.0}

    mag_den = max(np.linalg.norm(hs[-1] - hs[0], ord=2), 1e-12)
    repdiff = np.array([hs[i + 1] - hs[i] for i in range(len(hs) - 1)])
    repdiff_norm = np.array([np.linalg.norm(item, ord=2) / mag_den for item in repdiff], dtype=np.float64)
    coe_mag = float(np.mean(repdiff_norm))

    first_last_den = max(np.linalg.norm(hs[-1], ord=2) * np.linalg.norm(hs[0], ord=2), 1e-12)
    first_last_cos = clamp_cosine(np.dot(hs[-1], hs[0]) / first_last_den)
    ang_den = max(math.acos(first_last_cos), 1e-12)
    semdiff = []
    for i in range(len(hs) - 1):
        a = hs[i + 1]
        b = hs[i]
        denom = max(np.linalg.norm(a, ord=2) * np.linalg.norm(b, ord=2), 1e-12)
        sim = clamp_cosine(np.dot(a, b) / denom)
        semdiff.append(math.acos(sim) / ang_den)
    semdiff_norm = np.array(semdiff, dtype=np.float64)
    coe_ang = float(np.mean(semdiff_norm))
    coe_r = float(coe_mag - coe_ang)

    x_list = np.array([repdiff_norm[i] * math.cos(semdiff_norm[i]) for i in range(len(semdiff_norm))])
    y_list = np.array([repdiff_norm[i] * math.sin(semdiff_norm[i]) for i in range(len(semdiff_norm))])
    coe_c = float(math.sqrt(float(np.mean(x_list)) ** 2 + float(np.mean(y_list)) ** 2))
    return {"Mag": coe_mag, "Ang": coe_ang, "R": coe_r, "C": coe_c}


def compute_output_scores(output_scores, batch_size, num_beams, score_beam_agg):
    if not output_scores:
        return [{metric: None for metric in OUTPUT_METRICS} for _ in range(batch_size)]

    rows = [[] for _ in range(batch_size)]
    for step_scores in output_scores:
        probs = F.softmax(step_scores.detach().float().cpu(), dim=-1)
        for b in range(batch_size):
            start = b * num_beams if probs.shape[0] >= batch_size * num_beams else b
            end = min(start + num_beams, probs.shape[0])
            sample_probs = probs[start:end]
            if sample_probs.numel() == 0:
                continue
            if score_beam_agg == "max":
                token_probs = sample_probs.max(dim=-1).values
                max_prob = float(token_probs.max().item())
                entropy_value = float((-(sample_probs * sample_probs.clamp_min(1e-12).log2()).sum(dim=-1)).min().item())
            else:
                beam_probs = sample_probs[0]
                max_prob = float(beam_probs.max().item())
                entropy_value = float((-(beam_probs * beam_probs.clamp_min(1e-12).log2()).sum()).item())
            rows[b].append((max_prob, entropy_value))

    metrics = []
    for row in rows:
        if not row:
            metrics.append({metric: None for metric in OUTPUT_METRICS})
            continue
        max_probs = np.array([item[0] for item in row], dtype=np.float64)
        entropies = np.array([item[1] for item in row], dtype=np.float64)
        metrics.append({
            "maxprob": float(np.mean(max_probs)),
            "ppl": float(-np.mean(np.log(np.maximum(max_probs, 1e-12)))),
            "entropy": float(np.mean(entropies)),
        })
    return metrics


def trace_batch(model, tracer, input_ids, source_mask, generated_ids, args, selected_layers, tokenizer):
    tracer.clear()
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

    enc_k, enc_v = tracer.get_layer_tensors(selected_layers)
    trajectories = {
        "encoder": encode_layers_all_tokens_batch(enc_k, enc_v, source_mask, args),
    }

    cache_parts = extract_cache_pkv(outputs.past_key_values, selected_layers)
    if "decoder_self" in cache_parts:
        k_layers, v_layers = cache_parts["decoder_self"]
        trajectories["decoder_self"] = encode_layers_all_tokens_batch(k_layers, v_layers, decoder_attention_mask, args)
    if "cross" in cache_parts:
        k_layers, v_layers = cache_parts["cross"]
        trajectories["cross"] = encode_layers_all_tokens_batch(k_layers, v_layers, source_mask, args)
    return trajectories


def coe_score_rows(split, examples, predictions, output_metrics, trajectories):
    rows = []
    scores_by_stream = {stream: [] for stream in trajectories}
    for i, example in enumerate(examples):
        for stream, values in trajectories.items():
            score = compute_coe_scores(values[i])
            scores_by_stream[stream].append(score)
            rows.append({
                "idx": int(example.idx),
                "split": split,
                "stream": stream,
                "is_backdoor": split == "backdoor",
                "attack_success": bool(predictions[i]["attack_success"]),
                **score,
                **output_metrics[i],
            })
    return rows, scores_by_stream


def run_split(name, examples, model, tokenizer, tracer, args, selected_layers, device):
    predictions = []
    score_rows = []
    output_metrics_all = []
    trajectories = {"encoder": [], "decoder_self": [], "cross": []}
    coe_scores = {"encoder": [], "decoder_self": [], "cross": []}

    for batch in batch_iter(examples, args.batch_size):
        sources = [example.source for example in batch]
        golds = [example.target.strip() for example in batch]
        ids = [int(example.idx) for example in batch]
        input_ids, source_mask = tokenize_sources(tokenizer, sources, args.max_source_length, device)

        tracer.clear()
        with torch.no_grad():
            generation_output = model.generate(
                input_ids,
                attention_mask=source_mask,
                use_cache=True,
                num_beams=args.beam_size,
                early_stopping=True,
                max_length=args.max_target_length,
                return_dict_in_generate=True,
                output_scores=True,
            )
        generated_ids = generation_output.sequences
        preds = decode_predictions(tokenizer, generated_ids)
        output_metrics = compute_output_scores(generation_output.scores, len(batch), args.beam_size, args.score_beam_agg)
        tracer.clear()

        with torch.no_grad():
            batch_trajectories = trace_batch(
                model, tracer, input_ids, source_mask, generated_ids.to(device), args, selected_layers, tokenizer
            )

        batch_predictions = []
        for idx, source, gold, pred in zip(ids, sources, golds, preds):
            batch_predictions.append({
                "idx": idx,
                "split": name,
                "is_backdoor": name == "backdoor",
                "source_len": len(source.split()),
                "gold": gold,
                "pred": pred,
                "exact_match": pred == gold,
                "attack_success": pred == BACKDOOR_TARGET,
            })
        batch_score_rows, batch_coe_scores = coe_score_rows(name, batch, batch_predictions, output_metrics, batch_trajectories)

        predictions.extend(batch_predictions)
        score_rows.extend(batch_score_rows)
        output_metrics_all.extend(output_metrics)
        for stream, value in batch_trajectories.items():
            if value is not None:
                trajectories[stream].append(value)
        for stream, values in batch_coe_scores.items():
            coe_scores[stream].extend(values)

    trajectories = {k: np.concatenate(v, axis=0) for k, v in trajectories.items() if v}
    coe_scores = {k: v for k, v in coe_scores.items() if v}
    return predictions, score_rows, output_metrics_all, trajectories, coe_scores


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean_or_none(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(np.mean(values))


def metric_stats(values):
    values = np.array(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean": None, "std": None}
    return {"count": int(values.size), "mean": float(np.mean(values)), "std": float(np.std(values))}


def score_group_stats(scores, indices):
    return {
        metric: metric_stats([scores[i][metric] for i in indices])
        for metric in COE_METRICS
    }


def output_group_stats(rows, indices):
    return {
        metric: metric_stats([rows[i][metric] for i in indices if rows[i][metric] is not None])
        for metric in OUTPUT_METRICS
    }


def summarize_scores(clean_predictions, backdoor_predictions, clean_output, backdoor_output, clean_coe, backdoor_coe):
    success = np.array([row["attack_success"] for row in backdoor_predictions], dtype=bool)
    clean_indices = list(range(len(clean_predictions)))
    poison_activated_indices = np.where(success)[0].tolist()
    poison_inactivated_indices = np.where(~success)[0].tolist()

    summary = {
        "groups": {
            "clean": {"count": len(clean_predictions)},
            "poison_activated": {"count": len(poison_activated_indices)},
            "poison_inactivated": {"count": len(poison_inactivated_indices)},
        },
        "output_score_stats": {
            "clean": output_group_stats(clean_output, clean_indices),
            "poison_activated": output_group_stats(backdoor_output, poison_activated_indices),
            "poison_inactivated": output_group_stats(backdoor_output, poison_inactivated_indices),
        },
        "coe_score_stats": {},
    }

    for stream, clean_scores in clean_coe.items():
        if stream not in backdoor_coe:
            continue
        backdoor_scores = backdoor_coe[stream]
        n = min(len(clean_scores), len(backdoor_scores))
        active_indices = [i for i in poison_activated_indices if i < n]
        inactive_indices = [i for i in poison_inactivated_indices if i < n]
        summary["coe_score_stats"][stream] = {
            "clean": score_group_stats(clean_scores, list(range(n))),
            "poison_activated": score_group_stats(backdoor_scores, active_indices),
            "poison_inactivated": score_group_stats(backdoor_scores, inactive_indices),
        }
    return summary


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    config, model, tokenizer, clean_examples, backdoor_examples, device = load_model_and_data(args)
    selected_layers = parse_selected_layers(args.selected_layers, config.num_layers)

    tracer = T5EncoderKVTracer(model, selected_layers)
    tracer.attach()
    try:
        clean_predictions, clean_score_rows, clean_output, clean_traj, clean_coe = run_split(
            "clean", clean_examples, model, tokenizer, tracer, args, selected_layers, device
        )
        backdoor_predictions, backdoor_score_rows, backdoor_output, backdoor_traj, backdoor_coe = run_split(
            "backdoor", backdoor_examples, model, tokenizer, tracer, args, selected_layers, device
        )
    finally:
        tracer.remove()

    predictions = clean_predictions + backdoor_predictions
    score_rows = clean_score_rows + backdoor_score_rows
    write_jsonl(os.path.join(args.out_dir, "predictions.jsonl"), predictions)
    write_jsonl(os.path.join(args.out_dir, "coe_scores.jsonl"), score_rows)

    npz_payload = {}
    for split, trajectories in [("clean", clean_traj), ("backdoor", backdoor_traj)]:
        for stream, value in trajectories.items():
            npz_payload[f"{split}_{stream}"] = value
    np.savez_compressed(os.path.join(args.out_dir, "coe_trajectories.npz"), **npz_payload)

    score_summary = summarize_scores(
        clean_predictions, backdoor_predictions, clean_output, backdoor_output, clean_coe, backdoor_coe
    )

    clean_em = np.mean([row["exact_match"] for row in clean_predictions]) if clean_predictions else 0.0
    backdoor_asr = np.mean([row["attack_success"] for row in backdoor_predictions]) if backdoor_predictions else 0.0
    summary = {
        "checkpoint_path": args.checkpoint_path,
        "data_file": args.data_file,
        "device": str(device),
        "num_clean": len(clean_predictions),
        "num_backdoor": len(backdoor_predictions),
        "selected_layers": selected_layers,
        "kv_part": args.kv_part,
        "head_agg": args.head_agg,
        "token_agg": args.token_agg,
        "clean_exact_match": float(clean_em),
        "backdoor_asr": float(backdoor_asr),
        **score_summary,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2, ensure_ascii=False)

    print(json.dumps({
        "out_dir": args.out_dir,
        "clean_exact_match": summary["clean_exact_match"],
        "backdoor_asr": summary["backdoor_asr"],
        "groups": summary["groups"],
        "coe_score_stats": summary["coe_score_stats"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
