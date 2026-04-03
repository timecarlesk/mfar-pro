"""
Dump q @ W logit range from the trained mFAR checkpoint.

Encodes a sample of queries and computes the field weight logits
(q @ W) to determine the natural scale. This informs the α sweep
range for negation memory bias.

If logits are in [-1, +1], then α=1.0 with Qwen3 score=8 gives
bias=8 — far too large. If logits are in [-10, +10], then α=1.0
is reasonable.

Run from project root:
  python failure_analysis/type_b_memory/dump_logit_range.py
  python failure_analysis/type_b_memory/dump_logit_range.py --n_queries 200
"""

import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from failure_analysis.utils import load_queries

DATA_DIR = "data/prime"
ANALYSIS_DIR = "output/failure_analysis/type_b_memory/analysis"


def main():
    parser = argparse.ArgumentParser(description="Dump q @ W logit range")
    parser.add_argument("--checkpoint_dir", default="output/prime",
                        help="Checkpoint directory (default: output/prime)")
    parser.add_argument("--n_queries", type=int, default=100,
                        help="Number of queries to sample (default: 100)")
    parser.add_argument("--split", default="val",
                        help="Split to sample queries from (default: val)")
    parser.add_argument("--dataset_name", default="prime")
    parser.add_argument("--model_name", default="facebook/contriever-msmarco")
    parser.add_argument("--field_names", default="all_dense,all_sparse,single_dense,single_sparse")
    args = parser.parse_args()

    import torch
    from mfar.data.schema import resolve_fields
    from mfar.modeling.util import prepare_model
    from mfar.modeling.contrastive import RetrievalTrainingModule

    # Load model
    field_info = resolve_fields(args.field_names, args.dataset_name)
    tokenizer, encoder, _ = prepare_model(args.model_name, normalize=False, with_decoder=False)

    best_ckpt_path = os.path.join(args.checkpoint_dir, "best.txt")
    with open(best_ckpt_path) as f:
        checkpoint_suffix = f.read().strip().split("/")[-1]
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_suffix)
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint weights only (no corpus/indices needed)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Extract mixture_of_fields_layer weights
    state_dict = ckpt.get("state_dict", ckpt)
    weight_key = None
    for k in state_dict:
        if "mixture_of_fields_layer.weight" in k:
            weight_key = k
            break

    if weight_key is None:
        print("ERROR: Could not find mixture_of_fields_layer.weight in checkpoint")
        print("Available keys:", [k for k in state_dict if "weight" in k.lower()])
        return

    W = state_dict[weight_key]  # [emb_size, num_fields]
    print(f"\nW shape: {W.shape}")
    print(f"W stats: min={W.min():.4f}, max={W.max():.4f}, mean={W.mean():.4f}, std={W.std():.4f}")

    # Load and encode sample queries
    queries = load_queries(DATA_DIR, args.split)
    qids = list(queries.keys())[:args.n_queries]
    query_texts = [queries[qid] for qid in qids]

    print(f"\nEncoding {len(qids)} queries...")
    encoder.eval()
    all_logits = []

    with torch.no_grad():
        for i in range(0, len(query_texts), 16):
            batch_texts = query_texts[i:i+16]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            q_emb = encoder(encoded)["sentence_embedding"]  # [batch, emb]
            logits = q_emb @ W  # [batch, num_fields]
            all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=0)  # [n_queries, num_fields]
    print(f"\nLogits shape: {all_logits.shape}")

    # Global stats
    print(f"\n{'='*60}")
    print(f"  q @ W LOGIT STATISTICS ({len(qids)} queries, {W.shape[1]} fields)")
    print(f"{'='*60}")
    print(f"  Global min:    {all_logits.min():.4f}")
    print(f"  Global max:    {all_logits.max():.4f}")
    print(f"  Global mean:   {all_logits.mean():.4f}")
    print(f"  Global std:    {all_logits.std():.4f}")
    print(f"  Per-query range (mean): {(all_logits.max(dim=1).values - all_logits.min(dim=1).values).mean():.4f}")

    # Per-field stats
    field_names = list(field_info.keys())
    print(f"\n  Per-field logit statistics:")
    print(f"  {'Field':<40} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*72}")
    for idx, fname in enumerate(field_names):
        if idx >= all_logits.shape[1]:
            break
        col = all_logits[:, idx]
        print(f"  {fname:<40} {col.mean():>8.4f} {col.std():>8.4f} {col.min():>8.4f} {col.max():>8.4f}")

    # Softmax weight distribution
    weights_dist = torch.softmax(all_logits, dim=1)  # [n_queries, num_fields]
    print(f"\n  Top-5 fields by mean softmax weight:")
    mean_weights = weights_dist.mean(dim=0)
    top_fields = torch.topk(mean_weights, k=min(5, len(field_names)))
    for rank, (val, idx) in enumerate(zip(top_fields.values, top_fields.indices)):
        fname = field_names[idx] if idx < len(field_names) else f"field_{idx}"
        print(f"    {rank+1}. {fname}: {val:.4f}")

    # Key fields for negation
    print(f"\n  Key fields for negation routing:")
    for fname in ["contraindication_dense", "indication_dense",
                   "contraindication_sparse", "indication_sparse"]:
        if fname in field_names:
            idx = field_names.index(fname)
            col = all_logits[:, idx]
            w_col = weights_dist[:, idx]
            print(f"    {fname}: logit mean={col.mean():.4f}, "
                  f"softmax weight mean={w_col.mean():.6f}")

    # α recommendation
    logit_range = all_logits.max().item() - all_logits.min().item()
    per_query_range = (all_logits.max(dim=1).values - all_logits.min(dim=1).values).mean().item()
    print(f"\n  RECOMMENDATION:")
    print(f"    Logit range (global): {logit_range:.2f}")
    print(f"    Logit range (per-query mean): {per_query_range:.2f}")
    print(f"    For Qwen3 score in [-10, +10]:")
    print(f"      α=0.1 → max bias = ±1.0 (subtle nudge)")
    print(f"      α={per_query_range/20:.2f} → max bias = ±{per_query_range/2:.2f} (half of range)")
    print(f"      α={per_query_range/10:.2f} → max bias = ±{per_query_range:.2f} (full range override)")
    print(f"    Suggested sweep: α ∈ {{0.1, {per_query_range/20:.2f}, {per_query_range/10:.2f}, {per_query_range/5:.2f}}}")

    # Save results
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    results = {
        "W_shape": list(W.shape),
        "n_queries": len(qids),
        "global_min": float(all_logits.min()),
        "global_max": float(all_logits.max()),
        "global_mean": float(all_logits.mean()),
        "global_std": float(all_logits.std()),
        "per_query_range_mean": float(per_query_range),
        "suggested_alpha_sweep": [
            0.1,
            round(per_query_range / 20, 3),
            round(per_query_range / 10, 3),
            round(per_query_range / 5, 3),
        ],
    }
    out_path = os.path.join(ANALYSIS_DIR, "logit_range.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
