"""
Memory as Reward: Fine-tune mFAR's field weight matrix W on negation queries.

Uses memory KG's BOOSTS weights as target distribution for KL divergence loss.
Only W changes; encoder is frozen. This "internalizes" memory into model params.

Commands:
  encode  — pre-compute query embeddings for negation queries (needs encoder)
  train   — fine-tune W using KL divergence (CPU, ~5 min)
  eval    — compare softmax(q @ W_old) vs softmax(q @ W_new) for sanity check

Run from project root:
  $PY failure_analysis/type_b_memory/rerank/train_memory/finetune_W.py encode
  $PY failure_analysis/type_b_memory/rerank/train_memory/finetune_W.py train --lr 1e-3 --epochs 50
  $PY failure_analysis/type_b_memory/rerank/train_memory/finetune_W.py eval
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from failure_analysis.type_b_memory.rerank.shared.memory_kg import MemoryKG
from mfar.data.schema import resolve_fields

# ── Paths ────────────────────────────────���───────────────────────────────────

CHECKPOINT_PATH = "output/prime"
ANALYSIS_DIR = "output/failure_analysis/type_b_memory/analysis"
CACHE_DIR = "output/failure_analysis/type_b_memory/cache"
OUTPUT_DIR = "output/prime_W_finetune"
DATA_DIR = "data/prime"

# ── Field Index Mapping ──────────────────────────────────────────────────────

def _model_tag(model_name):
    return model_name.replace(":", "_").replace("/", "_")

def get_field_name_to_idx():
    """Get mapping from field base name to (dense_idx, sparse_idx)."""
    fi = resolve_fields("all_dense,all_sparse,single_dense,single_sparse", "prime")
    field_keys = list(fi.keys())

    base_to_indices = {}
    for idx, key in enumerate(field_keys):
        for suffix in ("_dense", "_sparse"):
            if key.endswith(suffix):
                base = key[:-len(suffix)]
                base_to_indices.setdefault(base, []).append(idx)
                break

    return base_to_indices, field_keys


# ── Load Data ────────────────────────────────────────────────────────────────

def load_negation_queries(detect_model="qwen3:8b"):
    """Load negation queries from train Qwen3 cache."""
    model_tag = _model_tag(detect_model)
    search_paths = [os.path.join(CACHE_DIR, "stage12", model_tag, "shared", "qwen3_cache_train.jsonl")]
    if model_tag == "qwen3_8b":
        search_paths.extend([
            os.path.join(CACHE_DIR, "stage12", "shared", "qwen3_cache_train.jsonl"),
            os.path.join(CACHE_DIR, "qwen3_8b", "qwen3_cache_train.jsonl"),
            os.path.join(CACHE_DIR, "qwen3_cache_train.jsonl"),
        ])
    cache_path = next((p for p in search_paths if os.path.exists(p)), None)
    if cache_path is None:
        raise FileNotFoundError(f"No qwen3 train cache found for detect_model={detect_model}")
    queries = []
    with open(cache_path) as f:
        for line in f:
            e = json.loads(line)
            if e.get("needs_reroute"):
                queries.append({
                    "qid": e["qid"],
                    "query": e.get("query", ""),
                    "answer_type": e.get("answer_type", "unknown"),
                    "negation_pattern": e.get("negation_pattern", "other"),
                })
    print(f"  Loaded {len(queries)} negation queries from train")
    return queries


def load_W():
    """Load W matrix and full checkpoint."""
    best_file = os.path.join(CHECKPOINT_PATH, "best.txt")
    with open(best_file) as f:
        ckpt_name = f.read().strip().split("/")[-1]
    ckpt_path = os.path.join(CHECKPOINT_PATH, ckpt_name)
    print(f"  Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    W = ckpt["state_dict"]["mixture_of_fields_layer.weight"].clone()
    print(f"  W shape: {W.shape}")
    return W, ckpt, ckpt_path


def load_memory_kg():
    """Load memory KG."""
    kg_path = os.path.join(ANALYSIS_DIR, "memory_kg.json")
    if os.path.exists(kg_path):
        kg = MemoryKG.from_json(kg_path)
        print(f"  {kg.summary()}")
        return kg
    print(f"  ERROR: {kg_path} not found")
    return None


# ── Target Weight Construction ───────────────────────────────────────────────

def get_target_weights(answer_type, neg_pattern, memory_kg, base_to_indices, num_fields=46):
    """Get target field weight distribution from memory KG BOOSTS."""
    result = memory_kg.query(answer_type, neg_pattern)
    if result is None:
        return None

    target = torch.zeros(num_fields)
    for field_name, weight in result.boost_fields:
        indices = base_to_indices.get(field_name, [])
        for idx in indices:
            target[idx] = weight

    if target.sum() > 0:
        target = target / target.sum()
        return target
    return None


# ── Encode Command ───────────────────────────────────────────────────────────

def cmd_encode(args):
    """Pre-compute query embeddings for all negation queries."""
    from sentence_transformers import SentenceTransformer

    neg_queries = load_negation_queries(args.detect_model)

    print(f"  Loading encoder: facebook/contriever-msmarco")
    encoder = SentenceTransformer("facebook/contriever-msmarco")

    print(f"  Encoding {len(neg_queries)} queries...")
    embeddings = {}
    texts = [q["query"] for q in neg_queries]
    qids = [q["qid"] for q in neg_queries]

    # Batch encode
    embs = encoder.encode(texts, batch_size=64, show_progress_bar=True,
                          convert_to_tensor=True)

    for qid, emb in zip(qids, embs):
        embeddings[qid] = emb.cpu()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    emb_path = os.path.join(OUTPUT_DIR, "negation_query_embeddings.pt")
    torch.save(embeddings, emb_path)
    print(f"  Saved {len(embeddings)} embeddings to {emb_path}")


# ── Train Command ────────────────────────────────────────────────────────────

def cmd_train(args):
    """Fine-tune W on negation queries using KL divergence."""
    # Load everything
    W_original, ckpt, ckpt_path = load_W()
    memory_kg = load_memory_kg()
    neg_queries = load_negation_queries(args.detect_model)
    base_to_indices, field_keys = get_field_name_to_idx()

    # Load pre-computed embeddings
    emb_path = os.path.join(OUTPUT_DIR, "negation_query_embeddings.pt")
    if not os.path.exists(emb_path):
        print(f"  ERROR: {emb_path} not found. Run 'encode' first.")
        return
    embeddings = torch.load(emb_path, map_location="cpu", weights_only=False)
    print(f"  Loaded {len(embeddings)} pre-computed embeddings")

    # Build (query_embedding, target_weights) pairs
    pairs = []
    for q in neg_queries:
        if q["qid"] not in embeddings:
            continue
        target = get_target_weights(
            q["answer_type"], q["negation_pattern"],
            memory_kg, base_to_indices, num_fields=W_original.shape[1]
        )
        if target is None:
            continue
        pairs.append((embeddings[q["qid"]], target))

    print(f"  Training pairs: {len(pairs)} (from {len(neg_queries)} negation queries)")

    # Fine-tune
    W_param = nn.Parameter(W_original.clone())
    W_frozen = W_original.clone()  # for elastic constraint
    optimizer = Adam([W_param], lr=args.lr)

    for epoch in range(args.epochs):
        total_kl = 0
        total_elastic = 0

        for q_emb, target in pairs:
            logits = q_emb.unsqueeze(0) @ W_param  # [1, 46]
            log_actual = F.log_softmax(logits, dim=-1)  # [1, 46]

            # KL divergence: target is the reference distribution
            kl_loss = F.kl_div(log_actual, target.unsqueeze(0), reduction="batchmean")

            # Elastic constraint: don't move W too far
            elastic_loss = F.mse_loss(W_param, W_frozen)

            loss = args.lambda_kl * kl_loss + args.lambda_elastic * elastic_loss
            loss.backward()
            total_kl += kl_loss.item()
            total_elastic += elastic_loss.item()

        optimizer.step()
        optimizer.zero_grad()

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            avg_kl = total_kl / len(pairs)
            avg_el = total_elastic / len(pairs)
            # Check how much W moved
            w_delta = (W_param.data - W_frozen).norm().item()
            print(f"  Epoch {epoch:>3}: KL={avg_kl:.4f}, Elastic={avg_el:.6f}, "
                  f"W_delta={w_delta:.4f}")

    # Save modified checkpoint
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    W_new = W_param.data
    ckpt["state_dict"]["mixture_of_fields_layer.weight"] = W_new
    ckpt["state_dict"]["hybrid_contrastive_loss_fn.mixture_of_fields_layer.weight"] = W_new

    new_ckpt_name = os.path.basename(ckpt_path)
    new_ckpt_path = os.path.join(OUTPUT_DIR, new_ckpt_name)
    torch.save(ckpt, new_ckpt_path)

    # Also write best.txt pointing to the new checkpoint
    with open(os.path.join(OUTPUT_DIR, "best.txt"), "w") as f:
        f.write(new_ckpt_name)

    print(f"\n  Saved modified checkpoint to {new_ckpt_path}")
    print(f"  W moved by {(W_new - W_frozen).norm().item():.4f} (L2 norm)")


# ── Eval Command ─────────────────────────────────────────────────────────────

def cmd_eval(args):
    """Compare weight distributions before and after fine-tuning."""
    W_original, _, _ = load_W()
    memory_kg = load_memory_kg()
    neg_queries = load_negation_queries(args.detect_model)
    base_to_indices, field_keys = get_field_name_to_idx()

    # Load fine-tuned W
    ft_ckpt_path = os.path.join(OUTPUT_DIR, "best.txt")
    if not os.path.exists(ft_ckpt_path):
        print("  ERROR: No fine-tuned checkpoint found. Run 'train' first.")
        return

    with open(ft_ckpt_path) as f:
        ft_name = f.read().strip()
    ft_ckpt = torch.load(os.path.join(OUTPUT_DIR, ft_name), map_location="cpu",
                          weights_only=False)
    W_new = ft_ckpt["state_dict"]["mixture_of_fields_layer.weight"]

    # Load embeddings
    emb_path = os.path.join(OUTPUT_DIR, "negation_query_embeddings.pt")
    embeddings = torch.load(emb_path, map_location="cpu", weights_only=False)

    # Show a few examples
    print(f"\n  {'QID':<8} {'Pattern':<30} {'Old Top-3 Fields':<50} {'New Top-3 Fields'}")
    print(f"  {'-'*140}")

    for q in neg_queries[:20]:
        if q["qid"] not in embeddings:
            continue
        q_emb = embeddings[q["qid"]].unsqueeze(0)

        old_weights = F.softmax(q_emb @ W_original, dim=-1).squeeze()
        new_weights = F.softmax(q_emb @ W_new, dim=-1).squeeze()

        old_top = torch.topk(old_weights, 3)
        new_top = torch.topk(new_weights, 3)

        old_str = ", ".join(f"{field_keys[i]}({v:.3f})" for v, i in zip(old_top.values, old_top.indices))
        new_str = ", ".join(f"{field_keys[i]}({v:.3f})" for v, i in zip(new_top.values, new_top.indices))
        pat = f"{q['answer_type']}|{q['negation_pattern']}"

        print(f"  {q['qid']:<8} {pat:<30} {old_str:<50} {new_str}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune W on negation queries")
    parser.add_argument("--detect_model", default="qwen3:8b",
                        help="Model used for Stage 1+2 detection (default: qwen3:8b)")
    subparsers = parser.add_subparsers(dest="command")

    # Encode
    subparsers.add_parser("encode", help="Pre-compute query embeddings")

    # Train
    t_parser = subparsers.add_parser("train", help="Fine-tune W")
    t_parser.add_argument("--lr", type=float, default=1e-3)
    t_parser.add_argument("--epochs", type=int, default=50)
    t_parser.add_argument("--lambda_kl", type=float, default=1.0,
                          help="Weight for KL divergence loss")
    t_parser.add_argument("--lambda_elastic", type=float, default=0.1,
                          help="Weight for elastic constraint (prevent W from moving too far)")

    # Eval
    subparsers.add_parser("eval", help="Compare old vs new weight distributions")

    args = parser.parse_args()

    if args.command == "encode":
        cmd_encode(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
