"""
Step 8: Meta-Harness Main Loop.

Orchestrates the full optimization cycle:
  Round 0: Run baseline config → collect traces
  Round 1-N: Propose new config → score → merge → evaluate → accept/reject

Usage:
    # Full loop (requires Ollama for Qwen3 + ANTHROPIC_API_KEY for proposer)
    python failure_analysis/type_b_memory/meta_harness/loop.py \
        --max_rounds 5 \
        --endpoints http://127.0.0.1:11434

    # Step-by-step (manual control)
    python failure_analysis/type_b_memory/meta_harness/loop.py \
        --round 0 --step baseline
    python failure_analysis/type_b_memory/meta_harness/loop.py \
        --round 1 --step propose
    python failure_analysis/type_b_memory/meta_harness/loop.py \
        --round 1 --step eval
"""

import argparse
import json
import os
import shutil
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")


def run_baseline(split, model, endpoints, workers):
    """Round 0: Score and merge with baseline config, collect traces."""
    from failure_analysis.type_b_memory.meta_harness.harness_config import baseline_config, save_config
    from failure_analysis.type_b_memory.meta_harness.config_rerank import (
        batch_score_with_config, merge_with_config, load_qwen3_cache,
    )
    from failure_analysis.type_b_memory.meta_harness.collect_traces import collect_traces

    config = baseline_config()
    config_path = os.path.join(CONFIGS_DIR, "round_0_baseline.json")
    save_config(config, config_path)

    # Score
    print("\n" + "="*60)
    print("  ROUND 0: Baseline scoring")
    print("="*60)
    rerank_scores, _ = batch_score_with_config(
        split, model, endpoints, workers, config, round_num=0)

    # Merge
    qwen3_cache = load_qwen3_cache(split)
    merge_with_config(split, rerank_scores, qwen3_cache, config, round_num=0)

    # Collect traces
    print("\n  Collecting traces...")
    traces = collect_traces(split, round_num=0)
    traces_path = os.path.join(CONFIGS_DIR, "traces_round_0.json")
    with open(traces_path, "w") as f:
        json.dump(traces, f, indent=2)

    improved = sum(1 for t in traces if t["improved"])
    print(f"  Traces: {len(traces)} total, {improved} improved")

    # Evaluate (baseline vs mFAR raw baseline)
    from failure_analysis.type_b_memory.meta_harness.apply_and_eval import evaluate_round
    # For round 0, compare against mFAR raw baseline (no reranking)
    # We do this by computing MRR on baseline .qres directly
    from failure_analysis.utils import load_qrels, load_retrieved
    from failure_analysis.type_b_memory.meta_harness.apply_and_eval import (
        compute_mrr_for_qids, bootstrap_ci,
    )
    from failure_analysis.type_b_memory.meta_harness.config_rerank import BASELINE_QRES

    qrels = load_qrels("data/prime", split)
    rerouted_qids = [qid for qid, e in qwen3_cache.items() if e.get("needs_reroute")]

    mfar_ret = load_retrieved(BASELINE_QRES[split])
    mfar_rr = compute_mrr_for_qids(rerouted_qids, qrels, mfar_ret)
    mfar_mrr, mfar_lo, mfar_hi = bootstrap_ci(mfar_rr)

    round0_qres = os.path.join("output/failure_analysis/type_b_memory/runs/meta_harness/round_0",
                               f"reranked_{split}.qres")
    round0_ret = load_retrieved(round0_qres)
    round0_rr = compute_mrr_for_qids(rerouted_qids, qrels, round0_ret)
    round0_mrr, round0_lo, round0_hi = bootstrap_ci(round0_rr)

    print(f"\n  Baseline results on {split} rerouted ({len(mfar_rr)} queries):")
    print(f"    mFAR raw MRR:   {mfar_mrr:.4f} [{mfar_lo:.4f}, {mfar_hi:.4f}]")
    print(f"    Round 0 MRR:    {round0_mrr:.4f} [{round0_lo:.4f}, {round0_hi:.4f}]")
    print(f"    Delta:          {round0_mrr - mfar_mrr:+.4f}")


def run_propose(round_num, proposer_model="claude-sonnet-4-20250514"):
    """Run proposer to generate config for this round."""
    from failure_analysis.type_b_memory.meta_harness.propose import (
        build_proposer_prompt, call_proposer, parse_config_from_response,
    )
    from failure_analysis.type_b_memory.meta_harness.harness_config import load_config, save_config

    # Use traces from the best accepted round (not necessarily prev_round)
    # Find the most recent traces file
    traces_path = None
    for r in range(round_num - 1, -1, -1):
        candidate = os.path.join(CONFIGS_DIR, f"traces_round_{r}.json")
        if os.path.exists(candidate):
            traces_path = candidate
            break
    if traces_path is None:
        raise FileNotFoundError("No traces found from any previous round")

    # Use the best config as baseline for proposer
    best_path = os.path.join(CONFIGS_DIR, "best.json")
    if os.path.exists(best_path):
        config_path = best_path
    else:
        config_path = os.path.join(CONFIGS_DIR, "round_0_baseline.json")

    with open(traces_path) as f:
        traces = json.load(f)
    config = load_config(config_path)

    # Append info about previously rejected rounds
    rejected_info = ""
    for r in range(1, round_num):
        rej_config = os.path.join(CONFIGS_DIR, f"round_{r}.json")
        rej_metrics = os.path.join(CONFIGS_DIR, f"metrics_round_{r}.json")
        if os.path.exists(rej_config) and os.path.exists(rej_metrics):
            with open(rej_metrics) as f:
                m = json.load(f)
            with open(rej_config) as f:
                rc = json.load(f)
            status = "ACCEPTED" if m.get("accept") else "REJECTED"
            rejected_info += (f"\nRound {r} ({status}): delta={m.get('delta',0):+.4f}, "
                              f"rationale: {rc.get('rationale','?')[:200]}")

    prompt = build_proposer_prompt(traces, config)
    if rejected_info:
        prompt += f"\n\nPrevious rounds attempted:{rejected_info}\nDo NOT repeat rejected approaches. Try a DIFFERENT aspect."

    print(f"\n  Proposer prompt length: {len(prompt)} chars")
    print(f"  Calling {proposer_model}...")

    response = call_proposer(prompt, proposer_model)

    print(f"\n  Response preview: {response[:300]}")

    new_config = parse_config_from_response(response)
    new_config.round = round_num

    output_path = os.path.join(CONFIGS_DIR, f"round_{round_num}.json")
    save_config(new_config, output_path)
    print(f"  Config saved → {output_path}")
    print(f"  Rationale: {new_config.rationale}")

    # Save full response for debugging
    with open(output_path.replace(".json", "_response.txt"), "w") as f:
        f.write(response)

    return new_config


def run_eval(round_num, split, model, endpoints, workers, baseline_round=0):
    """Score, merge, evaluate for a round. Returns accept/reject."""
    from failure_analysis.type_b_memory.meta_harness.harness_config import load_config
    from failure_analysis.type_b_memory.meta_harness.apply_and_eval import main as eval_main

    config_path = os.path.join(CONFIGS_DIR, f"round_{round_num}.json")

    # Monkey-patch sys.argv for apply_and_eval
    sys.argv = [
        "apply_and_eval.py",
        "--config", config_path,
        "--round", str(round_num),
        "--split", split,
        "--baseline_round", str(baseline_round),
        "--model", model,
        "--endpoints", ",".join(endpoints),
        "--workers", str(workers),
    ]
    metrics = eval_main()

    # Collect traces for next round
    if metrics and metrics.get("accept"):
        from failure_analysis.type_b_memory.meta_harness.collect_traces import collect_traces
        traces = collect_traces(split, round_num, config_path)
        traces_path = os.path.join(CONFIGS_DIR, f"traces_round_{round_num}.json")
        with open(traces_path, "w") as f:
            json.dump(traces, f, indent=2)
        print(f"  Traces for next round saved → {traces_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Meta-Harness main loop")
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--start_round", type=int, default=0,
                        help="Resume from this round (skip earlier rounds)")
    parser.add_argument("--split", default="val")
    parser.add_argument("--model", default="qwen3:8b", help="Qwen3 model for scoring")
    parser.add_argument("--endpoints", default="http://127.0.0.1:11434")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--proposer_model", default="claude-sonnet-4-20250514")

    # Step-by-step mode
    parser.add_argument("--round", type=int, default=None,
                        help="Run a specific round (step-by-step mode)")
    parser.add_argument("--step", choices=["baseline", "propose", "eval"],
                        help="Which step to run for --round")

    args = parser.parse_args()

    endpoints = [ep.strip() for ep in args.endpoints.split(",")]
    workers = args.workers or len(endpoints)
    os.makedirs(CONFIGS_DIR, exist_ok=True)

    # Step-by-step mode
    if args.round is not None and args.step:
        if args.step == "baseline":
            run_baseline(args.split, args.model, endpoints, workers)
        elif args.step == "propose":
            run_propose(args.round, args.proposer_model)
        elif args.step == "eval":
            run_eval(args.round, args.split, args.model, endpoints, workers)
        return

    # Full loop mode
    print("\n" + "="*70)
    print("  META-HARNESS OPTIMIZATION LOOP")
    print("="*70)

    start = args.start_round
    best_round = 0

    if start == 0:
        # Round 0: Baseline
        run_baseline(args.split, args.model, endpoints, workers)
        best_src = os.path.join(CONFIGS_DIR, "round_0_baseline.json")
        best_dst = os.path.join(CONFIGS_DIR, "best.json")
        shutil.copy2(best_src, best_dst)
    else:
        # Resuming: find best accepted round so far
        print(f"  Resuming from round {start}")
        for r in range(start - 1, 0, -1):
            m_path = os.path.join(CONFIGS_DIR, f"metrics_round_{r}.json")
            if os.path.exists(m_path):
                with open(m_path) as f:
                    m = json.load(f)
                if m.get("accept"):
                    best_round = r
                    break
        print(f"  Best round so far: {best_round}")

    for round_num in range(max(1, start), args.max_rounds + 1):
        print(f"\n\n{'='*70}")
        print(f"  ROUND {round_num}")
        print(f"{'='*70}")

        # Propose
        try:
            run_propose(round_num, args.proposer_model)
        except Exception as e:
            print(f"  Proposer failed: {e}")
            break

        # Evaluate
        metrics = run_eval(round_num, args.split, args.model, endpoints, workers,
                           baseline_round=best_round)

        if metrics and metrics.get("accept"):
            print(f"\n  >>> ACCEPTED round {round_num} (delta={metrics['delta']:+.4f})")
            best_round = round_num

            # Copy to best.json
            best_src = os.path.join(CONFIGS_DIR, f"round_{round_num}.json")
            best_dst = os.path.join(CONFIGS_DIR, "best.json")
            shutil.copy2(best_src, best_dst)
        else:
            print(f"\n  >>> REJECTED round {round_num}")
            if metrics:
                print(f"      delta={metrics['delta']:+.4f}, threshold={metrics['threshold']:.4f}")

    # Summary
    print(f"\n\n{'='*70}")
    print(f"  LOOP COMPLETE")
    print(f"  Best round: {best_round}")
    best_path = os.path.join(CONFIGS_DIR, f"round_{best_round}.json"
                             if best_round > 0 else "round_0_baseline.json")
    print(f"  Best config: {best_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
