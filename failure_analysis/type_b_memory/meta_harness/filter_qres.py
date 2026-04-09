"""
Filter a .qres file to keep only queries in a specified qid set.

Usage:
    python failure_analysis/type_b_memory/meta_harness/filter_qres.py \
        --qres output/prime_train_eval/final-all-0.qres \
        --keep_qids data/prime/train-dev.queries \
        --output output/prime_eval/final-train-dev-all-0.qres
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qres", required=True, help="Input .qres file")
    parser.add_argument("--keep_qids", required=True,
                        help="Path to .queries file; only these qids are kept")
    parser.add_argument("--output", required=True, help="Output .qres file")
    args = parser.parse_args()

    # Load qid set from queries file
    keep = set()
    with open(args.keep_qids) as f:
        for line in f:
            qid = line.strip().split("\t", 1)[0]
            if qid:
                keep.add(qid)
    print(f"Keeping {len(keep)} qids from {args.keep_qids}")

    # Filter qres
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    kept = 0
    total = 0
    with open(args.qres) as fin, open(args.output, "w") as fout:
        for line in fin:
            total += 1
            qid = line.split("\t", 1)[0]
            if qid in keep:
                fout.write(line)
                kept += 1

    print(f"Filtered: {kept}/{total} lines kept → {args.output}")
    print(f"Unique qids in output: {len(keep)} (requested), check with wc -l")


if __name__ == "__main__":
    main()
