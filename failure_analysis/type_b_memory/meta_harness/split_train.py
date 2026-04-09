"""
Split train.queries / train.qrels into train-build (5000) + train-dev (1162).

train-build: used to construct memory (field confusion, memory context).
train-dev:   held-out for Meta-Harness evaluation signal.

Usage:
    python failure_analysis/type_b_memory/meta_harness/split_train.py [--seed 42]
"""

import argparse
import os
import random

DATA_DIR = "data/prime"


def load_queries(path):
    """Load queries file → dict[qid → text]."""
    queries = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    return queries


def load_qrels(path):
    """Load qrels file → list of (qid, rest_of_line) tuples."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                qid = line.split("\t", 1)[0]
                rows.append((qid, line))
    return rows


def write_queries(queries_dict, qids, path):
    with open(path, "w") as f:
        for qid in qids:
            f.write(f"{qid}\t{queries_dict[qid]}\n")


def write_qrels(qrel_rows, qid_set, path):
    with open(path, "w") as f:
        for qid, line in qrel_rows:
            if qid in qid_set:
                f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--build_size", type=int, default=5000)
    args = parser.parse_args()

    queries_path = os.path.join(DATA_DIR, "train.queries")
    qrels_path = os.path.join(DATA_DIR, "train.qrels")

    queries = load_queries(queries_path)
    qrel_rows = load_qrels(qrels_path)

    all_qids = sorted(queries.keys())
    print(f"Total train queries: {len(all_qids)}")

    random.seed(args.seed)
    random.shuffle(all_qids)

    build_qids = all_qids[:args.build_size]
    dev_qids = all_qids[args.build_size:]

    build_set = set(build_qids)
    dev_set = set(dev_qids)

    assert len(build_set & dev_set) == 0, "Overlap detected!"
    assert len(build_set) + len(dev_set) == len(all_qids)

    print(f"train-build: {len(build_qids)}")
    print(f"train-dev:   {len(dev_qids)}")

    # Write split files
    write_queries(queries, build_qids, os.path.join(DATA_DIR, "train-build.queries"))
    write_qrels(qrel_rows, build_set, os.path.join(DATA_DIR, "train-build.qrels"))

    write_queries(queries, dev_qids, os.path.join(DATA_DIR, "train-dev.queries"))
    write_qrels(qrel_rows, dev_set, os.path.join(DATA_DIR, "train-dev.qrels"))

    print(f"\nWrote to {DATA_DIR}:")
    print(f"  train-build.queries, train-build.qrels")
    print(f"  train-dev.queries,   train-dev.qrels")

    # Verify qrel counts
    build_qrel_count = sum(1 for qid, _ in qrel_rows if qid in build_set)
    dev_qrel_count = sum(1 for qid, _ in qrel_rows if qid in dev_set)
    print(f"\nQrel rows: build={build_qrel_count}, dev={dev_qrel_count}")


if __name__ == "__main__":
    main()
