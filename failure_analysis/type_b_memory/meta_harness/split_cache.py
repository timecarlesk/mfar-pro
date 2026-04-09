"""
Split existing Qwen3 stage12 cache (full train) into train-build and train-dev caches.

Avoids re-running Qwen3 — just filters by qid sets.

Usage:
    python failure_analysis/type_b_memory/meta_harness/split_cache.py
"""

import json
import os

DATA_DIR = "data/prime"
CACHE_DIR = "output/failure_analysis/type_b_memory/cache"
FULL_CACHE = os.path.join(CACHE_DIR, "stage12", "shared", "qwen3_cache_train.jsonl")


def load_qids(queries_path):
    qids = set()
    with open(queries_path) as f:
        for line in f:
            qid = line.strip().split("\t", 1)[0]
            if qid:
                qids.add(qid)
    return qids


def main():
    build_qids = load_qids(os.path.join(DATA_DIR, "train-build.queries"))
    dev_qids = load_qids(os.path.join(DATA_DIR, "train-dev.queries"))

    print(f"train-build qids: {len(build_qids)}")
    print(f"train-dev qids:   {len(dev_qids)}")

    # Read full cache
    entries = []
    with open(FULL_CACHE) as f:
        for line in f:
            entries.append(json.loads(line))
    print(f"Full train cache:  {len(entries)} entries")

    # Write split caches to stage12/shared/
    out_dir = os.path.join(CACHE_DIR, "stage12", "shared")
    for name, qid_set in [("train-build", build_qids), ("train-dev", dev_qids)]:
        out_path = os.path.join(out_dir, f"qwen3_cache_{name}.jsonl")
        count = 0
        with open(out_path, "w") as f:
            for entry in entries:
                if entry["qid"] in qid_set:
                    f.write(json.dumps(entry) + "\n")
                    count += 1
        print(f"Wrote {count} entries → {out_path}")

    # Verify
    build_count = sum(1 for e in entries if e["qid"] in build_qids)
    dev_count = sum(1 for e in entries if e["qid"] in dev_qids)
    assert build_count + dev_count == len(entries), f"Mismatch: {build_count}+{dev_count} != {len(entries)}"
    print(f"\nVerified: {build_count} + {dev_count} = {len(entries)}")


if __name__ == "__main__":
    main()
