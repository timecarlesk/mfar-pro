"""Analyze retrieval results: find success and failure examples."""
import json
from collections import defaultdict

DATA_DIR = "data/prime"
EVAL_DIR = "output/contriever/prime_eval"

# 1. Load queries
queries = {}
with open(f"{DATA_DIR}/test.queries") as f:
    for line in f:
        qid, text = line.strip().split("\t", 1)
        queries[qid] = text

# 2. Load ground truth (qrels)
qrels = defaultdict(set)
with open(f"{DATA_DIR}/test.qrels") as f:
    for line in f:
        parts = line.strip().split("\t")
        qid, _, docid, rel = parts
        if float(rel) > 0:
            qrels[qid].add(docid)

# 3. Load corpus (id -> name/type for display)
corpus = {}
with open(f"{DATA_DIR}/corpus") as f:
    for line in f:
        idx, json_str = line.strip().split("\t", 1)
        doc = json.loads(json_str)
        corpus[idx] = {
            "name": doc.get("name", ""),
            "type": doc.get("type", ""),
        }

# 4. Load retrieval results (qres)
retrieved = defaultdict(list)
with open(f"{EVAL_DIR}/final-additional-all-0.qres") as f:
    for line in f:
        parts = line.strip().split("\t")
        qid, _, docid, _, score, _ = parts
        retrieved[qid].append((docid, float(score)))

# 5. Analyze
successes = []  # hit@1
failures = []   # miss in top 20
partial = []    # hit in top 20 but not top 1

for qid in queries:
    if qid not in qrels:
        continue
    gold = qrels[qid]
    top_docs = retrieved.get(qid, [])[:20]
    top_ids = [d[0] for d in top_docs]

    top1_hit = len(gold & set(top_ids[:1])) > 0
    top5_hit = len(gold & set(top_ids[:5])) > 0
    top20_hit = len(gold & set(top_ids[:20])) > 0
    recall_20 = len(gold & set(top_ids)) / len(gold) if gold else 0

    info = {
        "qid": qid,
        "query": queries[qid],
        "num_relevant": len(gold),
        "top1_hit": top1_hit,
        "top5_hit": top5_hit,
        "top20_hit": top20_hit,
        "recall@20": round(recall_20, 3),
        "top5_retrieved": [(did, corpus.get(did, {}).get("name", "?"), corpus.get(did, {}).get("type", "?")) for did in top_ids[:5]],
        "gold_samples": [(did, corpus.get(did, {}).get("name", "?"), corpus.get(did, {}).get("type", "?")) for did in list(gold)[:5]],
    }

    if top1_hit:
        successes.append(info)
    elif not top20_hit:
        failures.append(info)
    else:
        partial.append(info)

# 6. Print summary
total = len(successes) + len(failures) + len(partial)
print(f"{'='*70}")
print(f"Prime Test Set Evaluation Summary")
print(f"{'='*70}")
print(f"Total queries:    {total}")
print(f"Hit@1 (success):  {len(successes)} ({100*len(successes)/total:.1f}%)")
print(f"Hit@5 (partial):  {len(successes)+len(partial)-len([p for p in partial if not p['top5_hit']])} ({100*(len(successes)+len([p for p in partial if p['top5_hit']]))/total:.1f}%)")
print(f"Hit@20:           {len(successes)+len(partial)} ({100*(len(successes)+len(partial))/total:.1f}%)")
print(f"Complete miss:    {len(failures)} ({100*len(failures)/total:.1f}%)")

print(f"\n{'='*70}")
print(f"SUCCESS EXAMPLES (Top-1 Hit)")
print(f"{'='*70}")
for ex in successes[:3]:
    print(f"\nQuery [{ex['qid']}]: {ex['query']}")
    print(f"  Relevant docs: {ex['num_relevant']}, Recall@20: {ex['recall@20']}")
    print(f"  Top-5 retrieved:")
    for rank, (did, name, dtype) in enumerate(ex['top5_retrieved'], 1):
        marker = " ✓" if did in qrels[ex['qid']] else ""
        print(f"    {rank}. [{dtype}] {name} (id={did}){marker}")
    print(f"  Gold samples:")
    for did, name, dtype in ex['gold_samples'][:3]:
        print(f"    - [{dtype}] {name} (id={did})")

print(f"\n{'='*70}")
print(f"FAILURE EXAMPLES (Complete Miss in Top-20)")
print(f"{'='*70}")
for ex in failures[:15]:
    print(f"\nQuery [{ex['qid']}]: {ex['query']}")
    print(f"  Relevant docs: {ex['num_relevant']}, Recall@20: {ex['recall@20']}")
    print(f"  Top-5 retrieved:")
    for rank, (did, name, dtype) in enumerate(ex['top5_retrieved'], 1):
        print(f"    {rank}. [{dtype}] {name} (id={did})")
    print(f"  Gold samples (should have found):")
    for did, name, dtype in ex['gold_samples'][:3]:
        print(f"    - [{dtype}] {name} (id={did})")

print(f"\n{'='*70}")
print(f"PARTIAL SUCCESS (Hit in Top-20 but not Top-1)")
print(f"{'='*70}")
for ex in partial[:3]:
    print(f"\nQuery [{ex['qid']}]: {ex['query']}")
    print(f"  Relevant docs: {ex['num_relevant']}, Recall@20: {ex['recall@20']}")
    print(f"  Top-5 retrieved:")
    for rank, (did, name, dtype) in enumerate(ex['top5_retrieved'], 1):
        marker = " ✓" if did in qrels[ex['qid']] else ""
        print(f"    {rank}. [{dtype}] {name} (id={did}){marker}")
