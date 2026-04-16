"""Print detailed negation failure examples."""
import json, re
from collections import defaultdict

NEGATION_PATTERN = re.compile(
    r"\b(?:not|no|without|lack(?:s|ing)?|absent|neither|nor|never|"
    r"cannot|can't|don't|doesn't|didn't|won't|isn't|aren't|wasn't|weren't|"
    r"exclude[ds]?|non[\-])\b|"
    r"\bun(?:express|relat|associat|link|affect|involv)\w*\b",
    re.IGNORECASE,
)

DATA_DIR = "data/prime"
EVAL_DIR = "output/contriever/prime_eval"

# Load data
corpus = {}
with open(f"{DATA_DIR}/corpus") as f:
    for line in f:
        idx, js = line.strip().split("\t", 1)
        doc = json.loads(js)
        corpus[idx] = doc

queries = {}
with open(f"{DATA_DIR}/val.queries") as f:
    for line in f:
        qid, text = line.strip().split("\t", 1)
        queries[qid] = text

qrels = defaultdict(set)
with open(f"{DATA_DIR}/val.qrels") as f:
    for line in f:
        parts = line.strip().split("\t")
        qid, _, docid, rel = parts
        if float(rel) > 0:
            qrels[qid].add(docid)

retrieved = defaultdict(list)
with open(f"{EVAL_DIR}/final-all-0.qres") as f:
    for line in f:
        parts = line.strip().split("\t")
        qid, _, docid, _, score, _ = parts
        retrieved[qid].append((docid, float(score)))

# Collect negation query results
examples = []
for qid, text in queries.items():
    if not NEGATION_PATTERN.search(text):
        continue
    if qid not in qrels:
        continue
    gold = qrels[qid]
    docs = retrieved.get(qid, [])
    top_ids = [d[0] for d in docs[:100]]
    first_rel = next((i for i, did in enumerate(top_ids) if did in gold), -1)
    rr = 0.0 if first_rel < 0 else 1.0 / (first_rel + 1)
    neg_tokens = NEGATION_PATTERN.findall(text.lower())

    examples.append({
        "qid": qid, "query": text, "rr": rr,
        "rank": first_rel + 1 if first_rel >= 0 else -1,
        "neg_tokens": neg_tokens, "gold_ids": list(gold),
        "top3": [(d, s) for d, s in docs[:3]],
    })

# Sort: complete misses first, then worst rank
examples.sort(key=lambda x: (x["rank"] == -1, -x["rank"] if x["rank"] > 0 else 0), reverse=True)

printed = 0
for ex in examples:
    if printed >= 10:
        break
    qid = ex["qid"]
    rank = ex["rank"]
    neg = ex["neg_tokens"]

    print("=" * 80)
    rank_str = "MISS" if rank < 0 else str(rank)
    print(f"Query [{qid}]  Rank={rank_str}  RR={ex['rr']:.3f}")
    print(f"Negation tokens: {neg}")
    print(f"Query: {ex['query'][:250]}")
    print()

    for gid in list(ex["gold_ids"])[:2]:
        gdoc = corpus.get(gid, {})
        gtype = gdoc.get("type", "?")
        gname = gdoc.get("name", "?")
        print(f"  GOLD [{gid}]: {gtype} | {gname}")
        details = gdoc.get("details", "")
        if details:
            print(f"    details: {str(details)[:180]}")

    print()
    for i, (did, score) in enumerate(ex["top3"]):
        rdoc = corpus.get(did, {})
        rtype = rdoc.get("type", "?")
        rname = rdoc.get("name", "?")
        tag = "CORRECT" if did in ex["gold_ids"] else "WRONG"
        print(f"  Retrieved #{i+1} (score={score:.3f}) [{tag}]: {rtype} | {rname}")
        details = rdoc.get("details", "")
        if details:
            print(f"    details: {str(details)[:180]}")
    print()
    printed += 1
