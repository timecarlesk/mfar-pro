"""
Shared data-loading utilities for failure analysis scripts.
"""

import json
import math
from collections import defaultdict

# ── PRIME relation fields ────────────────────────────────────────────────────
RELATION_FIELDS = [
    "ppi", "carrier", "enzyme", "target", "transporter",
    "contraindication", "indication", "off-label use",
    "synergistic interaction", "associated with", "parent-child",
    "phenotype absent", "phenotype present", "side effect",
    "interacts with", "linked to", "expression present", "expression absent",
]
BASIC_FIELDS = ["name", "type", "source", "details"]
ALL_FIELDS = BASIC_FIELDS + RELATION_FIELDS


def load_corpus(data_dir, relation_fields=None):
    """Load corpus with metadata and populated-field info."""
    if relation_fields is None:
        relation_fields = RELATION_FIELDS
    corpus = {}
    with open(f"{data_dir}/corpus") as f:
        for line in f:
            idx, json_str = line.strip().split("\t", 1)
            doc = json.loads(json_str)
            populated = {rel for rel in relation_fields if rel in doc}
            has_details = bool(doc.get("details"))
            corpus[idx] = {
                "name": doc.get("name", ""),
                "type": doc.get("type", ""),
                "source": doc.get("source", ""),
                "fields": populated,
                "has_details": has_details,
                "field_count": len(populated) + (1 if has_details else 0),
            }
    print(f"  Loaded {len(corpus):,} corpus documents")
    return corpus


def load_corpus_full(data_dir, relation_fields=None):
    """Load corpus with full absent-field values for Type A verification."""
    if relation_fields is None:
        relation_fields = RELATION_FIELDS
    corpus = {}
    with open(f"{data_dir}/corpus") as f:
        for line in f:
            idx, json_str = line.strip().split("\t", 1)
            doc = json.loads(json_str)
            populated = {rel for rel in relation_fields if rel in doc}
            has_details = bool(doc.get("details"))
            corpus[idx] = {
                "name": doc.get("name", ""),
                "type": doc.get("type", ""),
                "source": doc.get("source", ""),
                "fields": populated,
                "has_details": has_details,
                "field_count": len(populated) + (1 if has_details else 0),
                "expression_absent": doc.get("expression absent", {}),
                "phenotype_absent": doc.get("phenotype absent", {}),
            }
    print(f"  Loaded {len(corpus):,} corpus documents (full)")
    return corpus


def load_queries(data_dir, split="val"):
    queries = {}
    with open(f"{data_dir}/{split}.queries") as f:
        for line in f:
            qid, text = line.strip().split("\t", 1)
            queries[qid] = text
    print(f"  Loaded {len(queries):,} {split} queries")
    return queries


def load_qrels(data_dir, split="val"):
    qrels = defaultdict(set)
    with open(f"{data_dir}/{split}.qrels") as f:
        for line in f:
            parts = line.strip().split("\t")
            qid, _, docid, rel = parts
            if float(rel) > 0:
                qrels[qid].add(docid)
    print(f"  Loaded qrels for {len(qrels):,} queries")
    return qrels


def load_retrieved(path):
    retrieved = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            qid, _, docid, _, score, _ = parts
            retrieved[qid].append((docid, float(score)))
    print(f"  Loaded retrieved results for {len(retrieved):,} queries")
    return retrieved


def dcg(gains, k):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))
