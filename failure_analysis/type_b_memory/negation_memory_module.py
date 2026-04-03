"""
Negation Memory Module for mFAR.

Loads pre-computed Qwen3 classifications (from batch_qwen3_inference.py)
to produce per-query logit biases that re-route field weights.

Pipeline:
  1. Look up query ID in Qwen3 cache
  2. If needs_reroute=True and boost_fields non-empty: apply bias
  3. All boosted fields get +α, all suppressed fields get -α
  4. Return bias for injection into LinearWeights.forward()
"""

import json
import os
import torch
from typing import Dict, Optional


class NegationMemory:
    """Provides per-query logit biases from pre-computed Qwen3 negation classifications."""

    def __init__(self, cache_paths, field_name_to_idx, alpha=1.0, use_boost=True, use_suppress=False):
        """
        Args:
            cache_paths: Path(s) to qwen3_cache_{split}.jsonl — comma-separated string or single path
            field_name_to_idx: Dict mapping mFAR field keys (e.g., "contraindication_dense") to indices
            alpha: Fixed bias magnitude.
            use_boost: Apply +α to boost fields
            use_suppress: Apply -α to suppress fields
        """
        self.alpha = alpha
        self.use_boost = use_boost
        self.use_suppress = use_suppress
        self.field_name_to_idx = field_name_to_idx

        # Build reverse lookup: base field name → [dense_idx, sparse_idx]
        # e.g., "contraindication" → indices of "contraindication_dense" and "contraindication_sparse"
        self.base_to_indices = {}
        for key, idx in field_name_to_idx.items():
            for suffix in ("_dense", "_sparse"):
                if key.endswith(suffix):
                    base = key[:-len(suffix)]
                    self.base_to_indices.setdefault(base, []).append(idx)
                    break

        # Load Qwen3 cache(s) — supports comma-separated paths
        self.qwen3_cache = {}
        if isinstance(cache_paths, str):
            cache_paths = [p.strip() for p in cache_paths.split(",")]
        for cache_path in cache_paths:
            if os.path.exists(cache_path):
                with open(cache_path) as f:
                    for line in f:
                        entry = json.loads(line)
                        self.qwen3_cache[entry["qid"]] = entry

        # Count how many queries will be rerouted
        rerouted = sum(1 for e in self.qwen3_cache.values()
                       if e.get("needs_reroute") and (e.get("boost_fields") or e.get("suppress_fields")))
        modes = []
        if use_boost: modes.append("boost")
        if use_suppress: modes.append("suppress")
        print(f"  NegationMemory loaded: {len(self.qwen3_cache)} queries, "
              f"{rerouted} with re-routing, alpha={alpha}, mode={'+'.join(modes) or 'none'}")

    def get_logit_bias(self, qid, num_fields):
        """
        Get logit bias tensor for a query.

        Args:
            qid: Query ID to look up in cache
            num_fields: Total number of fields (for tensor shape)

        Returns:
            torch.Tensor of shape [1, num_fields] or None if no bias needed
        """
        entry = self.qwen3_cache.get(qid)
        if entry is None:
            return None
        if not entry.get("needs_reroute"):
            return None

        boost_fields = entry.get("boost_fields") or []
        suppress_fields = entry.get("suppress_fields") or []

        if not boost_fields and not suppress_fields:
            return None

        bias = torch.zeros(1, num_fields)

        if self.use_boost and boost_fields:
            # Only boost the top-1 field (most confident recommendation)
            top_field = boost_fields[0]
            for idx in self.base_to_indices.get(top_field, []):
                bias[0, idx] = self.alpha

        if self.use_suppress:
            for field_name in suppress_fields:
                for idx in self.base_to_indices.get(field_name, []):
                    bias[0, idx] = -self.alpha

        if bias.abs().sum() == 0:
            return None
        return bias


def load_negation_memory(cache_path, field_info, alpha=1.0, use_boost=True, use_suppress=False):
    """
    Factory function to create NegationMemory from mFAR field_info dict.

    Args:
        cache_path: Path(s) to qwen3_cache_{split}.jsonl (comma-separated)
        field_info: OrderedDict from mFAR's resolve_fields() — keys are field names, order = index
        alpha: Fixed bias magnitude
        use_boost: Apply +α to boost fields
        use_suppress: Apply -α to suppress fields
    """
    field_name_to_idx = {name: idx for idx, name in enumerate(field_info.keys())}
    return NegationMemory(cache_path, field_name_to_idx, alpha, use_boost, use_suppress)
