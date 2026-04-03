"""
Generate Memory Entries for Negation Field Routing.

Creates negation_memory.json containing field pair metadata.
Memory entries define WHICH fields to boost/suppress for each
negation pattern. The actual bias magnitude comes from Qwen3
per-query scoring at inference time.

Run from project root:
  python failure_analysis/type_b_memory/generate_memory.py

This generates a starter memory based on known field pairs.
Refine after reviewing Step 2 field confusion data.
"""

import json
import os

ANALYSIS_DIR = "output/failure_analysis/type_b_memory/analysis"


def generate_memory_entries():
    """Generate memory entries for known negation field pairs."""
    entries = [
        {
            "rule_id": "contraindication_negation",
            "field_pair_key": "indication/contraindication",
            "field_pair": ["indication", "contraindication"],
            "description": (
                "Query contains negation about drug indication "
                "(e.g., 'should not treat', 'not suitable for'). "
                "The relevant information is in the contraindication field, "
                "not the indication field."
            ),
            "example_queries": [
                "drugs that should NOT be used to treat diabetes",
                "medications not suitable for hypertension management",
                "which drugs should not be prescribed for asthma",
            ],
            "boost_fields": [
                "contraindication_dense",
                "contraindication_sparse",
            ],
            "suppress_fields": [
                "indication_dense",
                "indication_sparse",
            ],
            "source_subtype": "contraindication",
        },
        # B.indication — TBD after Step 2 field confusion analysis
        # The gold docs for B.indication queries have 0% indication field populated.
        # We need Step 2 data to determine which fields to boost instead.
        # Placeholder entry — update boost/suppress fields after reviewing
        # field_confusion_train.json.
        {
            "rule_id": "indication_negation",
            "field_pair_key": "indication/contraindication_reverse",
            "field_pair": ["contraindication", "indication"],
            "description": (
                "Query contains negation about contraindication "
                "(e.g., 'lacking treatment', 'no drug for'). "
                "PLACEHOLDER: gold docs may not have indication field populated. "
                "Update after Step 2 analysis."
            ),
            "example_queries": [
                "diseases lacking approved treatment options",
                "conditions without effective drug therapy",
            ],
            "boost_fields": [
                # TBD — may be details_dense, associated_with_dense, etc.
                # Depends on Step 2 field confusion data
            ],
            "suppress_fields": [],
            "source_subtype": "indication",
            "_status": "placeholder_pending_step2",
        },
        {
            "rule_id": "target_negation",
            "field_pair_key": "target",
            "field_pair": ["target", "target"],
            "description": (
                "Query contains negation about drug/protein targets "
                "(e.g., 'does not target', 'no binding'). "
                "May need to suppress target field or boost other fields."
            ),
            "example_queries": [
                "proteins that do not target kinase",
                "drugs without binding to receptor",
            ],
            "boost_fields": [],
            "suppress_fields": [
                "target_dense",
                "target_sparse",
            ],
            "source_subtype": "target",
            "_status": "placeholder_low_count",
        },
        {
            "rule_id": "side_effect_negation",
            "field_pair_key": "side_effect",
            "field_pair": ["side_effect", "side_effect"],
            "description": (
                "Query contains negation about side effects "
                "(e.g., 'without adverse effects', 'no toxicity'). "
            ),
            "example_queries": [
                "drugs without serious side effects",
                "medications with no adverse reactions",
            ],
            "boost_fields": [],
            "suppress_fields": [
                "side effect_dense",
                "side effect_sparse",
            ],
            "source_subtype": "side_effect",
            "_status": "placeholder_low_count",
        },
        {
            "rule_id": "associated_with_negation",
            "field_pair_key": "associated_with",
            "field_pair": ["associated with", "associated with"],
            "description": (
                "Query contains negation about gene-disease association "
                "(e.g., 'not associated with', 'unrelated to'). "
            ),
            "example_queries": [
                "genes not associated with cancer",
                "proteins unrelated to inflammation",
            ],
            "boost_fields": [],
            "suppress_fields": [
                "associated with_dense",
                "associated with_sparse",
            ],
            "source_subtype": "associated_with",
            "_status": "placeholder_low_count",
        },
        {
            "rule_id": "ppi_negation",
            "field_pair_key": "ppi",
            "field_pair": ["ppi", "ppi"],
            "description": (
                "Query contains negation about protein-protein interactions "
                "(e.g., 'does not interact', 'no interaction'). "
            ),
            "example_queries": [
                "proteins that do not interact with p53",
                "genes without known PPI",
            ],
            "boost_fields": [],
            "suppress_fields": [
                "ppi_dense",
                "ppi_sparse",
            ],
            "source_subtype": "ppi",
            "_status": "placeholder_low_count",
        },
    ]

    return entries


def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    entries = generate_memory_entries()

    out_path = os.path.join(ANALYSIS_DIR, "negation_memory.json")
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"Generated {len(entries)} memory entries → {out_path}")
    print()
    for entry in entries:
        status = entry.get("_status", "active")
        print(f"  [{status}] {entry['rule_id']}: {entry['field_pair_key']}")
        print(f"    boost:    {entry['boost_fields']}")
        print(f"    suppress: {entry['suppress_fields']}")
        print()


if __name__ == "__main__":
    main()
