"""
Build Memory KG from Training Data.

Reads field_confusion_train.json and verification_rates.json,
constructs a structured Knowledge Graph, and optionally visualizes it.

Commands:
  build     — build KG from training data → memory_kg.json
  visualize — generate KG visualization → memory_kg.png

Run from project root:
  $PY failure_analysis/type_b_memory/rerank/train_memory/build_memory_kg.py build
  $PY failure_analysis/type_b_memory/rerank/train_memory/build_memory_kg.py visualize
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from failure_analysis.type_b_memory.rerank.shared.memory_kg import (
    MemoryKG, PatternNode, FieldNode, AnswerTypeNode, DatasetNode, Edge,
)
from failure_analysis.utils import RELATION_FIELDS

ANALYSIS_DIR = "output/failure_analysis/type_b_memory/analysis"

# ── Schema knowledge: which entity types have which fields ───────────────────

ENTITY_TYPE_FIELDS = {
    "drug": [
        "indication", "contraindication", "side effect", "target",
        "carrier", "enzyme", "transporter", "synergistic interaction",
        "off-label use", "interacts with",
    ],
    "disease": [
        "associated with", "phenotype present", "phenotype absent",
        "parent-child", "indication", "contraindication", "off-label use",
        "linked to",
    ],
    "gene/protein": [
        "ppi", "interacts with", "expression present", "expression absent",
        "associated with", "target", "parent-child",
    ],
    "anatomy": [
        "expression present", "expression absent", "parent-child",
    ],
    "phenotype": [
        "phenotype present", "phenotype absent", "parent-child",
        "side effect",
    ],
    "pathway": [
        "interacts with", "parent-child",
    ],
    "cellular_component": [
        "interacts with", "parent-child", "ppi",
    ],
    "biological_process": [
        "interacts with", "parent-child",
    ],
    "exposure": [
        "interacts with", "linked to", "parent-child",
    ],
}

NEGATION_PATTERN_DESCRIPTIONS = {
    "not_indicated": "Drug/treatment should not be used for a condition",
    "not_expressed": "Gene/protein not expressed or lacking expression in tissue",
    "lacking_treatment": "No approved drugs or treatments exist for a condition",
    "avoid_contraindicated": "Drug is contraindicated, should be avoided",
    "not_associated": "Gene/entity not associated or related to a condition",
    "phenotype_absent": "Phenotype or symptom not observed",
    "no_side_effect": "Drug without side effects or adverse reactions",
    "other": "Other negation or semantic constraint pattern",
}


# ── KG Builder ───────────────────────────────────────────────────────────────

def build_kg(field_confusion_path, verification_rates_path=None, dataset_name="PRIME"):
    """Build MemoryKG from training data statistics."""
    kg = MemoryKG()
    kg.metadata = {
        "dataset": dataset_name,
        "generated_from": field_confusion_path,
    }

    # Load field confusion data
    with open(field_confusion_path) as f:
        confusion_data = json.load(f)

    # Load verification rates if available
    verification_rates = {}
    if verification_rates_path and os.path.exists(verification_rates_path):
        with open(verification_rates_path) as f:
            verification_rates = json.load(f)
        kg.metadata["verification_source"] = verification_rates_path

    # ── Dataset node ──
    all_entity_types = set()
    kg.add_dataset(DatasetNode(
        id=dataset_name,
        field_count=len(RELATION_FIELDS),
        entity_types=[],  # populated below
    ))

    # ── Field nodes ──
    for field_name in RELATION_FIELDS:
        # Find which entity types have this field
        entity_types = [
            et for et, fields in ENTITY_TYPE_FIELDS.items()
            if field_name in fields
        ]
        kg.add_field(FieldNode(
            id=field_name,
            description=f"Relation field: {field_name}",
            entity_types=entity_types,
        ))

    # ── Pattern nodes + edges from field confusion groups ──
    groups = confusion_data.get("groups", {})
    per_query = confusion_data.get("per_query_details", [])

    # Collect example queries per group
    group_examples = {}
    for detail in per_query[:500]:  # limit for memory
        gkey = f"{detail.get('qwen3_answer_type', 'unknown')}|{detail.get('negation_pattern', 'other')}"
        # Use gold type for key (consistent with groups)
        for gold_doc in detail.get("gold_docs", []):
            gold_type = gold_doc.get("type", "unknown")
            gkey_gold = f"{gold_type}|{detail.get('negation_pattern', 'other')}"
            if gkey_gold not in group_examples:
                group_examples[gkey_gold] = []
            if len(group_examples[gkey_gold]) < 3:
                group_examples[gkey_gold].append(detail.get("query", "")[:150])

    for group_key, group_info in groups.items():
        query_count = group_info.get("query_count", 0)
        if query_count < 3:
            continue

        parts = group_key.split("|", 1)
        if len(parts) != 2:
            continue
        answer_type, neg_pattern = parts

        all_entity_types.add(answer_type)

        # Verification rate
        vr = verification_rates.get(group_key, {})
        v_rate = vr.get("rate") if vr else None

        # Description
        base_desc = NEGATION_PATTERN_DESCRIPTIONS.get(neg_pattern, "Negation constraint")
        description = f"{base_desc} (answer type: {answer_type})"

        # Create pattern node
        pattern = PatternNode(
            id=group_key,
            answer_type=answer_type,
            negation_pattern=neg_pattern,
            query_count=query_count,
            verification_rate=v_rate,
            description=description,
            example_queries=group_examples.get(group_key, []),
            dataset=dataset_name,
        )
        kg.add_pattern(pattern)

        # BOOSTS edges from gold field distribution
        field_dist = group_info.get("gold_field_distribution", {})
        total_field_count = sum(field_dist.values())
        if total_field_count > 0:
            for field_name, count in field_dist.items():
                weight = count / total_field_count
                if weight >= 0.05:  # only include fields with >= 5% presence
                    kg.add_edge(Edge(
                        source=group_key,
                        target=field_name,
                        relation="BOOSTS",
                        weight=round(weight, 3),
                    ))

        # ANSWER_IS edge (always 1.0 since grouping is by gold type)
        kg.add_edge(Edge(
            source=group_key,
            target=answer_type,
            relation="ANSWER_IS",
            weight=1.0,
        ))

        # FROM_DATASET edge
        kg.add_edge(Edge(
            source=group_key,
            target=dataset_name,
            relation="FROM_DATASET",
            weight=1.0,
        ))

    # ── AnswerType nodes + HAS_FIELD edges ──
    for entity_type in all_entity_types:
        available = ENTITY_TYPE_FIELDS.get(entity_type, [])
        kg.add_answer_type(AnswerTypeNode(
            id=entity_type,
            available_fields=available,
        ))
        for field_name in available:
            kg.add_edge(Edge(
                source=entity_type,
                target=field_name,
                relation="HAS_FIELD",
                weight=1.0,
            ))

    # Update dataset node with entity types
    kg.datasets[dataset_name].entity_types = sorted(all_entity_types)

    return kg


# ── Visualization ────────────────────────────────────────────────────────────

def visualize_kg(kg, output_path):
    """Generate a visualization of the KG using networkx + matplotlib."""
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  ERROR: networkx and matplotlib required for visualization")
        print("  pip install networkx matplotlib")
        return

    G = nx.DiGraph()

    # Color scheme
    colors = {
        "pattern": "#4A90D9",
        "field": "#50C878",
        "answer_type": "#FF8C42",
        "dataset": "#9B59B6",
    }

    # Add nodes
    node_colors = []
    node_sizes = []
    labels = {}

    for p in kg.patterns.values():
        G.add_node(p.id)
        node_colors.append(colors["pattern"])
        node_sizes.append(200 + p.query_count * 2)
        labels[p.id] = f"{p.answer_type}\n{p.negation_pattern}\n(n={p.query_count})"

    for f in kg.fields.values():
        # Only add fields that have BOOSTS edges
        if kg.get_incoming_edges(f.id, "BOOSTS"):
            G.add_node(f.id)
            node_colors.append(colors["field"])
            node_sizes.append(150)
            labels[f.id] = f.id

    for at in kg.answer_types.values():
        G.add_node(at.id)
        node_colors.append(colors["answer_type"])
        node_sizes.append(300)
        labels[at.id] = at.id

    # Add edges (only BOOSTS with weight >= 0.1 to reduce clutter)
    edge_colors = []
    edge_widths = []
    for e in kg.edges:
        if e.relation == "BOOSTS" and e.weight >= 0.1:
            if e.source in G.nodes and e.target in G.nodes:
                G.add_edge(e.source, e.target)
                edge_colors.append("#4A90D9")
                edge_widths.append(e.weight * 5)
        elif e.relation == "ANSWER_IS":
            if e.source in G.nodes and e.target in G.nodes:
                G.add_edge(e.source, e.target)
                edge_colors.append("#FF8C42")
                edge_widths.append(1.5)

    # Layout
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                           alpha=0.5, arrows=True, arrowsize=10, ax=ax)

    # Legend
    legend_patches = [
        mpatches.Patch(color=colors["pattern"], label="Pattern (answer_type|neg_pattern)"),
        mpatches.Patch(color=colors["field"], label="Field"),
        mpatches.Patch(color=colors["answer_type"], label="Answer Type"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8)
    ax.set_title("Memory Knowledge Graph", fontsize=14)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Visualization saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build Memory KG from training data")
    subparsers = parser.add_subparsers(dest="command")

    # Build command
    b_parser = subparsers.add_parser("build", help="Build KG from field confusion data")
    b_parser.add_argument("--field_confusion",
                          default=os.path.join(ANALYSIS_DIR, "field_confusion_train.json"))
    b_parser.add_argument("--verification_rates",
                          default=os.path.join(ANALYSIS_DIR, "verification_rates.json"))
    b_parser.add_argument("--output",
                          default=os.path.join(ANALYSIS_DIR, "memory_kg.json"))
    b_parser.add_argument("--dataset", default="PRIME")

    # Visualize command
    v_parser = subparsers.add_parser("visualize", help="Visualize KG")
    v_parser.add_argument("--kg", default=os.path.join(ANALYSIS_DIR, "memory_kg.json"))
    v_parser.add_argument("--output", default=os.path.join(ANALYSIS_DIR, "memory_kg.png"))

    args = parser.parse_args()

    if args.command == "build":
        print(f"Building KG from {args.field_confusion}...")
        kg = build_kg(args.field_confusion, args.verification_rates, args.dataset)
        kg.to_json(args.output)
        print(f"\n{kg.summary()}")
        print(f"\nSaved to {args.output}")

        # Test a query
        print(f"\nTest query (disease, not_indicated):")
        result = kg.format_for_prompt("disease", "not_indicated")
        print(result)

    elif args.command == "visualize":
        print(f"Loading KG from {args.kg}...")
        kg = MemoryKG.from_json(args.kg)
        print(kg.summary())
        visualize_kg(kg, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
