"""
Memory Knowledge Graph for Agent Memory.

Structured KG storing negation pattern → field routing rules learned
from training data. Replaces flat text memory_context with graph-based
retrieval.

Node types: Pattern, Field, AnswerType, Dataset
Edge types: BOOSTS, ANSWER_IS, HAS_FIELD, FROM_DATASET, SIMILAR_TO
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple


@dataclass
class PatternNode:
    """A negation query pattern observed in training data."""
    id: str                              # "disease|not_indicated"
    answer_type: str                     # "disease"
    negation_pattern: str                # "not_indicated"
    query_count: int = 0                 # number of training examples
    verification_rate: Optional[float] = None  # from feedback loop
    description: str = ""                # human-readable description
    example_queries: List[str] = field(default_factory=list)
    dataset: str = "PRIME"


@dataclass
class FieldNode:
    """A searchable field in the knowledge base."""
    id: str                              # "contraindication"
    description: str = ""                # "Diseases a drug should NOT treat"
    entity_types: List[str] = field(default_factory=list)  # which entity types have this


@dataclass
class AnswerTypeNode:
    """An entity type that can be the answer."""
    id: str                              # "disease"
    available_fields: List[str] = field(default_factory=list)


@dataclass
class DatasetNode:
    """A dataset/domain."""
    id: str                              # "PRIME"
    field_count: int = 0
    entity_types: List[str] = field(default_factory=list)


@dataclass
class Edge:
    """A directed relationship between two nodes."""
    source: str
    target: str
    relation: str                        # BOOSTS, ANSWER_IS, HAS_FIELD, FROM_DATASET, SIMILAR_TO
    weight: float = 1.0                  # 0.0-1.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class MemoryResult:
    """Result of querying the KG for a specific pattern."""
    boost_fields: List[Tuple[str, float]]  # [(field_name, weight), ...]
    confidence: Optional[float]            # verification rate
    query_count: int
    description: str
    similar_patterns: List[Dict] = field(default_factory=list)


class MemoryKG:
    """Knowledge Graph for agent memory."""

    def __init__(self):
        self.patterns: Dict[str, PatternNode] = {}
        self.fields: Dict[str, FieldNode] = {}
        self.answer_types: Dict[str, AnswerTypeNode] = {}
        self.datasets: Dict[str, DatasetNode] = {}
        self.edges: List[Edge] = []
        self.metadata: Dict = {}

        # Index for fast edge lookup
        self._edges_by_source: Dict[str, List[Edge]] = {}
        self._edges_by_target: Dict[str, List[Edge]] = {}

    def add_pattern(self, node: PatternNode):
        self.patterns[node.id] = node

    def add_field(self, node: FieldNode):
        self.fields[node.id] = node

    def add_answer_type(self, node: AnswerTypeNode):
        self.answer_types[node.id] = node

    def add_dataset(self, node: DatasetNode):
        self.datasets[node.id] = node

    def add_edge(self, edge: Edge):
        self.edges.append(edge)
        self._edges_by_source.setdefault(edge.source, []).append(edge)
        self._edges_by_target.setdefault(edge.target, []).append(edge)

    def get_edges(self, source: str, relation: str = None) -> List[Edge]:
        """Get all edges from a source node, optionally filtered by relation."""
        edges = self._edges_by_source.get(source, [])
        if relation:
            edges = [e for e in edges if e.relation == relation]
        return edges

    def get_incoming_edges(self, target: str, relation: str = None) -> List[Edge]:
        """Get all edges pointing to a target node."""
        edges = self._edges_by_target.get(target, [])
        if relation:
            edges = [e for e in edges if e.relation == relation]
        return edges

    # ── Query ────────────────────────────────────────────────────────────

    def query(self, answer_type: str, negation_pattern: str) -> Optional[MemoryResult]:
        """Query KG for a specific (answer_type, negation_pattern) combination.

        Returns structured MemoryResult or None if no match.
        """
        pattern_id = f"{answer_type}|{negation_pattern}"
        pattern = self.patterns.get(pattern_id)

        if pattern:
            return self._build_result(pattern)

        # Fallback: try patterns with same answer_type
        fallback_patterns = [
            p for p in self.patterns.values()
            if p.answer_type == answer_type
        ]
        if fallback_patterns:
            # Pick the one with most training examples
            best = max(fallback_patterns, key=lambda p: p.query_count)
            result = self._build_result(best)
            result.description = f"[Fallback from {best.id}] " + result.description
            return result

        return None

    def _build_result(self, pattern: PatternNode) -> MemoryResult:
        """Build a MemoryResult from a PatternNode by traversing edges."""
        # Get boost fields via BOOSTS edges
        boost_edges = self.get_edges(pattern.id, "BOOSTS")
        boost_fields = sorted(
            [(e.target, e.weight) for e in boost_edges],
            key=lambda x: -x[1]
        )

        # Get similar patterns
        similar_edges = self.get_edges(pattern.id, "SIMILAR_TO")
        similar = []
        for e in similar_edges:
            sim_pattern = self.patterns.get(e.target)
            if sim_pattern:
                similar.append({
                    "id": sim_pattern.id,
                    "description": sim_pattern.description,
                    "dataset": sim_pattern.dataset,
                    "similarity": e.weight,
                })

        return MemoryResult(
            boost_fields=boost_fields,
            confidence=pattern.verification_rate,
            query_count=pattern.query_count,
            description=pattern.description,
            similar_patterns=similar,
        )

    def query_all_for_type(self, answer_type: str) -> List[MemoryResult]:
        """Get all patterns for an answer type."""
        patterns = [p for p in self.patterns.values() if p.answer_type == answer_type]
        return [self._build_result(p) for p in sorted(patterns, key=lambda p: -p.query_count)]

    # ── Prompt Generation ────────────────────────────────────────────────

    def format_for_prompt(self, answer_type: str, negation_pattern: str) -> str:
        """Query KG and format result as natural language for Stage 2 prompt."""
        result = self.query(answer_type, negation_pattern)

        if result is None:
            # No match at all — return entity type field inventory
            at_node = self.answer_types.get(answer_type)
            if at_node and at_node.available_fields:
                return (
                    f"No matching pattern found for answer_type={answer_type}, "
                    f"pattern={negation_pattern}.\n"
                    f"Known fields for {answer_type}: {at_node.available_fields}\n"
                    f"Reason about which fields would contain the answer."
                )
            return "No matching pattern found. Reason from the query semantics."

        lines = []
        lines.append(f"Matched pattern: {result.description}")
        lines.append(f"Based on {result.query_count} training examples", )

        if result.confidence is not None:
            lines[-1] += f" (verification confidence: {result.confidence:.0%})"

        lines.append("Recommended boost fields (ranked by evidence strength):")
        for field_name, weight in result.boost_fields[:5]:
            lines.append(f"  - {field_name}: {weight:.0%} of gold documents have this field")

        if result.confidence is not None and result.confidence < 0.5:
            lines.append("WARNING: Low verification confidence. Consider reasoning from entity type field inventory instead.")
            at_node = self.answer_types.get(answer_type)
            if at_node:
                lines.append(f"Known fields for {answer_type}: {at_node.available_fields}")

        if result.similar_patterns:
            lines.append("Related patterns from other domains:")
            for sim in result.similar_patterns:
                lines.append(f"  - [{sim['dataset']}] {sim['description']} (similarity: {sim['similarity']:.0%})")

        return "\n".join(lines)

    def format_structured_for_prompt(self, answer_type: str, negation_pattern: str) -> str:
        """Compact structured format instead of natural language verbalization."""
        result = self.query(answer_type, negation_pattern)

        if result is None:
            at_node = self.answer_types.get(answer_type)
            fields = at_node.available_fields if at_node else []
            return f"MEMORY_MATCH: none\nentity_fields: {fields}"

        boost_str = ", ".join(f"{f}({w:.2f})" for f, w in result.boost_fields[:5])
        lines = [
            "MEMORY_MATCH:",
            f"  pattern: {answer_type}|{negation_pattern}",
            f"  n_examples: {result.query_count}",
        ]
        if result.confidence is not None:
            lines.append(f"  confidence: {result.confidence:.2f}")
        else:
            lines.append("  confidence: N/A")
        lines.append(f"  boost: {boost_str}")

        at_node = self.answer_types.get(answer_type)
        if at_node:
            lines.append(f"  entity_fields: {at_node.available_fields}")
        if result.confidence is not None and result.confidence < 0.5:
            lines.append("  WARNING: low confidence")

        return "\n".join(lines)

    def format_full_context(self) -> str:
        """Format entire KG as text for prompt (fallback when no match)."""
        lines = ["Retrieval re-routing rules from training data (KG-structured):"]
        lines.append("")

        for pattern in sorted(self.patterns.values(), key=lambda p: -p.query_count):
            if pattern.query_count < 3:
                continue
            boost_edges = self.get_edges(pattern.id, "BOOSTS")
            boost_fields = sorted([(e.target, e.weight) for e in boost_edges], key=lambda x: -x[1])
            fields_str = ", ".join(f"{f} ({w:.0%})" for f, w in boost_fields[:5])

            conf_str = ""
            if pattern.verification_rate is not None:
                conf_str = f", confidence: {pattern.verification_rate:.0%}"

            lines.append(f"When answer_type={pattern.answer_type} and pattern={pattern.negation_pattern} "
                         f"({pattern.query_count} examples{conf_str}):")
            lines.append(f"  Boost: {fields_str}")
            lines.append("")

        # Entity type field inventory
        lines.append("Entity type field inventory:")
        for at in sorted(self.answer_types.values(), key=lambda a: a.id):
            if at.available_fields:
                lines.append(f"  {at.id}: {at.available_fields}")

        return "\n".join(lines)

    # ── Serialization ────────────────────────────────────────────────────

    def to_json(self, path: str):
        """Save KG to JSON file."""
        data = {
            "nodes": {
                "patterns": {k: asdict(v) for k, v in self.patterns.items()},
                "fields": {k: asdict(v) for k, v in self.fields.items()},
                "answer_types": {k: asdict(v) for k, v in self.answer_types.items()},
                "datasets": {k: asdict(v) for k, v in self.datasets.items()},
            },
            "edges": [asdict(e) for e in self.edges],
            "metadata": self.metadata,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "MemoryKG":
        """Load KG from JSON file."""
        with open(path) as f:
            data = json.load(f)

        kg = cls()
        kg.metadata = data.get("metadata", {})

        for k, v in data["nodes"].get("patterns", {}).items():
            kg.add_pattern(PatternNode(**v))
        for k, v in data["nodes"].get("fields", {}).items():
            kg.add_field(FieldNode(**v))
        for k, v in data["nodes"].get("answer_types", {}).items():
            kg.add_answer_type(AnswerTypeNode(**v))
        for k, v in data["nodes"].get("datasets", {}).items():
            kg.add_dataset(DatasetNode(**v))

        for e in data.get("edges", []):
            kg.add_edge(Edge(**e))

        return kg

    # ── Stats ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Print KG summary statistics."""
        lines = [
            f"MemoryKG: {len(self.patterns)} patterns, {len(self.fields)} fields, "
            f"{len(self.answer_types)} answer types, {len(self.datasets)} datasets, "
            f"{len(self.edges)} edges",
        ]
        for rel in set(e.relation for e in self.edges):
            count = sum(1 for e in self.edges if e.relation == rel)
            lines.append(f"  {rel}: {count} edges")
        return "\n".join(lines)
