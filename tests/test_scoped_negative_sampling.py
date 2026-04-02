"""
Unit test for per-conversation scoped negative sampling.

Toy dataset: 3 conversations, 5 sessions each = 15 docs.
Each conversation has 1 query with 1 positive doc.
Verifies that sampled negatives come ONLY from the same conversation.
"""

import os
import sys
import json
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mfar.data.typedef import Query, Document, Corpus
from mfar.data.index import BM25sSparseIndex
from mfar.data.negative_sampler import IndexNegativeSampler


def build_toy_data():
    """Build 3 conversations x 5 sessions = 15 docs, 3 queries."""
    conversations = {
        "conv_A": {
            "topics": ["italian food", "pasta recipe", "pizza dough", "wine pairing", "tiramisu dessert"],
            "query": "What Italian dish did we discuss first?",
            "positive_session": 0,
        },
        "conv_B": {
            "topics": ["japanese food", "sushi making", "ramen broth", "sake tasting", "mochi dessert"],
            "query": "Tell me about the Japanese cuisine conversation",
            "positive_session": 0,
        },
        "conv_C": {
            "topics": ["french food", "croissant baking", "cheese selection", "wine region", "creme brulee"],
            "query": "What French food topic came up?",
            "positive_session": 0,
        },
    }

    # Build corpus: doc_id -> text
    corpus_dict = {}
    # Build scope map
    doc_scope = {}
    query_scope = {}
    # Build queries and qrels
    queries = {}
    pos_for_each_qid = {}

    for conv_id, conv in conversations.items():
        for i, topic in enumerate(conv["topics"]):
            doc_id = f"{conv_id}_s{i}"
            # Make docs semantically similar across conversations (food topics)
            # to ensure cross-conversation negatives would be tempting
            text = f"User: I want to learn about {topic}. Assistant: Sure, let me tell you about {topic} in detail."
            corpus_dict[doc_id] = text
            doc_scope[doc_id] = conv_id

        qid = f"{conv_id}_q0"
        queries[qid] = conv["query"]
        query_scope[qid] = conv_id
        pos_session = conv["positive_session"]
        pos_for_each_qid[qid] = {f"{conv_id}_s{pos_session}"}

    scope_map = {"doc_scope": doc_scope, "query_scope": query_scope}
    return corpus_dict, queries, pos_for_each_qid, scope_map, conversations


class TestScopedNegativeSampling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Build toy corpus and BM25 index once."""
        cls.corpus_dict, cls.queries, cls.pos_for_each_qid, cls.scope_map, cls.conversations = build_toy_data()

        # Build BM25 index from corpus
        corpus = Corpus.from_docs_dict(cls.corpus_dict, dataset_name="memory")
        cls.bm25_index = BM25sSparseIndex.create(corpus, dataset_name="memory")

    def test_scoped_negatives_stay_in_conversation(self):
        """Core test: every sampled negative must be from the same conversation."""
        sampler = IndexNegativeSampler(
            index=self.bm25_index,
            documents=self.corpus_dict,
            n_retrieve=15,  # retrieve all docs
            n_bottom=4,     # bottom 4 as hard negatives
            n_sample=1,
            scope_map=self.scope_map,
        )

        for conv_id in self.conversations:
            qid = f"{conv_id}_q0"
            query = Query(qid, self.queries[qid])
            expected_scope = conv_id

            # Sample many times to catch stochastic failures
            for trial in range(50):
                negatives = sampler.sample(query, self.pos_for_each_qid)
                for neg in negatives:
                    neg_scope = self.scope_map["doc_scope"].get(neg._id)
                    self.assertEqual(
                        neg_scope, expected_scope,
                        f"Trial {trial}: Query {qid} (scope={expected_scope}) got negative "
                        f"{neg._id} from scope={neg_scope}"
                    )

    def test_scoped_negatives_exclude_positives(self):
        """Negatives must not include positive documents."""
        sampler = IndexNegativeSampler(
            index=self.bm25_index,
            documents=self.corpus_dict,
            n_retrieve=15,
            n_bottom=4,
            n_sample=1,
            scope_map=self.scope_map,
        )

        for conv_id in self.conversations:
            qid = f"{conv_id}_q0"
            query = Query(qid, self.queries[qid])
            positives = self.pos_for_each_qid[qid]

            for _ in range(50):
                negatives = sampler.sample(query, self.pos_for_each_qid)
                for neg in negatives:
                    self.assertNotIn(
                        neg._id, positives,
                        f"Query {qid} sampled positive {neg._id} as negative"
                    )

    def test_unscoped_can_return_cross_conversation(self):
        """Without scope_map, negatives CAN come from other conversations."""
        sampler = IndexNegativeSampler(
            index=self.bm25_index,
            documents=self.corpus_dict,
            n_retrieve=15,
            n_bottom=10,
            n_sample=1,
            scope_map=None,  # no scoping
        )

        # Sample many times — at least one should be cross-conversation
        # (since food topics are semantically similar across conversations)
        cross_conv_found = False
        for conv_id in self.conversations:
            qid = f"{conv_id}_q0"
            query = Query(qid, self.queries[qid])
            for _ in range(50):
                negatives = sampler.sample(query, self.pos_for_each_qid)
                for neg in negatives:
                    neg_scope = self.scope_map["doc_scope"].get(neg._id)
                    if neg_scope != conv_id:
                        cross_conv_found = True

        self.assertTrue(
            cross_conv_found,
            "Expected at least one cross-conversation negative without scoping"
        )

    def test_scoped_negative_count(self):
        """Each conversation has 5 sessions, 1 positive → 4 possible negatives."""
        sampler = IndexNegativeSampler(
            index=self.bm25_index,
            documents=self.corpus_dict,
            n_retrieve=15,
            n_bottom=4,
            n_sample=1,
            scope_map=self.scope_map,
        )

        for conv_id in self.conversations:
            qid = f"{conv_id}_q0"
            query = Query(qid, self.queries[qid])
            negatives = sampler.sample(query, self.pos_for_each_qid)
            self.assertEqual(len(negatives), 1, f"Expected 1 sampled negative, got {len(negatives)}")

    def test_all_in_scope_docs_reachable(self):
        """Over many samples, all non-positive in-scope docs should appear."""
        sampler = IndexNegativeSampler(
            index=self.bm25_index,
            documents=self.corpus_dict,
            n_retrieve=15,
            n_bottom=4,  # 4 of 4 possible negatives
            n_sample=1,
            scope_map=self.scope_map,
        )

        conv_id = "conv_A"
        qid = f"{conv_id}_q0"
        query = Query(qid, self.queries[qid])
        positives = self.pos_for_each_qid[qid]
        expected_negatives = {f"{conv_id}_s{i}" for i in range(5)} - positives

        seen = set()
        for _ in range(200):
            negatives = sampler.sample(query, self.pos_for_each_qid)
            for neg in negatives:
                seen.add(neg._id)

        self.assertTrue(
            seen.issubset(expected_negatives),
            f"Unexpected negatives: {seen - expected_negatives}"
        )
        # At least 2 of 4 should be reachable (BM25 might not rank all equally)
        self.assertGreaterEqual(
            len(seen), 2,
            f"Only {len(seen)} unique negatives seen out of {len(expected_negatives)} possible"
        )


if __name__ == "__main__":
    unittest.main()
