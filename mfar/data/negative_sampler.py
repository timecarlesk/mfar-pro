from abc import ABC
from typing import AbstractSet, List, Set, Mapping, Tuple, Optional, Dict

import random

from mfar.data.index import Index
from mfar.data.typedef import Query, Document


class NegativeSampler(ABC):
    @property
    def n_sample(self) -> int:
        raise NotImplementedError

    def sample(self, query: Query, pos_for_each_qid: Mapping[str, AbstractSet[str]]) -> List[Document]:
        raise NotImplementedError

    def sample_batch(self, queries: List[Query], pos_for_each_qid: Mapping[str, AbstractSet[str]]) -> List[List[Document]]:
        raise NotImplementedError


class IndexNegativeSampler(NegativeSampler):
    def __init__(self,
                 index: Index,
                 documents: Mapping[str, str],
                 n_retrieve: int = 50,
                 n_bottom: int = 5,
                 n_sample: int = 1,
                 scope_map: Optional[Dict] = None,
                 ):
        self.index = index
        self.documents = documents
        self.n_retrieve = n_retrieve
        self.n_bottom = n_bottom
        self._n_sample = n_sample
        # scope_map: {"doc_scope": {doc_id: scope_id}, "query_scope": {qid: scope_id}}
        # When set, negatives are restricted to documents in the same scope.
        self.scope_map = scope_map
        if scope_map is not None:
            # Pre-build scope -> set of doc_ids for fast filtering
            self._scope_to_docs: Dict[str, Set[str]] = {}
            for doc_id, scope_id in scope_map.get("doc_scope", {}).items():
                if scope_id is not None:
                    self._scope_to_docs.setdefault(scope_id, set()).add(doc_id)

    @property
    def n_sample(self) -> int:
        return self._n_sample

    def _get_scope_docs(self, query: Query) -> Optional[Set[str]]:
        """Return the set of doc_ids in the same scope as the query, or None."""
        if self.scope_map is None:
            return None
        q_scope = self.scope_map.get("query_scope", {}).get(query._id)
        if q_scope is None:
            return None
        return self._scope_to_docs.get(q_scope)

    def sample(self, query: Query, pos_for_each_qid: Mapping[str, AbstractSet[str]]) -> List[Document]:
        scope_docs = self._get_scope_docs(query)

        neg_cand_with_scores: List[Tuple[str, float]] = [
            (doc_id, score)
            for doc_id, score in self.index.retrieve(query.text, top_k=self.n_retrieve)
            if doc_id not in pos_for_each_qid[query._id]  # remove correct samples
            and (scope_docs is None or doc_id in scope_docs)  # scope filter
        ]
        if len(neg_cand_with_scores) == 0:
            new_n_retrieve = len(pos_for_each_qid[query._id]) + self.n_bottom
            if scope_docs is not None:
                # Need to retrieve more to find enough in-scope negatives
                new_n_retrieve = max(new_n_retrieve, len(scope_docs))
            neg_cand_with_scores = [
                (doc_id, score)
                for doc_id, score in self.index.retrieve(query.text, top_k=new_n_retrieve)
                if doc_id not in pos_for_each_qid[query._id]
                and (scope_docs is None or doc_id in scope_docs)
            ]
        neg_cand_with_scores.sort(key=lambda x: x[1], reverse=True)
        neg_cand_ids = [doc_id for doc_id, _ in neg_cand_with_scores[-self.n_bottom:]]
        if len(neg_cand_ids) == 0:
            # Fallback: random in-scope negative
            if scope_docs is not None:
                fallback = list(scope_docs - pos_for_each_qid.get(query._id, set()))
                if fallback:
                    neg_cand_ids = [random.choice(fallback)]
                else:
                    neg_cand_ids = [list(scope_docs)[0]] if scope_docs else []
            else:
                # Truly empty — pick any document
                neg_cand_ids = [list(self.documents.keys())[0]]

        n_to_sample = min(self.n_sample, len(neg_cand_ids))
        sampled_neg_cand_ids = [neg_cand_ids[i] for i in random.sample(range(len(neg_cand_ids)), n_to_sample)]
        sampled_neg_cands = [
            Document(i, self.documents.get(i, ""))
            for i in sampled_neg_cand_ids
        ]
        return sampled_neg_cands

    def sample_batch(self, queries: List[Query], pos_for_each_qid: Mapping[str, AbstractSet[str]]) -> List[List[Document]]:
        # TODO: implement batch sampling
        return [self.sample(q, pos_for_each_qid) for q in queries]