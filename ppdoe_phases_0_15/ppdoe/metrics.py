"""
Relationship recovery metrics exactly as specified in the test program.
"""

from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from .ontology import Ontology, Relationship


@dataclass
class RelationshipMetrics:
    precision: float
    recall: float
    f1: float
    group_accuracy: float
    fine_accuracy: float
    candidate_edges: int
    confirmed_edges: int
    false_discovery_rate: float
    rejection_rate: float
    tp: int = 0
    fp: int = 0
    fn: int = 0


def _edge_set(rels: List[Relationship], status_filter: Optional[str] = "confirmed") -> Set[Tuple[str, str, str]]:
    s = set()
    for r in rels:
        if status_filter is None or r.status == status_filter:
            # undirected for hierarchy matching
            a, b = sorted([r.source_id, r.target_id])
            s.add((a, b, r.rel_type))
    return s


def gold_edge_set(fine_to_coarse: Dict[int, int], domain: str = "cifar100") -> Set[Tuple[str, str, str]]:
    """Build gold is_a edges from the official hierarchy."""
    edges = set()
    for fine, coarse in fine_to_coarse.items():
        src = f"{domain}_class_{fine}"
        tgt = f"{domain}_coarse_{coarse}"
        a, b = sorted([src, tgt])
        edges.add((a, b, "is_a"))
    return edges


def compute_metrics(
    ontology: Ontology,
    gold_edges: Set[Tuple[str, str, str]],
    group_preds: Optional[Dict[str, int]] = None,
    group_true: Optional[Dict[str, int]] = None,
    fine_preds: Optional[Dict[str, int]] = None,
    fine_true: Optional[Dict[str, int]] = None,
) -> RelationshipMetrics:
    """
    Compute the exact metrics required by the test program.
    Gold edges are used ONLY for evaluation.
    """
    confirmed = [r for r in ontology.list_relationships() if r.status == "confirmed"]
    candidates = [r for r in ontology.list_relationships() if r.status == "candidate"]
    rejected = [r for r in ontology.list_relationships() if r.status == "rejected"]

    pred_edges = _edge_set(confirmed, status_filter="confirmed")
    # also allow candidates that have high confidence to be counted if desired;
    # primary evaluation is on confirmed

    tp = len(pred_edges & gold_edges)
    fp = len(pred_edges - gold_edges)
    fn = len(gold_edges - pred_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    rejection_rate = len(rejected) / max(1, len(confirmed) + len(candidates) + len(rejected))

    # group / fine accuracy (optional, for classification side)
    group_acc = 0.0
    if group_preds and group_true:
        correct = sum(1 for k, v in group_preds.items() if group_true.get(k) == v)
        group_acc = correct / max(1, len(group_true))
    fine_acc = 0.0
    if fine_preds and fine_true:
        correct = sum(1 for k, v in fine_preds.items() if fine_true.get(k) == v)
        fine_acc = correct / max(1, len(fine_true))

    return RelationshipMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        group_accuracy=group_acc,
        fine_accuracy=fine_acc,
        candidate_edges=len(candidates),
        confirmed_edges=len(confirmed),
        false_discovery_rate=fdr,
        rejection_rate=rejection_rate,
        tp=tp,
        fp=fp,
        fn=fn,
    )


def jaccard_edge_sets(set_a: Set, set_b: Set) -> float:
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0
