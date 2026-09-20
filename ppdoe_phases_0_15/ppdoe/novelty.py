"""
Semantic novelty detection (paper: θ = 0.5).
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
from .ontology import Ontology, Concept, Relationship


class NoveltyDetector:
    """
    Detects concepts / relationships that are not adequately represented
    in the current ontology. Novelty score > θ triggers evolution.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def concept_novelty(self, concept: Concept, ontology: Ontology) -> float:
        """
        Simple novelty: 1.0 if concept name never seen, else based on
        attribute / embedding distance to nearest existing concept.
        """
        existing = ontology.get_concept_by_name(concept.name)
        if existing is None:
            # check domain + attribute overlap
            same_domain = [c for c in ontology.list_concepts(domain=concept.domain)]
            if not same_domain:
                return 1.0
            # attribute Jaccard-like
            scores = []
            for c in same_domain:
                keys_a = set(concept.attributes.keys())
                keys_b = set(c.attributes.keys())
                if not keys_a and not keys_b:
                    scores.append(0.0)
                    continue
                inter = len(keys_a & keys_b)
                union = len(keys_a | keys_b)
                scores.append(1.0 - (inter / union if union else 0.0))
            return float(np.mean(scores)) if scores else 1.0
        return 0.0  # already present

    def relationship_novelty(self, rel: Relationship, ontology: Ontology) -> float:
        """
        Novelty of a candidate relationship.
        High if no similar edge exists between the two concepts (or their ancestors).
        """
        # exact match?
        for existing in ontology.list_relationships():
            if (existing.source_id == rel.source_id and
                existing.target_id == rel.target_id and
                existing.rel_type == rel.rel_type):
                return 0.0
        # soft: any edge between the same pair?
        for existing in ontology.list_relationships():
            if existing.source_id == rel.source_id and existing.target_id == rel.target_id:
                return 0.3
        return 1.0

    def score(self, candidates: List[Relationship], ontology: Ontology) -> List[Tuple[Relationship, float]]:
        scored = []
        for rel in candidates:
            nov = self.relationship_novelty(rel, ontology)
            scored.append((rel, nov))
        return scored

    def should_evolve(self, novelty_score: float) -> bool:
        return novelty_score > self.threshold
