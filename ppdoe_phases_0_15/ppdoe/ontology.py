"""
Ontology and concept / relationship primitives.
Faithful to the OWL/RDF-inspired design in the PPDOE paper.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import copy
import hashlib
import json
from datetime import datetime


@dataclass
class Concept:
    """A concept / entity in the ontology."""
    concept_id: str
    name: str
    domain: str                          # e.g. "cifar100", "healthcare", "aviation"
    attributes: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None
    version_created: int = 0
    is_gold: bool = False                # only set for evaluation; never used in prediction

    def to_dict(self) -> dict:
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "domain": self.domain,
            "attributes": self.attributes,
            "version_created": self.version_created,
            "is_gold": self.is_gold,
        }


@dataclass
class Relationship:
    """A directed relationship between two concepts."""
    rel_id: str
    source_id: str
    target_id: str
    rel_type: str                        # e.g. "is_a", "has_condition", "linked_to"
    confidence: float = 0.0
    evidence: Dict[str, float] = field(default_factory=dict)  # structural, terminological, contextual
    version_created: int = 0
    status: str = "candidate"            # candidate | confirmed | rejected | deprecated
    is_gold: bool = False                # evaluation only

    def to_dict(self) -> dict:
        return {
            "rel_id": self.rel_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "rel_type": self.rel_type,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "version_created": self.version_created,
            "status": self.status,
            "is_gold": self.is_gold,
        }


class Ontology:
    """
    Versioned ontology holding concepts and relationships.
    Matches the paper's O / O' design.
    """

    def __init__(self, name: str = "ppdoe_ontology"):
        self.name = name
        self.version: int = 0
        self.concepts: Dict[str, Concept] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.history: List[dict] = []
        self._concept_name_index: Dict[str, str] = {}   # name -> concept_id

    # ------------------------------------------------------------------
    # Concept management
    # ------------------------------------------------------------------
    def add_concept(self, concept: Concept, force_new_version: bool = False) -> str:
        if concept.concept_id in self.concepts and not force_new_version:
            # update attributes only
            existing = self.concepts[concept.concept_id]
            existing.attributes.update(concept.attributes)
            if concept.embeddings is not None:
                existing.embeddings = concept.embeddings
            return concept.concept_id
        concept.version_created = self.version
        self.concepts[concept.concept_id] = concept
        self._concept_name_index[concept.name.lower()] = concept.concept_id
        return concept.concept_id

    def get_concept(self, concept_id: str) -> Optional[Concept]:
        return self.concepts.get(concept_id)

    def get_concept_by_name(self, name: str) -> Optional[Concept]:
        cid = self._concept_name_index.get(name.lower())
        return self.concepts.get(cid) if cid else None

    def list_concepts(self, domain: Optional[str] = None) -> List[Concept]:
        if domain is None:
            return list(self.concepts.values())
        return [c for c in self.concepts.values() if c.domain == domain]

    # ------------------------------------------------------------------
    # Relationship management
    # ------------------------------------------------------------------
    def add_relationship(self, rel: Relationship) -> str:
        rel.version_created = self.version
        self.relationships[rel.rel_id] = rel
        return rel.rel_id

    def get_relationship(self, rel_id: str) -> Optional[Relationship]:
        return self.relationships.get(rel_id)

    def list_relationships(self, status: Optional[str] = None) -> List[Relationship]:
        if status is None:
            return list(self.relationships.values())
        return [r for r in self.relationships.values() if r.status == status]

    def confirm_relationship(self, rel_id: str, confidence: float) -> None:
        if rel_id in self.relationships:
            self.relationships[rel_id].status = "confirmed"
            self.relationships[rel_id].confidence = confidence

    def reject_relationship(self, rel_id: str) -> None:
        if rel_id in self.relationships:
            self.relationships[rel_id].status = "rejected"

    def deprecate_relationship(self, rel_id: str) -> None:
        if rel_id in self.relationships:
            self.relationships[rel_id].status = "deprecated"

    # ------------------------------------------------------------------
    # Versioning (paper: commit updated ontology as a new version)
    # ------------------------------------------------------------------
    def commit_version(self, note: str = "") -> int:
        snapshot = {
            "version": self.version,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "note": note,
            "n_concepts": len(self.concepts),
            "n_relationships": len(self.relationships),
            "n_confirmed": sum(1 for r in self.relationships.values() if r.status == "confirmed"),
            "n_candidates": sum(1 for r in self.relationships.values() if r.status == "candidate"),
        }
        self.history.append(snapshot)
        self.version += 1
        return self.version

    def current_state_hash(self) -> str:
        """Deterministic hash of current concepts + confirmed relationships (for audit)."""
        payload = {
            "concepts": sorted([c.to_dict() for c in self.concepts.values()], key=lambda x: x["concept_id"]),
            "confirmed_rels": sorted(
                [r.to_dict() for r in self.relationships.values() if r.status == "confirmed"],
                key=lambda x: x["rel_id"],
            ),
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def clone(self) -> "Ontology":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "concepts": {cid: c.to_dict() for cid, c in self.concepts.items()},
            "relationships": {rid: r.to_dict() for rid, r in self.relationships.items()},
            "history": self.history,
        }
