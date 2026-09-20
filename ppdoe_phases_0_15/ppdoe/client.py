"""
Federated client that keeps raw data local and releases only
permitted model updates + semantic evidence.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from .ontology import Concept, Relationship
from .privacy import DifferentialPrivacy


class FederatedClient:
    """
    One federated domain / client.
    Holds private data D_k, maps to ontology concepts, computes local updates.
    """

    def __init__(
        self,
        client_id: str,
        domain: str,
        data: Any,                       # local samples (images, features, or JSON records)
        labels: Optional[np.ndarray] = None,
        feature_dim: int = 64,
        seed: Optional[int] = None,
    ):
        self.client_id = client_id
        self.domain = domain
        self.data = data
        self.labels = labels
        self.feature_dim = feature_dim
        self.rng = np.random.RandomState(seed)
        self.local_concepts: Dict[str, Concept] = {}
        self.local_prototypes: Dict[str, np.ndarray] = {}
        self.round = 0

    def map_to_concepts(self, ontology_version: int = 0) -> List[Concept]:
        """
        Map local entities / classes to Concept objects.
        For CIFAR-100 style: each unique fine label becomes a concept.
        For JSON-style multi-domain: each entity becomes a concept.
        """
        concepts = []
        if self.labels is not None:
            # image / classification domain
            unique = np.unique(self.labels)
            for lab in unique:
                cid = f"{self.domain}_class_{int(lab)}"
                name = f"{self.domain}_{int(lab)}"
                c = Concept(
                    concept_id=cid,
                    name=name,
                    domain=self.domain,
                    attributes={"label": int(lab), "client": self.client_id},
                    version_created=ontology_version,
                )
                concepts.append(c)
                self.local_concepts[cid] = c
        else:
            # assume list of dict records (paper style)
            for i, rec in enumerate(self.data):
                entity = rec.get("entity", f"entity_{i}")
                cid = f"{self.domain}_{entity.replace(' ', '_')}"
                c = Concept(
                    concept_id=cid,
                    name=entity,
                    domain=self.domain,
                    attributes=rec,
                    version_created=ontology_version,
                )
                concepts.append(c)
                self.local_concepts[cid] = c
        return concepts

    def compute_prototypes(self, features: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Compute class / entity prototypes (mean feature vectors).
        """
        prototypes = {}
        if features is not None and self.labels is not None:
            for lab in np.unique(self.labels):
                mask = self.labels == lab
                proto = features[mask].mean(axis=0)
                cid = f"{self.domain}_class_{int(lab)}"
                prototypes[cid] = proto
                self.local_prototypes[cid] = proto
        else:
            # random or attribute-based vectors for non-image domains
            for cid, concept in self.local_concepts.items():
                vec = self.rng.randn(self.feature_dim).astype(np.float32)
                vec /= (np.linalg.norm(vec) + 1e-8)
                prototypes[cid] = vec
                self.local_prototypes[cid] = vec
        return prototypes

    def local_update(
        self,
        global_weights: Optional[np.ndarray] = None,
        privacy: Optional[DifferentialPrivacy] = None,
    ) -> Dict[str, Any]:
        """
        Produce a local model update (or prototype summary) and apply privacy.
        Returns only the permitted information.
        """
        self.round += 1
        # simple prototype-based update for this reconstruction
        protos = self.compute_prototypes()
        update_vec = np.stack(list(protos.values())) if protos else np.zeros((1, self.feature_dim))
        mean_update = update_vec.mean(axis=0)

        if privacy is not None:
            mean_update = privacy.privatize_update(mean_update)
            privatized_protos = {
                cid: privacy.privatize_update(p) for cid, p in protos.items()
            }
        else:
            privatized_protos = protos

        return {
            "client_id": self.client_id,
            "domain": self.domain,
            "round": self.round,
            "mean_update": mean_update,
            "prototypes": privatized_protos,
            "n_samples": len(self.data) if hasattr(self.data, "__len__") else 0,
            "concepts": list(self.local_concepts.keys()),
        }

    def semantic_evidence(self) -> List[dict]:
        """Evidence released for ontology aggregation (no raw records)."""
        evidence = []
        for cid, concept in self.local_concepts.items():
            evidence.append({
                "concept_id": cid,
                "name": concept.name,
                "domain": concept.domain,
                "attributes": {k: v for k, v in concept.attributes.items()
                               if k not in ("raw_data", "image")},
            })
        return evidence
