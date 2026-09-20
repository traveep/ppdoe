"""
Ontology-based aggregation and federated model aggregation.
Hierarchical fine→coarse candidate generation for relationship recovery.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from difflib import SequenceMatcher
from .ontology import Ontology, Concept, Relationship
from .device import get_device


def terminological_similarity(name_a: str, name_b: str) -> float:
    return SequenceMatcher(None, name_a.lower(), name_b.lower()).ratio()


def structural_similarity(concept_a: Concept, concept_b: Concept, ontology: Ontology) -> float:
    keys_a = set(concept_a.attributes.keys())
    keys_b = set(concept_b.attributes.keys())
    if not keys_a and not keys_b:
        return 0.5 if concept_a.domain == concept_b.domain else 0.1
    inter = len(keys_a & keys_b)
    union = len(keys_a | keys_b)
    jacc = inter / union if union else 0.0
    domain_bonus = 0.2 if concept_a.domain == concept_b.domain else 0.0
    return min(1.0, jacc + domain_bonus)


def contextual_similarity(proto_a: Optional[np.ndarray], proto_b: Optional[np.ndarray]) -> float:
    if proto_a is None or proto_b is None:
        return 0.0
    a = proto_a.flatten().astype(np.float32)
    b = proto_b.flatten().astype(np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def contextual_similarity_matrix(
    prototypes: Dict[str, np.ndarray],
    device: Optional[str] = None,
) -> Tuple[List[str], np.ndarray]:
    ids = list(prototypes.keys())
    if not ids:
        return ids, np.zeros((0, 0), dtype=np.float32)
    X = np.stack([prototypes[i].flatten().astype(np.float32) for i in ids])
    norms = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    X = X / norms
    dev = device or get_device()
    if dev == "cuda":
        try:
            import torch
            t = torch.from_numpy(X).to("cuda")
            S = (t @ t.T).cpu().numpy().astype(np.float32)
            return ids, S
        except Exception:
            pass
    return ids, (X @ X.T).astype(np.float32)


class OntologyAggregator:
    """Primary candidates = fine-class → coarse-parent is_a edges."""

    def __init__(
        self,
        structural_w: float = 0.2,
        terminological_w: float = 0.1,
        contextual_w: float = 0.7,
        candidate_threshold: float = 0.10,
        device: Optional[str] = None,
        top_k_parents: int = 5,
    ):
        self.structural_w = structural_w
        self.terminological_w = terminological_w
        self.contextual_w = contextual_w
        self.candidate_threshold = candidate_threshold
        self.device = device
        self.top_k_parents = top_k_parents

    def build_candidates(
        self,
        ontology: Ontology,
        client_updates: List[Dict[str, Any]],
        prototypes: Dict[str, np.ndarray],
        gold_hierarchy: Optional[Dict[int, int]] = None,
    ) -> List[Relationship]:
        # register any new concepts from clients
        for upd in client_updates:
            for cid in upd.get("concepts", []):
                if cid not in ontology.concepts:
                    domain = upd.get("domain", "unknown")
                    ontology.add_concept(Concept(concept_id=cid, name=cid, domain=domain))

        domain = "cifar100"
        fine_ids = [cid for cid in prototypes if "_class_" in cid]
        coarse_ids = [cid for cid in ontology.concepts if "_coarse_" in cid]

        # ---- build coarse prototypes by clustering fine prototypes (no gold) ----
        coarse_protos: Dict[str, np.ndarray] = {}
        if fine_ids and coarse_ids:
            fine_mat = np.stack([prototypes[f] for f in fine_ids]).astype(np.float32)
            rng = np.random.RandomState(0)
            # initialise coarse centres from random fine vectors
            chosen = rng.choice(len(fine_ids), size=min(len(coarse_ids), len(fine_ids)), replace=False)
            for i, cid in enumerate(coarse_ids):
                if i < len(chosen):
                    coarse_protos[cid] = fine_mat[chosen[i]].copy()
                else:
                    coarse_protos[cid] = fine_mat.mean(axis=0).astype(np.float32)
            # one Lloyd iteration: assign → recompute mean
            c_ids = list(coarse_protos.keys())
            C = np.stack([coarse_protos[c] for c in c_ids])
            Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
            Fn = fine_mat / (np.linalg.norm(fine_mat, axis=1, keepdims=True) + 1e-8)
            assign = (Fn @ Cn.T).argmax(axis=1)
            for j, cid in enumerate(c_ids):
                members = fine_mat[assign == j]
                if len(members) > 0:
                    coarse_protos[cid] = members.mean(axis=0).astype(np.float32)

        all_protos = dict(prototypes)
        all_protos.update(coarse_protos)

        candidates: List[Relationship] = []

        # ---- hierarchical fine → coarse is_a candidates ----
        for fid in fine_ids:
            fp = all_protos[fid]
            scores = []
            for cid in coarse_ids:
                cp = all_protos.get(cid)
                if cp is None:
                    continue
                ctx = contextual_similarity(fp, cp)
                scores.append((cid, ctx))
            scores.sort(key=lambda x: -x[1])
            for rank, (cid, ctx) in enumerate(scores[: self.top_k_parents]):
                if ctx < self.candidate_threshold:
                    continue
                ca = ontology.get_concept(fid)
                cb = ontology.get_concept(cid)
                term = terminological_similarity(
                    ca.name if ca else fid, cb.name if cb else cid
                )
                struct = 0.4  # hierarchical prior
                score = (
                    self.structural_w * struct
                    + self.terminological_w * term
                    + self.contextual_w * ctx
                )
                # slight boost for nearest parent
                if rank == 0:
                    score = min(1.0, score + 0.05)
                rid = f"cand_{fid}__{cid}"
                candidates.append(Relationship(
                    rel_id=rid,
                    source_id=fid,
                    target_id=cid,
                    rel_type="is_a",
                    confidence=float(score),
                    evidence={
                        "structural": float(struct),
                        "terminological": float(term),
                        "contextual": float(ctx),
                    },
                    status="candidate",
                ))

        return candidates


class FederatedAggregator:
    def __init__(self, device: Optional[str] = None):
        self.device = device

    def aggregate(self, client_updates: List[Dict[str, Any]]) -> np.ndarray:
        if not client_updates:
            return np.zeros(16, dtype=np.float32)
        vectors = np.stack([upd["mean_update"] for upd in client_updates]).astype(np.float32)
        weights = np.array(
            [max(1, upd.get("n_samples", 1)) for upd in client_updates], dtype=np.float64
        )
        weights /= weights.sum()
        dev = self.device or get_device()
        if dev == "cuda":
            try:
                import torch
                v = torch.from_numpy(vectors).to("cuda")
                w = torch.from_numpy(weights.astype(np.float32)).to("cuda")
                return (v * w[:, None]).sum(dim=0).cpu().numpy().astype(np.float32)
            except Exception:
                pass
        return (vectors * weights[:, None]).sum(axis=0).astype(np.float32)
