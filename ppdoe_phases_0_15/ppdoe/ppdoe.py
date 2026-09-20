"""
Main PPDOE orchestrator.
Implements the exact high-level workflow supplied by the user.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import json
import hashlib
from datetime import datetime

from .ontology import Ontology, Concept, Relationship
from .graph import SemanticGraph
from .client import FederatedClient
from .aggregator import OntologyAggregator, FederatedAggregator
from .novelty import NoveltyDetector
from .rl_optimizer import RLGraphOptimizer
from .privacy import DifferentialPrivacy
from .metrics import compute_metrics, gold_edge_set, RelationshipMetrics


class PPDOE:
    """
    Privacy-Preserving Dynamic Ontology Evolution.

    Faithful executable realization of the reconstructed algorithm:
      local mapping → candidate relationships → ontology aggregation
      → novelty detection → RL graph evolution → versioned commit
    """

    def __init__(
        self,
        n_clients: int = 10,
        active_fraction: float = 0.8,
        theta: float = 0.5,
        epsilon: float = 0.1,
        delta: float = 1e-5,
        n_rounds: int = 10,
        seed: int = 0,
        enable_ontology: bool = True,
        enable_dynamic: bool = True,
        enable_semantic: bool = True,
        enable_structural: bool = True,
        enable_contextual: bool = True,
        enable_rl: bool = True,
        enable_privacy: bool = True,
        confirmation_threshold: float = 0.55,
        device: Optional[str] = None,
    ):
        from .device import set_device, get_device
        self.device = set_device(device) if device else get_device()

        self.n_clients = n_clients
        self.active_fraction = active_fraction
        self.theta = theta
        self.epsilon = epsilon
        self.delta = delta
        self.n_rounds = n_rounds
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # component switches for ablation (Phase 2)
        self.enable_ontology = enable_ontology
        self.enable_dynamic = enable_dynamic
        self.enable_semantic = enable_semantic
        self.enable_structural = enable_structural
        self.enable_contextual = enable_contextual
        self.enable_rl = enable_rl
        self.enable_privacy = enable_privacy
        self.confirmation_threshold = confirmation_threshold

        # core state
        self.ontology = Ontology(name="ppdoe")
        self.graph = SemanticGraph(ontology=self.ontology)
        self.global_weights = np.zeros(16, dtype=np.float32)  # matches feature dim
        self.clients: List[FederatedClient] = []
        self.logs: List[dict] = []
        self.round = 0

        # sub-modules (device propagated to numerical parts)
        self.privacy = DifferentialPrivacy(
            epsilon=epsilon, delta=delta, seed=seed, device=self.device
        ) if enable_privacy else None
        self.novelty = NoveltyDetector(threshold=theta)
        self.onto_agg = OntologyAggregator(
            structural_w=0.3 if enable_structural else 0.0,
            terminological_w=0.3 if enable_semantic else 0.0,
            contextual_w=0.4 if enable_contextual else 0.0,
            device=self.device,
        )
        self.fed_agg = FederatedAggregator(device=self.device)
        self.rl = RLGraphOptimizer(seed=seed) if enable_rl else None

        # frozen mode (Phase 3)
        self.frozen = False
        self.frozen_after_round = 1

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup_cifar_clients(
        self,
        features: np.ndarray,
        fine_labels: np.ndarray,
        client_indices: List[np.ndarray],
        domain: str = "cifar100",
    ) -> None:
        self.clients = []
        for i, idxs in enumerate(client_indices):
            client = FederatedClient(
                client_id=f"client_{i}",
                domain=domain,
                data=features[idxs],
                labels=fine_labels[idxs],
                feature_dim=features.shape[1],
                seed=self.seed + i,
            )
            self.clients.append(client)

        # seed ontology with coarse concepts (parents) – these are not gold edges
        for c in range(20):
            cid = f"{domain}_coarse_{c}"
            self.ontology.add_concept(Concept(
                concept_id=cid,
                name=f"coarse_{c}",
                domain=domain,
                attributes={"level": "coarse", "id": c},
            ))
        self.ontology.commit_version(note="initial coarse concepts")
        self.graph.sync_from_ontology()

    def setup_multidomain_clients(self, domain_data: Dict[str, List[dict]]) -> None:
        self.clients = []
        for i, (domain, records) in enumerate(domain_data.items()):
            client = FederatedClient(
                client_id=f"{domain}_node",
                domain=domain,
                data=records,
                labels=None,
                feature_dim=16,
                seed=self.seed + i,
            )
            self.clients.append(client)
        self.ontology.commit_version(note="empty multi-domain start")
        self.graph.sync_from_ontology()

    # ------------------------------------------------------------------
    # One federated round (the core of the supplied algorithm)
    # ------------------------------------------------------------------
    def run_round(self, participating: Optional[List[int]] = None) -> dict:
        self.round += 1
        t = self.round
        if participating is None:
            n_active = max(1, int(self.active_fraction * len(self.clients)))
            participating = self.rng.choice(
                len(self.clients), size=n_active, replace=False
            ).tolist()

        round_log = {
            "round": t,
            "participating": participating,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # 2.1 Local client processing
        client_updates = []
        all_prototypes: Dict[str, np.ndarray] = {}
        for idx in participating:
            client = self.clients[idx]
            # a-c. map local data to concepts
            concepts = client.map_to_concepts(ontology_version=self.ontology.version)
            for c in concepts:
                if self.enable_ontology:
                    self.ontology.add_concept(c)
            # d-e. local update + privacy
            upd = client.local_update(
                global_weights=self.global_weights,
                privacy=self.privacy if self.enable_privacy else None,
            )
            client_updates.append(upd)
            all_prototypes.update(upd["prototypes"])

        # 2.2 Ontology-based aggregation → candidate relationships
        # Score novelty AGAINST CURRENT ontology BEFORE inserting candidates
        # (otherwise every candidate looks already present → novelty = 0).
        candidates: List[Relationship] = []
        novel_candidates: List[Relationship] = []
        if self.enable_ontology:
            candidates = self.onto_agg.build_candidates(
                ontology=self.ontology,
                client_updates=client_updates,
                prototypes=all_prototypes,
                gold_hierarchy=None,  # deliberately never passed
            )
            scored = self.novelty.score(candidates, self.ontology)
            for rel, nov in scored:
                if self.novelty.should_evolve(nov):
                    novel_candidates.append(rel)
            # now insert all candidates into the ontology store
            for rel in candidates:
                if rel.rel_id not in self.ontology.relationships:
                    self.ontology.add_relationship(rel)

        # 2.3 Federated model aggregation
        self.global_weights = self.fed_agg.aggregate(client_updates)

        # 2.4 Structural updates (confirmation of hierarchical candidates)
        # IMPORTANT: "dynamic" = structural evolution continues.
        # When frozen, structural updates stop after frozen_after_round.
        # θ remains a FIXED hyper-parameter; it is never adapted online.
        structural_change = False
        # Static ontology: allow structure only on the first round, then freeze.
        # Dynamic: continue updating every round (unless explicitly frozen).
        if not self.enable_dynamic:
            allow_structural = self.enable_ontology and (t <= 1)
        else:
            allow_structural = (
                self.enable_ontology
                and not (self.frozen and t > self.frozen_after_round)
            )

        if allow_structural:
            # Confirm at most one hierarchical is_a parent per fine class
            # (the highest-confidence candidate above threshold).
            best_parent: Dict[str, Any] = {}
            pool = novel_candidates if novel_candidates else candidates
            for rel in pool:
                if rel.rel_type != "is_a":
                    continue
                if "_class_" not in rel.source_id or "_coarse_" not in rel.target_id:
                    continue
                if rel.confidence < self.confirmation_threshold:
                    continue
                src = rel.source_id
                if src not in best_parent or rel.confidence > best_parent[src].confidence:
                    best_parent[src] = rel
            for src, rel in best_parent.items():
                # deprecate any previously confirmed parent for this fine class
                for existing in self.ontology.list_relationships(status="confirmed"):
                    if (existing.source_id == src and existing.rel_type == "is_a"
                            and existing.rel_id != rel.rel_id):
                        self.ontology.deprecate_relationship(existing.rel_id)
                self.ontology.confirm_relationship(rel.rel_id, rel.confidence)
                structural_change = True

        # 2.5 RL-based graph optimization (additional edits when enabled)
        reward = 0.0
        action = {"type": "noop"}
        if self.enable_rl and allow_structural:
            self.graph.sync_from_ontology()
            action = self.rl.select_action(self.graph, novel_candidates or candidates)
            self.graph, reward = self.rl.apply_and_reward(self.graph, action)
            for u, v, k, d in self.graph.G.edges(keys=True, data=True):
                if d.get("status") == "confirmed" and k in self.ontology.relationships:
                    self.ontology.confirm_relationship(k, d.get("confidence", 0.6))
                    structural_change = True
        # else: frozen → no structural change after freeze point

        # 2.6 Commit new version (only when dynamic structural evolution is active)
        if allow_structural:
            self.ontology.commit_version(
                note=f"round_{t}_action_{action.get('type')}_struct={structural_change}"
            )
        self.graph.sync_from_ontology()

        # 2.7 Logging
        round_log.update({
            "n_candidates": len(candidates),
            "n_novel": len(novel_candidates),
            "n_confirmed": len(self.ontology.list_relationships(status="confirmed")),
            "ontology_version": self.ontology.version,
            "concept_count": len(self.ontology.concepts),
            "relationship_count": len(self.ontology.relationships),
            "reward": reward,
            "action": action.get("type"),
            "structural_change": structural_change,
            "frozen": bool(self.frozen and t > self.frozen_after_round),
            "theta_fixed": self.theta,          # θ is never adapted
            "privacy_epsilon_spent": self.privacy.spent_epsilon if self.privacy else 0.0,
            "graph_hash": self.graph.state_hash()[:16],
            "ontology_hash": self.ontology.current_state_hash()[:16],
        })
        self.logs.append(round_log)
        return round_log

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------
    def run(
        self,
        arrival_order: Optional[List[int]] = None,
        evaluate_every: int = 1,
        gold_hierarchy: Optional[Dict[int, int]] = None,
    ) -> List[dict]:
        """
        Execute T rounds. If arrival_order is given, clients are introduced
        sequentially (Phase 3 temporal experiment).
        """
        results = []
        if arrival_order is None:
            for t in range(self.n_rounds):
                log = self.run_round()
                if gold_hierarchy is not None and (t + 1) % evaluate_every == 0:
                    gold = gold_edge_set(gold_hierarchy)
                    metrics = compute_metrics(self.ontology, gold)
                    log["metrics"] = metrics.__dict__
                results.append(log)
        else:
            # sequential client arrival
            present = []
            for client_idx in arrival_order:
                present.append(client_idx)
                log = self.run_round(participating=present)
                if gold_hierarchy is not None:
                    gold = gold_edge_set(gold_hierarchy)
                    metrics = compute_metrics(self.ontology, gold)
                    log["metrics"] = metrics.__dict__
                results.append(log)
        return results

    def freeze(self, after_round: int = 1) -> None:
        """Phase 3: freeze structural ontology after a given round."""
        self.frozen = True
        self.frozen_after_round = after_round

    # ------------------------------------------------------------------
    # Audit helpers (Phase 0)
    # ------------------------------------------------------------------
    def input_hashes(self) -> dict:
        """Hashes of inputs for the equal-input audit."""
        client_ids = [c.client_id for c in self.clients]
        return {
            "client_ids": hashlib.sha256(json.dumps(client_ids).encode()).hexdigest(),
            "n_clients": len(self.clients),
            "seed": self.seed,
            "n_rounds": self.n_rounds,
            "theta": self.theta,
            "epsilon": self.epsilon,
            "confirmation_threshold": self.confirmation_threshold,
            "ontology_hash": self.ontology.current_state_hash(),
            "graph_hash": self.graph.state_hash(),
        }

    def final_state(self) -> dict:
        return {
            "ontology": self.ontology.to_dict(),
            "logs": self.logs,
            "global_weights_norm": float(np.linalg.norm(self.global_weights)),
            "privacy_budget": self.privacy.current_budget() if self.privacy else (0.0, 0.0),
        }
