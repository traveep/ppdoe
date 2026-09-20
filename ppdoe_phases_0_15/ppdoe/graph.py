"""
Semantic graph G used by PPDOE.
Wraps NetworkX for topology operations while staying close to the paper's
Neo4j / RDF graph description.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any
import networkx as nx
import hashlib
import json
from .ontology import Ontology, Concept, Relationship


class SemanticGraph:
    """
    Directed multigraph that mirrors the ontology relationships.
    Supports the RL actions: add, modify, align, reorganize.
    """

    def __init__(self, ontology: Optional[Ontology] = None):
        self.G = nx.MultiDiGraph()
        self.ontology = ontology
        if ontology is not None:
            self.sync_from_ontology()

    def sync_from_ontology(self) -> None:
        """Rebuild graph nodes/edges from the current ontology state."""
        self.G.clear()
        if self.ontology is None:
            return
        for cid, concept in self.ontology.concepts.items():
            self.G.add_node(
                cid,
                name=concept.name,
                domain=concept.domain,
                attributes=concept.attributes,
                version=concept.version_created,
            )
        for rid, rel in self.ontology.relationships.items():
            if rel.status in ("confirmed", "candidate"):
                self.G.add_edge(
                    rel.source_id,
                    rel.target_id,
                    key=rid,
                    rel_type=rel.rel_type,
                    confidence=rel.confidence,
                    status=rel.status,
                    version=rel.version_created,
                )

    def add_node_from_concept(self, concept: Concept) -> None:
        self.G.add_node(
            concept.concept_id,
            name=concept.name,
            domain=concept.domain,
            attributes=concept.attributes,
            version=concept.version_created,
        )

    def add_edge_from_relationship(self, rel: Relationship) -> None:
        self.G.add_edge(
            rel.source_id,
            rel.target_id,
            key=rel.rel_id,
            rel_type=rel.rel_type,
            confidence=rel.confidence,
            status=rel.status,
            version=rel.version_created,
        )

    # ------------------------------------------------------------------
    # RL action primitives (paper: ApplyAction)
    # ------------------------------------------------------------------
    def apply_action(self, action: dict) -> "SemanticGraph":
        """
        Apply an RL action and return the new graph state.
        Supported actions:
          - {"type": "add_edge", "source": ..., "target": ..., "rel_type": ..., "confidence": ...}
          - {"type": "remove_edge", "rel_id": ...}
          - {"type": "confirm_edge", "rel_id": ..., "confidence": ...}
          - {"type": "align", "concept_a": ..., "concept_b": ...}  (merge / link)
          - {"type": "reorganize", "nodes": [...], "new_parent": ...}
        """
        new_graph = SemanticGraph(ontology=self.ontology)
        new_graph.G = self.G.copy()

        atype = action.get("type")
        if atype == "add_edge":
            rid = action.get("rel_id", f"rl_{action['source']}_{action['target']}")
            new_graph.G.add_edge(
                action["source"],
                action["target"],
                key=rid,
                rel_type=action.get("rel_type", "linked_to"),
                confidence=action.get("confidence", 0.5),
                status="candidate",
                version=self.ontology.version if self.ontology else 0,
            )
        elif atype == "remove_edge":
            rid = action["rel_id"]
            for u, v, k in list(new_graph.G.edges(keys=True)):
                if k == rid:
                    new_graph.G.remove_edge(u, v, key=k)
                    break
        elif atype == "confirm_edge":
            rid = action["rel_id"]
            for u, v, k, data in new_graph.G.edges(keys=True, data=True):
                if k == rid:
                    data["status"] = "confirmed"
                    data["confidence"] = action.get("confidence", data.get("confidence", 0.7))
                    break
        elif atype == "align":
            a, b = action["concept_a"], action["concept_b"]
            rid = f"align_{a}_{b}"
            new_graph.G.add_edge(a, b, key=rid, rel_type="aligned_with",
                                 confidence=action.get("confidence", 0.6), status="candidate")
        elif atype == "reorganize":
            parent = action.get("new_parent")
            for n in action.get("nodes", []):
                if parent and n != parent:
                    rid = f"reorg_{n}_{parent}"
                    new_graph.G.add_edge(n, parent, key=rid, rel_type="is_a",
                                         confidence=0.55, status="candidate")
        return new_graph

    # ------------------------------------------------------------------
    # Graph metrics used by the RL reward
    # ------------------------------------------------------------------
    def coherence(self) -> float:
        """Simple coherence proxy: fraction of confirmed edges among all edges."""
        edges = list(self.G.edges(keys=True, data=True))
        if not edges:
            return 0.0
        confirmed = sum(1 for _, _, _, d in edges if d.get("status") == "confirmed")
        return confirmed / len(edges)

    def scalability_proxy(self) -> float:
        """Proxy for scalability: inverse of average degree (lower density better for large graphs)."""
        n = self.G.number_of_nodes()
        if n < 2:
            return 1.0
        avg_deg = sum(dict(self.G.degree()).values()) / n
        # normalize to [0,1] roughly
        return max(0.0, 1.0 - (avg_deg / (n - 1)))

    def reward(self, coherence_w: float = 0.6, scalability_w: float = 0.4) -> float:
        """Paper reward: coherence × 0.6 + scalability × 0.4"""
        return coherence_w * self.coherence() + scalability_w * self.scalability_proxy()

    def state_hash(self) -> str:
        edges = sorted(
            [(u, v, k, d.get("status"), round(d.get("confidence", 0), 4))
             for u, v, k, d in self.G.edges(keys=True, data=True)]
        )
        raw = json.dumps(edges, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def number_of_nodes(self) -> int:
        return self.G.number_of_nodes()

    def number_of_edges(self) -> int:
        return self.G.number_of_edges()

    def confirmed_edges(self) -> List[Tuple]:
        return [(u, v, k) for u, v, k, d in self.G.edges(keys=True, data=True)
                if d.get("status") == "confirmed"]

    def candidate_edges(self) -> List[Tuple]:
        return [(u, v, k) for u, v, k, d in self.G.edges(keys=True, data=True)
                if d.get("status") == "candidate"]
