"""
RL-based graph optimization (paper: reward = coherence×0.6 + scalability×0.4).
Because the exact policy is not specified, we implement a simple
ε-greedy policy over a small discrete action set derived from candidates.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from .graph import SemanticGraph
from .ontology import Relationship


class RLGraphOptimizer:
    """
    Minimal RL controller that selects graph-edit actions.
    Policy is intentionally simple so that the rest of the PPDOE loop
    remains the focus; a full deep RL agent can later replace this class.
    """

    def __init__(
        self,
        coherence_weight: float = 0.6,
        scalability_weight: float = 0.4,
        exploration: float = 0.2,
        seed: Optional[int] = None,
    ):
        self.coherence_w = coherence_weight
        self.scalability_w = scalability_weight
        self.exploration = exploration
        self.rng = np.random.RandomState(seed)
        self.q_table: Dict[str, float] = {}   # crude state-action values
        self.last_reward = 0.0
        self.last_state_hash = ""

    def _state_key(self, graph: SemanticGraph) -> str:
        return graph.state_hash()[:16]

    def propose_actions(self, graph: SemanticGraph, candidate_rels: List[Relationship]) -> List[dict]:
        """Generate a small set of legal actions from current candidates."""
        actions = []
        for rel in candidate_rels[:20]:  # limit action space
            actions.append({
                "type": "confirm_edge",
                "rel_id": rel.rel_id,
                "confidence": rel.confidence,
                "source": rel.source_id,
                "target": rel.target_id,
            })
            actions.append({
                "type": "add_edge",
                "rel_id": rel.rel_id,
                "source": rel.source_id,
                "target": rel.target_id,
                "rel_type": rel.rel_type,
                "confidence": rel.confidence,
            })
        # always include a no-op
        actions.append({"type": "noop"})
        return actions

    def select_action(self, graph: SemanticGraph, candidate_rels: List[Relationship]) -> dict:
        actions = self.propose_actions(graph, candidate_rels)
        if not actions:
            return {"type": "noop"}

        state = self._state_key(graph)
        if self.rng.rand() < self.exploration:
            return actions[self.rng.randint(len(actions))]

        # greedy on crude Q
        best_a = actions[0]
        best_q = -np.inf
        for a in actions:
            key = f"{state}|{a['type']}|{a.get('rel_id', '')}"
            q = self.q_table.get(key, 0.0)
            if q > best_q:
                best_q = q
                best_a = a
        return best_a

    def apply_and_reward(self, graph: SemanticGraph, action: dict) -> Tuple[SemanticGraph, float]:
        if action.get("type") == "noop":
            new_graph = graph
        else:
            new_graph = graph.apply_action(action)
        reward = new_graph.reward(self.coherence_w, self.scalability_w)
        # simple Q update
        state = self._state_key(graph)
        key = f"{state}|{action.get('type')}|{action.get('rel_id', '')}"
        old_q = self.q_table.get(key, 0.0)
        self.q_table[key] = old_q + 0.1 * (reward - old_q)
        self.last_reward = reward
        self.last_state_hash = new_graph.state_hash()
        return new_graph, reward
