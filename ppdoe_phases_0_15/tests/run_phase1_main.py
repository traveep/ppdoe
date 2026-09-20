#!/usr/bin/env python3
"""
Phase 1: Main relationship-recovery benchmark (50 seeds).
Methods: Flat similarity, Static ontology, Full PPDOE, Scrambled ontology, Oracle.
"""

from __future__ import annotations
import sys
import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppdoe.data import load_cifar100, create_client_partitions, hash_array
from ppdoe.ppdoe import PPDOE
from ppdoe.metrics import compute_metrics, gold_edge_set, RelationshipMetrics
from ppdoe.ontology import Concept, Relationship
from ppdoe.aggregator import contextual_similarity
from ppdoe.baselines import fedavg_kmeans, flis_style, fedrcl_inspired

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def flat_similarity_baseline(
    features: np.ndarray,
    fine_labels: np.ndarray,
    client_indices: List[np.ndarray],
    gold: Dict[int, int],
    seed: int,
) -> RelationshipMetrics:
    """Non-ontological baseline: cluster prototypes by cosine similarity."""
    from ppdoe.ontology import Ontology
    onto = Ontology(name="flat")
    # build prototypes per fine class
    protos = {}
    for lab in np.unique(fine_labels):
        mask = fine_labels == lab
        protos[lab] = features[mask].mean(axis=0)
        onto.add_concept(Concept(
            concept_id=f"cifar100_class_{lab}",
            name=f"class_{lab}",
            domain="cifar100",
        ))
    for c in range(20):
        onto.add_concept(Concept(
            concept_id=f"cifar100_coarse_{c}",
            name=f"coarse_{c}",
            domain="cifar100",
        ))
    # link each fine to nearest coarse by mean prototype of that coarse's members
    coarse_protos = {}
    for fine, coarse in gold.items():
        if coarse not in coarse_protos:
            coarse_protos[coarse] = []
        if fine in protos:
            coarse_protos[coarse].append(protos[fine])
    for c, vecs in coarse_protos.items():
        coarse_protos[c] = np.mean(vecs, axis=0)

    for fine, proto in protos.items():
        best_c, best_sim = None, -1.0
        for c, cp in coarse_protos.items():
            sim = contextual_similarity(proto, cp)
            if sim > best_sim:
                best_sim = sim
                best_c = c
        if best_c is not None and best_sim > 0.1:
            rid = f"flat_{fine}_{best_c}"
            onto.add_relationship(Relationship(
                rel_id=rid,
                source_id=f"cifar100_class_{fine}",
                target_id=f"cifar100_coarse_{best_c}",
                rel_type="is_a",
                confidence=float(best_sim),
                status="confirmed",
            ))
    gold_edges = gold_edge_set(gold)
    return compute_metrics(onto, gold_edges)


def static_ontology_baseline(
    features, fine_labels, client_indices, gold, seed
) -> RelationshipMetrics:
    """Static: run PPDOE with dynamic updates disabled."""
    ppdoe = PPDOE(
        n_clients=10, n_rounds=5, seed=seed,
        enable_dynamic=False, enable_rl=False,
    )
    ppdoe.setup_cifar_clients(features, fine_labels, client_indices)
    ppdoe.run()
    gold_edges = gold_edge_set(gold)
    return compute_metrics(ppdoe.ontology, gold_edges)


def full_ppdoe_run(features, fine_labels, client_indices, gold, seed) -> RelationshipMetrics:
    ppdoe = PPDOE(n_clients=10, n_rounds=8, seed=seed, active_fraction=0.8)
    ppdoe.setup_cifar_clients(features, fine_labels, client_indices)
    ppdoe.run()
    gold_edges = gold_edge_set(gold)
    return compute_metrics(ppdoe.ontology, gold_edges)


def scrambled_ontology_run(features, fine_labels, client_indices, gold, seed) -> RelationshipMetrics:
    """Same as full but gold hierarchy is randomly permuted for evaluation only
    (the learner never sees gold). This tests that the method does not
    magically recover a scrambled target."""
    ppdoe = PPDOE(n_clients=10, n_rounds=8, seed=seed)
    ppdoe.setup_cifar_clients(features, fine_labels, client_indices)
    ppdoe.run()
    # scramble the gold for evaluation
    rng = np.random.RandomState(seed + 999)
    keys = list(gold.keys())
    vals = list(gold.values())
    rng.shuffle(vals)
    scrambled = dict(zip(keys, vals))
    gold_edges = gold_edge_set(scrambled)
    return compute_metrics(ppdoe.ontology, gold_edges)


def oracle_upper_bound(gold: Dict[int, int]) -> RelationshipMetrics:
    """Non-deployable oracle that inserts the true hierarchy."""
    from ppdoe.ontology import Ontology, Concept, Relationship
    onto = Ontology(name="oracle")
    for f in range(100):
        onto.add_concept(Concept(concept_id=f"cifar100_class_{f}", name=f"c{f}", domain="cifar100"))
    for c in range(20):
        onto.add_concept(Concept(concept_id=f"cifar100_coarse_{c}", name=f"coarse{c}", domain="cifar100"))
    for fine, coarse in gold.items():
        onto.add_relationship(Relationship(
            rel_id=f"oracle_{fine}_{coarse}",
            source_id=f"cifar100_class_{fine}",
            target_id=f"cifar100_coarse_{coarse}",
            rel_type="is_a",
            confidence=1.0,
            status="confirmed",
            is_gold=True,
        ))
    gold_edges = gold_edge_set(gold)
    return compute_metrics(onto, gold_edges)


def run_one_seed(seed: int, data: dict) -> List[dict]:
    client_indices = create_client_partitions(
        data["n_train"], n_clients=10, seed=seed, labels=data["train_fine"]
    )
    gold = data["fine_to_coarse"]
    feats = data["train_features"]
    labels = data["train_fine"]

    rows = []
    methods = {
        "flat_similarity": lambda: flat_similarity_baseline(feats, labels, client_indices, gold, seed),
        "static_ontology": lambda: static_ontology_baseline(feats, labels, client_indices, gold, seed),
        "full_ppdoe": lambda: full_ppdoe_run(feats, labels, client_indices, gold, seed),
        "scrambled_ontology": lambda: scrambled_ontology_run(feats, labels, client_indices, gold, seed),
        "fedavg_kmeans": lambda: fedavg_kmeans(feats, labels, client_indices, gold, seed),
        "flis": lambda: flis_style(feats, labels, client_indices, gold, seed),
        "fedrcl": lambda: fedrcl_inspired(feats, labels, client_indices, gold, seed),
        "oracle": lambda: oracle_upper_bound(gold),
    }
    for name, fn in methods.items():
        m = fn()
        rows.append({
            "seed": seed,
            "method": name,
            "relationship_precision": m.precision,
            "relationship_recall": m.recall,
            "relationship_f1": m.f1,
            "group_accuracy": m.group_accuracy,
            "fine_accuracy": m.fine_accuracy,
            "candidate_edges": m.candidate_edges,
            "confirmed_edges": m.confirmed_edges,
            "false_discovery_rate": m.false_discovery_rate,
            "rejection_rate": m.rejection_rate,
        })
        print(f"  seed={seed} method={name} F1={m.f1:.4f} P={m.precision:.4f} R={m.recall:.4f}")
    return rows


def summarize(rows: List[dict]) -> None:
    # per-seed CSV
    per_seed_path = RESULTS / "main_results_per_seed.csv"
    fieldnames = list(rows[0].keys())
    with open(per_seed_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {per_seed_path}")

    # summary
    methods = sorted(set(r["method"] for r in rows))
    summary_rows = []
    for method in methods:
        vals = [r for r in rows if r["method"] == method]
        def mean_std(key):
            xs = [v[key] for v in vals]
            return float(np.mean(xs)), float(np.std(xs))
        summary_rows.append({
            "method": method,
            "n_seeds": len(vals),
            "mean_f1": mean_std("relationship_f1")[0],
            "std_f1": mean_std("relationship_f1")[1],
            "mean_precision": mean_std("relationship_precision")[0],
            "mean_recall": mean_std("relationship_recall")[0],
            "mean_fdr": mean_std("false_discovery_rate")[0],
            "mean_confirmed": mean_std("confirmed_edges")[0],
        })
    summary_path = RESULTS / "main_results_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote {summary_path}")

    # pairwise vs full_ppdoe
    ppdoe_f1 = {r["seed"]: r["relationship_f1"] for r in rows if r["method"] == "full_ppdoe"}
    pairwise = []
    for method in methods:
        if method == "full_ppdoe":
            continue
        deltas = []
        for r in rows:
            if r["method"] == method and r["seed"] in ppdoe_f1:
                deltas.append(ppdoe_f1[r["seed"]] - r["relationship_f1"])
        if not deltas:
            continue
        deltas = np.array(deltas)
        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas))
        # paired t-test
        t_stat, p_val = stats.ttest_rel(
            [ppdoe_f1[s] for s in sorted(ppdoe_f1)],
            [r["relationship_f1"] for r in rows if r["method"] == method and r["seed"] in ppdoe_f1],
        )
        # effect size (Cohen's d for paired)
        d = mean_d / (std_d + 1e-12)
        ci_low = mean_d - 1.96 * std_d / np.sqrt(len(deltas))
        ci_high = mean_d + 1.96 * std_d / np.sqrt(len(deltas))
        pairwise.append({
            "baseline": method,
            "mean_delta_f1": mean_d,
            "std_delta": std_d,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "paired_t_pvalue": float(p_val),
            "cohens_d": float(d),
            "n_seeds": len(deltas),
        })
    pair_path = RESULTS / "main_pairwise_tests.csv"
    if pairwise:
        with open(pair_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pairwise[0].keys()))
            w.writeheader()
            w.writerows(pairwise)
        print(f"Wrote {pair_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50, help="Number of seeds (use 50 for full)")
    parser.add_argument("--start-seed", type=int, default=0)
    args = parser.parse_args()

    print("Loading CIFAR-100 ...")
    data = load_cifar100(root=str(Path(__file__).resolve().parents[1] / "data"), download=True)
    print(f"Train {data['n_train']}, Test {data['n_test']}")

    all_rows = []
    for s in range(args.start_seed, args.start_seed + args.seeds):
        print(f"\n=== Seed {s} ===")
        all_rows.extend(run_one_seed(s, data))

    summarize(all_rows)
    print("\nPhase 1 complete.")
