#!/usr/bin/env python3
"""
Phase 9: Formal-ish privacy utility curve over ε budgets.
Note: uses the package Gaussian mechanism; not a full RDP accountant.
"""
from __future__ import annotations
import sys, csv
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppdoe.data import load_cifar100, create_client_partitions
from ppdoe.ppdoe import PPDOE
from ppdoe.metrics import compute_metrics, gold_edge_set

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
EPSILONS = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]


def main(n_seeds=5):
    print(f"Phase 9 – Privacy utility ({n_seeds} seeds)")
    print("Privacy unit: client-level Gaussian mechanism (approximate; not formal RDP).")
    data = load_cifar100(prefer_synthetic=False)
    gold = gold_edge_set(data["fine_to_coarse"])
    rows = []
    for seed in range(n_seeds):
        idxs = create_client_partitions(
            data["n_train"], 10, seed=seed, labels=data["train_fine"]
        )
        for eps in EPSILONS:
            p = PPDOE(n_clients=10, n_rounds=5, seed=seed, epsilon=eps,
                      confirmation_threshold=0.20, theta=0.5, enable_privacy=True)
            p.setup_cifar_clients(data["train_features"], data["train_fine"], idxs)
            p.run()
            m = compute_metrics(p.ontology, gold)
            spent, delta = p.privacy.current_budget() if p.privacy else (0.0, 0.0)
            rows.append({
                "seed": seed,
                "epsilon_target": eps,
                "epsilon_spent_approx": spent,
                "delta": delta,
                "relationship_f1": m.f1,
                "precision": m.precision,
                "recall": m.recall,
                "fdr": m.false_discovery_rate,
                "group_accuracy": m.group_accuracy,
                "confirmed_edges": m.confirmed_edges,
            })
            print(f"  seed={seed} ε={eps} F1={m.f1:.4f}")
    out = RESULTS / "privacy_utility.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    main(ap.parse_args().seeds)
