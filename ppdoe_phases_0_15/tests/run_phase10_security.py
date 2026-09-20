#!/usr/bin/env python3
"""
Phase 10: Privacy/security probes – membership inference proxy + poisoning.
Simplified but runnable probes (not full attack papers).
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


def membership_proxy(p, member_feats, nonmember_feats):
    """
    Simple threshold attack on prototype norms / distances.
    Returns attack accuracy (higher = more leakage).
    """
    # use global weight norm as a crude signal
    g = p.global_weights
    def score(X):
        return np.array([np.linalg.norm(x - g) for x in X])
    s_m = score(member_feats[:200])
    s_n = score(nonmember_feats[:200])
    # lower distance → more likely member under this crude model
    thresh = np.median(np.concatenate([s_m, s_n]))
    pred_m = (s_m <= thresh).mean()
    pred_n = (s_n > thresh).mean()
    acc = 0.5 * (pred_m + pred_n)
    return float(acc)


def main(n_seeds=5):
    print(f"Phase 10 – Security probes ({n_seeds} seeds)")
    data = load_cifar100(prefer_synthetic=False)
    gold = gold_edge_set(data["fine_to_coarse"])
    rows = []
    for seed in range(n_seeds):
        idxs = create_client_partitions(
            data["n_train"], 10, seed=seed, labels=data["train_fine"]
        )
        # clean
        for privacy, eps in [(False, 0.0), (True, 0.1), (True, 1.0), (True, 8.0)]:
            p = PPDOE(n_clients=10, n_rounds=4, seed=seed, epsilon=max(eps, 0.1),
                      enable_privacy=privacy, confirmation_threshold=0.20, theta=0.5)
            p.setup_cifar_clients(data["train_features"], data["train_fine"], idxs)
            p.run()
            m = compute_metrics(p.ontology, gold)
            mem_acc = membership_proxy(
                p, data["train_features"][idxs[0]], data["test_features"]
            )
            rows.append({
                "seed": seed, "attack": "membership_proxy", "setting": f"privacy={privacy}_eps={eps}",
                "relationship_f1": m.f1, "attack_accuracy": mem_acc,
                "attack_advantage": mem_acc - 0.5,
            })
            print(f"  seed={seed} priv={privacy} eps={eps} F1={m.f1:.3f} attack_acc={mem_acc:.3f}")

        # label-flip poisoning on client 0
        poison_labels = data["train_fine"].copy()
        poison_labels[idxs[0]] = (poison_labels[idxs[0]] + 50) % 100
        p = PPDOE(n_clients=10, n_rounds=4, seed=seed, confirmation_threshold=0.20, theta=0.5)
        p.setup_cifar_clients(data["train_features"], poison_labels, idxs)
        p.run()
        m = compute_metrics(p.ontology, gold)
        rows.append({
            "seed": seed, "attack": "label_flip_client0", "setting": "poisoned",
            "relationship_f1": m.f1, "attack_accuracy": None, "attack_advantage": None,
        })
        print(f"  seed={seed} label_flip F1={m.f1:.3f}")

    out = RESULTS / "security_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    main(ap.parse_args().seeds)
