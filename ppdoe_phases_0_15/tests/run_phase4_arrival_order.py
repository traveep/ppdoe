#!/usr/bin/env python3
"""
Phase 4: Arrival-order sensitivity.
Dynamic vs frozen under multiple client arrival orders.
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
N_CLIENTS = 10


def auc(vals):
    if len(vals) < 2:
        return float(vals[0]) if vals else 0.0
    return sum((vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1))


def make_orders(seed, labels, idxs):
    rng = np.random.RandomState(seed)
    base = list(range(N_CLIENTS))
    orders = {
        "random": rng.permutation(N_CLIENTS).tolist(),
        "reverse": list(reversed(base)),
        "sequential": base,
    }
    # sorted by label entropy (diversity)
    diversities = []
    for i, ix in enumerate(idxs):
        labs = labels[ix]
        _, cnts = np.unique(labs, return_counts=True)
        p = cnts / cnts.sum()
        diversities.append((i, -np.sum(p * np.log(p + 1e-12))))
    diversities.sort(key=lambda x: x[1])
    orders["sorted_by_label_div"] = [i for i, _ in diversities]
    orders["adversarial_non_iid"] = [i for i, _ in reversed(diversities)]
    return orders


def run_traj(seed, data, order, frozen):
    idxs = create_client_partitions(
        data["n_train"], N_CLIENTS, seed=seed, labels=data["train_fine"]
    )
    p = PPDOE(n_clients=N_CLIENTS, n_rounds=len(order), seed=seed,
              theta=0.55, confirmation_threshold=0.20, active_fraction=1.0)
    p.setup_cifar_clients(data["train_features"], data["train_fine"], idxs)
    if frozen:
        p.freeze(after_round=1)
    gold = gold_edge_set(data["fine_to_coarse"])
    present, f1s, gas, edits = [], [], [], 0
    for c in order:
        present.append(c)
        log = p.run_round(participating=list(present))
        m = compute_metrics(p.ontology, gold)
        f1s.append(m.f1)
        gas.append(m.group_accuracy)
        if log.get("structural_change"):
            edits += 1
    return {
        "seed": seed,
        "arrival_order": "",  # filled by caller
        "condition": "frozen" if frozen else "dynamic",
        "relationship_f1_auc": auc(f1s),
        "group_accuracy_auc": auc(gas),
        "final_f1": f1s[-1],
        "final_group_accuracy": gas[-1],
        "ontology_edit_count": edits,
    }


def main(n_seeds=5):
    print(f"Phase 4 – Arrival-order sensitivity ({n_seeds} seeds)")
    data = load_cifar100(prefer_synthetic=False)
    rows = []
    for seed in range(n_seeds):
        idxs = create_client_partitions(
            data["n_train"], N_CLIENTS, seed=seed, labels=data["train_fine"]
        )
        orders = make_orders(seed, data["train_fine"], idxs)
        print(f"  seed {seed}")
        for oname, order in orders.items():
            for frozen in (False, True):
                r = run_traj(seed, data, order, frozen)
                r["arrival_order"] = oname
                rows.append(r)
                print(f"    {oname}/{r['condition']}: F1-AUC={r['relationship_f1_auc']:.4f}")
    out = RESULTS / "arrival_order_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    main(ap.parse_args().seeds)
