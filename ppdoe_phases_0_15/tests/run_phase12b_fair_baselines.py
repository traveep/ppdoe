#!/usr/bin/env python3
"""
Phase 12b: Fair head-to-head — PPDOE vs clustering WITH and WITHOUT
Hungarian alignment.

Addresses the evaluation asymmetry:
  - Clustering baselines in Phase 1 get Hungarian matching at eval time.
  - PPDOE is scored on fixed coarse_0…19 IDs.

This script reports both scoring modes for FedAvg+k-means and FLIS,
plus PPDOE (never aligned), on identical data/partitions.
"""
from __future__ import annotations
import sys
import csv
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppdoe.data import load_cifar100, create_client_partitions
from ppdoe.ppdoe import PPDOE
from ppdoe.metrics import compute_metrics, gold_edge_set
from ppdoe.baselines import (
    _fine_prototypes, _kmeans, _hungarian_align, _to_metrics,
)
from ppdoe.ontology import Ontology, Concept, Relationship

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def clustering_metrics(features, labels, idxs, gold, seed, style: str, aligned: bool):
    protos = {}
    for ix in idxs:
        for lab, p in _fine_prototypes(features, labels, [ix]).items():
            protos.setdefault(lab, []).append(p)
    protos = {lab: np.mean(vs, axis=0).astype(np.float32) for lab, vs in protos.items()}
    labs = sorted(protos.keys())
    X = np.stack([protos[l] for l in labs])
    _, cids = _kmeans(X, k=20, seed=seed if style == "fedavg_kmeans" else seed + 1)
    pred = {labs[i]: int(cids[i]) for i in range(len(labs))}
    if aligned:
        pred = _hungarian_align(pred, gold)
    return _to_metrics(pred, gold, seed)


def ppdoe_metrics(features, labels, idxs, gold, seed):
    p = PPDOE(
        n_clients=10, n_rounds=6, seed=seed,
        confirmation_threshold=0.20, theta=0.5, enable_rl=True,
    )
    p.setup_cifar_clients(features, labels, idxs)
    p.run()
    return compute_metrics(p.ontology, gold_edge_set(gold))


def main(n_seeds: int = 5):
    print(f"Phase 12b – Fair baselines (aligned vs raw) ({n_seeds} seeds)")
    data = load_cifar100(prefer_synthetic=True)
    gold = data["fine_to_coarse"]
    rows = []

    methods = [
        ("full_ppdoe", None),
        ("fedavg_kmeans_raw", False),
        ("fedavg_kmeans_aligned", True),
        ("flis_raw", False),
        ("flis_aligned", True),
    ]

    for seed in range(n_seeds):
        idxs = create_client_partitions(
            data["n_train"], 10, seed=seed, labels=data["train_fine"]
        )
        print(f"  seed {seed}")
        for name, aligned in methods:
            if name == "full_ppdoe":
                m = ppdoe_metrics(
                    data["train_features"], data["train_fine"], idxs, gold, seed
                )
            else:
                style = "fedavg_kmeans" if "fedavg" in name else "flis"
                m = clustering_metrics(
                    data["train_features"], data["train_fine"], idxs, gold, seed,
                    style=style, aligned=aligned,
                )
            rows.append({
                "seed": seed,
                "method": name,
                "hungarian_aligned": aligned if aligned is not None else False,
                "relationship_f1": m.f1,
                "precision": m.precision,
                "recall": m.recall,
                "confirmed_edges": m.confirmed_edges,
                "false_discovery_rate": m.false_discovery_rate,
            })
            print(f"    {name}: F1={m.f1:.4f} P={m.precision:.4f} R={m.recall:.4f}")

    # per-seed CSV
    out = RESULTS / "fair_baselines_per_seed.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")

    # summary + paired deltas vs PPDOE
    summary = []
    ppdoe_f1 = {r["seed"]: r["relationship_f1"] for r in rows if r["method"] == "full_ppdoe"}
    for name, _ in methods:
        vals = [r for r in rows if r["method"] == name]
        f1s = np.array([v["relationship_f1"] for v in vals])
        deltas = np.array([
            ppdoe_f1[v["seed"]] - v["relationship_f1"]
            for v in vals if v["seed"] in ppdoe_f1
        ])
        mean_d = float(np.mean(deltas)) if len(deltas) else 0.0
        std_d = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
        se = std_d / np.sqrt(len(deltas)) if len(deltas) else 0.0
        p_val = None
        if len(deltas) > 1 and name != "full_ppdoe":
            p_val = float(stats.ttest_1samp(deltas, 0.0).pvalue)
        summary.append({
            "method": name,
            "n_seeds": len(vals),
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)),
            "mean_precision": float(np.mean([v["precision"] for v in vals])),
            "mean_recall": float(np.mean([v["recall"] for v in vals])),
            "mean_delta_f1_vs_ppdoe": mean_d if name != "full_ppdoe" else 0.0,
            "ci95_low": mean_d - 1.96 * se if name != "full_ppdoe" else 0.0,
            "ci95_high": mean_d + 1.96 * se if name != "full_ppdoe" else 0.0,
            "p_value_vs_ppdoe": p_val,
        })

    out2 = RESULTS / "fair_baselines_summary.csv"
    with open(out2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"Wrote {out2}")

    report = {
        "note": (
            "PPDOE is scored without Hungarian alignment (fixed ontology IDs). "
            "Clustering is reported both RAW (k-means labels as-is) and ALIGNED "
            "(Hungarian match to gold coarse IDs). Fair ontology comparison "
            "should prefer RAW clustering vs PPDOE; ALIGNED is an upper-bound "
            "for partition recovery, not a deployable ontology export."
        ),
        "summary": summary,
    }
    out3 = RESULTS / "fair_baselines_report.json"
    with open(out3, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out3}")
    print(json.dumps({s["method"]: s["mean_f1"] for s in summary}, indent=2))
    print("Phase 12b complete.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    main(ap.parse_args().seeds)
