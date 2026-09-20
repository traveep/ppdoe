#!/usr/bin/env python3
"""
Phase 8: Confidence bins and abstention thresholds.
"""
from __future__ import annotations
import sys, csv
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppdoe.data import load_cifar100, create_client_partitions
from ppdoe.ppdoe import PPDOE
from ppdoe.metrics import gold_edge_set, _edge_set

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
BINS = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]


def main(n_seeds=5):
    print(f"Phase 8 – Confidence / abstention ({n_seeds} seeds)")
    data = load_cifar100(prefer_synthetic=False)
    gold = gold_edge_set(data["fine_to_coarse"])
    bin_rows, thr_rows = [], []
    for seed in range(n_seeds):
        idxs = create_client_partitions(
            data["n_train"], 10, seed=seed, labels=data["train_fine"]
        )
        p = PPDOE(n_clients=10, n_rounds=6, seed=seed,
                  confirmation_threshold=0.15, theta=0.3)  # lower so we get confidences
        p.setup_cifar_clients(data["train_features"], data["train_fine"], idxs)
        p.run()
        # use all candidates + confirmed with confidence scores
        rels = list(p.ontology.relationships.values())
        for lo, hi in BINS:
            subset = [r for r in rels if lo <= r.confidence < hi]
            pred = _edge_set(subset, status_filter=None)
            # treat as predicted positives
            tp = len(pred & gold)
            fp = len(pred - gold)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            fdr = fp / (tp + fp) if (tp + fp) else 0.0
            bin_rows.append({
                "seed": seed, "bin_low": lo, "bin_high": hi,
                "n_predictions": len(subset),
                "empirical_precision": prec,
                "empirical_fdr": fdr,
                "calibration_error": abs(prec - (lo + hi) / 2),
            })
        for tau in THRESHOLDS:
            subset = [r for r in rels if r.confidence >= tau]
            pred = _edge_set(subset, status_filter=None)
            tp = len(pred & gold)
            fp = len(pred - gold)
            fn = len(gold - pred)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            fdr = fp / (tp + fp) if (tp + fp) else 0.0
            thr_rows.append({
                "seed": seed, "threshold": tau,
                "n_predictions": len(subset),
                "precision": prec, "recall": rec, "fdr": fdr,
                "abstention_rate": 1.0 - len(subset) / max(1, len(rels)),
            })
        print(f"  seed {seed} done")
    for name, rows in [("confidence_bins.csv", bin_rows), ("confidence_thresholds.csv", thr_rows)]:
        out = RESULTS / name
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"Wrote {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    main(ap.parse_args().seeds)
