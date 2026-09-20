#!/usr/bin/env python3
"""
Phase 5: Missing-edge recovery.
Hide a fraction of gold edges; evaluate recovery only on held-out edges.
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
RATES = [0.10, 0.20, 0.30, 0.40, 0.50]


def rank_metrics(pred_edges, held_out, ranked_confs):
    """ranked_confs: list of (edge_tuple, confidence) sorted desc."""
    if not held_out:
        return {k: 0.0 for k in ("precision", "recall", "f1", "mrr", "hits@1", "hits@5", "hits@10", "fdr")}
    hits = pred_edges & held_out
    tp, fp = len(hits), len(pred_edges - held_out)
    fn = len(held_out - pred_edges)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fdr = fp / (tp + fp) if (tp + fp) else 0.0
    # ranking
    rr, h1, h5, h10 = [], 0, 0, 0
    for rank, (e, _) in enumerate(ranked_confs, 1):
        if e in held_out:
            rr.append(1.0 / rank)
            if rank <= 1: h1 += 1
            if rank <= 5: h5 += 1
            if rank <= 10: h10 += 1
    mrr = float(np.mean(rr)) if rr else 0.0
    n = max(1, len(held_out))
    return {
        "precision": prec, "recall": rec, "f1": f1, "fdr": fdr,
        "mrr": mrr, "hits@1": h1 / n, "hits@5": h5 / n, "hits@10": h10 / n,
    }


def main(n_seeds=5):
    print(f"Phase 5 – Missing-edge recovery ({n_seeds} seeds)")
    data = load_cifar100(prefer_synthetic=False)
    all_gold = gold_edge_set(data["fine_to_coarse"])
    gold_list = sorted(all_gold)
    rows = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        idxs = create_client_partitions(
            data["n_train"], 10, seed=seed, labels=data["train_fine"]
        )
        for rate in RATES:
            n_hide = max(1, int(rate * len(gold_list)))
            hide_idx = rng.choice(len(gold_list), size=n_hide, replace=False)
            held_out = {gold_list[i] for i in hide_idx}
            # run PPDOE (never sees gold)
            p = PPDOE(n_clients=10, n_rounds=6, seed=seed,
                      confirmation_threshold=0.20, theta=0.5)
            p.setup_cifar_clients(data["train_features"], data["train_fine"], idxs)
            p.run()
            conf = p.ontology.list_relationships(status="confirmed")
            pred = _edge_set(conf, "confirmed")
            ranked = sorted(
                [((min(r.source_id, r.target_id), max(r.source_id, r.target_id), r.rel_type), r.confidence)
                 for r in conf],
                key=lambda x: -x[1],
            )
            m = rank_metrics(pred, held_out, ranked)
            row = {"seed": seed, "missing_rate": rate, "n_held_out": len(held_out), **m}
            rows.append(row)
            print(f"  seed={seed} rate={rate:.0%} F1={m['f1']:.4f} MRR={m['mrr']:.4f}")
    out = RESULTS / "missing_edge_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    main(ap.parse_args().seeds)
