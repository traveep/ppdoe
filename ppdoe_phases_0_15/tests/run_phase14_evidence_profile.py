#!/usr/bin/env python3
"""
Phase 14: Evidence profiles for recovered relationships.

For every candidate / confirmed edge, records the structural,
terminological, and contextual scores that drove confirmation.
Also bins edges by evidence composition and reports precision
within each bin (calibration of the multi-signal score).

Aligned with ontology discovery: the *why* of each edge matters
as much as whether it matches gold.
"""
from __future__ import annotations
import sys
import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppdoe.data import load_cifar100, create_client_partitions
from ppdoe.ppdoe import PPDOE
from ppdoe.metrics import gold_edge_set, _edge_set

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def edge_key(r) -> Tuple[str, str, str]:
    a, b = sorted([r.source_id, r.target_id])
    return (a, b, r.rel_type)


def main(n_seeds: int = 5):
    print(f"Phase 14 – Evidence profiles ({n_seeds} seeds)")
    data = load_cifar100(prefer_synthetic=True)
    gold = gold_edge_set(data["fine_to_coarse"])

    all_edges: List[dict] = []
    bin_rows: List[dict] = []
    summary_rows: List[dict] = []

    # Evidence composition bins (which signal dominates)
    DOMINANCE = [
        ("contextual_dominant", lambda e: e["contextual"] >= e["structural"] and e["contextual"] >= e["terminological"]),
        ("structural_dominant", lambda e: e["structural"] > e["contextual"] and e["structural"] >= e["terminological"]),
        ("terminological_dominant", lambda e: e["terminological"] > e["contextual"] and e["terminological"] > e["structural"]),
        ("balanced", lambda e: True),  # fallback
    ]

    for seed in range(n_seeds):
        idxs = create_client_partitions(
            data["n_train"], 10, seed=seed, labels=data["train_fine"]
        )
        p = PPDOE(
            n_clients=10, n_rounds=6, seed=seed,
            confirmation_threshold=0.15, theta=0.3,  # lower → more scored candidates
            enable_rl=True,
        )
        p.setup_cifar_clients(data["train_features"], data["train_fine"], idxs)
        p.run()

        rels = list(p.ontology.relationships.values())
        print(f"  seed {seed}: {len(rels)} relationships")

        for r in rels:
            struct = float(r.evidence.get("structural", 0.0))
            term = float(r.evidence.get("terminological", 0.0))
            ctx = float(r.evidence.get("contextual", 0.0))
            ek = edge_key(r)
            is_correct = ek in gold
            row = {
                "seed": seed,
                "rel_id": r.rel_id,
                "source_id": r.source_id,
                "target_id": r.target_id,
                "rel_type": r.rel_type,
                "status": r.status,
                "confidence": float(r.confidence),
                "structural": struct,
                "terminological": term,
                "contextual": ctx,
                "is_gold_edge": is_correct,
                "version_created": int(r.version_created),
            }
            # dominance label
            for name, pred in DOMINANCE:
                if pred(row):
                    row["evidence_dominance"] = name
                    break
            all_edges.append(row)

        # Per-seed evidence summary for confirmed edges
        conf = [r for r in all_edges if r["seed"] == seed and r["status"] == "confirmed"]
        if conf:
            correct = [r for r in conf if r["is_gold_edge"]]
            summary_rows.append({
                "seed": seed,
                "n_confirmed": len(conf),
                "n_correct": len(correct),
                "precision": len(correct) / len(conf) if conf else 0.0,
                "mean_confidence": float(np.mean([r["confidence"] for r in conf])),
                "mean_structural": float(np.mean([r["structural"] for r in conf])),
                "mean_terminological": float(np.mean([r["terminological"] for r in conf])),
                "mean_contextual": float(np.mean([r["contextual"] for r in conf])),
                "frac_contextual_dominant": sum(1 for r in conf if r["evidence_dominance"] == "contextual_dominant") / len(conf),
                "frac_structural_dominant": sum(1 for r in conf if r["evidence_dominance"] == "structural_dominant") / len(conf),
                "frac_terminological_dominant": sum(1 for r in conf if r["evidence_dominance"] == "terminological_dominant") / len(conf),
            })

    # Confidence × correctness calibration bins
    CONF_BINS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    for lo, hi in CONF_BINS:
        subset = [r for r in all_edges if lo <= r["confidence"] < hi]
        if not subset:
            continue
        correct = sum(1 for r in subset if r["is_gold_edge"])
        confirmed = [r for r in subset if r["status"] == "confirmed"]
        bin_rows.append({
            "bin_type": "confidence",
            "bin_low": lo,
            "bin_high": hi,
            "n_edges": len(subset),
            "n_confirmed": len(confirmed),
            "empirical_precision_all": correct / len(subset),
            "empirical_precision_confirmed": (
                sum(1 for r in confirmed if r["is_gold_edge"]) / len(confirmed)
                if confirmed else 0.0
            ),
            "mean_structural": float(np.mean([r["structural"] for r in subset])),
            "mean_contextual": float(np.mean([r["contextual"] for r in subset])),
        })

    for name, _ in DOMINANCE[:-1]:  # skip balanced fallback
        subset = [r for r in all_edges if r.get("evidence_dominance") == name]
        if not subset:
            continue
        correct = sum(1 for r in subset if r["is_gold_edge"])
        confirmed = [r for r in subset if r["status"] == "confirmed"]
        bin_rows.append({
            "bin_type": "dominance",
            "bin_low": name,
            "bin_high": name,
            "n_edges": len(subset),
            "n_confirmed": len(confirmed),
            "empirical_precision_all": correct / len(subset) if subset else 0.0,
            "empirical_precision_confirmed": (
                sum(1 for r in confirmed if r["is_gold_edge"]) / len(confirmed)
                if confirmed else 0.0
            ),
            "mean_structural": float(np.mean([r["structural"] for r in subset])),
            "mean_contextual": float(np.mean([r["contextual"] for r in subset])),
        })

    # Write full edge evidence table
    out1 = RESULTS / "evidence_edges.csv"
    if all_edges:
        with open(out1, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_edges[0].keys()))
            w.writeheader()
            w.writerows(all_edges)
        print(f"Wrote {out1} ({len(all_edges)} rows)")

    out2 = RESULTS / "evidence_bins.csv"
    if bin_rows:
        with open(out2, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(bin_rows[0].keys()))
            w.writeheader()
            w.writerows(bin_rows)
        print(f"Wrote {out2}")

    out3 = RESULTS / "evidence_summary.csv"
    if summary_rows:
        with open(out3, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"Wrote {out3}")

    report = {
        "n_seeds": n_seeds,
        "total_scored_edges": len(all_edges),
        "mean_precision_confirmed": float(np.mean([s["precision"] for s in summary_rows])) if summary_rows else 0.0,
        "evidence_signal_means_confirmed": {
            "structural": float(np.mean([s["mean_structural"] for s in summary_rows])) if summary_rows else 0.0,
            "terminological": float(np.mean([s["mean_terminological"] for s in summary_rows])) if summary_rows else 0.0,
            "contextual": float(np.mean([s["mean_contextual"] for s in summary_rows])) if summary_rows else 0.0,
        },
        "note": (
            "Each edge carries structural / terminological / contextual evidence. "
            "Dominance bins show which signal drove recovery. "
            "Confidence bins allow calibration checks (empirical precision vs reported confidence)."
        ),
    }
    out4 = RESULTS / "evidence_report.json"
    with open(out4, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out4}")
    print(json.dumps(report, indent=2))
    print("Phase 14 complete.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    main(ap.parse_args().seeds)
