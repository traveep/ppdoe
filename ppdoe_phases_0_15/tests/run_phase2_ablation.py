#!/usr/bin/env python3
"""
Phase 2: Genuine component ablation.
One component disabled at a time; all other inputs identical.
"""
from __future__ import annotations
import sys, csv, json
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppdoe.data import load_cifar100, create_client_partitions
from ppdoe.ppdoe import PPDOE
from ppdoe.metrics import compute_metrics, gold_edge_set

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

CONDITIONS = [
    ("full_ppdoe", {}),
    ("no_ontology", {"enable_ontology": False}),
    ("no_dynamic", {"enable_dynamic": False}),
    ("no_semantic", {"enable_semantic": False}),
    ("no_structural", {"enable_structural": False}),
    ("no_contextual", {"enable_contextual": False}),
    ("no_rl", {"enable_rl": False}),
    ("no_privacy", {"enable_privacy": False}),
    ("scrambled_eval", {}),  # same run, gold scrambled at eval only
]


def run_one(seed, data, name, kwargs):
    idxs = create_client_partitions(
        data["n_train"], 10, seed=seed, labels=data["train_fine"]
    )
    p = PPDOE(n_clients=10, n_rounds=6, seed=seed, confirmation_threshold=0.20,
              theta=0.5, device="cpu", **kwargs)
    p.setup_cifar_clients(data["train_features"], data["train_fine"], idxs)
    p.run()
    gold = dict(data["fine_to_coarse"])
    if name == "scrambled_eval":
        rng = np.random.RandomState(seed + 999)
        keys, vals = list(gold.keys()), list(gold.values())
        rng.shuffle(vals)
        gold = dict(zip(keys, vals))
    m = compute_metrics(p.ontology, gold_edge_set(gold))
    return {
        "seed": seed,
        "component_condition": name,
        "relationship_f1": m.f1,
        "relationship_precision": m.precision,
        "relationship_recall": m.recall,
        "group_accuracy": m.group_accuracy,
        "false_discovery_rate": m.false_discovery_rate,
        "confirmed_edges": m.confirmed_edges,
        "candidate_edges": m.candidate_edges,
    }


def main(n_seeds=10):
    print(f"Phase 2 – Component ablation ({n_seeds} seeds)")
    data = load_cifar100(prefer_synthetic=False)
    rows = []
    for seed in range(n_seeds):
        print(f"  seed {seed}")
        for name, kw in CONDITIONS:
            r = run_one(seed, data, name, kw)
            rows.append(r)
            print(f"    {name}: F1={r['relationship_f1']:.4f}")

    out = RESULTS / "ablation_per_seed.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out}")

    # summary vs full
    full = {r["seed"]: r["relationship_f1"] for r in rows if r["component_condition"] == "full_ppdoe"}
    summary = []
    for name, _ in CONDITIONS:
        vals = [r for r in rows if r["component_condition"] == name]
        f1s = np.array([v["relationship_f1"] for v in vals])
        deltas = np.array([full[v["seed"]] - v["relationship_f1"] for v in vals if v["seed"] in full])
        mean_d = float(np.mean(deltas)) if len(deltas) else 0.0
        std_d = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
        se = std_d / np.sqrt(len(deltas)) if len(deltas) else 0.0
        p = float(stats.ttest_1samp(deltas, 0.0).pvalue) if len(deltas) > 1 else None
        summary.append({
            "component_condition": name,
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)),
            "mean_precision": float(np.mean([v["relationship_precision"] for v in vals])),
            "mean_recall": float(np.mean([v["relationship_recall"] for v in vals])),
            "mean_fdr": float(np.mean([v["false_discovery_rate"] for v in vals])),
            "paired_delta_vs_full": mean_d,
            "ci95_low": mean_d - 1.96 * se,
            "ci95_high": mean_d + 1.96 * se,
            "p_value": p,
            "n_seeds": len(vals),
        })
    out2 = RESULTS / "ablation_summary.csv"
    with open(out2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    main(ap.parse_args().seeds)
