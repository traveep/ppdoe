#!/usr/bin/env python3
"""
Phase 3: Temporal / client-arrival ablation.

Corrected protocol (fixed θ, structural evolution is the only dynamic factor):

  same client evidence
  same client arrival order
  same fixed threshold θ = 0.55
  same final concept vocabulary
        ↓
  dynamic : structural updates continue after every arrival
  frozen  : structural updates stop after round 1

θ is NEVER adapted.  No adaptive-threshold rule exists in the supplied
PPDOE material; any claim that θ itself was dynamic is unsupported.
"""

from __future__ import annotations
import sys
import csv
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppdoe.data import load_cifar100, create_client_partitions
from ppdoe.ppdoe import PPDOE
from ppdoe.metrics import compute_metrics, gold_edge_set

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

# Corrected temporal experiment uses a fixed θ = 0.55
FIXED_THETA = 0.55
N_SEEDS = 50         # match Phase 1 scale
N_CLIENTS = 10


def auc_trapezoid(values: List[float]) -> float:
    """AUC = sum_{t=1}^{T-1} (m_t + m_{t+1}) / 2"""
    if len(values) < 2:
        return float(values[0]) if values else 0.0
    s = 0.0
    for i in range(len(values) - 1):
        s += (values[i] + values[i + 1]) / 2.0
    return s


def run_condition(
    seed: int,
    data: dict,
    arrival_order: List[int],
    frozen: bool,
) -> List[dict]:
    """One seed, one condition (dynamic or frozen)."""
    ppdoe = PPDOE(
        n_clients=N_CLIENTS,
        n_rounds=len(arrival_order),
        seed=seed,
        theta=FIXED_THETA,                 # fixed, never adapted
        confirmation_threshold=0.55,
        enable_dynamic=True,
        enable_rl=True,
        active_fraction=1.0,               # all present clients participate
    )
    client_indices = create_client_partitions(
        data["n_train"], n_clients=N_CLIENTS, seed=seed, labels=data["train_fine"]
    )
    ppdoe.setup_cifar_clients(
        data["train_features"], data["train_fine"], client_indices
    )
    if frozen:
        ppdoe.freeze(after_round=1)

    gold = data["fine_to_coarse"]
    gold_edges = gold_edge_set(gold)

    # sequential arrival
    present: List[int] = []
    trajectory = []
    for client_idx in arrival_order:
        present.append(client_idx)
        log = ppdoe.run_round(participating=list(present))
        metrics = compute_metrics(ppdoe.ontology, gold_edges)
        row = {
            "seed": seed,
            "round": log["round"],
            "condition": "frozen" if frozen else "dynamic",
            "relationship_precision": metrics.precision,
            "relationship_recall": metrics.recall,
            "relationship_f1": metrics.f1,
            "group_accuracy": metrics.group_accuracy,
            "fine_accuracy": metrics.fine_accuracy,
            "candidate_edges": metrics.candidate_edges,
            "confirmed_edges": metrics.confirmed_edges,
            "structural_change": log.get("structural_change", False),
            "ontology_version": log["ontology_version"],
            "concept_count": log["concept_count"],
            "relationship_count": log["relationship_count"],
            "theta_fixed": FIXED_THETA,
            "frozen_flag": log.get("frozen", False),
        }
        trajectory.append(row)
    return trajectory


def main(n_seeds: int = N_SEEDS):
    print(f"Phase 3 – Temporal ablation (fixed θ={FIXED_THETA})")
    print("Dynamic = structural updates continue | Frozen = stop after round 1")
    print("θ is fixed; it is never adapted online.\n")

    data = load_cifar100(prefer_synthetic=True)
    all_rows: List[dict] = []
    seed_summaries = []

    for seed in range(n_seeds):
        # identical arrival order for both conditions
        rng = np.random.RandomState(seed)
        arrival_order = rng.permutation(N_CLIENTS).tolist()

        print(f"Seed {seed}  arrival={arrival_order}")
        dyn = run_condition(seed, data, arrival_order, frozen=False)
        frz = run_condition(seed, data, arrival_order, frozen=True)
        all_rows.extend(dyn)
        all_rows.extend(frz)

        # AUCs
        dyn_f1 = [r["relationship_f1"] for r in dyn]
        frz_f1 = [r["relationship_f1"] for r in frz]
        dyn_ga = [r["group_accuracy"] for r in dyn]
        frz_ga = [r["group_accuracy"] for r in frz]

        delta_f1 = auc_trapezoid(dyn_f1) - auc_trapezoid(frz_f1)
        delta_ga = auc_trapezoid(dyn_ga) - auc_trapezoid(frz_ga)
        n_struct_dyn = sum(1 for r in dyn if r["structural_change"])
        n_struct_frz = sum(1 for r in frz if r["structural_change"])

        seed_summaries.append({
            "seed": seed,
            "delta_f1_auc": delta_f1,
            "delta_group_acc_auc": delta_ga,
            "struct_changes_dynamic": n_struct_dyn,
            "struct_changes_frozen": n_struct_frz,
            "final_f1_dynamic": dyn_f1[-1],
            "final_f1_frozen": frz_f1[-1],
            "final_concept_count_dynamic": dyn[-1]["concept_count"],
            "final_concept_count_frozen": frz[-1]["concept_count"],
        })
        print(f"  ΔF1-AUC={delta_f1:+.4f}  ΔGA-AUC={delta_ga:+.4f}  "
              f"struct_dyn={n_struct_dyn} struct_frz={n_struct_frz}")

    # write per-round trajectory
    traj_path = RESULTS / "temporal_trajectory.csv"
    with open(traj_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {traj_path}")

    # write seed-level deltas
    delta_path = RESULTS / "temporal_seed_deltas.csv"
    with open(delta_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seed_summaries[0].keys()))
        w.writeheader()
        w.writerows(seed_summaries)
    print(f"Wrote {delta_path}")

    # statistical summary
    deltas_f1 = np.array([s["delta_f1_auc"] for s in seed_summaries])
    deltas_ga = np.array([s["delta_group_acc_auc"] for s in seed_summaries])

    def paired_stats(deltas):
        mean = float(np.mean(deltas))
        std = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
        n = len(deltas)
        se = std / np.sqrt(n) if n else 0.0
        ci = (mean - 1.96 * se, mean + 1.96 * se)
        # Wilcoxon
        if n >= 5 and np.any(deltas != 0):
            try:
                w_stat, w_p = stats.wilcoxon(deltas)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
        else:
            w_stat, w_p = np.nan, np.nan
        # paired t
        if n >= 2:
            t_stat, t_p = stats.ttest_1samp(deltas, 0.0)
        else:
            t_stat, t_p = np.nan, np.nan
        return {
            "mean": mean, "std": std, "ci95": ci,
            "n_favor_dynamic": int(np.sum(deltas > 0)),
            "n_favor_frozen": int(np.sum(deltas < 0)),
            "median": float(np.median(deltas)),
            "min": float(np.min(deltas)), "max": float(np.max(deltas)),
            "wilcoxon_p": float(w_p) if w_p == w_p else None,
            "ttest_p": float(t_p) if t_p == t_p else None,
            "cohens_d": mean / (std + 1e-12),
        }

    summary = {
        "theta_fixed": FIXED_THETA,
        "note": (
            "θ is a fixed hyper-parameter. The only dynamic factor tested is "
            "online structural evolution of the ontology/graph. No adaptive "
            "threshold rule is present in the supplied PPDOE material."
        ),
        "relationship_f1_auc_delta": paired_stats(deltas_f1),
        "group_accuracy_auc_delta": paired_stats(deltas_ga),
        "n_seeds": n_seeds,
    }
    sum_path = RESULTS / "temporal_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {sum_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    args = p.parse_args()
    main(n_seeds=args.seeds)
