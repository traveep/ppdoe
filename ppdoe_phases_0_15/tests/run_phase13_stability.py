#!/usr/bin/env python3
"""
Phase 13: Ontology stability across seeds and arrival orders.

Measures whether recovered edge sets are path-dependent or stable:
  - Jaccard of confirmed edges across seeds (same method)
  - Jaccard across arrival orders (dynamic PPDOE)
  - Concept/relationship count variance
  - Parent-assignment agreement (before any Hungarian alignment)

Aligned with ontology discovery: a useful ontology should be
reproducible under reasonable variation, or at least quantify
how much structure changes with seed / order.
"""
from __future__ import annotations
import sys
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppdoe.data import load_cifar100, create_client_partitions
from ppdoe.ppdoe import PPDOE
from ppdoe.metrics import _edge_set, gold_edge_set, compute_metrics, jaccard_edge_sets
from ppdoe.baselines import fedavg_kmeans, flis_style, _fine_prototypes, _kmeans, _hungarian_align
from ppdoe.ontology import Ontology, Concept, Relationship

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def confirmed_edges(ontology) -> Set[Tuple[str, str, str]]:
    conf = ontology.list_relationships(status="confirmed")
    return _edge_set(conf, status_filter="confirmed")


def parent_map(ontology, domain: str = "cifar100") -> Dict[int, int]:
    """Map fine class id → coarse id from confirmed is_a edges (no alignment)."""
    out = {}
    for r in ontology.list_relationships(status="confirmed"):
        if r.rel_type != "is_a":
            continue
        if f"{domain}_class_" in r.source_id and f"{domain}_coarse_" in r.target_id:
            try:
                fine = int(r.source_id.split("_class_")[-1])
                coarse = int(r.target_id.split("_coarse_")[-1])
                out[fine] = coarse
            except ValueError:
                pass
    return out


def parent_agreement(map_a: Dict[int, int], map_b: Dict[int, int]) -> float:
    keys = set(map_a) & set(map_b)
    if not keys:
        return 0.0
    return sum(1 for k in keys if map_a[k] == map_b[k]) / len(keys)


def run_ppdoe(seed, data, idxs, frozen=False, arrival=None):
    n_rounds = 6 if arrival is None else len(arrival)
    p = PPDOE(
        n_clients=10, n_rounds=n_rounds, seed=seed,
        confirmation_threshold=0.20, theta=0.5,
        enable_rl=True, active_fraction=1.0 if arrival else 0.8,
    )
    p.setup_cifar_clients(data["train_features"], data["train_fine"], idxs)
    if frozen:
        p.freeze(after_round=1)
    if arrival is None:
        p.run()
    else:
        present = []
        for c in arrival:
            present.append(c)
            p.run_round(participating=list(present))
    return p.ontology


def clustering_ontology(features, labels, idxs, gold, seed, style: str):
    protos = {}
    for ix in idxs:
        for lab, p in _fine_prototypes(features, labels, [ix]).items():
            protos.setdefault(lab, []).append(p)
    protos = {lab: np.mean(vs, axis=0) for lab, vs in protos.items()}
    labs = sorted(protos.keys())
    X = np.stack([protos[l] for l in labs])
    _, cids = _kmeans(X, k=20, seed=seed)
    pred = {labs[i]: int(cids[i]) for i in range(len(labs))}
    # export BOTH raw and Hungarian-aligned for stability comparison
    onto_raw = Ontology(name=f"{style}_raw")
    onto_aligned = Ontology(name=f"{style}_aligned")
    for f in range(100):
        onto_raw.add_concept(Concept(f"cifar100_class_{f}", f"c{f}", "cifar100"))
        onto_aligned.add_concept(Concept(f"cifar100_class_{f}", f"c{f}", "cifar100"))
    for c in range(20):
        onto_raw.add_concept(Concept(f"cifar100_coarse_{c}", f"coarse{c}", "cifar100"))
        onto_aligned.add_concept(Concept(f"cifar100_coarse_{c}", f"coarse{c}", "cifar100"))
    aligned = _hungarian_align(pred, gold)
    for fine, parent in pred.items():
        onto_raw.add_relationship(Relationship(
            rel_id=f"raw_{fine}_{parent}",
            source_id=f"cifar100_class_{fine}",
            target_id=f"cifar100_coarse_{parent}",
            rel_type="is_a", confidence=1.0, status="confirmed",
        ))
    for fine, parent in aligned.items():
        onto_aligned.add_relationship(Relationship(
            rel_id=f"al_{fine}_{parent}",
            source_id=f"cifar100_class_{fine}",
            target_id=f"cifar100_coarse_{parent}",
            rel_type="is_a", confidence=1.0, status="confirmed",
        ))
    return onto_raw, onto_aligned


def pairwise_jaccard(edge_sets: List[Set]) -> Dict[str, float]:
    if len(edge_sets) < 2:
        return {"mean_jaccard": 1.0, "min_jaccard": 1.0, "std_jaccard": 0.0, "n_pairs": 0}
    js = []
    for i in range(len(edge_sets)):
        for j in range(i + 1, len(edge_sets)):
            js.append(jaccard_edge_sets(edge_sets[i], edge_sets[j]))
    return {
        "mean_jaccard": float(np.mean(js)),
        "min_jaccard": float(np.min(js)),
        "std_jaccard": float(np.std(js)),
        "n_pairs": len(js),
    }


def main(n_seeds: int = 5):
    print(f"Phase 13 – Ontology stability ({n_seeds} seeds)")
    data = load_cifar100(prefer_synthetic=True)
    gold = data["fine_to_coarse"]
    gold_edges = gold_edge_set(gold)

    # --- A. Cross-seed stability (fixed random arrival / standard run) ---
    method_edges: Dict[str, List[Set]] = {
        "full_ppdoe": [],
        "static_ontology": [],
        "fedavg_kmeans_raw": [],
        "fedavg_kmeans_aligned": [],
        "flis_raw": [],
        "flis_aligned": [],
    }
    method_parents: Dict[str, List[Dict[int, int]]] = {k: [] for k in method_edges}
    per_seed_rows = []

    for seed in range(n_seeds):
        idxs = create_client_partitions(
            data["n_train"], 10, seed=seed, labels=data["train_fine"]
        )
        print(f"  seed {seed}")

        onto = run_ppdoe(seed, data, idxs, frozen=False)
        e = confirmed_edges(onto)
        method_edges["full_ppdoe"].append(e)
        method_parents["full_ppdoe"].append(parent_map(onto))
        m = compute_metrics(onto, gold_edges)
        per_seed_rows.append({
            "seed": seed, "method": "full_ppdoe", "arrival": "default",
            "n_confirmed": len(e), "f1": m.f1, "n_concepts": len(onto.concepts),
            "version": onto.version, "state_hash": onto.current_state_hash()[:16],
        })

        onto_s = run_ppdoe(seed, data, idxs, frozen=True)
        e = confirmed_edges(onto_s)
        method_edges["static_ontology"].append(e)
        method_parents["static_ontology"].append(parent_map(onto_s))
        m = compute_metrics(onto_s, gold_edges)
        per_seed_rows.append({
            "seed": seed, "method": "static_ontology", "arrival": "default",
            "n_confirmed": len(e), "f1": m.f1, "n_concepts": len(onto_s.concepts),
            "version": onto_s.version, "state_hash": onto_s.current_state_hash()[:16],
        })

        for style in ("fedavg_kmeans", "flis"):
            raw, aligned = clustering_ontology(
                data["train_features"], data["train_fine"], idxs, gold, seed, style
            )
            for tag, onto_b in (("raw", raw), ("aligned", aligned)):
                key = f"{style}_{tag}"
                e = confirmed_edges(onto_b)
                method_edges[key].append(e)
                method_parents[key].append(parent_map(onto_b))
                m = compute_metrics(onto_b, gold_edges)
                per_seed_rows.append({
                    "seed": seed, "method": key, "arrival": "n/a",
                    "n_confirmed": len(e), "f1": m.f1, "n_concepts": len(onto_b.concepts),
                    "version": onto_b.version, "state_hash": onto_b.current_state_hash()[:16],
                })

    # --- B. Arrival-order sensitivity (PPDOE dynamic only) ---
    order_names = ["random", "sequential", "reverse"]
    order_edges: Dict[str, List[Set]] = {o: [] for o in order_names}
    order_parents: Dict[str, List[Dict]] = {o: [] for o in order_names}

    for seed in range(n_seeds):
        idxs = create_client_partitions(
            data["n_train"], 10, seed=seed, labels=data["train_fine"]
        )
        rng = np.random.RandomState(seed)
        orders = {
            "random": rng.permutation(10).tolist(),
            "sequential": list(range(10)),
            "reverse": list(reversed(range(10))),
        }
        for oname, order in orders.items():
            onto = run_ppdoe(seed, data, idxs, frozen=False, arrival=order)
            e = confirmed_edges(onto)
            order_edges[oname].append(e)
            order_parents[oname].append(parent_map(onto))
            m = compute_metrics(onto, gold_edges)
            per_seed_rows.append({
                "seed": seed, "method": "full_ppdoe", "arrival": oname,
                "n_confirmed": len(e), "f1": m.f1, "n_concepts": len(onto.concepts),
                "version": onto.version, "state_hash": onto.current_state_hash()[:16],
            })
            print(f"    arrival={oname}: confirmed={len(e)} F1={m.f1:.4f}")

    # --- Summaries ---
    stability_summary = []
    for method, sets in method_edges.items():
        j = pairwise_jaccard(sets)
        # parent agreement
        pa = []
        maps = method_parents[method]
        for i in range(len(maps)):
            for k in range(i + 1, len(maps)):
                pa.append(parent_agreement(maps[i], maps[k]))
        stability_summary.append({
            "scope": "cross_seed",
            "method": method,
            "mean_edge_jaccard": j["mean_jaccard"],
            "min_edge_jaccard": j["min_jaccard"],
            "std_edge_jaccard": j["std_jaccard"],
            "mean_parent_agreement": float(np.mean(pa)) if pa else 0.0,
            "n_seeds": n_seeds,
            "n_pairs": j["n_pairs"],
        })

    for oname, sets in order_edges.items():
        j = pairwise_jaccard(sets)
        pa = []
        maps = order_parents[oname]
        for i in range(len(maps)):
            for k in range(i + 1, len(maps)):
                pa.append(parent_agreement(maps[i], maps[k]))
        stability_summary.append({
            "scope": "cross_seed_same_arrival",
            "method": f"full_ppdoe_arrival_{oname}",
            "mean_edge_jaccard": j["mean_jaccard"],
            "min_edge_jaccard": j["min_jaccard"],
            "std_edge_jaccard": j["std_jaccard"],
            "mean_parent_agreement": float(np.mean(pa)) if pa else 0.0,
            "n_seeds": n_seeds,
            "n_pairs": j["n_pairs"],
        })

    # Within-seed: Jaccard between different arrival orders (same seed)
    within_order_js = []
    for seed in range(n_seeds):
        seed_sets = []
        for oname in order_names:
            # recover from per_seed? easier recompute from stored lists
            seed_sets.append(order_edges[oname][seed])
        for i in range(len(seed_sets)):
            for k in range(i + 1, len(seed_sets)):
                within_order_js.append(jaccard_edge_sets(seed_sets[i], seed_sets[k]))
    if within_order_js:
        stability_summary.append({
            "scope": "within_seed_across_arrivals",
            "method": "full_ppdoe",
            "mean_edge_jaccard": float(np.mean(within_order_js)),
            "min_edge_jaccard": float(np.min(within_order_js)),
            "std_edge_jaccard": float(np.std(within_order_js)),
            "mean_parent_agreement": None,
            "n_seeds": n_seeds,
            "n_pairs": len(within_order_js),
        })

    # Write
    out1 = RESULTS / "stability_per_seed.csv"
    with open(out1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
        w.writeheader()
        w.writerows(per_seed_rows)
    print(f"Wrote {out1}")

    out2 = RESULTS / "stability_summary.csv"
    with open(out2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stability_summary[0].keys()))
        w.writeheader()
        w.writerows(stability_summary)
    print(f"Wrote {out2}")

    report = {
        "note": (
            "Edge Jaccard and parent agreement measure structural stability. "
            "Clustering baselines are reported both RAW (k-means labels as-is) "
            "and HUNGARIAN-aligned (best match to gold coarse IDs). "
            "PPDOE is never aligned — fixed coarse_0..19 IDs."
        ),
        "n_seeds": n_seeds,
        "cross_seed_highlights": {
            r["method"]: {
                "mean_edge_jaccard": r["mean_edge_jaccard"],
                "mean_parent_agreement": r["mean_parent_agreement"],
            }
            for r in stability_summary if r["scope"] == "cross_seed"
        },
    }
    out3 = RESULTS / "stability_report.json"
    with open(out3, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out3}")
    print(json.dumps(report["cross_seed_highlights"], indent=2))
    print("Phase 13 complete.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    main(ap.parse_args().seeds)
