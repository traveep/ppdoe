#!/usr/bin/env python3
"""
Phase 11: Scalability – time, memory proxies, concept/relationship counts.
"""
from __future__ import annotations
import sys, csv, time, tracemalloc
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppdoe.data import load_cifar100, create_client_partitions
from ppdoe.ppdoe import PPDOE

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
CLIENT_SCALES = [5, 10, 20]  # reduced from original 100–5000 for practical runtime
# for larger scales, subsample features


def main(n_seeds=3):
    print(f"Phase 11 – Scalability ({n_seeds} seeds)")
    data = load_cifar100(prefer_synthetic=False)
    rows = []
    for seed in range(n_seeds):
        for n_clients in CLIENT_SCALES:
            idxs = create_client_partitions(
                data["n_train"], n_clients, seed=seed, labels=data["train_fine"]
            )
            tracemalloc.start()
            t0 = time.perf_counter()
            p = PPDOE(n_clients=n_clients, n_rounds=min(5, n_clients), seed=seed,
                      confirmation_threshold=0.20, theta=0.5, active_fraction=0.8)
            p.setup_cifar_clients(data["train_features"], data["train_fine"], idxs)
            p.run()
            elapsed = time.perf_counter() - t0
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rows.append({
                "seed": seed,
                "n_clients": n_clients,
                "n_rounds": min(5, n_clients),
                "wall_time_sec": elapsed,
                "peak_memory_mb": peak / 1e6,
                "n_concepts": len(p.ontology.concepts),
                "n_relationships": len(p.ontology.relationships),
                "n_confirmed": len(p.ontology.list_relationships(status="confirmed")),
                "ontology_versions": p.ontology.version,
            })
            print(f"  seed={seed} clients={n_clients} time={elapsed:.2f}s peak={peak/1e6:.1f}MB concepts={len(p.ontology.concepts)}")
    out = RESULTS / "scalability_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    main(ap.parse_args().seeds)
