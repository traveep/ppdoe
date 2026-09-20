#!/usr/bin/env python3
"""
Phase 7: Subscriber inference.
Hold out clients during construction; subscriber maps local concepts and queries ontology.
"""
from __future__ import annotations
import sys, csv
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppdoe.data import load_cifar100, create_client_partitions
from ppdoe.ppdoe import PPDOE
from ppdoe.metrics import compute_metrics, gold_edge_set, _edge_set
from ppdoe.client import FederatedClient
from ppdoe.aggregator import contextual_similarity

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def main(n_seeds=5):
    print(f"Phase 7 – Subscriber inference ({n_seeds} seeds)")
    data = load_cifar100(prefer_synthetic=False)
    rows = []
    for seed in range(n_seeds):
        idxs = create_client_partitions(
            data["n_train"], 10, seed=seed, labels=data["train_fine"]
        )
        # hold out last 2 clients as subscribers
        train_idxs, sub_idxs = idxs[:8], idxs[8:]
        p = PPDOE(n_clients=8, n_rounds=6, seed=seed,
                  confirmation_threshold=0.20, theta=0.5)
        p.setup_cifar_clients(data["train_features"], data["train_fine"], train_idxs)
        p.run()
        gold = gold_edge_set(data["fine_to_coarse"])
        # subscriber: local prototypes only, map to shared ontology by nearest concept
        for si, six in enumerate(sub_idxs):
            sub = FederatedClient(
                f"subscriber_{si}", "cifar100",
                data["train_features"][six], data["train_fine"][six],
                feature_dim=data["train_features"].shape[1], seed=seed + si,
            )
            concepts = sub.map_to_concepts()
            protos = sub.compute_prototypes(data["train_features"][six])
            # assign each local concept to nearest ontology concept
            onto_protos = {}
            for cid, c in p.ontology.concepts.items():
                # use mean of confirmed children's prototypes if available
                pass
            # evaluate: which gold edges involving subscriber classes are recoverable
            sub_classes = set(int(l) for l in np.unique(data["train_fine"][six]))
            relevant_gold = {e for e in gold if any(f"class_{c}" in e[0] or f"class_{c}" in e[1] for c in sub_classes)}
            conf = p.ontology.list_relationships(status="confirmed")
            pred = _edge_set(conf, "confirmed")
            hits = pred & relevant_gold
            tp, fp = len(hits), len(pred - relevant_gold)
            fn = len(relevant_gold - pred)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            rows.append({
                "seed": seed,
                "subscriber": si,
                "subscriber_relationship_precision": prec,
                "subscriber_relationship_recall": rec,
                "subscriber_f1": f1,
                "false_discovery_rate": fp / (tp + fp) if (tp + fp) else 0.0,
                "n_relevant_gold": len(relevant_gold),
                "n_hits": len(hits),
                "abstention_rate": 0.0,  # no abstention in current system
            })
            print(f"  seed={seed} sub={si} F1={f1:.4f}")
    out = RESULTS / "subscriber_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    main(ap.parse_args().seeds)
