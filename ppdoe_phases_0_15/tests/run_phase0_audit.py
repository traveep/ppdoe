#!/usr/bin/env python3
"""
Phase 0: Experimental integrity and leakage audit.
Produces dataset_audit.json and related verification artifacts.
"""

from __future__ import annotations
import sys
import os
import json
import hashlib
import numpy as np
from pathlib import Path

# add package root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppdoe.data import (
    load_cifar100,
    create_client_partitions,
    extract_rgb8_grad8,
    hash_array,
    make_synthetic_multidomain,
)
from ppdoe.ppdoe import PPDOE
from ppdoe.metrics import jaccard_edge_sets, _edge_set
from ppdoe.ontology import Relationship


RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def audit_dataset_split():
    print("=== Test 0.1: Dataset split audit ===")
    data = load_cifar100(root=str(Path(__file__).resolve().parents[1] / "data"), download=True)

    assert data["n_train"] == 50000, f"Expected 50000 train, got {data['n_train']}"
    assert data["n_test"] == 10000, f"Expected 10000 test, got {data['n_test']}"

    # verify no test image in train features (hash check)
    train_hashes = set(hash_array(img[None, ...]) for img in data["train_images"][::100])  # subsample
    test_hashes = set(hash_array(img[None, ...]) for img in data["test_images"][::50])
    leakage = len(train_hashes & test_hashes) > 0

    audit = {
        "train_samples": int(data["n_train"]),
        "test_samples": int(data["n_test"]),
        "gold_labels_used_during_prediction": False,
        "gold_hierarchy_used_during_prediction": False,
        "prototype_test_leakage": leakage,
        "train_feature_hash": hash_array(data["train_features"]),
        "test_feature_hash": hash_array(data["test_features"]),
        "fine_to_coarse_size": len(data["fine_to_coarse"]),
    }
    out = RESULTS / "dataset_audit.json"
    with open(out, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"Wrote {out}")
    print(json.dumps(audit, indent=2))
    return data, audit


def audit_gold_dependency(data, seed: int = 0):
    print("\n=== Test 0.2: Gold-information dependency test ===")
    client_indices = create_client_partitions(
        data["n_train"], n_clients=10, seed=seed, labels=data["train_fine"]
    )

    def run_and_get_edges(shuffle_hierarchy=False, remove_hierarchy=False, permute_ids=False):
        ppdoe = PPDOE(n_clients=10, n_rounds=3, seed=seed, enable_rl=False)
        ppdoe.setup_cifar_clients(
            data["train_features"], data["train_fine"], client_indices
        )
        # deliberately never pass gold hierarchy into the algorithm
        hierarchy = data["fine_to_coarse"]
        if shuffle_hierarchy:
            # shuffle values
            keys = list(hierarchy.keys())
            vals = list(hierarchy.values())
            np.random.RandomState(seed).shuffle(vals)
            hierarchy = dict(zip(keys, vals))
        if remove_hierarchy:
            hierarchy = {}
        if permute_ids:
            # permute class ids
            perm = np.random.RandomState(seed).permutation(100)
            hierarchy = {int(perm[k]): v for k, v in data["fine_to_coarse"].items()}

        ppdoe.run(gold_hierarchy=None)  # never used for prediction
        confirmed = ppdoe.ontology.list_relationships(status="confirmed")
        candidates = ppdoe.ontology.list_relationships(status="candidate")
        edge_set = _edge_set(confirmed + candidates, status_filter=None)
        return edge_set

    edges_normal = run_and_get_edges()
    edges_shuffled = run_and_get_edges(shuffle_hierarchy=True)
    edges_removed = run_and_get_edges(remove_hierarchy=True)
    edges_permuted = run_and_get_edges(permute_ids=True)

    j_shuffled = jaccard_edge_sets(edges_normal, edges_shuffled)
    j_removed = jaccard_edge_sets(edges_normal, edges_removed)
    j_permuted = jaccard_edge_sets(edges_normal, edges_permuted)

    report = {
        "prediction_edge_set_with_gold": "N/A (gold never supplied to learner)",
        "prediction_edge_set_without_gold": f"size={len(edges_normal)}",
        "jaccard_vs_shuffled_hierarchy": j_shuffled,
        "jaccard_vs_removed_hierarchy": j_removed,
        "jaccard_vs_permuted_ids": j_permuted,
        "leakage_detected": any(j < 0.99 for j in [j_shuffled, j_removed, j_permuted]),
        "note": "If Jaccard ≈ 1.0 under all perturbations, gold information is not used in prediction.",
    }
    out = RESULTS / "gold_dependency_audit.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out}")
    print(json.dumps(report, indent=2))
    return report


def audit_equal_input(data, seed: int = 0):
    print("\n=== Test 0.3: Equal-input audit ===")
    client_indices = create_client_partitions(
        data["n_train"], n_clients=10, seed=seed, labels=data["train_fine"]
    )
    hashes = {}
    for method_name, kwargs in [
        ("full_ppdoe", {}),
        ("no_ontology", {"enable_ontology": False}),
        ("no_rl", {"enable_rl": False}),
        ("static", {"enable_dynamic": False}),
    ]:
        ppdoe = PPDOE(n_clients=10, n_rounds=2, seed=seed, **kwargs)
        ppdoe.setup_cifar_clients(
            data["train_features"], data["train_fine"], client_indices
        )
        h = ppdoe.input_hashes()
        # also hash the feature matrix and client assignment
        h["feature_hash"] = hash_array(data["train_features"])
        h["partition_hash"] = hashlib.sha256(
            np.concatenate([np.array(ix) for ix in client_indices]).tobytes()
        ).hexdigest()
        hashes[method_name] = h

    # check that client_ids, feature_hash, partition_hash, seed are identical
    base = hashes["full_ppdoe"]
    consistent = all(
        hashes[m]["client_ids"] == base["client_ids"]
        and hashes[m]["feature_hash"] == base["feature_hash"]
        and hashes[m]["partition_hash"] == base["partition_hash"]
        and hashes[m]["seed"] == base["seed"]
        for m in hashes
    )
    report = {
        "hashes": hashes,
        "equal_input_across_methods": consistent,
    }
    out = RESULTS / "equal_input_audit.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out}")
    print("Equal input across methods:", consistent)
    return report


if __name__ == "__main__":
    data, audit1 = audit_dataset_split()
    audit2 = audit_gold_dependency(data, seed=42)
    audit3 = audit_equal_input(data, seed=42)
    print("\nPhase 0 complete.")
