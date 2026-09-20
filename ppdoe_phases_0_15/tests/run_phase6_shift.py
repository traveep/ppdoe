#!/usr/bin/env python3
"""
Phase 6: Distribution-shift adaptation.
Label shift / feature shift / concept shift on later clients.
"""
from __future__ import annotations
import sys, csv
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppdoe.data import load_cifar100, create_client_partitions
from ppdoe.ppdoe import PPDOE
from ppdoe.metrics import compute_metrics, gold_edge_set

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
N_CLIENTS = 10
SHIFT_AFTER = 5  # first 5 clients "early", rest "late"


def apply_label_shift(labels, idxs, seed):
    """Late clients: reweight toward a subset of classes."""
    rng = np.random.RandomState(seed)
    new_idxs = list(idxs)
    prefer = rng.choice(100, size=30, replace=False)
    for i in range(SHIFT_AFTER, N_CLIENTS):
        ix = idxs[i]
        labs = labels[ix]
        mask = np.isin(labs, prefer)
        if mask.sum() < 10:
            continue
        # keep preferred, subsample others
        keep = np.where(mask)[0]
        drop = np.where(~mask)[0]
        drop = drop[: max(0, len(drop) // 2)]
        new_idxs[i] = np.sort(ix[np.concatenate([keep, drop])])
    return new_idxs


def apply_feature_shift(features, idxs, seed):
    """Late clients: additive noise / scale on features."""
    rng = np.random.RandomState(seed)
    feats = features.copy()
    for i in range(SHIFT_AFTER, N_CLIENTS):
        ix = idxs[i]
        feats[ix] = feats[ix] * (1.0 + 0.3 * rng.randn(*feats[ix].shape).astype(np.float32))
    return feats


def run_condition(seed, data, shift, frozen):
    idxs = create_client_partitions(
        data["n_train"], N_CLIENTS, seed=seed, labels=data["train_fine"]
    )
    feats = data["train_features"]
    if shift == "label":
        idxs = apply_label_shift(data["train_fine"], idxs, seed)
    elif shift == "feature":
        feats = apply_feature_shift(feats, idxs, seed)
    # concept shift ≈ label shift with different preferred set
    elif shift == "concept":
        idxs = apply_label_shift(data["train_fine"], idxs, seed + 77)

    p = PPDOE(n_clients=N_CLIENTS, n_rounds=N_CLIENTS, seed=seed,
              theta=0.55, confirmation_threshold=0.20, active_fraction=1.0)
    p.setup_cifar_clients(feats, data["train_fine"], idxs)
    if frozen:
        p.freeze(after_round=1)
    gold = gold_edge_set(data["fine_to_coarse"])
    order = list(range(N_CLIENTS))
    present, pre_f1, post_f1, edits = [], [], [], 0
    for t, c in enumerate(order):
        present.append(c)
        log = p.run_round(participating=list(present))
        m = compute_metrics(p.ontology, gold)
        if t < SHIFT_AFTER:
            pre_f1.append(m.f1)
        else:
            post_f1.append(m.f1)
        if log.get("structural_change"):
            edits += 1
    return {
        "seed": seed,
        "shift": shift,
        "condition": "frozen" if frozen else "dynamic",
        "f1_before_shift": float(np.mean(pre_f1)) if pre_f1 else 0.0,
        "f1_after_shift": float(np.mean(post_f1)) if post_f1 else 0.0,
        "final_f1": post_f1[-1] if post_f1 else 0.0,
        "structural_edits": edits,
    }


def main(n_seeds=5):
    print(f"Phase 6 – Distribution shift ({n_seeds} seeds)")
    data = load_cifar100(prefer_synthetic=False)
    rows = []
    for seed in range(n_seeds):
        print(f"  seed {seed}")
        for shift in ("label", "feature", "concept"):
            for frozen in (False, True):
                r = run_condition(seed, data, shift, frozen)
                rows.append(r)
                print(f"    {shift}/{r['condition']}: pre={r['f1_before_shift']:.3f} post={r['f1_after_shift']:.3f}")
    out = RESULTS / "shift_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    main(ap.parse_args().seeds)
