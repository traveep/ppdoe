"""
Competitive baselines for hierarchy recovery — GPU-accelerated when available.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from .ontology import Ontology, Concept, Relationship
from .metrics import compute_metrics, gold_edge_set, RelationshipMetrics
from .device import get_device


def _torch():
    import torch
    return torch


def _kmeans(X: np.ndarray, k: int, seed: int = 0, n_iter: int = 30, device: Optional[str] = None):
    """Lloyd k-means on CUDA if available."""
    dev = device or get_device()
    n, d = X.shape
    if n == 0:
        return np.zeros((k, d), dtype=np.float32), np.array([], dtype=int)
    k = min(k, n)

    if dev == "cuda":
        try:
            torch = _torch()
            torch.manual_seed(seed)
            t = torch.from_numpy(X.astype(np.float32)).to("cuda")
            # k-means++ style init
            idx = torch.randperm(n, device="cuda")[:k]
            C = t[idx].clone()
            labels = torch.zeros(n, dtype=torch.long, device="cuda")
            for _ in range(n_iter):
                # (n, k) squared distances
                d2 = ((t.unsqueeze(1) - C.unsqueeze(0)) ** 2).sum(dim=2)
                labels = d2.argmin(dim=1)
                for j in range(k):
                    mask = labels == j
                    if mask.any():
                        C[j] = t[mask].mean(dim=0)
            return C.cpu().numpy(), labels.cpu().numpy()
        except Exception:
            pass

    # CPU fallback
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=k, replace=False)
    C = X[idx].astype(np.float32).copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                C[j] = members.mean(axis=0)
    return C, labels


def _hungarian_align(pred: Dict[int, int], gold: Dict[int, int], k: int = 20) -> Dict[int, int]:
    cont = np.zeros((k, k), dtype=np.int64)
    for fine, pp in pred.items():
        if fine in gold:
            gp = gold[fine]
            if 0 <= pp < k and 0 <= gp < k:
                cont[pp, gp] += 1
    try:
        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(-cont)
        mapping = {int(r): int(c) for r, c in zip(row, col)}
    except Exception:
        mapping, used = {}, set()
        for r in range(k):
            for c in np.argsort(-cont[r]):
                if c not in used:
                    mapping[r] = int(c)
                    used.add(c)
                    break
    return {f: mapping.get(p, p) for f, p in pred.items()}


def _fine_prototypes(features, fine_labels, client_indices=None, device=None):
    if client_indices is not None:
        mask = np.zeros(len(fine_labels), dtype=bool)
        for ix in client_indices:
            mask[ix] = True
        features, fine_labels = features[mask], fine_labels[mask]

    dev = device or get_device()
    if dev == "cuda":
        try:
            torch = _torch()
            t = torch.from_numpy(features.astype(np.float32)).to("cuda")
            y = torch.from_numpy(fine_labels.astype(np.int64)).to("cuda")
            protos = {}
            for lab in torch.unique(y).tolist():
                protos[int(lab)] = t[y == lab].mean(dim=0).cpu().numpy()
            return protos
        except Exception:
            pass

    protos = {}
    for lab in np.unique(fine_labels):
        protos[int(lab)] = features[fine_labels == lab].mean(axis=0).astype(np.float32)
    return protos


def _to_metrics(fine_to_parent, gold, seed=0):
    aligned = _hungarian_align(fine_to_parent, gold)
    domain = "cifar100"
    onto = Ontology(name="baseline")
    for f in range(100):
        onto.add_concept(Concept(f"{domain}_class_{f}", f"c{f}", domain))
    for c in range(20):
        onto.add_concept(Concept(f"{domain}_coarse_{c}", f"coarse{c}", domain))
    for fine, parent in aligned.items():
        onto.add_relationship(Relationship(
            rel_id=f"b_{fine}_{parent}",
            source_id=f"{domain}_class_{fine}",
            target_id=f"{domain}_coarse_{parent}",
            rel_type="is_a",
            confidence=1.0,
            status="confirmed",
        ))
    return compute_metrics(onto, gold_edge_set(gold))


def fedavg_kmeans(features, fine_labels, client_indices, gold, seed=0, k=20):
    dev = get_device()
    accum = {}
    for ix in client_indices:
        for lab, p in _fine_prototypes(features, fine_labels, [ix], device=dev).items():
            accum.setdefault(lab, []).append(p)
    protos = {lab: np.mean(vs, axis=0).astype(np.float32) for lab, vs in accum.items()}
    labs = sorted(protos.keys())
    X = np.stack([protos[l] for l in labs])
    _, cids = _kmeans(X, k=k, seed=seed, device=dev)
    pred = {labs[i]: int(cids[i]) for i in range(len(labs))}
    return _to_metrics(pred, gold, seed)


def flis_style(features, fine_labels, client_indices, gold, seed=0, n_client_clusters=3, k=20):
    dev = get_device()
    sigs = []
    for ix in client_indices:
        local = _fine_prototypes(features, fine_labels, [ix], device=dev)
        sigs.append(np.mean(list(local.values()), axis=0) if local else np.zeros(features.shape[1], np.float32))
    S = np.stack(sigs)
    _kmeans(S, k=min(n_client_clusters, len(sigs)), seed=seed, device=dev)
    accum = {}
    for ix in client_indices:
        for lab, p in _fine_prototypes(features, fine_labels, [ix], device=dev).items():
            accum.setdefault(lab, []).append(p)
    protos = {lab: np.mean(vs, axis=0).astype(np.float32) for lab, vs in accum.items()}
    labs = sorted(protos.keys())
    X = np.stack([protos[l] for l in labs])
    _, cids = _kmeans(X, k=k, seed=seed + 1, device=dev)
    pred = {labs[i]: int(cids[i]) for i in range(len(labs))}
    return _to_metrics(pred, gold, seed)


def fedrcl_inspired(features, fine_labels, client_indices, gold, seed=0, k=20, n_refine=5):
    """Contrastive-style refine on GPU when available."""
    dev = get_device()
    rng = np.random.RandomState(seed)
    accum = {}
    for ix in client_indices:
        for lab, p in _fine_prototypes(features, fine_labels, [ix], device=dev).items():
            accum.setdefault(lab, []).append(p)
    protos = {lab: np.mean(vs, axis=0).astype(np.float32) for lab, vs in accum.items()}

    sample_idx = rng.choice(len(fine_labels), size=min(5000, len(fine_labels)), replace=False)
    X_np = features[sample_idx].astype(np.float32)
    y_np = fine_labels[sample_idx]

    if dev == "cuda":
        try:
            torch = _torch()
            X = torch.from_numpy(X_np).to("cuda")
            y = torch.from_numpy(y_np.astype(np.int64)).to("cuda")
            labs = sorted(protos.keys())
            for _ in range(n_refine):
                means = torch.stack([torch.from_numpy(protos[l]).to("cuda") for l in labs])
                lab_to_i = {l: i for i, l in enumerate(labs)}
                new_sum = {l: torch.zeros_like(means[0]) for l in labs}
                new_cnt = {l: 0 for l in labs}
                # vectorised-ish loop over unique labels
                for lab in labs:
                    mask = y == lab
                    if not mask.any():
                        continue
                    pts = X[mask]
                    own = means[lab_to_i[lab]]
                    d = ((means.unsqueeze(0) - pts.unsqueeze(1)) ** 2).sum(dim=2)  # (n_lab, k)
                    d[:, lab_to_i[lab]] = float("inf")
                    other_idx = d.argmin(dim=1)
                    other = means[other_idx]
                    refined = pts + 0.1 * (own - pts) - 0.05 * (other - pts)
                    new_sum[lab] = refined.sum(dim=0)
                    new_cnt[lab] = refined.shape[0]
                for lab in labs:
                    if new_cnt[lab] > 0:
                        protos[lab] = (new_sum[lab] / new_cnt[lab]).cpu().numpy()
            labs = sorted(protos.keys())
            Xm = np.stack([protos[l] for l in labs])
            _, cids = _kmeans(Xm, k=k, seed=seed, device=dev)
            pred = {labs[i]: int(cids[i]) for i in range(len(labs))}
            return _to_metrics(pred, gold, seed)
        except Exception:
            pass

    # CPU path
    X, y = X_np, y_np
    for _ in range(n_refine):
        labs = sorted(protos.keys())
        means = np.stack([protos[l] for l in labs])
        lab_to_i = {l: i for i, l in enumerate(labs)}
        new_accum = {l: [] for l in labs}
        for i in range(len(X)):
            lab = int(y[i])
            if lab not in lab_to_i:
                continue
            own = means[lab_to_i[lab]]
            d = ((means - X[i]) ** 2).sum(axis=1)
            d[lab_to_i[lab]] = np.inf
            other = means[d.argmin()]
            new_accum[lab].append(X[i] + 0.1 * (own - X[i]) - 0.05 * (other - X[i]))
        for lab in labs:
            if new_accum[lab]:
                protos[lab] = np.mean(new_accum[lab], axis=0).astype(np.float32)
    labs = sorted(protos.keys())
    Xm = np.stack([protos[l] for l in labs])
    _, cids = _kmeans(Xm, k=k, seed=seed, device="cpu")
    pred = {labs[i]: int(cids[i]) for i in range(len(labs))}
    return _to_metrics(pred, gold, seed)
