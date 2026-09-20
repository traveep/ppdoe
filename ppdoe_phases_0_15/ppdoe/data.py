"""
Data loading for CIFAR-100 and multi-domain synthetic streams.
Uses real CIFAR-100 official split when available. Falls back to a
structure-preserving synthetic hierarchical dataset for fast testing.
Features are deterministic so that all methods receive identical inputs.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import hashlib
import json


def _download_cifar100():
    try:
        from torchvision.datasets import CIFAR100
        return CIFAR100
    except ImportError:
        return None


def extract_rgb8_grad8(images: np.ndarray) -> np.ndarray:
    """
    Deterministic feature extraction approximating rgb8_grad8.
    images: (N, 32, 32, 3) uint8 or float
    returns: (N, 16) float32
    """
    if images.dtype != np.float32:
        imgs = images.astype(np.float32) / 255.0
    else:
        imgs = images
    N = imgs.shape[0]
    feats = np.zeros((N, 16), dtype=np.float32)
    for i in range(N):
        img = imgs[i]
        h, w = img.shape[:2]
        q = [
            img[:h//2, :w//2].mean(axis=(0, 1)),
            img[:h//2, w//2:].mean(axis=(0, 1)),
            img[h//2:, :w//2].mean(axis=(0, 1)),
            img[h//2:, w//2:].mean(axis=(0, 1)),
        ]
        rgb = np.concatenate(q)[:8]
        gx = np.abs(np.diff(img.mean(axis=2), axis=1)).flatten()
        gy = np.abs(np.diff(img.mean(axis=2), axis=0)).flatten()
        g = np.concatenate([gx, gy])
        hist, _ = np.histogram(g, bins=8, range=(0.0, 1.0), density=True)
        feats[i, :8] = rgb
        feats[i, 8:] = hist.astype(np.float32)
    return feats


def _standard_fine_to_coarse() -> Dict[int, int]:
    """Official CIFAR-100 fine → coarse mapping (0-99 → 0-19)."""
    mapping = {}
    groups = [
        [4, 30, 55, 72, 95],          # 0 aquatic mammals
        [1, 32, 67, 73, 91],          # 1 fish
        [54, 62, 70, 82, 92],         # 2 flowers
        [9, 10, 16, 28, 61],          # 3 food containers
        [0, 51, 53, 57, 83],          # 4 fruit and vegetables
        [22, 39, 40, 86, 87],         # 5 household electrical
        [5, 20, 25, 84, 94],          # 6 household furniture
        [6, 7, 14, 18, 24],           # 7 insects
        [3, 42, 43, 88, 97],          # 8 large carnivores
        [12, 17, 37, 68, 76],         # 9 large man-made outdoor
        [23, 33, 49, 60, 71],         # 10 large natural outdoor
        [15, 19, 21, 31, 38],         # 11 large omnivores/herbivores
        [34, 63, 64, 66, 75],         # 12 medium-sized mammals
        [26, 45, 77, 79, 99],         # 13 non-insect invertebrates
        [2, 11, 35, 46, 98],          # 14 people
        [27, 29, 44, 78, 93],         # 15 reptiles
        [36, 50, 65, 74, 80],         # 16 small mammals
        [47, 52, 56, 59, 96],         # 17 trees
        [8, 13, 48, 58, 90],          # 18 vehicles 1
        [41, 69, 81, 85, 89],         # 19 vehicles 2
    ]
    for coarse, fines in enumerate(groups):
        for f in fines:
            mapping[f] = coarse
    return mapping


def _synthetic_cifar100(n_train: int = 5000, n_test: int = 1000, seed: int = 0) -> Dict[str, Any]:
    """
    Fast synthetic hierarchical dataset preserving the official
    100-fine → 20-coarse structure. Class-conditional Gaussians make
    similarity-based recovery meaningful.
    """
    rng = np.random.RandomState(seed)
    fine_to_coarse = _standard_fine_to_coarse()
    dim = 16
    coarse_means = rng.randn(20, dim).astype(np.float32) * 2.0
    fine_means = np.zeros((100, dim), dtype=np.float32)
    for f, c in fine_to_coarse.items():
        fine_means[f] = coarse_means[c] + rng.randn(dim).astype(np.float32) * 0.4

    def _gen(n):
        fine = rng.randint(0, 100, size=n)
        feats = fine_means[fine] + rng.randn(n, dim).astype(np.float32) * 0.3
        coarse = np.array([fine_to_coarse[int(f)] for f in fine])
        imgs = rng.randint(0, 255, size=(n, 32, 32, 3), dtype=np.uint8)
        return imgs, fine, coarse, feats

    train_imgs, train_fine, train_coarse, train_feats = _gen(n_train)
    test_imgs, test_fine, test_coarse, test_feats = _gen(n_test)
    hierarchy = {int(f): int(c) for f, c in fine_to_coarse.items()}
    return {
        "train_images": train_imgs,
        "train_fine": train_fine,
        "train_coarse": train_coarse,
        "train_features": train_feats,
        "test_images": test_imgs,
        "test_fine": test_fine,
        "test_coarse": test_coarse,
        "test_features": test_feats,
        "fine_to_coarse": hierarchy,
        "n_train": n_train,
        "n_test": n_test,
        "n_fine": 100,
        "n_coarse": 20,
        "synthetic": True,
    }


def load_cifar100(
    root: str = "./data",
    download: bool = True,
    prefer_synthetic: bool = False,
) -> Dict[str, Any]:
    """
    Load official CIFAR-100 when possible; otherwise a structure-preserving
    synthetic set (for fast Phase-0 / CI runs).
    """
    if prefer_synthetic:
        return _synthetic_cifar100()

    CIFAR100 = _download_cifar100()
    if CIFAR100 is None:
        print("torchvision not available – using synthetic hierarchical data")
        return _synthetic_cifar100()

    try:
        train_set = CIFAR100(root=root, train=True, download=download)
        test_set = CIFAR100(root=root, train=False, download=download)

        def _to_numpy(ds):
            imgs, fine = [], []
            for img, target in ds:
                imgs.append(np.array(img, dtype=np.uint8))
                fine.append(target)
            return np.stack(imgs), np.array(fine)

        train_imgs, train_fine = _to_numpy(train_set)
        test_imgs, test_fine = _to_numpy(test_set)
        fine_to_coarse = _standard_fine_to_coarse()
        train_coarse = np.array([fine_to_coarse[f] for f in train_fine])
        test_coarse = np.array([fine_to_coarse[f] for f in test_fine])
        train_feats = extract_rgb8_grad8(train_imgs)
        test_feats = extract_rgb8_grad8(test_imgs)
        hierarchy = {int(f): int(c) for f, c in fine_to_coarse.items()}
        return {
            "train_images": train_imgs,
            "train_fine": train_fine,
            "train_coarse": train_coarse,
            "train_features": train_feats,
            "test_images": test_imgs,
            "test_fine": test_fine,
            "test_coarse": test_coarse,
            "test_features": test_feats,
            "fine_to_coarse": hierarchy,
            "n_train": len(train_fine),
            "n_test": len(test_fine),
            "n_fine": 100,
            "n_coarse": 20,
            "synthetic": False,
        }
    except Exception as e:
        print(f"CIFAR-100 load failed ({e}) – falling back to synthetic data")
        return _synthetic_cifar100()


def create_client_partitions(
    n_samples: int,
    n_clients: int = 10,
    seed: int = 0,
    non_iid_alpha: float = 0.5,
    labels: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    if labels is None:
        indices = rng.permutation(n_samples)
        splits = np.array_split(indices, n_clients)
        return [np.sort(s) for s in splits]

    n_classes = int(labels.max()) + 1
    client_indices = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        cls_idx = np.where(labels == c)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet([non_iid_alpha] * n_clients)
        proportions = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        splits = np.split(cls_idx, proportions)
        for i, s in enumerate(splits):
            client_indices[i].extend(s.tolist())
    return [np.sort(np.array(ix, dtype=int)) for ix in client_indices]


def make_synthetic_multidomain(seed: int = 0) -> Dict[str, List[dict]]:
    rng = np.random.RandomState(seed)
    healthcare = []
    for i in range(50):
        healthcare.append({
            "entity": "Cardiovascular System",
            "condition": rng.choice(["Elevated Blood Pressure", "Normal", "Arrhythmia"]),
            "value": f"{rng.randint(110, 180)}/{rng.randint(70, 110)} mmHg",
            "severity": rng.choice(["Low", "Moderate", "High"]),
            "timestamp": "2025-10-19T17:26:00-04:00",
            "source": f"HealthcareNode_{i % 3}",
            "patient_context": f"Caribbean Man, Age {rng.randint(30, 70)}",
        })
    aviation = []
    for i in range(40):
        aviation.append({
            "entity": rng.choice(["Left Aileron", "Right Aileron", "Engine", "Landing Gear"]),
            "condition": rng.choice(["Mechanical Fault", "Normal", "Overheating"]),
            "severity": rng.choice(["Low", "Moderate", "High"]),
            "timestamp": "2025-10-19T17:26:00-04:00",
            "source": f"AviationNode_{i % 2}",
            "vehicle": "Diamond DA-40",
        })
    manufacturing = []
    for i in range(30):
        manufacturing.append({
            "entity": "Conveyor Belt Motor",
            "condition": rng.choice(["Overheating", "Normal", "Vibration"]),
            "value": f"{rng.randint(40, 95)}C",
            "severity": rng.choice(["Low", "Moderate", "High"]),
            "timestamp": "2025-10-19T17:26:00-04:00",
            "source": f"ManufacturingNode_{i % 2}",
            "equipment_id": f"CB-Motor-{i:03d}",
        })
    return {
        "healthcare": healthcare,
        "aviation": aviation,
        "manufacturing": manufacturing,
    }


def hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()
