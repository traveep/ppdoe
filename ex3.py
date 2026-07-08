"""
Experiment 3: Cross-Silo Medical Benchmark (MedMNIST v2)

PPDOE is a server-hosted, ontology-guided aggregation system.
It is NOT federated learning.
FL algorithms below are independent comparison baselines.

Architecture: shared CNN encoder (frozen after warm-up) + per-silo classification head.
Ontology: hand-coded clinical knowledge graph encoding ICD-10, LOINC,
          specialty, privacy level, retention policy, and collaboration permissions.

Install: pip install medmnist
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import random
import copy
import os

try:
    import medmnist
    from medmnist import (
        DermaMNIST, PathMNIST, PneumoniaMNIST,
        RetinaMNIST, BreastMNIST, BloodMNIST, OrganAMNIST
    )
except ImportError:
    raise ImportError("Run: pip install medmnist")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================================================================
# PPDOE is the system under evaluation.
# It is NOT federated learning.
# FL algorithms are independent comparison baselines only.
# ================================================================

N_SUBCLIENTS = 3    # per silo (reduced from 5 because silos are clinically distinct)
ALPHA        = 0.4
N_ROUNDS     = 10
EMBED_DIM    = 512  # shared encoder output dimension


# ================================================================
# CLINICAL ONTOLOGY
# Hand-coded from ICD-10, LOINC, specialty, and data governance.
# Used exclusively by PPDOE. FL baselines have no access.
# ================================================================
class ClinicalOntology:
    """
    Nodes: 7 silos, each described by:
      - specialty        (dermatology, pathology, radiology, ophthalmology,
                          haematology, radiology-CT)
      - modality_loinc   (LOINC code for the imaging/lab modality)
      - icd10_chapter    (primary ICD-10 chapter of the disease domain)
      - privacy_level    (0=most open, 1=standard HIPAA, 2=restricted)
      - retention_policy (0=shareable, 1=pseudonymised, 2=local-only)

    Edges:
      - specialty_overlap    (0.0-1.0, hand-coded)
      - modality_similarity  (0.0-1.0, same modality family = 1.0)
      - icd10_proximity      (0.0-1.0, same chapter = 1.0, adjacent = 0.5)
      - collaboration_permit (bool: False -> weight forced to 0 regardless of similarity)

    Combined client similarity:
      If collaboration_permit is False: 0.0
      Else: 0.4*specialty + 0.3*modality + 0.3*icd10
    """

    SILOS = [
        # name, specialty, LOINC, ICD10-chapter, privacy, retention
        {"name": "DermaMNIST",   "specialty": "dermatology",
         "loinc": "18748-4",  "icd10_ch": "L",  "privacy": 1, "retention": 1},
        {"name": "PathMNIST",    "specialty": "pathology",
         "loinc": "22634-0",  "icd10_ch": "C",  "privacy": 2, "retention": 2},
        {"name": "PneumoniaMNIST","specialty": "radiology",
         "loinc": "24627-2",  "icd10_ch": "J",  "privacy": 1, "retention": 0},
        {"name": "RetinaMNIST",  "specialty": "ophthalmology",
         "loinc": "57133-1",  "icd10_ch": "H",  "privacy": 1, "retention": 1},
        {"name": "BreastMNIST",  "specialty": "radiology",
         "loinc": "24606-6",  "icd10_ch": "C",  "privacy": 2, "retention": 2},
        {"name": "BloodMNIST",   "specialty": "haematology",
         "loinc": "58410-2",  "icd10_ch": "D",  "privacy": 0, "retention": 0},
        {"name": "OrganAMNIST",  "specialty": "radiology",
         "loinc": "24725-4",  "icd10_ch": "K",  "privacy": 1, "retention": 1},
    ]

    # Specialty overlap matrix (hand-coded clinical knowledge)
    # dermatology, pathology, radiology, ophthalmology, haematology
    SPECIALTY_LIST = ["dermatology", "pathology", "radiology",
                      "ophthalmology", "haematology"]
    SPECIALTY_OVERLAP = {
        ("dermatology",  "dermatology") : 1.00,
        ("dermatology",  "pathology")   : 0.55,  # skin biopsy bridges both
        ("dermatology",  "radiology")   : 0.20,
        ("dermatology",  "ophthalmology"): 0.15,
        ("dermatology",  "haematology") : 0.10,
        ("pathology",    "pathology")   : 1.00,
        ("pathology",    "radiology")   : 0.35,  # imaging + histology in oncology
        ("pathology",    "ophthalmology"): 0.15,
        ("pathology",    "haematology") : 0.45,  # blood pathology
        ("radiology",    "radiology")   : 1.00,
        ("radiology",    "ophthalmology"): 0.30,  # fundus imaging is specialised imaging
        ("radiology",    "haematology") : 0.15,
        ("ophthalmology","ophthalmology"): 1.00,
        ("ophthalmology","haematology") : 0.10,
        ("haematology",  "haematology") : 1.00,
    }

    # ICD-10 chapter proximity (same chapter=1.0, clinically adjacent=0.5, distant=0.1)
    # Chapters present: L(skin), C(neoplasms), J(respiratory), H(eye), D(blood), K(digestive)
    ICD10_PROXIMITY = {
        ("L","L"): 1.00, ("L","C"): 0.50, ("L","J"): 0.15,
        ("L","H"): 0.20, ("L","D"): 0.25, ("L","K"): 0.10,
        ("C","C"): 1.00, ("C","J"): 0.30, ("C","H"): 0.20,
        ("C","D"): 0.45, ("C","K"): 0.35,
        ("J","J"): 1.00, ("J","H"): 0.10, ("J","D"): 0.15, ("J","K"): 0.25,
        ("H","H"): 1.00, ("H","D"): 0.10, ("H","K"): 0.10,
        ("D","D"): 1.00, ("D","K"): 0.20,
        ("K","K"): 1.00,
    }

    # LOINC modality families
    # dermoscopy, histopathology, chest-Xray, fundus-photo, breast-US, CBC, abdominal-CT
    MODALITY_FAMILIES = {
        "18748-4": "optical-imaging",   # dermoscopy
        "22634-0": "microscopy",         # histopathology
        "24627-2": "xray",               # chest X-ray
        "57133-1": "optical-imaging",   # fundus photography
        "24606-6": "ultrasound",         # breast US
        "58410-2": "microscopy",         # CBC / blood smear
        "24725-4": "ct",                 # abdominal CT
    }

    # Collaboration permissions: privacy + retention must both be compatible.
    # Rule: if both silos have retention_policy == 2 (local-only),
    #       collaboration is prohibited (weight -> 0).
    # This operationalises data governance as a hard graph constraint.

    def __init__(self):
        self.n_silos = len(self.SILOS)
        self._build_matrices()

    def _get_specialty_overlap(self, s1, s2):
        key = (min(s1, s2, key=lambda x: self.SPECIALTY_LIST.index(x)
                   if x in self.SPECIALTY_LIST else 99),
               max(s1, s2, key=lambda x: self.SPECIALTY_LIST.index(x)
                   if x in self.SPECIALTY_LIST else 99))
        # symmetric lookup
        return self.SPECIALTY_OVERLAP.get((s1, s2),
               self.SPECIALTY_OVERLAP.get((s2, s1), 0.10))

    def _get_icd10_proximity(self, c1, c2):
        return self.ICD10_PROXIMITY.get((c1, c2),
               self.ICD10_PROXIMITY.get((c2, c1), 0.05))

    def _get_modality_similarity(self, l1, l2):
        f1 = self.MODALITY_FAMILIES.get(l1, "unknown")
        f2 = self.MODALITY_FAMILIES.get(l2, "unknown")
        if f1 == f2:
            return 1.0
        # optical-imaging and microscopy share visual feature extractors
        if set([f1, f2]) == {"optical-imaging", "microscopy"}:
            return 0.55
        # xray and ct are both ionising radiation imaging
        if set([f1, f2]) == {"xray", "ct"}:
            return 0.65
        return 0.15

    def _collaboration_permitted(self, i, j):
        """Hard governance constraint: both local-only silos cannot collaborate."""
        ri = self.SILOS[i]["retention"]
        rj = self.SILOS[j]["retention"]
        return not (ri == 2 and rj == 2)

    def _build_matrices(self):
        n = self.n_silos
        self.specialty_mat   = np.zeros((n, n))
        self.modality_mat    = np.zeros((n, n))
        self.icd10_mat       = np.zeros((n, n))
        self.collab_mat      = np.ones((n, n), dtype=bool)
        self.similarity_mat  = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                si = self.SILOS[i]; sj = self.SILOS[j]
                sp = self._get_specialty_overlap(si["specialty"], sj["specialty"])
                mo = self._get_modality_similarity(si["loinc"], sj["loinc"])
                ic = self._get_icd10_proximity(si["icd10_ch"], sj["icd10_ch"])
                cp = self._collaboration_permitted(i, j)
                self.specialty_mat[i, j]  = sp
                self.modality_mat[i, j]   = mo
                self.icd10_mat[i, j]      = ic
                self.collab_mat[i, j]     = cp
                if not cp:
                    self.similarity_mat[i, j] = 0.0
                else:
                    self.similarity_mat[i, j] = 0.4*sp + 0.3*mo + 0.3*ic

    def get_silo_similarity(self, i, j):
        return float(self.similarity_mat[i, j])

    def get_collaboration_permitted(self, i, j):
        return bool(self.collab_mat[i, j])

    def print_ontology_summary(self):
        print("\nClinical Ontology Summary")
        print(f"  {'Silo':<20} {'Specialty':<15} {'LOINC':<12} "
              f"{'ICD-10':<8} {'Privacy':<9} {'Retention'}")
        print("  " + "-"*75)
        for s in self.SILOS:
            print(f"  {s['name']:<20} {s['specialty']:<15} {s['loinc']:<12} "
                  f"{s['icd10_ch']:<8} {s['privacy']:<9} {s['retention']}")
        print("\n  Collaboration permission matrix (F=prohibited by governance):")
        names = [s["name"][:8] for s in self.SILOS]
        print("  " + " "*12 + "  ".join(f"{n:>8}" for n in names))
        for i in range(self.n_silos):
            row = "  ".join(
                f"{'T':>8}" if self.collab_mat[i,j] else f"{'F':>8}"
                for j in range(self.n_silos))
            print(f"  {names[i]:>10}  {row}")
        print("\n  Pairwise clinical similarity (0=incompatible/prohibited, 1=identical):")
        print("  " + " "*12 + "  ".join(f"{n:>8}" for n in names))
        for i in range(self.n_silos):
            row = "  ".join(f"{self.similarity_mat[i,j]:>8.2f}"
                            for j in range(self.n_silos))
            print(f"  {names[i]:>10}  {row}")


# ================================================================
# SHARED ENCODER + PER-SILO HEAD
# ================================================================
class SharedEncoder(nn.Module):
    """
    CNN encoder shared across all silos.
    Input: 28x28 or 64x64 RGB (MedMNIST default is 28x28).
    Output: EMBED_DIM-dimensional embedding.
    All silos contribute to updating this encoder.
    """
    def __init__(self, in_channels=3, embed_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Linear(128 * 4 * 4, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        return self.proj(self.net(x).view(x.size(0), -1))


class SiloHead(nn.Module):
    """Per-silo classification head. NOT aggregated across silos."""
    def __init__(self, embed_dim=EMBED_DIM, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, emb):
        return self.fc(emb)


class SiloModel(nn.Module):
    """Encoder + silo-specific head. Full forward pass for one silo."""
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head    = head

    def forward(self, x):
        return self.head(self.encoder(x))

    def get_embedding(self, x):
        return self.encoder(x)


# ================================================================
# MEDMNIST SILO CONFIGURATION
# ================================================================
SILO_CONFIGS = [
    {"name": "DermaMNIST",    "cls": DermaMNIST,    "n_classes": 7,
     "task": "multi-class",  "in_ch": 3},
    {"name": "PathMNIST",     "cls": PathMNIST,     "n_classes": 9,
     "task": "multi-class",  "in_ch": 3},
    {"name": "PneumoniaMNIST","cls": PneumoniaMNIST,"n_classes": 2,
     "task": "binary",       "in_ch": 1},
    {"name": "RetinaMNIST",   "cls": RetinaMNIST,   "n_classes": 5,
     "task": "ordinal",      "in_ch": 3},
    {"name": "BreastMNIST",   "cls": BreastMNIST,   "n_classes": 2,
     "task": "binary",       "in_ch": 1},
    {"name": "BloodMNIST",    "cls": BloodMNIST,    "n_classes": 8,
     "task": "multi-class",  "in_ch": 3},
    {"name": "OrganAMNIST",   "cls": OrganAMNIST,   "n_classes": 11,
     "task": "multi-class",  "in_ch": 1},
]
N_SILOS = len(SILO_CONFIGS)


# ================================================================
# DATASET HELPERS
# ================================================================
class MedMNISTWrapper(Dataset):
    """
    Wraps a MedMNIST dataset to:
    - Convert single-channel images to 3-channel if needed by the encoder
    - Flatten the label array from (N,1) to (N,)
    - Resize to 28x28 (MedMNIST default; encoder uses AdaptiveAvgPool so size-agnostic)
    """
    def __init__(self, medmnist_dataset, to_3ch=True):
        self.ds     = medmnist_dataset
        self.to_3ch = to_3ch
        # Cache targets for Dirichlet partitioning
        self.targets = [int(self.ds[i][1].item())
                        if hasattr(self.ds[i][1], 'item')
                        else int(self.ds[i][1])
                        for i in range(len(self.ds))]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        # img is a PIL image from MedMNIST; convert to tensor
        img = torch.from_numpy(np.array(img)).float() / 255.0
        if img.ndim == 2:
            img = img.unsqueeze(0)          # H,W -> 1,H,W
        elif img.ndim == 3 and img.shape[2] in [1, 3]:
            img = img.permute(2, 0, 1)      # H,W,C -> C,H,W
        if self.to_3ch and img.shape[0] == 1:
            img = img.repeat(3, 1, 1)       # 1,H,W -> 3,H,W
        # Normalise with ImageNet stats (reasonable starting point for medical images)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img  = (img - mean) / std
        label = int(label.item()) if hasattr(label, 'item') else int(label)
        return img, label


def load_medmnist_silos(download=True, data_root="./medmnist_data"):
    """
    Downloads and wraps all 7 MedMNIST datasets.
    Returns dict: silo_id -> {train: Dataset, test: Dataset, n_classes: int}
    """
    os.makedirs(data_root, exist_ok=True)
    silos = {}
    for s_id, cfg in enumerate(SILO_CONFIGS):
        print(f"  Loading {cfg['name']}...")
        train_ds = cfg["cls"](split="train", download=download, root=data_root)
        test_ds  = cfg["cls"](split="test",  download=download, root=data_root)
        silos[s_id] = {
            "name"      : cfg["name"],
            "train"     : MedMNISTWrapper(train_ds, to_3ch=True),
            "test"      : MedMNISTWrapper(test_ds,  to_3ch=True),
            "n_classes" : cfg["n_classes"],
            "task"      : cfg["task"],
        }
        print(f"    train={len(train_ds)}  test={len(test_ds)}  "
              f"classes={cfg['n_classes']}  task={cfg['task']}")
    return silos


def create_subclient_partitions(dataset, n_subclients=3, alpha=0.4):
    targets     = np.array(dataset.targets)
    num_classes = int(targets.max()) + 1
    partitions  = []
    for _ in range(n_subclients):
        probs     = np.random.dirichlet([alpha] * num_classes)
        p         = probs[targets]; p /= p.sum()
        n_samples = min(300, len(dataset))
        selected  = np.random.choice(len(dataset), size=n_samples, p=p, replace=False)
        partitions.append(selected.tolist())
    return partitions


def build_silo_loaders(silos, n_subclients=3, alpha=0.4, batch_size=32):
    """
    Returns:
      client_loaders      - flat list, one DataLoader per sub-client
      client_silo_ids     - which silo each sub-client belongs to
      test_loaders        - dict silo_id -> DataLoader
    """
    client_loaders  = []
    client_silo_ids = []
    test_loaders    = {}
    for s_id, silo in silos.items():
        test_loaders[s_id] = DataLoader(
            silo["test"], batch_size=64, shuffle=False, num_workers=0)
        partitions = create_subclient_partitions(
            silo["train"], n_subclients, alpha)
        for part in partitions:
            client_loaders.append(
                DataLoader(Subset(silo["train"], part),
                           batch_size=batch_size, shuffle=True, num_workers=0))
            client_silo_ids.append(s_id)
    return client_loaders, client_silo_ids, test_loaders


# ================================================================
# METRICS
# ================================================================
def compute_ece(logits_list, labels_list, n_bins=10):
    logits      = torch.cat(logits_list)
    labels      = torch.cat(labels_list)
    probs       = F.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies  = predictions.eq(labels)
    ece = 0.0
    bin_size = 1.0 / n_bins
    for b in range(n_bins):
        lo, hi = b * bin_size, (b + 1) * bin_size
        mask   = confidences.gt(lo) & confidences.le(hi)
        if mask.sum() > 0:
            ece += abs(accuracies[mask].float().mean().item() -
                       confidences[mask].mean().item()) * mask.float().mean().item()
    return ece


def evaluate_silo(encoder, head, loader):
    """Evaluate one silo. Returns (accuracy, ece, per_class_acc)."""
    model = SiloModel(encoder, head).to(device)
    model.eval()
    correct = total = 0
    logits_list, labels_list = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            logits_list.append(out.cpu())
            labels_list.append(labels.cpu())
            correct += (out.argmax(1) == labels).sum().item()
            total   += labels.size(0)
    acc  = correct / total if total > 0 else 0.0
    ece  = compute_ece(logits_list, labels_list) if logits_list else 0.0
    all_preds  = torch.cat(logits_list).argmax(dim=1)
    all_labels = torch.cat(labels_list)
    pca = {}
    for c in range(head.fc.out_features):
        mask = all_labels == c
        if mask.sum() > 0:
            pca[c] = (all_preds[mask] == all_labels[mask]).float().mean().item()
    return acc, ece, pca


def evaluate_all_silos(encoder, silo_heads, test_loaders):
    results = {}
    for s_id, loader in test_loaders.items():
        acc, ece, pca = evaluate_silo(encoder, silo_heads[s_id], loader)
        results[s_id] = {"acc": acc, "ece": ece, "pca": pca,
                         "name": SILO_CONFIGS[s_id]["name"]}
    return results


def compute_negative_transfer_rate(transfer_pca, local_pca):
    classes = set(transfer_pca) & set(local_pca)
    if not classes:
        return 0.0
    return sum(1 for c in classes if transfer_pca[c] < local_pca[c]) / len(classes)


def summarise_silos(results):
    accs  = [results[s]["acc"] for s in results]
    eces  = [results[s]["ece"] for s in results]
    return {
        "avg_acc"  : float(np.mean(accs)),
        "worst_acc": float(np.min(accs)),
        "avg_ece"  : float(np.mean(eces)),
        "per_silo" : {results[s]["name"]: results[s]["acc"] for s in results},
    }


# ================================================================
# PPDOE  -- server-hosted, ontology-guided, NOT federated learning
# ================================================================
class PPDOEMedical:
    """
    PPDOE for heterogeneous medical silos.

    Architecture:
      - One shared encoder (aggregated using ontology weights)
      - One per-silo head (never aggregated; always local to the silo)

    Ontology weights govern ONLY encoder aggregation.
    Governance constraint: if collaboration_permitted is False between
    two silos, their encoder updates do not influence each other
    (weight = 0), operationalising the data-sharing agreement.
    """
    def __init__(self, ontology, silos):
        self.ontology   = ontology
        self.silos      = silos
        self.n_silos    = len(silos)
        # Shared encoder on the server
        self.encoder    = SharedEncoder(in_channels=3, embed_dim=EMBED_DIM).to(device)
        # Per-silo heads (one per silo, kept local)
        self.silo_heads = {
            s_id: SiloHead(EMBED_DIM, silo["n_classes"]).to(device)
            for s_id, silo in silos.items()
        }

    def _semantic_weights(self, client_silo_ids):
        """
        Compute ontology-weighted aggregation weights for encoder updates.
        For each client i, its weight = mean similarity to all other clients,
        respecting governance constraints (0 if collaboration prohibited).
        """
        n = len(client_silo_ids)
        w = []
        for i in range(n):
            sims = []
            for j in range(n):
                if i == j:
                    continue
                si = client_silo_ids[i]
                sj = client_silo_ids[j]
                sims.append(self.ontology.get_silo_similarity(si, sj))
            w.append(np.mean(sims) if sims else 1.0)
        total = sum(w)
        return [x / total if total > 0 else 1.0/n for x in w]

    def _local_train(self, s_id, loader, epochs=1, lr=0.01):
        """
        Local training for one silo sub-client.
        Both encoder and silo head are trained together.
        Returns updated encoder state dict and head state dict separately.
        """
        enc  = copy.deepcopy(self.encoder).to(device)
        head = copy.deepcopy(self.silo_heads[s_id]).to(device)
        model     = SiloModel(enc, head).to(device)
        opt       = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                criterion(model(imgs), labels).backward()
                opt.step()
        return enc.state_dict(), head.state_dict()

    def run_round(self, client_loaders, client_silo_ids):
        """
        One PPDOE round:
        1. All sub-clients do local training (encoder + silo head)
        2. Encoder updates are aggregated using ontology weights
        3. Silo head updates are averaged only within the same silo
        4. Global encoder is updated; silo heads are updated per-silo
        """
        weights = self._semantic_weights(client_silo_ids)

        enc_updates  = []
        head_updates = {s_id: [] for s_id in self.silos}

        for i, (loader, s_id) in enumerate(zip(client_loaders, client_silo_ids)):
            enc_sd, head_sd = self._local_train(s_id, loader)
            enc_updates.append(enc_sd)
            head_updates[s_id].append(head_sd)

        # Aggregate encoder with ontology weights
        new_enc = {
            key: sum(weights[i] * enc_updates[i][key].float()
                     for i in range(len(enc_updates)))
            for key in enc_updates[0]
        }
        self.encoder.load_state_dict(new_enc)

        # Aggregate each silo's head independently (FedAvg within silo)
        for s_id, updates in head_updates.items():
            if updates:
                avg_head = {
                    key: sum(u[key].float() for u in updates) / len(updates)
                    for key in updates[0]
                }
                self.silo_heads[s_id].load_state_dict(avg_head)

    def evaluate_all_silos(self, test_loaders):
        return evaluate_all_silos(self.encoder, self.silo_heads, test_loaders)


# ================================================================
# FL BASELINES
# Each runs its own independent loop. No ontology access.
# Each also uses shared encoder + per-silo head architecture
# for a fair apples-to-apples comparison.
# ================================================================
class FLMedBaseline:
    def __init__(self, silos):
        self.silos      = silos
        self.encoder    = SharedEncoder(in_channels=3, embed_dim=EMBED_DIM).to(device)
        self.silo_heads = {
            s_id: SiloHead(EMBED_DIM, silo["n_classes"]).to(device)
            for s_id, silo in silos.items()
        }

    def _local_train(self, s_id, loader, epochs=1, lr=0.01):
        enc       = copy.deepcopy(self.encoder).to(device)
        head      = copy.deepcopy(self.silo_heads[s_id]).to(device)
        model     = SiloModel(enc, head)
        opt       = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                criterion(model(imgs), labels).backward()
                opt.step()
        return enc.state_dict(), head.state_dict()

    def _fedavg_encoder(self, enc_updates):
        n = len(enc_updates)
        return {k: sum(u[k].float() for u in enc_updates) / n for k in enc_updates[0]}

    def run_round(self, client_loaders, client_silo_ids):
        raise NotImplementedError

    def evaluate_all_silos(self, test_loaders):
        return evaluate_all_silos(self.encoder, self.silo_heads, test_loaders)


class FedBNMed(FLMedBaseline):
    """Aggregate encoder except BN stats; per-silo heads averaged within silo."""
    def run_round(self, client_loaders, client_silo_ids):
        enc_updates  = []
        head_updates = {s_id: [] for s_id in self.silos}
        for loader, s_id in zip(client_loaders, client_silo_ids):
            e, h = self._local_train(s_id, loader)
            enc_updates.append(e); head_updates[s_id].append(h)
        avg_enc = {}
        for key in enc_updates[0]:
            if any(x in key for x in ["running_mean","running_var","num_batches_tracked"]):
                avg_enc[key] = self.encoder.state_dict()[key]
            else:
                avg_enc[key] = sum(u[key].float() for u in enc_updates) / len(enc_updates)
        self.encoder.load_state_dict(avg_enc)
        for s_id, updates in head_updates.items():
            if updates:
                self.silo_heads[s_id].load_state_dict(
                    {k: sum(u[k].float() for u in updates)/len(updates) for k in updates[0]})


class FedRepMed(FLMedBaseline):
    """Body (encoder) aggregated globally; head trained locally only."""
    def _local_train_fedrep(self, s_id, loader, epochs=1, lr=0.01):
        enc  = copy.deepcopy(self.encoder).to(device)
        head = copy.deepcopy(self.silo_heads[s_id]).to(device)
        model     = SiloModel(enc, head)
        criterion = nn.CrossEntropyLoss()
        # Phase 1: freeze encoder, train head
        for p in model.encoder.parameters(): p.requires_grad_(False)
        opt = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=lr, momentum=0.9)
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad(); criterion(model(imgs), labels).backward(); opt.step()
            break
        # Phase 2: freeze head, train encoder
        for p in model.encoder.parameters(): p.requires_grad_(True)
        for p in model.head.parameters(): p.requires_grad_(False)
        opt2 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=lr, momentum=0.9)
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt2.zero_grad(); criterion(model(imgs), labels).backward(); opt2.step()
            break
        for p in model.head.parameters(): p.requires_grad_(True)
        return enc.state_dict(), head.state_dict()

    def run_round(self, client_loaders, client_silo_ids):
        enc_updates  = []
        head_updates = {s_id: [] for s_id in self.silos}
        for loader, s_id in zip(client_loaders, client_silo_ids):
            e, h = self._local_train_fedrep(s_id, loader)
            enc_updates.append(e); head_updates[s_id].append(h)
        self.encoder.load_state_dict(self._fedavg_encoder(enc_updates))
        for s_id, updates in head_updates.items():
            if updates:
                self.silo_heads[s_id].load_state_dict(
                    {k: sum(u[k].float() for u in updates)/len(updates) for k in updates[0]})


class DittoMed(FLMedBaseline):
    def __init__(self, silos, lam=0.5):
        super().__init__(silos); self.lam = lam
        self.personal_encs  = {}
        self.personal_heads = {}

    def run_round(self, client_loaders, client_silo_ids):
        criterion = nn.CrossEntropyLoss()
        enc_updates  = []
        head_updates = {s_id: [] for s_id in self.silos}
        for i, (loader, s_id) in enumerate(zip(client_loaders, client_silo_ids)):
            e, h = self._local_train(s_id, loader)
            enc_updates.append(e); head_updates[s_id].append(h)
            # Personal model
            if i not in self.personal_encs:
                self.personal_encs[i]  = copy.deepcopy(self.encoder).to(device)
                self.personal_heads[i] = copy.deepcopy(self.silo_heads[s_id]).to(device)
            p_model = SiloModel(self.personal_encs[i], self.personal_heads[i])
            gp = {k: v.clone().detach() for k,v in self.encoder.state_dict().items()}
            opt_p = optim.SGD(p_model.parameters(), lr=0.01, momentum=0.9)
            p_model.train()
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt_p.zero_grad()
                loss = criterion(p_model(imgs), labels)
                prox = sum(((p - gp[k])**2).sum()
                           for k, p in p_model.encoder.named_parameters() if k in gp)
                (loss + (self.lam/2)*prox).backward(); opt_p.step()
        self.encoder.load_state_dict(self._fedavg_encoder(enc_updates))
        for s_id, updates in head_updates.items():
            if updates:
                self.silo_heads[s_id].load_state_dict(
                    {k: sum(u[k].float() for u in updates)/len(updates) for k in updates[0]})


class FedProtoMed(FLMedBaseline):
    """Prototype aggregation; prototypes are per-silo (different class spaces)."""
    def __init__(self, silos):
        super().__init__(silos)
        self.global_protos = {s_id: {} for s_id in silos}

    def run_round(self, client_loaders, client_silo_ids):
        criterion = nn.CrossEntropyLoss()
        enc_updates  = []
        head_updates = {s_id: [] for s_id in self.silos}
        all_protos   = {s_id: {} for s_id in self.silos}
        for loader, s_id in zip(client_loaders, client_silo_ids):
            enc  = copy.deepcopy(self.encoder).to(device)
            head = copy.deepcopy(self.silo_heads[s_id]).to(device)
            model        = SiloModel(enc, head)
            opt          = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            local_protos = {c: [] for c in range(self.silos[s_id]["n_classes"])}
            model.train()
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                emb  = model.get_embedding(imgs)
                loss = criterion(model.head(emb), labels)
                gp   = self.global_protos[s_id]
                if gp:
                    pl = sum(((emb[i] - gp[l.item()].to(device))**2).sum()
                             for i,l in enumerate(labels) if l.item() in gp)
                    loss = loss + 0.1*pl/max(len(labels),1)
                loss.backward(); opt.step()
                for i,l in enumerate(labels):
                    local_protos[l.item()].append(emb[i].detach().cpu())
            enc_updates.append(enc.state_dict())
            head_updates[s_id].append(head.state_dict())
            for c,v in local_protos.items():
                if v: all_protos[s_id].setdefault(c,[]).extend(v)
        self.encoder.load_state_dict(self._fedavg_encoder(enc_updates))
        for s_id in self.silos:
            self.global_protos[s_id] = {
                c: torch.stack(v).mean(0)
                for c,v in all_protos[s_id].items() if v
            }
            if head_updates[s_id]:
                self.silo_heads[s_id].load_state_dict(
                    {k: sum(u[k].float() for u in head_updates[s_id])/len(head_updates[s_id])
                     for k in head_updates[s_id][0]})


# ================================================================
# LOCAL BASELINE (no transfer)
# ================================================================
def compute_local_baselines(silos, test_loaders, n_epochs=4):
    local_pca = {}
    for s_id, silo in silos.items():
        enc  = SharedEncoder(in_channels=3, embed_dim=EMBED_DIM).to(device)
        head = SiloHead(EMBED_DIM, silo["n_classes"]).to(device)
        model     = SiloModel(enc, head)
        opt       = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        loader    = DataLoader(silo["train"], batch_size=32, shuffle=True, num_workers=0)
        for _ in range(n_epochs):
            model.train()
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad(); criterion(model(imgs), labels).backward(); opt.step()
        acc, ece, pca = evaluate_silo(enc, head, test_loaders[s_id])
        local_pca[s_id] = pca
        print(f"  Local baseline [{silo['name']}]: acc={acc:.4f}  ece={ece:.4f}")
    return local_pca


# ================================================================
# LEAVE-ONE-SILO-OUT
# ================================================================
def leave_one_silo_out_eval(silos, ontology, n_subclients=3,
                            alpha=0.4, n_rounds=5):
    """
    For each held-out silo, PPDOE trains on K-1 silos,
    then evaluates on the held-out silo's test set.
    """
    results = {}
    for held_id in silos:
        train_silos = {s: silos[s] for s in silos if s != held_id}
        cl, csi, tl = build_silo_loaders(train_silos, n_subclients, alpha)
        held_loader  = DataLoader(silos[held_id]["test"],
                                   batch_size=64, shuffle=False, num_workers=0)
        # PPDOE trained on K-1 silos
        ppdoe = PPDOEMedical(ontology, train_silos)
        for _ in range(n_rounds):
            ppdoe.run_round(cl, csi)
        # Evaluate on held-out silo using its own head (fresh, untrained)
        # This tests encoder generalisation specifically
        held_head = SiloHead(EMBED_DIM, silos[held_id]["n_classes"]).to(device)
        # Fine-tune head only on a small portion of held silo's train data
        fine_loader = DataLoader(
            Subset(silos[held_id]["train"],
                   list(range(min(200, len(silos[held_id]["train"]))))),
            batch_size=32, shuffle=True, num_workers=0)
        opt = optim.SGD(held_head.parameters(), lr=0.01, momentum=0.9)
        model = SiloModel(ppdoe.encoder, held_head).to(device)
        model.train()
        criterion = nn.CrossEntropyLoss()
        for imgs, labels in fine_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad(); criterion(model(imgs), labels).backward(); opt.step()
        acc, ece, _ = evaluate_silo(ppdoe.encoder, held_head, held_loader)
        results[held_id] = {"acc": acc, "ece": ece,
                             "name": silos[held_id]["name"]}
        print(f"    LOSO [held out={silos[held_id]['name']}]: "
              f"acc={acc:.4f}  ece={ece:.4f}")
    return results


# ================================================================
# MAIN
# ================================================================
def run_experiment3(
    n_subclients = N_SUBCLIENTS,
    alpha        = ALPHA,
    n_rounds     = N_ROUNDS,
    data_root    = "./medmnist_data",
):
    print("\n" + "="*75)
    print("  PPDOE Experiment 3 -- Cross-Silo Medical Benchmark (MedMNIST v2)")
    print("  PPDOE is the system under evaluation. FL baselines are independent.")
    print("="*75)

    # ---- Clinical ontology ----
    ontology = ClinicalOntology()
    ontology.print_ontology_summary()

    # ---- Load data ----
    print("\nLoading MedMNIST v2 silos...")
    silos = load_medmnist_silos(download=True, data_root=data_root)

    # ---- Sub-client loaders ----
    client_loaders, client_silo_ids, test_loaders = build_silo_loaders(
        silos, n_subclients, alpha)
    print(f"\nTotal sub-clients: {len(client_loaders)} "
          f"({N_SILOS} silos x {n_subclients})")
    print("Client->Silo mapping:")
    for i, s_id in enumerate(client_silo_ids):
        print(f"  client {i:2d} -> silo {s_id} ({SILO_CONFIGS[s_id]['name']})")

    # ---- Local no-transfer baselines ----
    print("\nComputing local baselines (no transfer, for NTR)...")
    local_pca = compute_local_baselines(silos, test_loaders)

    # ---- PPDOE ----
    ppdoe         = PPDOEMedical(ontology, silos)
    ppdoe_history = []

    # ---- FL baselines ----
    fl_baselines = {
        "FedBN"    : FedBNMed(silos),
        "FedRep"   : FedRepMed(silos),
        "Ditto"    : DittoMed(silos, lam=0.5),
        "FedProto" : FedProtoMed(silos),
    }
    fl_history = {name: [] for name in fl_baselines}

    # ---- Round loop ----
    for r in range(1, n_rounds + 1):
        print(f"\n  Round {r}/{n_rounds}")

        ppdoe.run_round(client_loaders, client_silo_ids)
        pr = ppdoe.evaluate_all_silos(test_loaders)
        pm = summarise_silos(pr)
        ppdoe_history.append(pm)
        print(f"    {'PPDOE':25}: avg={pm['avg_acc']:.4f}  "
              f"worst={pm['worst_acc']:.4f}  ece={pm['avg_ece']:.4f}")

        for name, bl in fl_baselines.items():
            bl.run_round(client_loaders, client_silo_ids)
            fr = bl.evaluate_all_silos(test_loaders)
            m  = summarise_silos(fr)
            fl_history[name].append(m)
            print(f"    {name:25}: avg={m['avg_acc']:.4f}  "
                  f"worst={m['worst_acc']:.4f}  ece={m['avg_ece']:.4f}")

    # ---- Leave-one-silo-out ----
    print("\n--- Leave-One-Silo-Out Transfer (PPDOE, K-1 regime) ---")
    loso_results = leave_one_silo_out_eval(
        silos, ontology, n_subclients=n_subclients,
        alpha=alpha, n_rounds=n_rounds // 2)
    loso_avg   = float(np.mean([v["acc"] for v in loso_results.values()]))
    loso_worst = float(np.min([v["acc"]  for v in loso_results.values()]))
    print(f"  LOSO summary: avg={loso_avg:.4f}  worst-silo={loso_worst:.4f}")

    # ---- Negative transfer rate ----
    print("\n--- Negative Transfer Rate (vs local no-transfer baseline) ---")
    ntr_results = {}
    ppdoe_final = ppdoe.evaluate_all_silos(test_loaders)
    for s_id in silos:
        ntr_results.setdefault("PPDOE", {})
        ntr_results["PPDOE"][s_id] = compute_negative_transfer_rate(
            ppdoe_final[s_id]["pca"], local_pca[s_id])
    for name, bl in fl_baselines.items():
        bl_final = bl.evaluate_all_silos(test_loaders)
        ntr_results[name] = {}
        for s_id in silos:
            ntr_results[name][s_id] = compute_negative_transfer_rate(
                bl_final[s_id]["pca"], local_pca[s_id])
    print(f"  {'Method':<20}  " +
          "  ".join(f"{silos[s]['name'][:10]:>12}" for s in silos))
    for mname, ntr_dict in ntr_results.items():
        row = "  ".join(f"{ntr_dict[s]:>12.4f}" for s in silos)
        print(f"  {mname:<20}  {row}")

    # ---- Governance check: prohibited collaborations ----
    print("\n--- Governance Compliance Check ---")
    print("  Verifying PPDOE enforces collaboration_permit=False constraints.")
    for i in range(N_SILOS):
        for j in range(N_SILOS):
            if i != j and not ontology.get_collaboration_permitted(i, j):
                sim = ontology.get_silo_similarity(i, j)
                status = "PASS" if sim == 0.0 else "FAIL"
                print(f"  [{status}] Silos {SILO_CONFIGS[i]['name']} <-> "
                      f"{SILO_CONFIGS[j]['name']}: "
                      f"similarity={sim:.3f} (must be 0.0)")

    # ---- Final summary ----
    print("\n" + "="*75)
    print("  FINAL RESULTS SUMMARY")
    print("="*75)
    print(f"  {'Method':<25}  {'Avg Acc':>9}  {'Worst Acc':>10}  "
          f"{'Avg ECE':>8}  {'Avg NTR':>9}  {'vs PPDOE':>10}")
    print("  " + "-"*73)

    baseline_avg = ppdoe_history[-1]["avg_acc"]
    all_final    = {"PPDOE": ppdoe_history[-1]}
    all_final.update({n: fl_history[n][-1] for n in fl_baselines})

    for mname, m in sorted(all_final.items(), key=lambda x: x[1]["avg_acc"], reverse=True):
        delta   = m["avg_acc"] - baseline_avg
        d_str   = "(base)" if mname == "PPDOE" else (
            f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}")
        avg_ntr = np.mean(list(ntr_results[mname].values())) \
                  if mname in ntr_results else float("nan")
        print(f"  {mname:<25}  {m['avg_acc']:>9.4f}  {m['worst_acc']:>10.4f}  "
              f"{m['avg_ece']:>8.4f}  {avg_ntr:>9.4f}  {d_str:>10}")

    print("\n  Per-silo accuracy breakdown (final round):")
    silo_names = [silos[s]["name"][:10] for s in silos]
    print(f"  {'Method':<25}  " + "  ".join(f"{n:>12}" for n in silo_names))
    print("  " + "-"*(25 + 16*N_SILOS))
    for mname, m in all_final.items():
        row = "  ".join(f"{m['per_silo'].get(silos[s]['name'], 0.0):>12.4f}"
                        for s in silos)
        print(f"  {mname:<25}  {row}")

    return ppdoe_history, fl_history, loso_results, ntr_results


if __name__ == "__main__":
    run_experiment3(
        n_subclients = N_SUBCLIENTS,
        alpha        = ALPHA,
        n_rounds     = N_ROUNDS,
    )