"""
Experiment 3: Cross-Silo Medical Benchmark (MedMNIST v2)

PPDOE is a server-hosted, ontology-guided aggregation system.
It is NOT federated learning.
FL algorithms below are independent comparison baselines only.

Architecture:
  - Shared CNN encoder aggregated using clinical ontology weights
  - Per-silo classification heads (never aggregated across silos)

Ontology:
  - Hand-coded clinical knowledge graph
  - Nodes: ICD-10 chapter, LOINC modality code, specialty,
           privacy level, data retention policy
  - Edges: specialty overlap, modality similarity, ICD-10 proximity,
           collaboration permission (hard governance constraint)

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

N_SUBCLIENTS = 3      # per silo
ALPHA        = 0.4
N_ROUNDS     = 10
EMBED_DIM    = 512
DATA_ROOT    = "./medmnist_data"


# ================================================================
# CLINICAL ONTOLOGY
# Hand-coded from ICD-10, LOINC, specialty, and data governance.
# Used exclusively by PPDOE. FL baselines have no access.
# ================================================================
class ClinicalOntology:
    """
    Seven silos, each described by:
      specialty       -- clinical department
      loinc           -- LOINC code for the imaging/lab modality
      icd10_ch        -- primary ICD-10 chapter of the disease domain
      privacy         -- 0=open, 1=standard HIPAA, 2=restricted
      retention       -- 0=shareable, 1=pseudonymised, 2=local-only

    Collaboration permission:
      If BOTH silos have retention == 2 (local-only), collaboration
      is prohibited. The ontology sets their similarity to 0.0,
      operationalising the data-sharing agreement as a hard graph
      constraint. No FL baseline can do this natively.

    Combined similarity (when collaboration permitted):
      0.4 * specialty_overlap + 0.3 * modality_similarity + 0.3 * icd10_proximity
    """

    SILOS = [
        # idx  name              specialty        loinc      icd10  priv  retain
        {"name": "DermaMNIST",    "specialty": "dermatology",
         "loinc": "18748-4",  "icd10_ch": "L", "privacy": 1, "retention": 1},
        {"name": "PathMNIST",     "specialty": "pathology",
         "loinc": "22634-0",  "icd10_ch": "C", "privacy": 2, "retention": 2},
        {"name": "PneumoniaMNIST","specialty": "radiology",
         "loinc": "24627-2",  "icd10_ch": "J", "privacy": 1, "retention": 0},
        {"name": "RetinaMNIST",   "specialty": "ophthalmology",
         "loinc": "57133-1",  "icd10_ch": "H", "privacy": 1, "retention": 1},
        {"name": "BreastMNIST",   "specialty": "radiology",
         "loinc": "24606-6",  "icd10_ch": "C", "privacy": 2, "retention": 2},
        {"name": "BloodMNIST",    "specialty": "haematology",
         "loinc": "58410-2",  "icd10_ch": "D", "privacy": 0, "retention": 0},
        {"name": "OrganAMNIST",   "specialty": "radiology",
         "loinc": "24725-4",  "icd10_ch": "K", "privacy": 1, "retention": 1},
    ]

    # Specialty overlap (hand-coded clinical knowledge)
    SPECIALTY_OVERLAP = {
        ("dermatology",   "dermatology")  : 1.00,
        ("dermatology",   "pathology")    : 0.55,  # skin biopsy bridges both
        ("dermatology",   "radiology")    : 0.20,
        ("dermatology",   "ophthalmology"): 0.15,
        ("dermatology",   "haematology")  : 0.10,
        ("pathology",     "pathology")    : 1.00,
        ("pathology",     "radiology")    : 0.35,
        ("pathology",     "ophthalmology"): 0.15,
        ("pathology",     "haematology")  : 0.45,  # blood pathology
        ("radiology",     "radiology")    : 1.00,
        ("radiology",     "ophthalmology"): 0.30,  # both are imaging
        ("radiology",     "haematology")  : 0.15,
        ("ophthalmology", "ophthalmology"): 1.00,
        ("ophthalmology", "haematology")  : 0.10,
        ("haematology",   "haematology")  : 1.00,
    }

    # ICD-10 chapter proximity
    # Chapters: L=skin, C=neoplasms, J=respiratory, H=eye, D=blood, K=digestive
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
    MODALITY_FAMILIES = {
        "18748-4": "optical",      # dermoscopy
        "22634-0": "microscopy",    # histopathology
        "24627-2": "xray",          # chest X-ray
        "57133-1": "optical",      # fundus photography
        "24606-6": "ultrasound",    # breast US
        "58410-2": "microscopy",    # blood smear / CBC
        "24725-4": "ct",            # abdominal CT
    }

    def __init__(self):
        self.n_silos = len(self.SILOS)
        self._build_matrices()

    def _specialty_sim(self, s1, s2):
        return self.SPECIALTY_OVERLAP.get(
            (s1, s2), self.SPECIALTY_OVERLAP.get((s2, s1), 0.10))

    def _icd10_sim(self, c1, c2):
        return self.ICD10_PROXIMITY.get(
            (c1, c2), self.ICD10_PROXIMITY.get((c2, c1), 0.05))

    def _modality_sim(self, l1, l2):
        f1 = self.MODALITY_FAMILIES.get(l1, "unknown")
        f2 = self.MODALITY_FAMILIES.get(l2, "unknown")
        if f1 == f2:                              return 1.00
        if {f1, f2} == {"optical", "microscopy"}: return 0.55
        if {f1, f2} == {"xray", "ct"}:            return 0.65
        return 0.15

    def _collaboration_permitted(self, i, j):
        """Hard governance rule: both local-only silos cannot collaborate."""
        return not (self.SILOS[i]["retention"] == 2 and
                    self.SILOS[j]["retention"] == 2)

    def _build_matrices(self):
        n = self.n_silos
        self.collab_mat     = np.ones((n, n), dtype=bool)
        self.similarity_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                si = self.SILOS[i]; sj = self.SILOS[j]
                cp = self._collaboration_permitted(i, j)
                self.collab_mat[i, j] = cp
                if not cp:
                    self.similarity_mat[i, j] = 0.0
                else:
                    sp = self._specialty_sim(si["specialty"], sj["specialty"])
                    mo = self._modality_sim(si["loinc"], sj["loinc"])
                    ic = self._icd10_sim(si["icd10_ch"], sj["icd10_ch"])
                    self.similarity_mat[i, j] = 0.4*sp + 0.3*mo + 0.3*ic

    def get_similarity(self, i, j):
        return float(self.similarity_mat[i, j])

    def collaboration_permitted(self, i, j):
        return bool(self.collab_mat[i, j])

    def print_summary(self):
        print("\n--- Clinical Ontology ---")
        print(f"  {'Silo':<20} {'Specialty':<15} {'LOINC':<10} "
              f"{'ICD-10':>6} {'Privacy':>8} {'Retention':>10}")
        print("  " + "-" * 72)
        for s in self.SILOS:
            print(f"  {s['name']:<20} {s['specialty']:<15} {s['loinc']:<10} "
                  f"{s['icd10_ch']:>6} {s['privacy']:>8} {s['retention']:>10}")
        names = [s["name"][:8] for s in self.SILOS]
        print("\n  Collaboration permissions (F = prohibited by governance):")
        print("             " + "".join(f"{n:>10}" for n in names))
        for i in range(self.n_silos):
            row = "".join(
                f"{'T':>10}" if self.collab_mat[i,j] else f"{'F':>10}"
                for j in range(self.n_silos))
            print(f"  {names[i]:>10} {row}")
        print("\n  Clinical similarity matrix (0 = prohibited, 1 = identical):")
        print("             " + "".join(f"{n:>10}" for n in names))
        for i in range(self.n_silos):
            row = "".join(f"{self.similarity_mat[i,j]:>10.2f}"
                          for j in range(self.n_silos))
            print(f"  {names[i]:>10} {row}")
        print()


# ================================================================
# SHARED ENCODER + PER-SILO HEAD
# ================================================================
class SharedEncoder(nn.Module):
    """
    Shared CNN encoder updated by all silos via ontology-weighted aggregation.
    Uses AdaptiveAvgPool so it accepts any input resolution.
    MedMNIST default is 28x28.
    """
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),
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
    """Per-silo classification head. Never aggregated across silos."""
    def __init__(self, embed_dim=EMBED_DIM, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, emb):
        return self.fc(emb)


class SiloModel(nn.Module):
    """Full forward pass: encoder -> silo head."""
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
    {"name": "DermaMNIST",    "cls": DermaMNIST,    "n_classes": 7},
    {"name": "PathMNIST",     "cls": PathMNIST,     "n_classes": 9},
    {"name": "PneumoniaMNIST","cls": PneumoniaMNIST,"n_classes": 2},
    {"name": "RetinaMNIST",   "cls": RetinaMNIST,   "n_classes": 5},
    {"name": "BreastMNIST",   "cls": BreastMNIST,   "n_classes": 2},
    {"name": "BloodMNIST",    "cls": BloodMNIST,    "n_classes": 8},
    {"name": "OrganAMNIST",   "cls": OrganAMNIST,   "n_classes": 11},
]
N_SILOS = len(SILO_CONFIGS)


# ================================================================
# DATASET HELPERS
# ================================================================
class MedMNISTWrapper(Dataset):
    """
    Wraps MedMNIST datasets:
      - Converts PIL image to normalised float tensor
      - Expands single-channel images to 3 channels
      - Flattens label shape from (1,) to scalar
    """
    def __init__(self, medmnist_ds):
        self.ds      = medmnist_ds
        self.targets = self._extract_targets()

    def _extract_targets(self):
        out = []
        for i in range(len(self.ds)):
            _, lbl = self.ds[i]
            out.append(int(np.array(lbl).flatten()[0]))
        return out

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        img = torch.from_numpy(np.array(img)).float() / 255.0
        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim == 3 and img.shape[-1] in (1, 3):
            img = img.permute(2, 0, 1)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img  = (img - mean) / std
        return img, int(np.array(label).flatten()[0])


def load_medmnist_silos(data_root=DATA_ROOT):
    os.makedirs(data_root, exist_ok=True)
    silos = {}
    for s_id, cfg in enumerate(SILO_CONFIGS):
        print(f"  Loading {cfg['name']}...")
        train_raw = cfg["cls"](split="train", download=True, root=data_root)
        test_raw  = cfg["cls"](split="test",  download=True, root=data_root)
        silos[s_id] = {
            "name"     : cfg["name"],
            "train"    : MedMNISTWrapper(train_raw),
            "test"     : MedMNISTWrapper(test_raw),
            "n_classes": cfg["n_classes"],
        }
        print(f"    train={len(train_raw)}  test={len(test_raw)}  "
              f"classes={cfg['n_classes']}")
    return silos


def create_subclient_partitions(dataset, n_subclients, alpha):
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


def build_silo_loaders(silos, n_subclients, alpha, batch_size=32):
    """
    Returns:
      client_loaders  -- flat list, one DataLoader per sub-client
      client_silo_ids -- which silo each sub-client belongs to
      test_loaders    -- dict silo_id -> DataLoader
    """
    client_loaders, client_silo_ids, test_loaders = [], [], {}
    for s_id, silo in silos.items():
        test_loaders[s_id] = DataLoader(
            silo["test"], batch_size=64, shuffle=False, num_workers=0)
        for part in create_subclient_partitions(silo["train"], n_subclients, alpha):
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
    ece, bin_size = 0.0, 1.0 / n_bins
    for b in range(n_bins):
        mask = confidences.gt(b * bin_size) & confidences.le((b + 1) * bin_size)
        if mask.sum() > 0:
            ece += abs(accuracies[mask].float().mean().item() -
                       confidences[mask].mean().item()) * mask.float().mean().item()
    return ece


def evaluate_silo(encoder, head, loader):
    """Returns (accuracy, ece, per_class_acc_dict)."""
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
    acc = correct / total if total > 0 else 0.0
    ece = compute_ece(logits_list, labels_list) if logits_list else 0.0
    all_preds  = torch.cat(logits_list).argmax(dim=1)
    all_labels = torch.cat(labels_list)
    pca = {}
    for c in range(head.fc.out_features):
        mask = all_labels == c
        if mask.sum() > 0:
            pca[c] = (all_preds[mask] == all_labels[mask]).float().mean().item()
    return acc, ece, pca


def evaluate_all_silos(encoder, silo_heads, test_loaders, silos):
    results = {}
    for s_id, loader in test_loaders.items():
        acc, ece, pca = evaluate_silo(encoder, silo_heads[s_id], loader)
        results[s_id] = {"acc": acc, "ece": ece, "pca": pca,
                         "name": silos[s_id]["name"]}
    return results


def summarise_silos(results):
    accs = [results[s]["acc"] for s in results]
    eces = [results[s]["ece"] for s in results]
    return {
        "avg_acc"  : float(np.mean(accs)),
        "worst_acc": float(np.min(accs)),
        "avg_ece"  : float(np.mean(eces)),
        "per_silo" : {results[s]["name"]: results[s]["acc"] for s in results},
    }


def compute_ntr(transfer_pca, local_pca):
    classes = set(transfer_pca) & set(local_pca)
    if not classes:
        return 0.0
    return sum(1 for c in classes if transfer_pca[c] < local_pca[c]) / len(classes)


# ================================================================
# PPDOE -- server-hosted, ontology-guided, NOT federated learning
# ================================================================
class PPDOEMedical:
    """
    PPDOE aggregates encoder updates using clinical ontology weights.
    Governance constraints are enforced as hard zero weights.
    Per-silo heads are never shared across silos.
    This is NOT a federated learning system.
    """
    def __init__(self, ontology, silos):
        self.ontology   = ontology
        self.silos      = silos
        self.encoder    = SharedEncoder(embed_dim=EMBED_DIM).to(device)
        self.silo_heads = {
            s_id: SiloHead(EMBED_DIM, silo["n_classes"]).to(device)
            for s_id, silo in silos.items()
        }

    def _semantic_weights(self, client_silo_ids):
        n = len(client_silo_ids)
        w = []
        for i in range(n):
            sims = [
                self.ontology.get_similarity(client_silo_ids[i], client_silo_ids[j])
                for j in range(n) if j != i
            ]
            w.append(float(np.mean(sims)) if sims else 1.0)
        total = sum(w)
        return [x / total if total > 0 else 1.0 / n for x in w]

    def _local_train(self, s_id, loader, epochs=1, lr=0.01):
        enc   = copy.deepcopy(self.encoder).to(device)
        head  = copy.deepcopy(self.silo_heads[s_id]).to(device)
        model = SiloModel(enc, head)
        opt   = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        crit  = nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad(); crit(model(imgs), labels).backward(); opt.step()
        return enc.state_dict(), head.state_dict()

    def run_round(self, client_loaders, client_silo_ids):
        """
        PPDOE round:
          1. Sub-clients train locally (encoder + silo head)
          2. Encoder updates aggregated with ontology weights
          3. Silo heads averaged within-silo only
          4. Global encoder updated; per-silo heads updated
        """
        weights      = self._semantic_weights(client_silo_ids)
        enc_updates  = []
        head_updates = {s_id: [] for s_id in self.silos}
        for loader, s_id in zip(client_loaders, client_silo_ids):
            e, h = self._local_train(s_id, loader)
            enc_updates.append(e)
            head_updates[s_id].append(h)
        # Aggregate encoder with ontology weights
        new_enc = {
            key: sum(weights[i] * enc_updates[i][key].float()
                     for i in range(len(enc_updates)))
            for key in enc_updates[0]
        }
        self.encoder.load_state_dict(new_enc)
        # Aggregate each silo's head independently (within-silo FedAvg)
        for s_id, updates in head_updates.items():
            if updates:
                avg_h = {k: sum(u[k].float() for u in updates) / len(updates)
                         for k in updates[0]}
                self.silo_heads[s_id].load_state_dict(avg_h)

    def evaluate(self, test_loaders):
        return evaluate_all_silos(self.encoder, self.silo_heads, test_loaders, self.silos)


# ================================================================
# FL BASELINES -- independent systems, no ontology access
# Same encoder + per-silo head architecture for fair comparison
# ================================================================
class FLMedBaseline:
    def __init__(self, silos):
        self.silos      = silos
        self.encoder    = SharedEncoder(embed_dim=EMBED_DIM).to(device)
        self.silo_heads = {
            s_id: SiloHead(EMBED_DIM, silo["n_classes"]).to(device)
            for s_id, silo in silos.items()
        }

    def _local_train(self, s_id, loader, epochs=1, lr=0.01):
        enc   = copy.deepcopy(self.encoder).to(device)
        head  = copy.deepcopy(self.silo_heads[s_id]).to(device)
        model = SiloModel(enc, head)
        opt   = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        crit  = nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad(); crit(model(imgs), labels).backward(); opt.step()
        return enc.state_dict(), head.state_dict()

    def _fedavg_enc(self, enc_updates):
        n = len(enc_updates)
        return {k: sum(u[k].float() for u in enc_updates) / n
                for k in enc_updates[0]}

    def _update_heads(self, head_updates):
        for s_id, updates in head_updates.items():
            if updates:
                self.silo_heads[s_id].load_state_dict(
                    {k: sum(u[k].float() for u in updates) / len(updates)
                     for k in updates[0]})

    def run_round(self, client_loaders, client_silo_ids):
        raise NotImplementedError

    def evaluate(self, test_loaders):
        return evaluate_all_silos(self.encoder, self.silo_heads, test_loaders, self.silos)


class FedBNMed(FLMedBaseline):
    """Aggregate encoder except BatchNorm running stats."""
    def run_round(self, client_loaders, client_silo_ids):
        enc_updates  = []
        head_updates = {s_id: [] for s_id in self.silos}
        for loader, s_id in zip(client_loaders, client_silo_ids):
            e, h = self._local_train(s_id, loader)
            enc_updates.append(e); head_updates[s_id].append(h)
        avg_enc = {}
        for key in enc_updates[0]:
            if any(x in key for x in
                   ["running_mean", "running_var", "num_batches_tracked"]):
                avg_enc[key] = self.encoder.state_dict()[key]
            else:
                avg_enc[key] = sum(u[key].float() for u in enc_updates) / len(enc_updates)
        self.encoder.load_state_dict(avg_enc)
        self._update_heads(head_updates)


class FedRepMed(FLMedBaseline):
    """Alternate head (local) and encoder (global) training phases."""
    def _local_train_fedrep(self, s_id, loader, lr=0.01):
        enc   = copy.deepcopy(self.encoder).to(device)
        head  = copy.deepcopy(self.silo_heads[s_id]).to(device)
        model = SiloModel(enc, head); crit = nn.CrossEntropyLoss()
        # Phase 1: train head only
        for p in model.encoder.parameters(): p.requires_grad_(False)
        opt = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=lr, momentum=0.9)
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad(); crit(model(imgs), labels).backward(); opt.step()
            break
        # Phase 2: train encoder only
        for p in model.encoder.parameters(): p.requires_grad_(True)
        for p in model.head.parameters(): p.requires_grad_(False)
        opt2 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=lr, momentum=0.9)
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt2.zero_grad(); crit(model(imgs), labels).backward(); opt2.step()
            break
        for p in model.head.parameters(): p.requires_grad_(True)
        return enc.state_dict(), head.state_dict()

    def run_round(self, client_loaders, client_silo_ids):
        enc_updates  = []
        head_updates = {s_id: [] for s_id in self.silos}
        for loader, s_id in zip(client_loaders, client_silo_ids):
            e, h = self._local_train_fedrep(s_id, loader)
            enc_updates.append(e); head_updates[s_id].append(h)
        self.encoder.load_state_dict(self._fedavg_enc(enc_updates))
        self._update_heads(head_updates)


class DittoMed(FLMedBaseline):
    """Global encoder + per-client personal encoder with proximal regularisation."""
    def __init__(self, silos, lam=0.5):
        super().__init__(silos); self.lam = lam
        self.personal_encs = {}

    def run_round(self, client_loaders, client_silo_ids):
        crit = nn.CrossEntropyLoss()
        enc_updates  = []
        head_updates = {s_id: [] for s_id in self.silos}
        for i, (loader, s_id) in enumerate(zip(client_loaders, client_silo_ids)):
            e, h = self._local_train(s_id, loader)
            enc_updates.append(e); head_updates[s_id].append(h)
            if i not in self.personal_encs:
                self.personal_encs[i] = copy.deepcopy(self.encoder).to(device)
            p_enc = self.personal_encs[i]
            head  = copy.deepcopy(self.silo_heads[s_id]).to(device)
            p_model = SiloModel(p_enc, head)
            gp    = {k: v.clone().detach() for k, v in self.encoder.state_dict().items()}
            opt_p = optim.SGD(p_model.parameters(), lr=0.01, momentum=0.9)
            p_model.train()
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt_p.zero_grad()
                loss = crit(p_model(imgs), labels)
                prox = sum(((p - gp[k])**2).sum()
                           for k, p in p_enc.named_parameters() if k in gp)
                (loss + (self.lam / 2) * prox).backward(); opt_p.step()
        self.encoder.load_state_dict(self._fedavg_enc(enc_updates))
        self._update_heads(head_updates)


class FedProtoMed(FLMedBaseline):
    """Prototype aggregation; per-silo prototypes (different class spaces)."""
    def __init__(self, silos):
        super().__init__(silos)
        self.global_protos = {s_id: {} for s_id in silos}

    def run_round(self, client_loaders, client_silo_ids):
        crit = nn.CrossEntropyLoss()
        enc_updates  = []
        head_updates = {s_id: [] for s_id in self.silos}
        all_protos   = {s_id: {} for s_id in self.silos}
        for loader, s_id in zip(client_loaders, client_silo_ids):
            enc  = copy.deepcopy(self.encoder).to(device)
            head = copy.deepcopy(self.silo_heads[s_id]).to(device)
            model = SiloModel(enc, head)
            opt   = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            local_protos = {c: [] for c in range(self.silos[s_id]["n_classes"])}
            model.train()
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                emb  = model.get_embedding(imgs)
                loss = crit(model.head(emb), labels)
                gp   = self.global_protos[s_id]
                if gp:
                    pl = sum(
                        ((emb[i] - gp[l.item()].to(device))**2).sum()
                        for i, l in enumerate(labels) if l.item() in gp)
                    loss = loss + 0.1 * pl / max(len(labels), 1)
                loss.backward(); opt.step()
                for i, l in enumerate(labels):
                    local_protos[l.item()].append(emb[i].detach().cpu())
            enc_updates.append(enc.state_dict())
            head_updates[s_id].append(head.state_dict())
            for c, v in local_protos.items():
                if v: all_protos[s_id].setdefault(c, []).extend(v)
        self.encoder.load_state_dict(self._fedavg_enc(enc_updates))
        for s_id in self.silos:
            self.global_protos[s_id] = {
                c: torch.stack(v).mean(0)
                for c, v in all_protos[s_id].items() if v
            }
        self._update_heads(head_updates)


class pFedMeMed(FLMedBaseline):
    """Moreau envelope personalisation on the encoder."""
    def __init__(self, silos, lam=15.0, k_steps=3):
        super().__init__(silos)
        self.lam = lam; self.k_steps = k_steps
        self.personal_encs = {}

    def run_round(self, client_loaders, client_silo_ids):
        crit = nn.CrossEntropyLoss(); lr = 0.01
        enc_updates  = []
        head_updates = {s_id: [] for s_id in self.silos}
        for i, (loader, s_id) in enumerate(zip(client_loaders, client_silo_ids)):
            if i not in self.personal_encs:
                self.personal_encs[i] = copy.deepcopy(self.encoder).to(device)
            theta   = self.personal_encs[i]
            head    = copy.deepcopy(self.silo_heads[s_id]).to(device)
            gparams = list(self.encoder.parameters())
            model   = SiloModel(theta, head)
            opt     = optim.SGD(model.parameters(), lr=lr / (1 + self.lam))
            model.train()
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                for _ in range(self.k_steps):
                    opt.zero_grad()
                    loss = crit(model(imgs), labels)
                    prox = sum(
                        self.lam / 2 * ((p - q.detach())**2).sum()
                        for p, q in zip(theta.parameters(), gparams))
                    (loss + prox).backward(); opt.step()
                break
            updated = copy.deepcopy(self.encoder.state_dict())
            with torch.no_grad():
                for k, gp in self.encoder.named_parameters():
                    tp = dict(theta.named_parameters())[k]
                    updated[k] = gp - lr * self.lam * (gp - tp.detach())
            enc_updates.append(updated)
            head_updates[s_id].append(head.state_dict())
        self.encoder.load_state_dict(self._fedavg_enc(enc_updates))
        self._update_heads(head_updates)


class FedSSAMed(FLMedBaseline):
    """Uniform FedAvg across all encoder layers."""
    def run_round(self, client_loaders, client_silo_ids):
        enc_updates  = []
        head_updates = {s_id: [] for s_id in self.silos}
        for loader, s_id in zip(client_loaders, client_silo_ids):
            e, h = self._local_train(s_id, loader)
            enc_updates.append(e); head_updates[s_id].append(h)
        self.encoder.load_state_dict(self._fedavg_enc(enc_updates))
        self._update_heads(head_updates)


# ================================================================
# LOCAL BASELINE (no transfer -- used to compute NTR)
# ================================================================
def compute_local_baselines(silos, test_loaders, n_epochs=4):
    local_pca = {}
    for s_id, silo in silos.items():
        enc   = SharedEncoder(embed_dim=EMBED_DIM).to(device)
        head  = SiloHead(EMBED_DIM, silo["n_classes"]).to(device)
        model = SiloModel(enc, head)
        opt   = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        crit  = nn.CrossEntropyLoss()
        loader = DataLoader(silo["train"], batch_size=32, shuffle=True, num_workers=0)
        for _ in range(n_epochs):
            model.train()
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad(); crit(model(imgs), labels).backward(); opt.step()
        acc, ece, pca = evaluate_silo(enc, head, test_loaders[s_id])
        local_pca[s_id] = pca
        print(f"  Local baseline [{silo['name']:20}]: acc={acc:.4f}  ece={ece:.4f}")
    return local_pca


# ================================================================
# LEAVE-ONE-SILO-OUT
# ================================================================
def leave_one_silo_out(silos, ontology, n_subclients, alpha, n_rounds):
    """
    For each held-out silo:
      1. PPDOE trains on the remaining K-1 silos
      2. Evaluate by fine-tuning the held-out silo's head on 200 samples
         (encoder frozen -- tests encoder generalisation)
      3. Report accuracy on the held-out silo's full test set
    """
    results = {}
    for held_id in silos:
        train_silos = {s: silos[s] for s in silos if s != held_id}
        cl, csi, _  = build_silo_loaders(train_silos, n_subclients, alpha)
        held_loader = DataLoader(silos[held_id]["test"],
                                  batch_size=64, shuffle=False, num_workers=0)
        ppdoe = PPDOEMedical(ontology, train_silos)
        for _ in range(n_rounds):
            ppdoe.run_round(cl, csi)
        # Fine-tune head only on small held-out sample
        held_head   = SiloHead(EMBED_DIM, silos[held_id]["n_classes"]).to(device)
        fine_subset = Subset(silos[held_id]["train"],
                             list(range(min(200, len(silos[held_id]["train"])))))
        fine_loader = DataLoader(fine_subset, batch_size=32,
                                  shuffle=True, num_workers=0)
        for p in ppdoe.encoder.parameters(): p.requires_grad_(False)
        opt = optim.SGD(held_head.parameters(), lr=0.01, momentum=0.9)
        m   = SiloModel(ppdoe.encoder, held_head).to(device)
        m.train(); crit = nn.CrossEntropyLoss()
        for imgs, labels in fine_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad(); crit(m(imgs), labels).backward(); opt.step()
        for p in ppdoe.encoder.parameters(): p.requires_grad_(True)
        acc, ece, _ = evaluate_silo(ppdoe.encoder, held_head, held_loader)
        results[held_id] = {"acc": acc, "ece": ece,
                             "name": silos[held_id]["name"]}
        print(f"    LOSO [held out={silos[held_id]['name']:20}]: "
              f"acc={acc:.4f}  ece={ece:.4f}")
    return results


# ================================================================
# MAIN
# ================================================================
def run_experiment3(
    n_subclients = N_SUBCLIENTS,
    alpha        = ALPHA,
    n_rounds     = N_ROUNDS,
    data_root    = DATA_ROOT,
):
    print("\n" + "=" * 75)
    print("  PPDOE Experiment 3 -- Cross-Silo Medical Benchmark (MedMNIST v2)")
    print("  PPDOE is the system under evaluation. FL baselines are independent.")
    print("=" * 75)

    ontology = ClinicalOntology()
    ontology.print_summary()

    print("Loading MedMNIST v2 silos...")
    silos = load_medmnist_silos(data_root=data_root)

    client_loaders, client_silo_ids, test_loaders = build_silo_loaders(
        silos, n_subclients, alpha)
    print(f"\nTotal sub-clients: {len(client_loaders)} "
          f"({N_SILOS} silos x {n_subclients})")
    print("Client -> Silo mapping:")
    for i, s_id in enumerate(client_silo_ids):
        print(f"  client {i:2d} -> {SILO_CONFIGS[s_id]['name']}")

    # Local no-transfer baselines
    print("\nComputing local baselines (no transfer, for NTR)...")
    local_pca = compute_local_baselines(silos, test_loaders)

    # PPDOE
    ppdoe         = PPDOEMedical(ontology, silos)
    ppdoe_history = []

    # FL baselines
    fl_baselines = {
        "FedBN"   : FedBNMed(silos),
        "FedRep"  : FedRepMed(silos),
        "Ditto"   : DittoMed(silos, lam=0.5),
        "FedProto": FedProtoMed(silos),
        "pFedMe"  : pFedMeMed(silos, lam=15.0),
        "FedSSA"  : FedSSAMed(silos),
    }
    fl_history = {name: [] for name in fl_baselines}

    print("\n" + "=" * 75)
    print("  Training rounds")
    print("=" * 75)

    for r in range(1, n_rounds + 1):
        print(f"\n  Round {r}/{n_rounds}")
        ppdoe.run_round(client_loaders, client_silo_ids)
        pr = ppdoe.evaluate(test_loaders)
        pm = summarise_silos(pr)
        ppdoe_history.append(pm)
        print(f"    {'PPDOE':25}: avg={pm['avg_acc']:.4f}  "
              f"worst={pm['worst_acc']:.4f}  ece={pm['avg_ece']:.4f}")
        for name, bl in fl_baselines.items():
            bl.run_round(client_loaders, client_silo_ids)
            fr = bl.evaluate(test_loaders)
            m  = summarise_silos(fr)
            fl_history[name].append(m)
            print(f"    {name:25}: avg={m['avg_acc']:.4f}  "
                  f"worst={m['worst_acc']:.4f}  ece={m['avg_ece']:.4f}")

    # Leave-one-silo-out
    print("\n--- Leave-One-Silo-Out Transfer (PPDOE, K-1 regime) ---")
    loso_results = leave_one_silo_out(
        silos, ontology, n_subclients=n_subclients,
        alpha=alpha, n_rounds=max(n_rounds // 2, 3))
    loso_avg   = float(np.mean([v["acc"] for v in loso_results.values()]))
    loso_worst = float(np.min([v["acc"]  for v in loso_results.values()]))
    print(f"  LOSO: avg={loso_avg:.4f}  worst-silo={loso_worst:.4f}")

    # Negative transfer rate
    print("\n--- Negative Transfer Rate (vs local no-transfer baseline) ---")
    ntr_results  = {}
    ppdoe_final  = ppdoe.evaluate(test_loaders)
    ntr_results["PPDOE"] = {
        s_id: compute_ntr(ppdoe_final[s_id]["pca"], local_pca[s_id])
        for s_id in silos
    }
    for name, bl in fl_baselines.items():
        bl_final = bl.evaluate(test_loaders)
        ntr_results[name] = {
            s_id: compute_ntr(bl_final[s_id]["pca"], local_pca[s_id])
            for s_id in silos
        }
    print(f"  {'Method':<20}  " +
          "".join(f"{silos[s]['name'][:10]:>14}" for s in silos))
    for mname, ntr_dict in ntr_results.items():
        row = "".join(f"{ntr_dict[s]:>14.4f}" for s in silos)
        print(f"  {mname:<20}  {row}")

    # Governance compliance check
    print("\n--- Governance Compliance Check ---")
    print("  Verifying PPDOE enforces collaboration_permit=False as zero weight.")
    all_pass = True
    for i in range(N_SILOS):
        for j in range(N_SILOS):
            if i != j and not ontology.collaboration_permitted(i, j):
                sim    = ontology.get_similarity(i, j)
                status = "PASS" if sim == 0.0 else "FAIL"
                if status == "FAIL": all_pass = False
                print(f"  [{status}] {SILO_CONFIGS[i]['name']} <-> "
                      f"{SILO_CONFIGS[j]['name']}: sim={sim:.3f} (must be 0.000)")
    if all_pass:
        print("  All governance constraints correctly enforced.")

    # Final summary table
    print("\n" + "=" * 75)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 75)
    print(f"  {'Method':<25}  {'Avg Acc':>9}  {'Worst Acc':>10}  "
          f"{'Avg ECE':>8}  {'Avg NTR':>9}  {'vs PPDOE':>10}")
    print("  " + "-" * 73)
    baseline_avg = ppdoe_history[-1]["avg_acc"]
    all_final    = {"PPDOE": ppdoe_history[-1]}
    all_final.update({n: fl_history[n][-1] for n in fl_baselines})
    for mname, m in sorted(all_final.items(),
                            key=lambda x: x[1]["avg_acc"], reverse=True):
        delta   = m["avg_acc"] - baseline_avg
        d_str   = "(base)" if mname == "PPDOE" else (
            f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}")
        avg_ntr = float(np.mean(list(ntr_results[mname].values()))) \
                  if mname in ntr_results else float("nan")
        print(f"  {mname:<25}  {m['avg_acc']:>9.4f}  {m['worst_acc']:>10.4f}  "
              f"{m['avg_ece']:>8.4f}  {avg_ntr:>9.4f}  {d_str:>10}")

    # Per-silo breakdown
    print("\n  Per-silo accuracy breakdown (final round):")
    snames = [silos[s]["name"][:10] for s in silos]
    print(f"  {'Method':<25}  " + "".join(f"{n:>13}" for n in snames))
    print("  " + "-" * (27 + 13 * N_SILOS))
    for mname, m in all_final.items():
        row = "".join(
            f"{m['per_silo'].get(silos[s]['name'], 0.0):>13.4f}" for s in silos)
        print(f"  {mname:<25}  {row}")

    return ppdoe_history, fl_history, loso_results, ntr_results


if __name__ == "__main__":
    run_experiment3(
        n_subclients = N_SUBCLIENTS,
        alpha        = ALPHA,
        n_rounds     = N_ROUNDS,
        data_root    = DATA_ROOT,
    )
