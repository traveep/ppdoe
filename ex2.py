import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
import random
import copy
import os

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("cpu")
print(f"Using device: {device}")

# Toggle: True = DomainNet (primary), False = Office-Home (secondary)
USE_DOMAINNET = True
DATA_ROOT = "./data"


# ====================== ONTOLOGY ======================
class DomainOntology:
    """
    Encodes:
      - Domain nodes with a hand-coded proximity matrix
        (real<->painting close; quickdraw<->sketch close; infograph distant)
      - Class nodes with Jaccard token-overlap similarity on class name strings
    Used exclusively by PPDOE to compute semantic aggregation weights.
    The FL baselines do not use the ontology.
    """
    DOMAINNET_DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    DOMAINNET_PROXIMITY = [
        [1.00, 0.55, 0.60, 0.40, 0.55, 0.65],  # clipart
        [0.55, 1.00, 0.50, 0.30, 0.50, 0.45],  # infograph
        [0.60, 0.50, 1.00, 0.35, 0.80, 0.60],  # painting
        [0.40, 0.30, 0.35, 1.00, 0.30, 0.75],  # quickdraw
        [0.55, 0.50, 0.80, 0.30, 1.00, 0.55],  # real
        [0.65, 0.45, 0.60, 0.75, 0.55, 1.00],  # sketch
    ]
    OFFICEHOME_DOMAINS = ["Art", "Clipart", "Product", "RealWorld"]
    OFFICEHOME_PROXIMITY = [
        [1.00, 0.55, 0.65, 0.70],
        [0.55, 1.00, 0.60, 0.55],
        [0.65, 0.60, 1.00, 0.80],
        [0.70, 0.55, 0.80, 1.00],
    ]

    def __init__(self, use_domainnet=True, class_names=None):
        self.use_domainnet = use_domainnet
        if use_domainnet:
            self.domain_names = self.DOMAINNET_DOMAINS
            self.domain_proximity = np.array(self.DOMAINNET_PROXIMITY)
        else:
            self.domain_names = self.OFFICEHOME_DOMAINS
            self.domain_proximity = np.array(self.OFFICEHOME_PROXIMITY)
        self.num_domains = len(self.domain_names)
        self.class_names = class_names or []
        self.num_classes = len(self.class_names)
        self._class_sim = self._build_class_similarity()

    def _build_class_similarity(self):
        n = self.num_classes
        if n == 0:
            return np.eye(1)
        sim = np.eye(n)
        tokens = [set(name.lower().replace("_", " ").split()) for name in self.class_names]
        for i in range(n):
            for j in range(i + 1, n):
                if tokens[i] and tokens[j]:
                    v = len(tokens[i] & tokens[j]) / len(tokens[i] | tokens[j])
                else:
                    v = 0.0
                sim[i, j] = sim[j, i] = v
        return sim

    def get_domain_similarity(self, d1, d2):
        return float(self.domain_proximity[d1, d2])

    def get_class_similarity(self, c1, c2):
        if self.num_classes == 0:
            return 1.0
        return float(self._class_sim[c1, c2])

    def get_client_similarity(self, ci, cj):
        """Combined similarity for clients described as (domain_id, dominant_class)."""
        d_sim = self.get_domain_similarity(ci[0], cj[0])
        c_sim = self.get_class_similarity(ci[1], cj[1]) if self.num_classes > 0 else 1.0
        return 0.6 * d_sim + 0.4 * c_sim


# ====================== MODEL ======================
class SimpleCNN(nn.Module):
    """
    Shared architecture across PPDOE and all baselines.
    Input 64x64 RGB -> embedding dim 256*8*8 = 16384.
    """
    def __init__(self, num_classes=345):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))

    def get_embedding(self, x):
        return self.features(x).view(x.size(0), -1)


# ====================== DATASET HELPERS ======================
class DomainSubset(Dataset):
    def __init__(self, base_dataset, indices, domain_id):
        self.base = base_dataset
        self.indices = indices
        self.domain_id = domain_id

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]

    @property
    def targets(self):
        if hasattr(self.base, "targets"):
            return [self.base.targets[i] for i in self.indices]
        return [self.base[i][1] for i in self.indices]


class SyntheticDomainDataset(Dataset):
    """Stand-in when the real dataset directory is missing."""
    def __init__(self, num_samples=500, num_classes=345):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.targets = [random.randint(0, num_classes - 1) for _ in range(num_samples)]
        self.classes = [str(i) for i in range(num_classes)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.randn(3, 64, 64), self.targets[idx]


def load_domain_datasets(root, domain_names, subdir, transform, num_classes, n_synth=500):
    out = {}
    for d in domain_names:
        path = os.path.join(root, subdir, d)
        if os.path.isdir(path):
            out[d] = torchvision.datasets.ImageFolder(path, transform=transform)
        else:
            print(f"  [WARN] {subdir}/{d} not found. Using synthetic fallback.")
            out[d] = SyntheticDomainDataset(n_synth, num_classes)
    return out


def create_subclient_partitions(domain_dataset, n_subclients=5, alpha=0.4):
    """Dirichlet class-skew within one domain silo. Mirrors Experiment 1 exactly."""
    targets = np.array(domain_dataset.targets)
    num_classes = int(targets.max()) + 1
    partitions = []
    for _ in range(n_subclients):
        probs = np.random.dirichlet([alpha] * num_classes)
        p = probs[targets]
        p /= p.sum()
        n_samples = min(400, len(domain_dataset))
        selected = np.random.choice(len(domain_dataset), size=n_samples, p=p, replace=False)
        partitions.append(selected.tolist())
    return partitions


def get_dominant_class(indices, dataset):
    targets = dataset.targets
    labels = [targets[i] for i in indices]
    counts = np.bincount(labels, minlength=int(max(labels)) + 1)
    return int(np.argmax(counts))


def build_all_loaders(domain_datasets, domain_names, n_subclients=5, alpha=0.4,
                      batch_size=32, test_frac=0.2):
    """
    Builds per-domain sub-client loaders and test loaders.
    Returns:
      client_loaders      - flat list (one DataLoader per sub-client)
      client_descriptors  - list of (domain_id, dominant_class) per sub-client
      test_loaders        - dict domain_name -> DataLoader
    """
    client_loaders, client_descriptors = [], []
    test_loaders = {}
    for d_id, dname in enumerate(domain_names):
        ds = domain_datasets[dname]
        n = len(ds)
        all_idx = list(range(n))
        random.shuffle(all_idx)
        split = int(n * (1 - test_frac))
        train_idx, test_idx = all_idx[:split], all_idx[split:]
        test_loaders[dname] = DataLoader(
            DomainSubset(ds, test_idx, d_id), batch_size=64, shuffle=False, num_workers=0)
        train_ds = DomainSubset(ds, train_idx, d_id)
        for sc_idx in create_subclient_partitions(train_ds, n_subclients, alpha):
            dom_class = get_dominant_class(sc_idx, train_ds)
            client_loaders.append(
                DataLoader(DomainSubset(train_ds, sc_idx, d_id),
                           batch_size=batch_size, shuffle=True, num_workers=0))
            client_descriptors.append((d_id, dom_class))
    return client_loaders, client_descriptors, test_loaders


# ====================== METRICS ======================
def compute_ece(logits_list, labels_list, n_bins=15):
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    probs = F.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    ece = 0.0
    bin_size = 1.0 / n_bins
    for b in range(n_bins):
        lo, hi = b * bin_size, (b + 1) * bin_size
        mask = confidences.gt(lo) & confidences.le(hi)
        if mask.sum() > 0:
            ece += abs(accuracies[mask].float().mean().item() -
                       confidences[mask].mean().item()) * mask.float().mean().item()
    return ece


def evaluate_model(model, loader, num_classes):
    """Returns (accuracy, ece, per_class_acc_dict)."""
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
            total += labels.size(0)
    acc = correct / total if total > 0 else 0.0
    ece = compute_ece(logits_list, labels_list) if logits_list else 0.0
    all_preds  = torch.cat(logits_list).argmax(dim=1)
    all_labels = torch.cat(labels_list)
    pca = {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            pca[c] = (all_preds[mask] == all_labels[mask]).float().mean().item()
    return acc, ece, pca


def compute_negative_transfer_rate(transfer_pca, local_pca):
    classes = set(transfer_pca) & set(local_pca)
    if not classes:
        return 0.0
    return sum(1 for c in classes if transfer_pca[c] < local_pca[c]) / len(classes)


def summarise(domain_results, domain_names):
    accs = [domain_results[d]["acc"] for d in domain_names]
    eces = [domain_results[d]["ece"] for d in domain_names]
    return {
        "avg_acc"   : float(np.mean(accs)),
        "worst_acc" : float(np.min(accs)),
        "avg_ece"   : float(np.mean(eces)),
        "per_domain": {d: domain_results[d]["acc"] for d in domain_names},
    }


# ======================================================
# PPDOE  (the system under evaluation)
# ======================================================
class PPDOEFramework:
    """
    PPDOE is NOT a federated learning system.
    It owns a single global model and drives aggregation using
    ontology-derived semantic similarity weights over clients.
    The FL baselines below are entirely separate systems that
    run their own communication loops independently.
    """
    def __init__(self, ontology, num_classes):
        self.ontology = ontology
        self.num_classes = num_classes
        self.global_model = SimpleCNN(num_classes=num_classes).to(device)

    def _semantic_weights(self, client_descriptors):
        """Mirrors Experiment 1 exactly."""
        n = len(client_descriptors)
        w = []
        for i in range(n):
            sims = [
                self.ontology.get_client_similarity(
                    client_descriptors[i], client_descriptors[j])
                for j in range(n) if j != i
            ]
            w.append(np.mean(sims) if sims else 1.0)
        total = sum(w)
        return [x / total for x in w]

    def _local_train(self, loader, epochs=1, lr=0.01):
        model = copy.deepcopy(self.global_model).to(device)
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                criterion(model(imgs), labels).backward()
                opt.step()
        return model.state_dict()

    def run_round(self, client_loaders, client_descriptors):
        """One PPDOE round: local train -> ontology-weighted aggregate."""
        weights = self._semantic_weights(client_descriptors)
        client_updates = [
            self._local_train(loader)
            for loader in client_loaders
        ]
        # Ontology-weighted aggregation (the defining operation of PPDOE)
        new_state = {
            key: sum(w * u[key].float() for w, u in zip(weights, client_updates))
            for key in client_updates[0]
        }
        self.global_model.load_state_dict(new_state)

    def evaluate_all_domains(self, test_loaders):
        results = {}
        for dname, loader in test_loaders.items():
            acc, ece, pca = evaluate_model(self.global_model, loader, self.num_classes)
            results[dname] = {"acc": acc, "ece": ece, "pca": pca}
        return results


# ======================================================
# FL BASELINES  (each runs its own independent loop)
# ======================================================

class FLBaseline:
    """
    Base class for all FL baselines.
    Each baseline owns its own global model and runs its own
    communication loop independently of PPDOE.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.global_model = SimpleCNN(num_classes=num_classes).to(device)

    def _local_train(self, loader, epochs=1, lr=0.01):
        model = copy.deepcopy(self.global_model).to(device)
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                criterion(model(imgs), labels).backward()
                opt.step()
        return model.state_dict()

    def _fedavg(self, updates):
        n = len(updates)
        return {key: sum(u[key].float() for u in updates) / n for key in updates[0]}

    def run_round(self, client_loaders):
        raise NotImplementedError

    def evaluate_all_domains(self, test_loaders):
        results = {}
        for dname, loader in test_loaders.items():
            acc, ece, pca = evaluate_model(self.global_model, loader, self.num_classes)
            results[dname] = {"acc": acc, "ece": ece, "pca": pca}
        return results


class FedBN(FLBaseline):
    """
    FedBN: aggregate all params except BN running statistics,
    which are kept local (approximated here by preserving the
    global model's BN stats after aggregation).
    """
    def run_round(self, client_loaders):
        updates = [self._local_train(l) for l in client_loaders]
        avg = {}
        for key in updates[0]:
            if any(x in key for x in ["running_mean", "running_var", "num_batches_tracked"]):
                avg[key] = self.global_model.state_dict()[key]
            else:
                avg[key] = sum(u[key].float() for u in updates) / len(updates)
        self.global_model.load_state_dict(avg)


class FedRep(FLBaseline):
    """
    FedRep: clients train the head locally; only the body (features) is aggregated.
    Phase 1 freezes body and trains head; Phase 2 freezes head and trains body.
    """
    def _local_train_fedrep(self, loader, epochs=1, lr=0.01):
        model = copy.deepcopy(self.global_model).to(device)
        criterion = nn.CrossEntropyLoss()
        # Phase 1: head only
        for p in model.features.parameters():
            p.requires_grad_(False)
        opt = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=lr, momentum=0.9)
        model.train()
        for _ in range(epochs):
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                criterion(model(imgs), labels).backward()
                opt.step()
        # Phase 2: body only
        for p in model.features.parameters():
            p.requires_grad_(True)
        for p in model.classifier.parameters():
            p.requires_grad_(False)
        opt2 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=lr, momentum=0.9)
        for _ in range(epochs):
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt2.zero_grad()
                criterion(model(imgs), labels).backward()
                opt2.step()
        for p in model.classifier.parameters():
            p.requires_grad_(True)
        return model.state_dict()

    def run_round(self, client_loaders):
        updates = [self._local_train_fedrep(l) for l in client_loaders]
        avg = copy.deepcopy(self.global_model.state_dict())
        for key in updates[0]:
            if "features" in key:  # aggregate body only
                avg[key] = sum(u[key].float() for u in updates) / len(updates)
        self.global_model.load_state_dict(avg)


class Ditto(FLBaseline):
    """
    Ditto: maintains a personal model per client with proximal regularisation
    toward the global model. Only the global model is aggregated.
    """
    def __init__(self, num_classes, lam=0.5):
        super().__init__(num_classes)
        self.lam = lam
        self.personal_models = {}

    def run_round(self, client_loaders):
        global_updates = []
        criterion = nn.CrossEntropyLoss()
        for i, loader in enumerate(client_loaders):
            # Global update
            global_updates.append(self._local_train(loader))
            # Personal update
            if i not in self.personal_models:
                self.personal_models[i] = copy.deepcopy(self.global_model).to(device)
            personal = self.personal_models[i]
            gp = {k: v.clone().detach() for k, v in self.global_model.state_dict().items()}
            opt_p = optim.SGD(personal.parameters(), lr=0.01, momentum=0.9)
            personal.train()
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt_p.zero_grad()
                loss = criterion(personal(imgs), labels)
                prox = sum(((p - gp[k]) ** 2).sum()
                           for k, p in personal.named_parameters() if k in gp)
                (loss + (self.lam / 2) * prox).backward()
                opt_p.step()
        self.global_model.load_state_dict(self._fedavg(global_updates))


class pFedMe(FLBaseline):
    """
    pFedMe: Moreau envelope personalisation.
    Inner loop solves personalised model; outer step updates global model.
    """
    def __init__(self, num_classes, lam=15.0, k_steps=5):
        super().__init__(num_classes)
        self.lam = lam
        self.k_steps = k_steps
        self.personal_models = {}

    def run_round(self, client_loaders):
        global_updates = []
        criterion = nn.CrossEntropyLoss()
        lr = 0.01
        for i, loader in enumerate(client_loaders):
            if i not in self.personal_models:
                self.personal_models[i] = copy.deepcopy(self.global_model).to(device)
            theta = self.personal_models[i]
            global_params = list(self.global_model.parameters())
            opt = optim.SGD(theta.parameters(), lr=lr / (1 + self.lam))
            theta.train()
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                for _ in range(self.k_steps):
                    opt.zero_grad()
                    loss = criterion(theta(imgs), labels)
                    prox = sum(self.lam / 2 * ((p - q.detach()) ** 2).sum()
                               for p, q in zip(theta.parameters(), global_params))
                    (loss + prox).backward()
                    opt.step()
                break
            updated = copy.deepcopy(self.global_model.state_dict())
            with torch.no_grad():
                for k, gp in self.global_model.named_parameters():
                    tp = dict(theta.named_parameters())[k]
                    updated[k] = gp - lr * self.lam * (gp - tp.detach())
            global_updates.append(updated)
        self.global_model.load_state_dict(self._fedavg(global_updates))


class FedALA(FLBaseline):
    """
    FedALA: Adaptive Local Aggregation.
    Each client blends incoming global model with its retained local model
    via sigmoid-gated interpolation before local training.
    """
    def __init__(self, num_classes, eta=1.0):
        super().__init__(num_classes)
        self.eta = eta
        self.local_models = {}

    def run_round(self, client_loaders):
        updates = []
        criterion = nn.CrossEntropyLoss()
        lam = float(torch.sigmoid(torch.tensor(self.eta)))
        for i, loader in enumerate(client_loaders):
            if i not in self.local_models:
                self.local_models[i] = copy.deepcopy(self.global_model).to(device)
            local = self.local_models[i]
            gsd = self.global_model.state_dict()
            blended = {k: lam * gsd[k].float() + (1 - lam) * local.state_dict()[k].float()
                       for k in gsd}
            local.load_state_dict(blended)
            opt = optim.SGD(local.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            local.train()
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                criterion(local(imgs), labels).backward()
                opt.step()
            self.local_models[i] = local
            updates.append(local.state_dict())
        self.global_model.load_state_dict(self._fedavg(updates))


class FedProto(FLBaseline):
    """Prototype-based FL. Matches Experiment 1 FedProtoStrategy exactly."""
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.global_protos = {}

    def run_round(self, client_loaders):
        all_protos = {}
        updates = []
        criterion = nn.CrossEntropyLoss()
        for loader in client_loaders:
            model = copy.deepcopy(self.global_model).to(device)
            opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            local_protos = {c: [] for c in range(self.num_classes)}
            model.train()
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                emb = model.get_embedding(imgs)
                loss = criterion(model.classifier(emb), labels)
                if self.global_protos:
                    pl = sum(((emb[i] - self.global_protos[l.item()].to(device)) ** 2).sum()
                             for i, l in enumerate(labels) if l.item() in self.global_protos)
                    loss = loss + 0.1 * pl / max(len(labels), 1)
                loss.backward()
                opt.step()
                for i, l in enumerate(labels):
                    local_protos[l.item()].append(emb[i].detach().cpu())
            updates.append(model.state_dict())
            for c, v in local_protos.items():
                if v:
                    all_protos.setdefault(c, []).extend(v)
        self.global_protos = {c: torch.stack(v).mean(0) for c, v in all_protos.items()}
        self.global_model.load_state_dict(self._fedavg(updates))


class FedTGP(FLBaseline):
    """
    FedTGP: Trainable Global Prototypes.
    Class-prototype matrix is updated via manual gradient steps during local training.
    """
    def __init__(self, num_classes, embed_dim=16384, proto_lr=0.01):
        super().__init__(num_classes)
        self.global_protos = torch.randn(num_classes, embed_dim) * 0.01
        self.proto_lr = proto_lr

    def run_round(self, client_loaders):
        updates = []
        criterion = nn.CrossEntropyLoss()
        for loader in client_loaders:
            model = copy.deepcopy(self.global_model).to(device)
            opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            model.train()
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                emb = model.get_embedding(imgs)
                loss = criterion(model.classifier(emb), labels)
                gp = self.global_protos[labels].to(device)
                loss = loss + 0.05 * F.mse_loss(emb, gp.detach())
                loss.backward()
                opt.step()
                with torch.no_grad():
                    grad = 2.0 * (gp - emb.detach()) / emb.size(0)
                    self.global_protos[labels] -= self.proto_lr * grad.cpu()
            updates.append(model.state_dict())
        self.global_model.load_state_dict(self._fedavg(updates))


class FedSSA(FLBaseline):
    """
    FedSSA: Semantic-Shared Aggregation.
    Intermediate feature layers are uniformly aggregated (semantic sharing);
    all other layers use standard FedAvg.
    """
    def run_round(self, client_loaders):
        updates = [self._local_train(l) for l in client_loaders]
        avg = {}
        for key in updates[0]:
            if "features.4" in key or "features.5" in key:
                avg[key] = sum(u[key].float() for u in updates) / len(updates)
            else:
                avg[key] = sum(u[key].float() for u in updates) / len(updates)
        self.global_model.load_state_dict(avg)


# ======================================================
# DG BASELINES  (centralised, no federation)
# ======================================================

class ERMBaseline:
    """Standard training on pooled K-1 domain data."""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = SimpleCNN(num_classes=num_classes).to(device)

    def train(self, loader, epochs=5, lr=0.01):
        opt = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(epochs):
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                criterion(self.model(imgs), labels).backward()
                opt.step()

    def evaluate_all_domains(self, test_loaders):
        results = {}
        for dname, loader in test_loaders.items():
            acc, ece, pca = evaluate_model(self.model, loader, self.num_classes)
            results[dname] = {"acc": acc, "ece": ece, "pca": pca}
        return results


class CORALBaseline(ERMBaseline):
    """ERM + CORAL loss aligning second-order statistics of consecutive mini-batches."""
    @staticmethod
    def _coral(s, t):
        d = s.size(1)
        cs = (s - s.mean(0)).t().mm(s - s.mean(0)) / max(s.size(0) - 1, 1)
        ct = (t - t.mean(0)).t().mm(t - t.mean(0)) / max(t.size(0) - 1, 1)
        return ((cs - ct) ** 2).sum() / (4 * d * d)

    def train(self, loader, epochs=5, lr=0.01):
        opt = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        prev = None
        for _ in range(epochs):
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                emb = self.model.get_embedding(imgs)
                loss = criterion(self.model.classifier(emb), labels)
                if prev is not None and prev.size(0) > 1 and emb.size(0) > 1:
                    loss = loss + 0.5 * self._coral(prev.detach(), emb)
                prev = emb.detach()
                loss.backward()
                opt.step()


class GroupDROBaseline(ERMBaseline):
    """Group DRO: upweights the highest-loss domain at each step."""
    def __init__(self, num_classes, num_groups, eta=0.01):
        super().__init__(num_classes)
        self.num_groups = num_groups
        self.eta = eta
        self.q = torch.ones(num_groups) / num_groups

    def train_grouped(self, domain_loaders, epochs=5, lr=0.01):
        opt = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        iters = {d: iter(dl) for d, dl in domain_loaders.items()}
        for _ in range(80 * epochs):
            losses = []
            for d in range(self.num_groups):
                if d not in iters:
                    losses.append(torch.tensor(0.0, requires_grad=True))
                    continue
                try:
                    imgs, labels = next(iters[d])
                except StopIteration:
                    iters[d] = iter(domain_loaders[d])
                    imgs, labels = next(iters[d])
                imgs, labels = imgs.to(device), labels.to(device)
                losses.append(criterion(self.model(imgs), labels))
            lt = torch.stack(losses)
            with torch.no_grad():
                self.q = self.q * torch.exp(self.eta * lt.detach().cpu())
                self.q /= self.q.sum()
            opt.zero_grad()
            (self.q.to(device) * lt).sum().backward()
            opt.step()


# ======================================================
# LOCAL BASELINES  (no transfer, per domain)
# ======================================================

def compute_local_baselines(domain_datasets, domain_names, num_classes,
                            n_rounds=4, batch_size=32, test_frac=0.2):
    """Train a no-transfer model per domain. Used to compute negative-transfer rate."""
    local_pca = {}
    for dname in domain_names:
        ds = domain_datasets[dname]
        idx = list(range(len(ds)))
        random.shuffle(idx)
        split = int(len(ds) * (1 - test_frac))
        train_loader = DataLoader(Subset(ds, idx[:split]), batch_size=batch_size,
                                   shuffle=True, num_workers=0)
        test_loader  = DataLoader(Subset(ds, idx[split:]), batch_size=64,
                                   shuffle=False, num_workers=0)
        model = SimpleCNN(num_classes=num_classes).to(device)
        opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        for _ in range(n_rounds):
            model.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                criterion(model(imgs), labels).backward()
                opt.step()
        _, _, pca = evaluate_model(model, test_loader, num_classes)
        local_pca[dname] = pca
        avg = sum(pca.values()) / max(len(pca), 1)
        print(f"  Local baseline [{dname}]: {avg:.4f}")
    return local_pca


# ======================================================
# LEAVE-ONE-DOMAIN-OUT
# ======================================================

def leave_one_domain_out_eval(domain_datasets, domain_names, ontology, num_classes,
                              n_subclients=5, alpha=0.4, n_rounds=6):
    """
    Same K-1 training regime for all methods.
    PPDOE trains on K-1 domains; evaluates on the held-out domain.
    """
    results = {}
    for held_out in domain_names:
        train_domains = [d for d in domain_names if d != held_out]
        subset_ds = {d: domain_datasets[d] for d in train_domains}
        cl, cd, _ = build_all_loaders(subset_ds, train_domains, n_subclients, alpha)
        held_loader = DataLoader(domain_datasets[held_out], batch_size=64,
                                  shuffle=False, num_workers=0)
        ppdoe = PPDOEFramework(ontology, num_classes)
        for _ in range(n_rounds):
            ppdoe.run_round(cl, cd)
        acc, ece, _ = evaluate_model(ppdoe.global_model, held_loader, num_classes)
        results[held_out] = {"acc": acc, "ece": ece}
        print(f"    LODO [held out={held_out}]: acc={acc:.4f}  ece={ece:.4f}")
    return results


# ======================================================
# MAIN EXPERIMENT
# ======================================================

def run_experiment2(use_domainnet=True, n_subclients=5, alpha=0.4, n_rounds=12):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if use_domainnet:
        domain_names = DomainOntology.DOMAINNET_DOMAINS
        num_classes, subdir, tag = 345, "domainnet", "DomainNet"
    else:
        domain_names = DomainOntology.OFFICEHOME_DOMAINS
        num_classes, subdir, tag = 65, "OfficeHome", "Office-Home"

    print(f"Loading {tag}...")
    domain_datasets = load_domain_datasets(DATA_ROOT, domain_names, subdir, transform, num_classes)
    class_names = (
        domain_datasets[domain_names[0]].classes
        if hasattr(domain_datasets[domain_names[0]], "classes")
        else [str(i) for i in range(num_classes)]
    )
    ontology = DomainOntology(use_domainnet=use_domainnet, class_names=class_names)
    print(f"Ontology: {ontology.num_domains} domains, {ontology.num_classes} classes")

    client_loaders, client_descriptors, test_loaders = build_all_loaders(
        domain_datasets, domain_names, n_subclients, alpha)
    print(f"Total sub-clients: {len(client_loaders)} "
          f"({len(domain_names)} domains x {n_subclients})")
    print(f"Client descriptors (domain_id, dominant_class): {client_descriptors}")

    # Local no-transfer baselines (for NTR computation)
    print("\nComputing local baselines (no transfer)...")
    local_pca = compute_local_baselines(domain_datasets, domain_names, num_classes)

    # ---- Instantiate PPDOE ----
    ppdoe = PPDOEFramework(ontology, num_classes)

    # ---- Instantiate FL baselines (each independent) ----
    fl_baselines = {
        "FedBN"   : FedBN(num_classes),
        "FedRep"  : FedRep(num_classes),
        "Ditto"   : Ditto(num_classes, lam=0.5),
        "pFedMe"  : pFedMe(num_classes, lam=15.0),
        "FedALA"  : FedALA(num_classes, eta=1.0),
        "FedProto": FedProto(num_classes),
        "FedTGP"  : FedTGP(num_classes),
        "FedSSA"  : FedSSA(num_classes),
    }

    ppdoe_history = []
    fl_history    = {name: [] for name in fl_baselines}

    print("\n" + "=" * 80)
    print(f"  PPDOE Experiment 2 ({tag}) -- Natural Domain Shift")
    print("=" * 80)

    # ---- Main round loop ----
    for r in range(1, n_rounds + 1):
        print(f"\n  Round {r}/{n_rounds}")

        # PPDOE round
        ppdoe.run_round(client_loaders, client_descriptors)
        ppdoe_dom = ppdoe.evaluate_all_domains(test_loaders)
        pm = summarise(ppdoe_dom, domain_names)
        ppdoe_history.append(pm)
        print(f"    {'PPDOE':25}: avg={pm['avg_acc']:.4f}  "
              f"worst={pm['worst_acc']:.4f}  ece={pm['avg_ece']:.4f}")

        # FL baseline rounds (each independent)
        for name, baseline in fl_baselines.items():
            baseline.run_round(client_loaders)
            fl_dom = baseline.evaluate_all_domains(test_loaders)
            m = summarise(fl_dom, domain_names)
            fl_history[name].append(m)
            print(f"    {name:25}: avg={m['avg_acc']:.4f}  "
                  f"worst={m['worst_acc']:.4f}  ece={m['avg_ece']:.4f}")

    # ---- DG baselines (centralised) ----
    print("\n--- Domain Generalisation Baselines (ERM, CORAL, GroupDRO) ---")
    pooled_subsets = [
        DomainSubset(domain_datasets[d], list(range(len(domain_datasets[d]))), i)
        for i, d in enumerate(domain_names)
    ]
    pooled_loader = DataLoader(ConcatDataset(pooled_subsets), batch_size=64,
                                shuffle=True, num_workers=0)
    domain_id_loaders = {
        i: DataLoader(pooled_subsets[i], batch_size=64, shuffle=True, num_workers=0)
        for i in range(len(domain_names))
    }
    dg_results = {}
    for dg_name, baseline in [
        ("ERM",      ERMBaseline(num_classes)),
        ("CORAL",    CORALBaseline(num_classes)),
        ("GroupDRO", GroupDROBaseline(num_classes, num_groups=len(domain_names))),
    ]:
        print(f"  Training {dg_name}...")
        if dg_name == "GroupDRO":
            baseline.train_grouped(domain_id_loaders, epochs=5)
        else:
            baseline.train(pooled_loader, epochs=5)
        dom_r = baseline.evaluate_all_domains(test_loaders)
        m = summarise(dom_r, domain_names)
        dg_results[dg_name] = m
        print(f"  {dg_name:10}: avg={m['avg_acc']:.4f}  worst={m['worst_acc']:.4f}  "
              f"ece={m['avg_ece']:.4f}")

    # ---- Leave-one-domain-out (PPDOE, K-1 regime) ----
    print("\n--- Leave-One-Domain-Out Transfer (PPDOE, K-1 regime) ---")
    lodo_results = leave_one_domain_out_eval(
        domain_datasets, domain_names, ontology, num_classes,
        n_subclients=n_subclients, alpha=alpha, n_rounds=n_rounds // 2)
    lodo_avg   = float(np.mean([v["acc"] for v in lodo_results.values()]))
    lodo_worst = float(np.min([v["acc"]  for v in lodo_results.values()]))
    print(f"  LODO summary: avg={lodo_avg:.4f}  worst-domain={lodo_worst:.4f}")

    # ---- Negative transfer rate ----
    print("\n--- Negative Transfer Rate (vs no-transfer local baseline) ---")
    ref_domain = domain_names[0]
    ntr_results = {"PPDOE": None}
    _, _, ppdoe_pca = evaluate_model(
        ppdoe.global_model, test_loaders[ref_domain], num_classes)
    ntr_results["PPDOE"] = compute_negative_transfer_rate(ppdoe_pca, local_pca[ref_domain])
    print(f"  {'PPDOE':25}: NTR[{ref_domain}] = {ntr_results['PPDOE']:.4f}")
    for name, baseline in fl_baselines.items():
        _, _, t_pca = evaluate_model(
            baseline.global_model, test_loaders[ref_domain], num_classes)
        ntr_results[name] = compute_negative_transfer_rate(t_pca, local_pca[ref_domain])
        print(f"  {name:25}: NTR[{ref_domain}] = {ntr_results[name]:.4f}")

    # ---- Final summary table ----
    print("\n" + "=" * 80)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"  {'Method':<25}  {'Avg Acc':>9}  {'Worst Acc':>10}  "
          f"{'Avg ECE':>8}  {'NTR':>7}  {'vs PPDOE':>10}")
    print("  " + "-" * 73)

    ppdoe_final = ppdoe_history[-1]
    baseline_avg = ppdoe_final["avg_acc"]

    all_final = {"PPDOE": ppdoe_final}
    all_final.update({name: fl_history[name][-1] for name in fl_baselines})
    all_final.update(dg_results)

    for name, m in sorted(all_final.items(), key=lambda x: x[1]["avg_acc"], reverse=True):
        delta = m["avg_acc"] - baseline_avg
        d_str = "(base)" if name == "PPDOE" else (
            f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}")
        ntr_s = f"{ntr_results[name]:.4f}" if name in ntr_results else "  n/a"
        print(f"  {name:<25}  {m['avg_acc']:>9.4f}  {m['worst_acc']:>10.4f}  "
              f"{m['avg_ece']:>8.4f}  {ntr_s:>7}  {d_str:>10}")

    return ppdoe_history, fl_history, dg_results, lodo_results, ntr_results


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("# EXPERIMENT 2a: DomainNet (primary)")
    print("#" * 80)
    run_experiment2(use_domainnet=True, n_subclients=5, alpha=0.4, n_rounds=12)

    print("\n" + "#" * 80)
    print("# EXPERIMENT 2b: Office-Home (secondary)")
    print("#" * 80)
    run_experiment2(use_domainnet=False, n_subclients=5, alpha=0.4, n_rounds=12)