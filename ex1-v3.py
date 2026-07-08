import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import random
import copy
from typing import Dict, List

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("cpu")
print(f"Using device: {device}")


# ====================== ONTOLOGY ======================
class CIFAR100Ontology:
    def __init__(self):
        self.fine_to_coarse = [
            4,1,14,8,0,6,7,18,3,3,14,9,18,7,11,3,9,7,11,6,11,5,10,7,6,13,15,3,15,0,
            11,1,10,12,14,16,9,11,5,5,19,8,17,2,4,7,7,4,11,14,18,1,2,5,12,12,16,12,
            16,17,13,14,10,18,5,10,15,6,0,17,8,14,19,9,10,13,4,15,16,6,7,13,2,4,8,11,
            3,17,0,16,9,1,15,12,19,2,10,19,8,13
        ]
        self.num_superclasses = 20

    def get_superclass(self, fine_label):
        return self.fine_to_coarse[fine_label]

    def get_similarity(self, sc1, sc2):
        return 1.0 if sc1 == sc2 else 0.25


# ====================== MODEL ======================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


# ====================== DATA ======================
def create_client_partitions(trainset, n_clients=12, alpha=0.4):
    labels = np.array(trainset.targets)
    ontology = CIFAR100Ontology()
    partitions = []
    for i in range(n_clients - 6):
        probs = np.random.dirichlet([alpha] * 100)
        p = probs[labels]; p /= p.sum()
        selected = np.random.choice(len(trainset), size=800, p=p, replace=False)
        partitions.append(selected.tolist())
    for i in range(n_clients - 6, n_clients - 3):
        sc = random.randint(0, 19)
        idx = [j for j in range(len(trainset)) if ontology.get_superclass(trainset.targets[j]) == sc]
        partitions.append(random.sample(idx, min(1000, len(idx))))
    for i in range(n_clients - 3, n_clients):
        dominant = random.randint(0, 99)
        idx = [j for j in range(len(trainset)) if trainset.targets[j] == dominant]
        partitions.append(idx[:800])
    return partitions


def get_client_dominant_superclass(indices, trainset, ontology):
    """Derive dominant superclass from actual client data (not random)."""
    labels = [trainset.targets[i] for i in indices]
    sc = [ontology.get_superclass(l) for l in labels]
    counts = np.bincount(sc, minlength=ontology.num_superclasses)
    return int(np.argmax(counts))


# ====================== PPDOE FRAMEWORK ======================
class PPDOEFramework:
    """
    PPDOE outer framework. Manages the global model, ontology-based
    client weighting, and evaluation. Each strategy plugs in via
    local_train() and aggregate().
    """
    def __init__(self, strategy, ontology):
        self.strategy = strategy
        self.ontology = ontology
        self.global_model = SimpleCNN().to(device)
        self.strategy.init(self.global_model)

    def run_round(self, client_loaders, client_superclasses):
        weights = self._semantic_weights(client_superclasses)
        client_updates = [
            self.strategy.local_train(i, loader, self.global_model)
            for i, loader in enumerate(client_loaders)
        ]
        new_state = self.strategy.aggregate(client_updates, weights, self.global_model)
        self.global_model.load_state_dict(new_state)
        self.strategy.init(self.global_model)  # sync strategy with new global model

    def _semantic_weights(self, client_superclasses):
        n = len(client_superclasses)
        w = []
        for i in range(n):
            sims = [self.ontology.get_similarity(client_superclasses[i], client_superclasses[j])
                    for j in range(n) if j != i]
            w.append(np.mean(sims) if sims else 1.0)
        total = sum(w)
        return [x / total for x in w]

    def evaluate(self, test_loader):
        self.global_model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                pred = self.global_model(images).argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        return correct / total


# ====================== STRATEGIES ======================
class BaseStrategy:
    """Default: standard local SGD training + weighted FedAvg aggregation."""
    def init(self, global_model):
        self.global_state = copy.deepcopy(global_model.state_dict())

    def local_train(self, client_id, loader, global_model, epochs=1, lr=0.01):
        model = SimpleCNN().to(device)
        model.load_state_dict(global_model.state_dict())
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                opt.zero_grad()
                criterion(model(images), labels).backward()
                opt.step()
        return model.state_dict()

    def aggregate(self, client_updates, weights, global_model):
        """PPDOE ontology-weighted aggregation (default for all strategies)."""
        avg = {}
        for key in client_updates[0]:
            avg[key] = sum(w * u[key] for w, u in zip(weights, client_updates))
        return avg


class FedAvgStrategy(BaseStrategy):
    """FedAvg: ignores PPDOE weights, uses uniform average."""
    def aggregate(self, client_updates, weights, global_model):
        n = len(client_updates)
        avg = {}
        for key in client_updates[0]:
            avg[key] = sum(u[key] for u in client_updates) / n
        return avg


class FedProxStrategy(BaseStrategy):
    def __init__(self, mu=0.01): self.mu = mu
    def local_train(self, client_id, loader, global_model, epochs=1, lr=0.01):
        model = SimpleCNN().to(device)
        model.load_state_dict(global_model.state_dict())
        global_params = {k: v.clone() for k, v in global_model.state_dict().items()}
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                opt.zero_grad()
                loss = criterion(model(images), labels)
                prox = sum(((p - global_params[k])**2).sum()
                           for k, p in model.named_parameters() if k in global_params)
                (loss + (self.mu/2)*prox).backward()
                opt.step()
        return model.state_dict()


class SCAFFOLDStrategy(BaseStrategy):
    def __init__(self, n_clients, lr=0.01):
        self.lr = lr; self.n_clients = n_clients
        self.server_c = None; self.client_c = [None]*n_clients
    def init(self, global_model):
        super().init(global_model)
        if self.server_c is None:
            self.server_c = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}
            self.client_c = [{k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}
                             for _ in range(self.n_clients)]
    def local_train(self, client_id, loader, global_model, epochs=1, lr=None):
        lr = lr or self.lr
        model = SimpleCNN().to(device)
        model.load_state_dict(global_model.state_dict())
        init_state = copy.deepcopy(model.state_dict())
        ci = self.client_c[client_id]; sc = self.server_c
        criterion = nn.CrossEntropyLoss()
        model.train(); steps = 0
        for _ in range(epochs):
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                loss = criterion(model(images), labels)
                grads = torch.autograd.grad(loss, model.parameters())
                with torch.no_grad():
                    for (name, p), g, c_i, c_s in zip(
                        model.named_parameters(), grads, ci.values(), sc.values()):
                        p -= lr * (g - c_i + c_s)
                steps += 1
        new_ci = {k: ci[k] - sc[k] + (init_state[k] - model.state_dict()[k]) / (steps*lr + 1e-8)
                  for k in ci}
        self.client_c[client_id] = new_ci
        return model.state_dict()
    def aggregate(self, client_updates, weights, global_model):
        n = len(client_updates)
        for key in self.server_c:
            self.server_c[key] += sum(self.client_c[i][key] for i in range(n)) / n
        avg = {key: sum(u[key] for u in client_updates)/n for key in client_updates[0]}
        return avg


class FedDynStrategy(BaseStrategy):
    def __init__(self, n_clients, alpha=0.01):
        self.alpha = alpha; self.n_clients = n_clients
        self.h = [None]*n_clients
    def init(self, global_model):
        super().init(global_model)
        for i in range(self.n_clients):
            if self.h[i] is None:
                self.h[i] = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}
    def local_train(self, client_id, loader, global_model, epochs=1, lr=0.01):
        model = SimpleCNN().to(device)
        model.load_state_dict(global_model.state_dict())
        global_state = {k: v.clone() for k, v in global_model.state_dict().items()}
        h_i = self.h[client_id]
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                opt.zero_grad()
                loss = criterion(model(images), labels)
                reg = sum((-h_i[k]*model.state_dict()[k]).sum() +
                          (self.alpha/2)*((model.state_dict()[k]-global_state[k])**2).sum()
                          for k in h_i)
                (loss + reg).backward()
                opt.step()
        with torch.no_grad():
            for k in h_i:
                self.h[client_id][k] = h_i[k] - self.alpha*(model.state_dict()[k] - global_state[k])
        return model.state_dict()


class FedBNStrategy(BaseStrategy):
    """Keep BN layers local; aggregate all other layers with PPDOE weights."""
    def aggregate(self, client_updates, weights, global_model):
        avg = {}
        for key in client_updates[0]:
            if any(x in key for x in ["running_", "num_batches"]):
                avg[key] = global_model.state_dict()[key]  # keep global BN stats
            else:
                avg[key] = sum(w * u[key] for w, u in zip(weights, client_updates))
        return avg


class FedProtoStrategy(BaseStrategy):
    def __init__(self, num_classes=100):
        self.num_classes = num_classes; self.global_protos = {}
    def local_train(self, client_id, loader, global_model, epochs=1, lr=0.01):
        model = SimpleCNN().to(device)
        model.load_state_dict(global_model.state_dict())
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        local_protos = {c: [] for c in range(self.num_classes)}
        model.train()
        for _ in range(epochs):
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                opt.zero_grad()
                emb = model.features(images).view(images.size(0), -1)
                loss = criterion(model.classifier(emb), labels)
                if self.global_protos:
                    pl = sum(((emb[i] - self.global_protos[l.item()].to(device))**2).sum()
                             for i, l in enumerate(labels) if l.item() in self.global_protos)
                    loss += 0.1 * pl / len(labels)
                loss.backward(); opt.step()
                for i, l in enumerate(labels):
                    local_protos[l.item()].append(emb[i].detach().cpu())
        client_protos = {c: torch.stack(v).mean(0) for c, v in local_protos.items() if v}
        return {"state_dict": model.state_dict(), "protos": client_protos}
    def aggregate(self, client_updates, weights, global_model):
        all_protos = {}
        for u in client_updates:
            for c, p in u["protos"].items():
                all_protos.setdefault(c, []).append(p)
        self.global_protos = {c: torch.stack(v).mean(0) for c, v in all_protos.items()}
        state_dicts = [u["state_dict"] for u in client_updates]
        avg = {key: sum(w*sd[key] for w, sd in zip(weights, state_dicts))
               for key in state_dicts[0]}
        return avg


class MOONStrategy(BaseStrategy):
    def __init__(self, mu=1.0, temperature=0.5):
        self.mu = mu; self.temperature = temperature
        self.prev_models = {}
    def local_train(self, client_id, loader, global_model, epochs=1, lr=0.01):
        model = SimpleCNN().to(device)
        model.load_state_dict(global_model.state_dict())
        prev = SimpleCNN().to(device)
        prev.load_state_dict(self.prev_models.get(client_id, global_model.state_dict()))
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        cos = nn.CosineSimilarity(dim=1)
        model.train()
        for _ in range(epochs):
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                opt.zero_grad()
                z = model.features(images).view(images.size(0), -1)
                z_g = global_model.features(images).view(images.size(0), -1).detach()
                z_p = prev.features(images).view(images.size(0), -1).detach()
                loss = criterion(model.classifier(z), labels)
                pos = cos(z, z_g)/self.temperature
                neg = cos(z, z_p)/self.temperature
                loss += self.mu * (-torch.log(torch.exp(pos)/(torch.exp(pos)+torch.exp(neg)+1e-8))).mean()
                loss.backward(); opt.step()
        self.prev_models[client_id] = copy.deepcopy(model.state_dict())
        return model.state_dict()


class PPDOENativeStrategy(BaseStrategy):
    """PPDOE's own aggregation: pure ontology-weighted average (the baseline)."""
    pass  # BaseStrategy.aggregate already does ontology-weighted avg


# ====================== EXPERIMENT ======================
def run_ppdoe_comparison(n_clients=12, n_rounds=12):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
    ])
    print("Downloading CIFAR-100...")
    trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    testset  = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    ontology = CIFAR100Ontology()

    client_indices = create_client_partitions(trainset, n_clients, alpha=0.4)
    client_loaders = [DataLoader(Subset(trainset, idx), batch_size=32, shuffle=True, num_workers=0)
                      for idx in client_indices]
    test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

    # Derive superclasses from actual client data (not random)
    client_superclasses = [
        get_client_dominant_superclass(client_indices[i], trainset, ontology)
        for i in range(n_clients)
    ]
    print(f"Client superclasses (from data): {client_superclasses}")

    strategies = {
        "PPDOE (native)" : PPDOENativeStrategy(),
        "PPDOE + FedAvg" : FedAvgStrategy(),
        "PPDOE + FedProx": FedProxStrategy(mu=0.01),
        "PPDOE + SCAFFOLD": SCAFFOLDStrategy(n_clients=n_clients),
        "PPDOE + FedDyn" : FedDynStrategy(n_clients=n_clients, alpha=0.01),
        "PPDOE + FedBN"  : FedBNStrategy(),
        "PPDOE + FedProto": FedProtoStrategy(num_classes=100),
        "PPDOE + MOON"   : MOONStrategy(mu=1.0),
    }

    # Each strategy gets its own persistent PPDOEFramework (model persists across rounds)
    frameworks = {
        name: PPDOEFramework(strategy=strat, ontology=ontology)
        for name, strat in strategies.items()
    }
    history = {name: [] for name in strategies}

    print("\n" + "="*70)
    print("  PPDOE Framework — Strategy Comparison (Non-IID + Cross-Domain)")
    print("="*70)

    for r in range(1, n_rounds + 1):
        print(f"\n Round {r}/{n_rounds}")
        for name, framework in frameworks.items():
            framework.run_round(client_loaders, client_superclasses)
            acc = framework.evaluate(test_loader)
            history[name].append(acc)
            print(f"  {name:25}: {acc:.4f}")

    # Final summary
    print("\n" + "="*70)
    print("  FINAL RESULTS")
    print("="*70)
    final = {name: scores[-1] for name, scores in history.items()}
    baseline = final["PPDOE (native)"]
    for name, score in sorted(final.items(), key=lambda x: x[1], reverse=True):
        delta = score - baseline
        tag = "  <- baseline" if name == "PPDOE (native)" else (f"  +{delta:.4f}" if delta >= 0 else f"  {delta:.4f}")
        print(f"  {name:25}: {score:.4f}{tag}")

    return history


if __name__ == "__main__":
    history = run_ppdoe_comparison(n_clients=12, n_rounds=12)