#!/usr/bin/env python
"""
PPDOE Experiment 1: Controlled Semantic Hierarchy Validation on CIFAR-100
Single-file script for Google Colab.
Run in Colab with GPU runtime.
"""

import copy
import random
import sys
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Standard CIFAR-100 fine -> coarse (superclass) mapping
FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0,
    11, 1, 10, 12, 14, 16, 9, 11, 5, 5, 19, 8, 17, 2, 4, 7, 7, 4, 11, 14, 18, 1, 2, 5, 12, 12, 16, 12,
    16, 17, 13, 14, 10, 18, 5, 10, 15, 6, 0, 17, 8, 14, 19, 9, 10, 13, 4, 15, 16, 6, 7, 13, 2, 4, 8, 11,
    3, 17, 0, 16, 9, 1, 15, 12, 19, 2, 10, 19, 8, 13,
]
cifar100_coarse = {fine: coarse for fine, coarse in enumerate(FINE_TO_COARSE)}

superclass_names = [
    "aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables",
    "household electrical devices", "household furniture", "insects", "large carnivores",
    "large man-made outdoor things", "large omnivores and herbivores", "medium-sized mammals",
    "non-insect invertebrates", "people", "reptiles", "small mammals", "trees",
    "vehicles 1", "vehicles 2", "vehicles 3",
]

trainset = None
testset = None
G = None


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def cosine_similarity_matrix(embeddings):
    """Pairwise cosine similarity without sklearn."""
    if isinstance(embeddings, torch.Tensor):
        norms = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (norms @ norms.T).cpu().numpy()
    norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
    return norms @ norms.T


def build_ontology_graph(class_names):
    ontology = nx.DiGraph()

    for i in range(100):
        ontology.add_node(f"class_{i}", type="fine", label=class_names[i])

    for sc in range(20):
        ontology.add_node(f"super_{sc}", type="coarse", label=superclass_names[sc])
        for fine, coarse in cifar100_coarse.items():
            if coarse == sc:
                ontology.add_edge(f"super_{sc}", f"class_{fine}", relation="is_a")

    try:
        from transformers import CLIPProcessor, CLIPModel

        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        class_texts = [class_names[i] for i in range(100)]
        inputs = clip_processor(text=class_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            text_emb = clip_model.get_text_features(**inputs)

        sim_matrix = cosine_similarity_matrix(text_emb)
        threshold = 0.65
        for i in range(100):
            for j in range(i + 1, 100):
                if sim_matrix[i, j] > threshold:
                    ontology.add_edge(
                        f"class_{i}", f"class_{j}", weight=float(sim_matrix[i, j]), type="similarity"
                    )
        print("CLIP similarity edges added.")
    except Exception as e:
        print("CLIP not available, using random similarities for demo:", e)
        rng = random.Random(42)
        for i in range(100):
            for j in range(i + 1, 100):
                if rng.random() > 0.9:
                    ontology.add_edge(f"class_{i}", f"class_{j}", weight=rng.random(), type="similarity")

    print(f"Ontology: {len(ontology.nodes())} nodes, {len(ontology.edges())} edges")
    return ontology


def load_datasets():
    global trainset, testset, G

    if trainset is not None and testset is not None and G is not None:
        return trainset, testset, G

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    G = build_ontology_graph(trainset.classes)
    print("CIFAR-100 loaded. Superclasses:", len(superclass_names))
    return trainset, testset, G


def get_client_dominant_superclass(indices, labels):
    coarse = [FINE_TO_COARSE[int(labels[i])] for i in indices]
    counts = np.bincount(coarse, minlength=20)
    return int(np.argmax(counts))


def semantic_weights(client_superclasses):
    weights = []
    for i, sc_i in enumerate(client_superclasses):
        sims = [1.0 if sc_i == client_superclasses[j] else 0.25 for j in range(len(client_superclasses)) if j != i]
        weights.append(float(np.mean(sims)) if sims else 1.0)
    total = sum(weights)
    return [w / total for w in weights] if total > 0 else [1.0 / len(client_superclasses)] * len(client_superclasses)


def partition_data(train_data, num_clients=20, partition_type="dirichlet", alpha=0.3, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    labels = np.array(train_data.targets)
    client_indices = [[] for _ in range(num_clients)]

    if partition_type == "dirichlet":
        min_size = 0
        while min_size < 10:
            client_indices = [[] for _ in range(num_clients)]
            for c in range(100):
                idx = np.where(labels == c)[0]
                if len(idx) == 0:
                    continue
                np.random.shuffle(idx)
                props = np.random.dirichlet([alpha] * num_clients)
                props = props / props.sum()
                split_points = (np.cumsum(props) * len(idx)).astype(int)[:-1]
                splits = np.split(idx, split_points)
                for client_idx, split in enumerate(splits):
                    client_indices[client_idx].extend(split.tolist())
            min_size = min(len(indices) for indices in client_indices)

    elif partition_type == "superclass_local":
        for sc in range(20):
            fine_classes = [f for f, coarse in cifar100_coarse.items() if coarse == sc]
            idx = np.concatenate([np.where(labels == f)[0] for f in fine_classes])
            np.random.shuffle(idx)
            client_indices[sc % num_clients].extend(idx.tolist())

    elif partition_type == "extreme_skew":
        for c in range(100):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                continue
            np.random.shuffle(idx)
            main_client = c % num_clients
            split = max(1, int(0.8 * len(idx)))
            client_indices[main_client].extend(idx[:split].tolist())
            remainder = idx[split:]
            if len(remainder) > 0:
                for other in random.sample(range(num_clients), min(3, num_clients)):
                    client_indices[other].extend(remainder.tolist())

    else:
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        split_size = len(train_data) // num_clients
        for i in range(num_clients):
            start = i * split_size
            end = (i + 1) * split_size if i < num_clients - 1 else len(train_data)
            client_indices[i] = indices[start:end]

    client_datasets = [Subset(train_data, idx) for idx in client_indices]
    sizes = [len(d) for d in client_datasets]
    print(f"Partition {partition_type}: client sizes (first 5) {sizes[:5]}, min={min(sizes)}")
    return client_datasets, client_indices


class SimpleResNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ClientModel:
    def __init__(self, model, client_id):
        self.model = model.to(device)
        self.client_id = client_id
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.prototypes = defaultdict(lambda: torch.zeros(512))

    def train_local(self, dataloader, epochs=1):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        for _ in range(epochs):
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        return self.model.state_dict()


def weighted_aggregation(client_states, weights):
    global_state = {}
    for key in client_states[0]:
        global_state[key] = sum(w * state[key] for w, state in zip(weights, client_states))
    return global_state


def fed_avg_aggregation(client_states):
    n = len(client_states)
    return weighted_aggregation(client_states, [1.0 / n] * n)


def ppdoe_aggregation(client_states, client_superclasses, ontology):
    weights = semantic_weights(client_superclasses)
    global_state = weighted_aggregation(client_states, weights)
    print(f"PPDOE: weighted aggregation over superclasses {client_superclasses}")
    return global_state


def run_federated(method="PPDOE", partition_type="dirichlet", num_clients=20, rounds=10, local_epochs=2):
    train_data, test_data, ontology = load_datasets()
    print(f"\n=== Running {method} on {partition_type} ===")
    clients_data, client_indices = partition_data(train_data, num_clients, partition_type)
    labels = np.array(train_data.targets)
    global_model = SimpleResNet().to(device)
    global_state = global_model.state_dict()

    accuracies = []
    for r in range(rounds):
        eligible_clients = [cid for cid in range(num_clients) if len(clients_data[cid]) > 0]
        if not eligible_clients:
            raise RuntimeError(f"No non-empty clients for partition {partition_type}")

        k = max(1, min(len(eligible_clients), max(4, num_clients // 4)))
        selected_clients = random.sample(eligible_clients, k=k)
        client_states = []
        selected_superclasses = []

        for cid in selected_clients:
            local_model = SimpleResNet()
            local_model.load_state_dict(global_state)
            client = ClientModel(local_model, cid)
            loader = DataLoader(clients_data[cid], batch_size=64, shuffle=True, num_workers=0)
            state = client.train_local(loader, local_epochs)
            client_states.append(state)
            selected_superclasses.append(get_client_dominant_superclass(client_indices[cid], labels))

        if method == "PPDOE":
            global_state = ppdoe_aggregation(client_states, selected_superclasses, ontology)
        else:
            global_state = fed_avg_aggregation(client_states)

        global_model.load_state_dict(global_state)
        global_model.eval()
        correct, total = 0, 0
        test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0)
        with torch.no_grad():
            for images, labels_batch in test_loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                outputs = global_model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()
        acc = 100 * correct / total
        accuracies.append(acc)
        print(f"Round {r + 1}/{rounds} - Acc: {acc:.2f}%")

    return accuracies


if __name__ == "__main__":
    print("=== PPDOE Experiment 1: CIFAR-100 Semantic Hierarchy ===")
    set_seed(42)
    print(f"Using device: {device}")

    quick = "--quick" in sys.argv
    rounds = 1 if quick else 5
    partition_types = ["dirichlet"] if quick else ["dirichlet", "superclass_local", "extreme_skew"]
    methods = ["FedAvg", "PPDOE"]

    results = {}
    for part in partition_types:
        results[part] = {}
        for meth in methods:
            accs = run_federated(meth, part, rounds=rounds)
            results[part][meth] = accs

    if HAS_MATPLOTLIB:
        plt.figure(figsize=(12, 6))
        for part in partition_types:
            for meth in methods:
                plt.plot(results[part][meth], label=f"{meth} - {part}")
        plt.xlabel("Round")
        plt.ylabel("Test Accuracy (%)")
        plt.title("PPDOE vs FedAvg on CIFAR-100 Partitions")
        plt.legend()
        plt.grid(True)
        plt.savefig("ppdoe_experiment1_results.png", dpi=150, bbox_inches="tight")
        print("Saved plot to ppdoe_experiment1_results.png")
        plt.show()
    else:
        print("matplotlib not installed; skipping plot. Results:", results)

    print("Experiment completed. Extend baselines, ontology validation, and full training for production use.")
