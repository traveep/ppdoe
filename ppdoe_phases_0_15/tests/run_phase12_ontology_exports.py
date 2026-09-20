#!/usr/bin/env python3
"""
Phase 12: Ontology export artifacts.

Produces machine-readable ontology snapshots (JSON + GraphML + edge lists)
so that recovered structure can be inspected, versioned, and shared
independent of F1 tables.

Aligned with the goal of *discovering* the ontology, not only scoring
parent assignment accuracy.
"""
from __future__ import annotations
import sys
import csv
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppdoe.data import load_cifar100, create_client_partitions
from ppdoe.ppdoe import PPDOE
from ppdoe.metrics import gold_edge_set, _edge_set, compute_metrics
from ppdoe.baselines import fedavg_kmeans, flis_style

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
EXPORTS = RESULTS / "ontology_exports"
EXPORTS.mkdir(exist_ok=True)


def ontology_to_edge_list(ontology) -> List[dict]:
    rows = []
    for r in ontology.list_relationships():
        rows.append({
            "rel_id": r.rel_id,
            "source_id": r.source_id,
            "target_id": r.target_id,
            "rel_type": r.rel_type,
            "confidence": float(r.confidence),
            "status": r.status,
            "version_created": int(r.version_created),
            "structural": float(r.evidence.get("structural", 0.0)),
            "terminological": float(r.evidence.get("terminological", 0.0)),
            "contextual": float(r.evidence.get("contextual", 0.0)),
            "is_gold": bool(r.is_gold),
        })
    return rows


def ontology_to_graphml(ontology, path: Path) -> None:
    """Minimal GraphML for confirmed + candidate is_a edges."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <key id="name" for="node" attr.name="name" attr.type="string"/>',
        '  <key id="domain" for="node" attr.name="domain" attr.type="string"/>',
        '  <key id="rel_type" for="edge" attr.name="rel_type" attr.type="string"/>',
        '  <key id="confidence" for="edge" attr.name="confidence" attr.type="double"/>',
        '  <key id="status" for="edge" attr.name="status" attr.type="string"/>',
        '  <graph id="G" edgedefault="directed">',
    ]
    for cid, c in ontology.concepts.items():
        name = (c.name or cid).replace('"', "'")
        domain = (c.domain or "").replace('"', "'")
        lines.append(f'    <node id="{cid}">')
        lines.append(f'      <data key="name">{name}</data>')
        lines.append(f'      <data key="domain">{domain}</data>')
        lines.append("    </node>")
    for r in ontology.list_relationships():
        if r.status not in ("confirmed", "candidate"):
            continue
        lines.append(
            f'    <edge id="{r.rel_id}" source="{r.source_id}" target="{r.target_id}">'
        )
        lines.append(f'      <data key="rel_type">{r.rel_type}</data>')
        lines.append(f'      <data key="confidence">{float(r.confidence):.6f}</data>')
        lines.append(f'      <data key="status">{r.status}</data>')
        lines.append("    </edge>")
    lines.append("  </graph>")
    lines.append("</graphml>")
    path.write_text("\n".join(lines), encoding="utf-8")


def export_method_ontology(
    method: str,
    seed: int,
    ontology,
    gold: Dict[int, int],
) -> dict:
    """Write JSON snapshot, edge CSV, GraphML; return summary row."""
    stem = f"{method}_seed{seed}"
    snap = {
        "method": method,
        "seed": seed,
        "ontology_name": ontology.name,
        "version": ontology.version,
        "n_concepts": len(ontology.concepts),
        "n_relationships": len(ontology.relationships),
        "n_confirmed": len(ontology.list_relationships(status="confirmed")),
        "n_candidates": len(ontology.list_relationships(status="candidate")),
        "n_rejected": len(ontology.list_relationships(status="rejected")),
        "state_hash": ontology.current_state_hash(),
        "concepts": {cid: c.to_dict() for cid, c in ontology.concepts.items()},
        "relationships": {rid: r.to_dict() for rid, r in ontology.relationships.items()},
        "history": ontology.history,
    }
    json_path = EXPORTS / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump(snap, f, indent=2)

    edges = ontology_to_edge_list(ontology)
    edge_path = EXPORTS / f"{stem}_edges.csv"
    if edges:
        with open(edge_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(edges[0].keys()))
            w.writeheader()
            w.writerows(edges)
    else:
        edge_path.write_text("rel_id,source_id,target_id,rel_type,confidence,status\n")

    graphml_path = EXPORTS / f"{stem}.graphml"
    ontology_to_graphml(ontology, graphml_path)

    gold_edges = gold_edge_set(gold)
    m = compute_metrics(ontology, gold_edges)
    return {
        "method": method,
        "seed": seed,
        "export_json": str(json_path.name),
        "export_edges_csv": str(edge_path.name),
        "export_graphml": str(graphml_path.name),
        "n_concepts": len(ontology.concepts),
        "n_confirmed": m.confirmed_edges,
        "n_candidates": m.candidate_edges,
        "relationship_f1": m.f1,
        "precision": m.precision,
        "recall": m.recall,
        "state_hash": ontology.current_state_hash()[:16],
        "ontology_version": ontology.version,
    }


def baseline_to_ontology(features, labels, idxs, gold, seed, style: str):
    """Run clustering baseline and return its ontology (already built in baselines)."""
    if style == "fedavg_kmeans":
        m = fedavg_kmeans(features, labels, idxs, gold, seed=seed)
    else:
        m = flis_style(features, labels, idxs, gold, seed=seed)
    # baselines return metrics only; rebuild a minimal ontology for export
    from ppdoe.ontology import Ontology, Concept, Relationship
    from ppdoe.baselines import _fine_prototypes, _kmeans, _hungarian_align
    protos = {}
    for ix in idxs:
        for lab, p in _fine_prototypes(features, labels, [ix]).items():
            protos.setdefault(lab, []).append(p)
    protos = {lab: np.mean(vs, axis=0) for lab, vs in protos.items()}
    labs = sorted(protos.keys())
    X = np.stack([protos[l] for l in labs])
    _, cids = _kmeans(X, k=20, seed=seed)
    pred = {labs[i]: int(cids[i]) for i in range(len(labs))}
    aligned = _hungarian_align(pred, gold)
    onto = Ontology(name=style)
    for f in range(100):
        onto.add_concept(Concept(f"cifar100_class_{f}", f"c{f}", "cifar100"))
    for c in range(20):
        onto.add_concept(Concept(f"cifar100_coarse_{c}", f"coarse{c}", "cifar100"))
    for fine, parent in aligned.items():
        onto.add_relationship(Relationship(
            rel_id=f"{style}_{fine}_{parent}",
            source_id=f"cifar100_class_{fine}",
            target_id=f"cifar100_coarse_{parent}",
            rel_type="is_a",
            confidence=1.0,
            status="confirmed",
        ))
    onto.commit_version(note=f"{style}_aligned")
    return onto


def main(n_seeds: int = 3):
    print(f"Phase 12 – Ontology exports ({n_seeds} seeds)")
    print("Artifacts: JSON snapshots, edge CSVs, GraphML under results/ontology_exports/")
    data = load_cifar100(prefer_synthetic=True)
    gold = data["fine_to_coarse"]
    rows = []

    for seed in range(n_seeds):
        idxs = create_client_partitions(
            data["n_train"], 10, seed=seed, labels=data["train_fine"]
        )
        print(f"  seed {seed}")

        # Full PPDOE
        p = PPDOE(
            n_clients=10, n_rounds=6, seed=seed,
            confirmation_threshold=0.20, theta=0.5, enable_rl=True,
        )
        p.setup_cifar_clients(data["train_features"], data["train_fine"], idxs)
        p.run()
        rows.append(export_method_ontology("full_ppdoe", seed, p.ontology, gold))
        print(f"    full_ppdoe: confirmed={rows[-1]['n_confirmed']} F1={rows[-1]['relationship_f1']:.4f}")

        # Static (no dynamic)
        p2 = PPDOE(
            n_clients=10, n_rounds=6, seed=seed,
            enable_dynamic=False, enable_rl=False,
            confirmation_threshold=0.20, theta=0.5,
        )
        p2.setup_cifar_clients(data["train_features"], data["train_fine"], idxs)
        p2.run()
        rows.append(export_method_ontology("static_ontology", seed, p2.ontology, gold))

        # Clustering baselines (with Hungarian alignment for fair export)
        for style in ("fedavg_kmeans", "flis"):
            onto = baseline_to_ontology(
                data["train_features"], data["train_fine"], idxs, gold, seed, style
            )
            rows.append(export_method_ontology(style, seed, onto, gold))
            print(f"    {style}: confirmed={rows[-1]['n_confirmed']} F1={rows[-1]['relationship_f1']:.4f}")

    out = RESULTS / "ontology_export_index.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")
    print(f"Export directory: {EXPORTS}")
    print("Phase 12 complete.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    main(ap.parse_args().seeds)
