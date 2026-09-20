#!/usr/bin/env python3
"""
Phase 15: Ontology version diffs and evolution trajectory.

Tracks every commit of the versioned ontology across rounds:
  - concepts / relationships added, confirmed, deprecated per version
  - edge-set delta (Jaccard vs previous version)
  - structural change events
  - final vs intermediate snapshots

Also compares dynamic vs frozen trajectories under identical client
arrival so that the *only* difference is whether structural updates
continue.

Aligned with ontology discovery: version history is a first-class
artifact of PPDOE, not a side effect of F1 evaluation.
"""
from __future__ import annotations
import sys
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppdoe.data import load_cifar100, create_client_partitions
from ppdoe.ppdoe import PPDOE
from ppdoe.metrics import _edge_set, gold_edge_set, compute_metrics, jaccard_edge_sets
from ppdoe.ontology import Ontology

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
DIFFS = RESULTS / "version_diffs"
DIFFS.mkdir(exist_ok=True)


def snapshot_edges(ontology: Ontology, status: Optional[str] = "confirmed") -> Set[Tuple]:
    rels = ontology.list_relationships(status=status) if status else ontology.list_relationships()
    return _edge_set(rels, status_filter=status)


def diff_edge_sets(prev: Set, curr: Set) -> dict:
    added = curr - prev
    removed = prev - curr
    kept = prev & curr
    return {
        "n_added": len(added),
        "n_removed": len(removed),
        "n_kept": len(kept),
        "jaccard_vs_prev": jaccard_edge_sets(prev, curr),
        "added_sample": [list(e) for e in list(added)[:5]],
        "removed_sample": [list(e) for e in list(removed)[:5]],
    }


def run_trajectory(
    seed: int,
    data: dict,
    idxs,
    arrival: List[int],
    frozen: bool,
) -> Tuple[List[dict], Ontology]:
    """Sequential arrival; record per-round ontology state."""
    p = PPDOE(
        n_clients=10,
        n_rounds=len(arrival),
        seed=seed,
        theta=0.55,
        confirmation_threshold=0.20,
        enable_dynamic=True,
        enable_rl=True,
        active_fraction=1.0,
    )
    p.setup_cifar_clients(data["train_features"], data["train_fine"], idxs)
    if frozen:
        p.freeze(after_round=1)

    gold = gold_edge_set(data["fine_to_coarse"])
    present: List[int] = []
    traj = []
    prev_confirmed: Set = set()
    prev_all: Set = set()

    for step, client_idx in enumerate(arrival):
        present.append(client_idx)
        log = p.run_round(participating=list(present))
        m = compute_metrics(p.ontology, gold)

        conf = snapshot_edges(p.ontology, "confirmed")
        cand = snapshot_edges(p.ontology, "candidate")
        all_e = snapshot_edges(p.ontology, None)

        d_conf = diff_edge_sets(prev_confirmed, conf)
        d_all = diff_edge_sets(prev_all, all_e)

        # history entry from ontology
        hist = p.ontology.history[-1] if p.ontology.history else {}

        row = {
            "seed": seed,
            "condition": "frozen" if frozen else "dynamic",
            "step": step,
            "round": log["round"],
            "client_arrived": client_idx,
            "n_present": len(present),
            "ontology_version": p.ontology.version,
            "n_concepts": len(p.ontology.concepts),
            "n_relationships": len(p.ontology.relationships),
            "n_confirmed": len(conf),
            "n_candidates": len(cand),
            "structural_change": bool(log.get("structural_change")),
            "frozen_flag": bool(log.get("frozen")),
            "relationship_f1": m.f1,
            "precision": m.precision,
            "recall": m.recall,
            "confirmed_added": d_conf["n_added"],
            "confirmed_removed": d_conf["n_removed"],
            "confirmed_jaccard_vs_prev": d_conf["jaccard_vs_prev"],
            "all_jaccard_vs_prev": d_all["jaccard_vs_prev"],
            "history_note": hist.get("note", ""),
            "state_hash": p.ontology.current_state_hash()[:16],
        }
        traj.append(row)
        prev_confirmed = conf
        prev_all = all_e

    return traj, p.ontology


def write_version_changelog(seed: int, condition: str, ontology: Ontology, traj: List[dict]) -> Path:
    """Human-readable changelog of ontology commits."""
    path = DIFFS / f"changelog_seed{seed}_{condition}.md"
    lines = [
        f"# Ontology version changelog — seed={seed} condition={condition}",
        "",
        f"Final version: **{ontology.version}**",
        f"Concepts: {len(ontology.concepts)} | Relationships: {len(ontology.relationships)}",
        f"Confirmed: {len(ontology.list_relationships(status='confirmed'))}",
        f"State hash: `{ontology.current_state_hash()[:32]}…`",
        "",
        "## Per-round evolution",
        "",
        "| Step | Version | Confirmed | Δ+ | Δ− | Jaccard | Struct? | F1 |",
        "|------|---------|-----------|----|----|---------|---------|----|",
    ]
    for r in traj:
        lines.append(
            f"| {r['step']} | {r['ontology_version']} | {r['n_confirmed']} | "
            f"{r['confirmed_added']} | {r['confirmed_removed']} | "
            f"{r['confirmed_jaccard_vs_prev']:.3f} | "
            f"{'Y' if r['structural_change'] else 'N'} | {r['relationship_f1']:.3f} |"
        )
    lines.append("")
    lines.append("## Ontology history snapshots")
    lines.append("")
    for h in ontology.history:
        lines.append(
            f"- v{h.get('version')}: concepts={h.get('n_concepts')} "
            f"rels={h.get('n_relationships')} confirmed={h.get('n_confirmed')} "
            f"— _{h.get('note', '')}_"
        )
    lines.append("")
    lines.append("## Confirmed is_a edges (final)")
    lines.append("")
    for r in ontology.list_relationships(status="confirmed"):
        if r.rel_type == "is_a":
            lines.append(
                f"- `{r.source_id}` → `{r.target_id}` "
                f"(conf={r.confidence:.3f}, v{r.version_created})"
            )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(n_seeds: int = 3):
    print(f"Phase 15 – Version diffs ({n_seeds} seeds)")
    print("Dynamic vs frozen trajectories; changelogs under results/version_diffs/")
    data = load_cifar100(prefer_synthetic=True)

    all_traj: List[dict] = []
    seed_summaries: List[dict] = []

    for seed in range(n_seeds):
        idxs = create_client_partitions(
            data["n_train"], 10, seed=seed, labels=data["train_fine"]
        )
        rng = np.random.RandomState(seed)
        arrival = rng.permutation(10).tolist()
        print(f"  seed {seed} arrival={arrival}")

        for frozen in (False, True):
            cond = "frozen" if frozen else "dynamic"
            traj, onto = run_trajectory(seed, data, idxs, arrival, frozen)
            all_traj.extend(traj)
            changelog = write_version_changelog(seed, cond, onto, traj)

            # dump final ontology JSON for this condition
            snap_path = DIFFS / f"final_seed{seed}_{cond}.json"
            with open(snap_path, "w") as f:
                json.dump(onto.to_dict(), f, indent=2)

            n_struct = sum(1 for r in traj if r["structural_change"])
            seed_summaries.append({
                "seed": seed,
                "condition": cond,
                "final_version": onto.version,
                "final_confirmed": len(onto.list_relationships(status="confirmed")),
                "final_concepts": len(onto.concepts),
                "structural_change_events": n_struct,
                "final_f1": traj[-1]["relationship_f1"] if traj else 0.0,
                "mean_confirmed_jaccard_step": float(np.mean(
                    [r["confirmed_jaccard_vs_prev"] for r in traj[1:]]
                )) if len(traj) > 1 else 1.0,
                "changelog": changelog.name,
                "final_snapshot": snap_path.name,
                "final_state_hash": onto.current_state_hash()[:16],
            })
            print(
                f"    {cond}: version={onto.version} confirmed={seed_summaries[-1]['final_confirmed']} "
                f"struct_events={n_struct} F1={seed_summaries[-1]['final_f1']:.4f}"
            )

    # Cross-condition: for each seed, Jaccard(final_dynamic, final_frozen)
    cross = []
    for seed in range(n_seeds):
        dyn = next(s for s in seed_summaries if s["seed"] == seed and s["condition"] == "dynamic")
        frz = next(s for s in seed_summaries if s["seed"] == seed and s["condition"] == "frozen")
        # reload edges from traj last rows
        dyn_traj = [r for r in all_traj if r["seed"] == seed and r["condition"] == "dynamic"]
        frz_traj = [r for r in all_traj if r["seed"] == seed and r["condition"] == "frozen"]
        # approximate via confirmed counts + hash difference
        cross.append({
            "seed": seed,
            "dynamic_final_f1": dyn["final_f1"],
            "frozen_final_f1": frz["final_f1"],
            "delta_f1": dyn["final_f1"] - frz["final_f1"],
            "dynamic_struct_events": dyn["structural_change_events"],
            "frozen_struct_events": frz["structural_change_events"],
            "same_final_hash": dyn["final_state_hash"] == frz["final_state_hash"],
            "dynamic_version": dyn["final_version"],
            "frozen_version": frz["final_version"],
        })

    out1 = RESULTS / "version_trajectory.csv"
    with open(out1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_traj[0].keys()))
        w.writeheader()
        w.writerows(all_traj)
    print(f"Wrote {out1}")

    out2 = RESULTS / "version_seed_summary.csv"
    with open(out2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seed_summaries[0].keys()))
        w.writeheader()
        w.writerows(seed_summaries)
    print(f"Wrote {out2}")

    out3 = RESULTS / "version_dynamic_vs_frozen.csv"
    with open(out3, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cross[0].keys()))
        w.writeheader()
        w.writerows(cross)
    print(f"Wrote {out3}")

    report = {
        "n_seeds": n_seeds,
        "mean_delta_f1_dynamic_minus_frozen": float(np.mean([c["delta_f1"] for c in cross])),
        "mean_struct_events_dynamic": float(np.mean([c["dynamic_struct_events"] for c in cross])),
        "mean_struct_events_frozen": float(np.mean([c["frozen_struct_events"] for c in cross])),
        "n_identical_final_hash": sum(1 for c in cross if c["same_final_hash"]),
        "note": (
            "Version diffs capture ontology evolution as a sequence of commits. "
            "Dynamic continues structural updates; frozen stops after round 1. "
            "Changelogs and final JSON snapshots are under results/version_diffs/."
        ),
        "artifacts_dir": str(DIFFS),
    }
    out4 = RESULTS / "version_diff_report.json"
    with open(out4, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out4}")
    print(json.dumps(report, indent=2))
    print("Phase 15 complete.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    main(ap.parse_args().seeds)
