# PPDOE Ontology Discovery Suite (Phases 12–15)

These phases measure **ontology discovery quality**, not only parent-assignment F1.
They address the fairness gap: accuracy-heavy tables under-emphasize exports,
stability, evidence, and version history.

## Quick start

```bash
# From package root (directory that contains ppdoe/ and tests/)
python tests/run_all.py --phases 12,13,14,15 --quick

# Full seed counts
python tests/run_all.py --phases 12,13,14,15
```

Or run individually:

```bash
python tests/run_phase12_ontology_exports.py --seeds 5
python tests/run_phase13_stability.py --seeds 5
python tests/run_phase14_evidence_profile.py --seeds 5
python tests/run_phase15_version_diffs.py --seeds 5
```

## What each phase produces

| Phase | Focus | Primary artifacts under `results/` |
|-------|--------|-------------------------------------|
| **12** Ontology exports | Machine-readable recovered structure | `ontology_exports/*.json`, `*_edges.csv`, `*.graphml`, `ontology_export_index.csv` |
| **13** Stability | Reproducibility across seeds & arrival orders | `stability_per_seed.csv`, `stability_summary.csv`, `stability_report.json` |
| **14** Evidence profiles | Why each edge was scored/confirmed | `evidence_edges.csv`, `evidence_bins.csv`, `evidence_summary.csv`, `evidence_report.json` |
| **15** Version diffs | Evolution trajectory & changelogs | `version_trajectory.csv`, `version_seed_summary.csv`, `version_dynamic_vs_frozen.csv`, `version_diffs/` |

## Fairness notes for clustering baselines

Phases 12–13 export **both**:

- **RAW** k-means parent IDs (no Hungarian alignment)
- **HUNGARIAN-aligned** parents (best match to gold coarse IDs)

PPDOE is **never** aligned: it uses fixed `coarse_0…19` ontology IDs.
Comparing PPDOE F1 only to *aligned* clustering overstates clustering’s advantage
for ontology discovery (the aligned labels are not a deployable ontology export).

## Alignment with “discover the ontology”

| Artifact | Why it matters beyond F1 |
|----------|---------------------------|
| GraphML / edge CSV / JSON | Shareable, inspectable ontology |
| Cross-seed Jaccard / parent agreement | Path dependence vs stability |
| Evidence (structural / term / contextual) | Explainability of recovered edges |
| Version changelogs | Lifecycle, evolution, dynamic vs frozen |

Existing Phases 0–11 remain unchanged and are still the accuracy / ablation suite.
