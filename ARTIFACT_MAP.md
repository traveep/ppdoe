# Artifact map 

| Paper item | What it shows | Primary inputs | How regenerated | Output / path |
|---|---|---|---|---|
| Table 3 MiniLM sweep | Name-embedding baseline vs 25-key | 50 labels, MiniLM scores, gold_relationships.json | scaffold notebook / baseline sweep | flat_embedding_baseline_results.csv |
| Table S theme sweep | Theme-lexicon threshold sweep | canonical scores (theme) | threshold on score_theme | from canonical_937_scores.csv |
| Table 4 head-to-head | Scaffold vs MiniLM at τ=0.50 | canonical combined + MiniLM | threshold_eval @ 0.50 | AP combined = 0.360 |
| Table 4b ablation | token / theme / combined / k2 | canonical_937_scores.csv | same file, four score columns | F1 0.000 / 0.444 / 0.449 / 0.500 |
| Table 5 RQ2 | Connected components vs 25 pairs | edges with score_combined ≥ τ | components + ARI/purity | descriptive only |
| Table 6 RQ4 | Status band vs key | status column in canonical file | gold-free assign_status | VALIDATED-band precision 0.643 |
| Figure 2 | F1 vs τ curves | Table S + Table 3 | plot from sweeps | figure |
| Figure 3–4 | Graph / domain-pair | selected edges @ 0.50 | from canonical selected flags | figure |
| Figure 5 | Status precision | Table 6 | bar chart | figure |
| Rater A check | Sensitivity on same 50 labels | rater list, frozen scorer | run_second_rater | F1 combined ≈ 0.182 |
| Hold-out 15 | Second list coverage | gold_holdout clean 15 | run_holdout | F1 combined ≈ 0.103 |
| Scoring recipe | Rebuild instructions | SCORING_RECIPE.md | human-readable | download/SCORING_RECIPE.md |
| Canonical scores | 937 rows all metrics | locked THEME_LEX + names | revision_eval locked scorer | canonical_scores/canonical_937_scores.csv |
| CIFAR Phase 0–3, 11 | Supporting only | earlier suite | provisional | not primary for main claim |


