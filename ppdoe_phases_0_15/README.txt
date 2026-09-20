PPDOE test suite — phases 0 through 15

Layout
  ppdoe/     library (aggregator, baselines, client, ontology, ...)
  tests/     run_phase0 ... run_phase15 + run_all.py

Run
  python tests/run_all.py --seeds 50
  python tests/run_phase12_ontology_exports.py --seeds 50
  python tests/run_phase13_stability.py --seeds 50
  python tests/run_phase14_evidence_profile.py --seeds 50
  python tests/run_phase15_version_diffs.py --seeds 50
