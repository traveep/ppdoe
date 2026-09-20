#!/usr/bin/env python3
"""
Master runner – execute Phases 0–15.

All seeded phases default to 50 seeds so a full suite run matches Phase 1 scale.
Use --quick for a reduced smoke test.

Usage:
  python tests/run_all.py                       # full blast (50 seeds everywhere)
  python tests/run_all.py --quick               # reduced seeds for smoke test
  python tests/run_all.py --phases 0,1,3        # selected phases only
  python tests/run_all.py --phases 12,12b,13,14,15
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent

# Full-blast defaults: every seeded phase uses 50 seeds (matches Phase 1).
PHASES = {
    "0":  ("run_phase0_audit.py", []),
    "1":  ("run_phase1_main.py", ["--seeds", "50"]),
    "2":  ("run_phase2_ablation.py", ["--seeds", "50"]),
    "3":  ("run_phase3_temporal.py", ["--seeds", "50"]),
    "4":  ("run_phase4_arrival_order.py", ["--seeds", "50"]),
    "5":  ("run_phase5_missing_edge.py", ["--seeds", "50"]),
    "6":  ("run_phase6_shift.py", ["--seeds", "50"]),
    "7":  ("run_phase7_subscriber.py", ["--seeds", "50"]),
    "8":  ("run_phase8_confidence.py", ["--seeds", "50"]),
    "9":  ("run_phase9_privacy.py", ["--seeds", "50"]),
    "10": ("run_phase10_security.py", ["--seeds", "50"]),
    "11": ("run_phase11_scalability.py", ["--seeds", "50"]),
    # Ontology discovery suite
    "12": ("run_phase12_ontology_exports.py", ["--seeds", "50"]),
    "12b": ("run_phase12b_fair_baselines.py", ["--seeds", "50"]),
    "13": ("run_phase13_stability.py", ["--seeds", "50"]),
    "14": ("run_phase14_evidence_profile.py", ["--seeds", "50"]),
    "15": ("run_phase15_version_diffs.py", ["--seeds", "50"]),
}

# Smoke-test overrides (still uniform where possible)
QUICK = {
    "0":  [],
    "1":  ["--seeds", "3"],
    "2":  ["--seeds", "2"],
    "3":  ["--seeds", "3"],
    "4":  ["--seeds", "2"],
    "5":  ["--seeds", "2"],
    "6":  ["--seeds", "2"],
    "7":  ["--seeds", "2"],
    "8":  ["--seeds", "2"],
    "9":  ["--seeds", "2"],
    "10": ["--seeds", "2"],
    "11": ["--seeds", "2"],
    "12": ["--seeds", "2"],
    "12b": ["--seeds", "2"],
    "13": ["--seeds", "2"],
    "14": ["--seeds", "2"],
    "15": ["--seeds", "2"],
}


def main():
    ap = argparse.ArgumentParser(description="Run PPDOE test phases 0–15 (50 seeds full)")
    ap.add_argument("--quick", action="store_true", help="Reduced seed counts")
    ap.add_argument(
        "--phases",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,12b,13,14,15",
        help="Comma-separated phase numbers",
    )
    args = ap.parse_args()
    selected = [p.strip() for p in args.phases.split(",") if p.strip() in PHASES]

    print("=" * 60)
    print("PPDOE full test suite")
    print(f"Phases: {selected}")
    print(f"Mode: {'QUICK' if args.quick else 'FULL BLAST (50 seeds)'}")
    print("=" * 60)

    failed = []
    for ph in selected:
        script, default_args = PHASES[ph]
        extra = QUICK[ph] if args.quick else default_args
        cmd = [sys.executable, str(TESTS / script)] + extra
        print(f"\n>>> Phase {ph}: {' '.join(cmd)}")
        print("-" * 60)
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            print(f"!!! Phase {ph} FAILED (exit {rc})")
            failed.append(ph)
        else:
            print(f"<<< Phase {ph} OK")

    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED phases: {failed}")
        sys.exit(1)
    print("All requested phases completed.")
    print(f"Results in: {ROOT / 'results'}")


if __name__ == "__main__":
    main()
