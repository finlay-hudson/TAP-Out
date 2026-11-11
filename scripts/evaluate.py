"""Batch evaluation of TAP-Out on TAPVid360-TAPOut.

Runs the same pipeline as ``run_tapout_release.py`` and, when all clips finish,
aggregates the per-clip JSON metrics into a single summary printed to stdout.

Example
-------
    # Full TAPVid360-TAPOut split (paths configured in configs/default.yaml).
    python scripts/evaluate.py

    # Ground-truth camera poses.
    python scripts/evaluate.py --use_gt_rot

    # A handful of clips only.
    python scripts/evaluate.py --split_file /tmp/my_clips.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make repo root importable when invoked as ``python scripts/evaluate.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from run_tapout_release import main  # noqa: E402
from runners.settings import get_args, load_config  # noqa: E402


def _aggregate(metrics_dir: Path) -> dict[str, float]:
    """Mean over per-clip *_metrics.json files."""
    all_files = sorted(metrics_dir.glob("*_metrics.json"))
    if not all_files:
        return {}
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for f in all_files:
        with open(f) as fh:
            m = json.load(fh)
        for k, v in m.items():
            if isinstance(v, (int, float)):
                sums[k] = sums.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


if __name__ == "__main__":
    args = get_args()
    settings = load_config(args.config)

    main(args, settings)

    metrics_dir = settings.paths.out_root / "metrics" / settings.ds_name / f"gt_poses-{args.use_gt_rot}"

    summary = _aggregate(metrics_dir)
    if summary:
        print("\n=== Aggregate metrics ===")
        for k, v in summary.items():
            print(f"  {k}: {v:.4f}")
        with open(metrics_dir / "AGGREGATE.json", "w") as f:
            json.dump(summary, f, indent=4)
        print(f"\nSaved: {metrics_dir / 'AGGREGATE.json'}")
    else:
        print(f"No metric files found under {metrics_dir}.")
