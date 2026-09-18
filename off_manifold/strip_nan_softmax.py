"""
strip_nan_softmax.py — remove NaN local_softmax_gap* fields from int.json files
so the local-gap sweep will recompute them with the fixed code.

    python off_manifold/strip_nan_softmax.py --results-root poly_results_new/results --dry-run
    python off_manifold/strip_nan_softmax.py --results-root poly_results_new/results
"""
import argparse
import glob
import json
import math
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="poly_results_new/results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fixed = 0
    scanned = 0
    for path in glob.glob(os.path.join(args.results_root, "**", "int.json"), recursive=True):
        scanned += 1
        try:
            r = json.load(open(path))
        except Exception:
            continue
        changed = False
        for k in ("local_softmax_gap", "local_softmax_gap_frac"):
            v = r.get(k)
            if v is not None and isinstance(v, float) and math.isnan(v):
                del r[k]
                changed = True
        if changed:
            fixed += 1
            if not args.dry_run:
                json.dump(r, open(path, "w"), indent=2)

    print(f"scanned {scanned} files, {'would strip' if args.dry_run else 'stripped'} "
          f"NaN softmax fields from {fixed}")


if __name__ == "__main__":
    main()
