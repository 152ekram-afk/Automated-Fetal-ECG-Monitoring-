#!/usr/bin/env python
"""Train and benchmark all models, then save them into models/ (what the Streamlit app reads).

Usage:
    python scripts/train_models.py --data path/to/csv_folder
    python scripts/train_models.py --data path/to/csv_folder --only random_forest svm_rbf
    python scripts/train_models.py --synthetic          # artificial data, only to check the install

Every CSV in the folder is one recording and must contain Abdomen_1..4 and Direct_1.
At least 3 recordings are needed (leave-one-recording-out evaluation).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from fecg import evaluation as ev  # noqa: E402
from fecg import models as fm  # noqa: E402
from fecg import pipeline as fp  # noqa: E402
from fecg import synthetic  # noqa: E402


def load_recordings(data_dir, fs):
    procs = {}
    for path in sorted(Path(data_dir).glob("*.csv")):
        try:
            proc = fp.process_record(path.read_bytes(), fs)
        except Exception as e:
            print(f"  skipped {path.name}: {e}")
            continue
        if proc["ref"] is None:
            print(f"  skipped {path.name}: no Direct_1 column (needed for training labels)")
            continue
        procs[path.stem] = proc
        print(f"  loaded {path.name}: {proc['n'] / fs:.1f} s")
    return procs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="folder with one CSV per recording")
    ap.add_argument("--out", type=Path, default=ROOT / "models", help="output folder (default: models/)")
    ap.add_argument("--fs", type=int, default=1000, help="sampling rate in Hz (default 1000)")
    ap.add_argument("--only", nargs="*", help="model keys to run, e.g. random_forest svm_rbf")
    ap.add_argument("--skip", nargs="*", default=[], help="model keys to benchmark but NOT save (e.g. random_forest)")
    ap.add_argument("--synthetic", action="store_true", help="use artificial data (install check only)")
    ap.add_argument("--pin-requirements", action="store_true",
                    help="pin scikit-learn in requirements.txt to the version used here")
    args = ap.parse_args()

    if args.synthetic:
        print("Using SYNTHETIC data - results are meaningless, this only checks the install.")
        procs = {k: fp.process_record(v, args.fs) for k, v in synthetic.make_dataset(5, 20, args.fs).items()}
    elif args.data:
        print(f"Loading recordings from {args.data} ...")
        procs = load_recordings(args.data, args.fs)
    else:
        ap.error("give --data FOLDER (or --synthetic)")

    if len(procs) < 3:
        sys.exit(f"Need at least 3 usable recordings, found {len(procs)}.")

    zoo = fm.build_zoo()
    if args.only:
        unknown = [k for k in args.only if k not in zoo]
        if unknown:
            sys.exit(f"Unknown model keys: {unknown}. Available: {list(zoo)}")
        zoo = {k: zoo[k] for k in args.only}
        if fm.OUR_MODEL_KEY not in zoo:
            print("Note: 'random_forest' (our model) is not included, so the app has no default model.")

    cs = ev.build_candidate_set(procs)
    print(f"\n{len(cs.y)} candidate peaks ({int(cs.y.sum())} fetal) from {len(procs)} recordings.\n")
    result = ev.benchmark_models(cs, zoo, progress=lambda m: print(" ", m))

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\nLeave-one-recording-out results (means over held-out recordings):\n")
    print(result["summary"].round(3).to_string(index=False))

    ev.save_run(args.out, result, cs.fs, cs.rec_ids, zoo, skip=args.skip)
    print(f"\nSaved {len(result['models']) - len([k for k in args.skip if k in result['models']])} models + benchmark tables to {args.out}")
    if args.pin_requirements:
        print("Pinned:", fm.pin_requirements(ROOT / "requirements.txt"))
    else:
        print(f"Tip: put  scikit-learn=={fm.environment_info()['sklearn']}  in requirements.txt "
              "(or rerun with --pin-requirements) so the deployed app matches these models.")


if __name__ == "__main__":
    main()
