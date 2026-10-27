"""Leave-one-recording-out benchmarking of several classifiers on the same fetal-beat task.

Why leave-one-recording-out: neighbouring beats of one recording look alike, so a random
train/test split would leak information. Each recording is held out in turn, and the
detection threshold is chosen using only the remaining (training) recordings.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from . import models as fm
from . import pipeline as fp

THR_GRID = np.arange(0.05, 0.91, 0.05)
BASELINES = {"union": "Baseline (every candidate, no filtering)", "rule": "Rules only (no ML)"}


class CandidateSet:
    """All candidate peaks of all training recordings, with labels from Direct_1."""

    def __init__(self, X, y, groups, pos, fs, n_samples, ref):
        self.X, self.y, self.groups, self.pos = X, y, groups, pos
        self.fs, self.n_samples, self.ref = fs, n_samples, ref

    @property
    def rec_ids(self):
        return list(dict.fromkeys(self.groups.tolist()))


def build_candidate_set(procs, tol_s=0.05):
    """procs: {record_id: output of fp.process_record}. Every record needs a Direct_1 channel."""
    X_l, y_l, g_l, p_l = [], [], [], []
    n_samples, ref, fs_set = {}, {}, set()
    for rid, proc in procs.items():
        if proc["ref"] is None:
            raise ValueError(f"Recording '{rid}' has no Direct_1 channel, which is needed for training labels.")
        fs = proc["fs"]
        fs_set.add(fs)
        n_samples[rid] = proc["n"]
        ref[rid] = fp.trim_edges(proc["ref"], proc["n"], fs)
        tol = int(tol_s * fs)
        for ch in fp.ABD:
            X, pk = proc["candidates"][ch]
            if len(pk) == 0:
                continue
            if len(proc["ref"]):
                dist = np.min(np.abs(pk[:, None] - proc["ref"][None, :]), axis=1)
            else:
                dist = np.full(len(pk), 10 ** 9)
            X_l.append(X)
            y_l.append((dist <= tol).astype(int))
            g_l += [rid] * len(pk)
            p_l.append(pk)
    if not X_l:
        raise ValueError("No candidate peaks were found in any recording.")
    if len(fs_set) != 1:
        raise ValueError(f"Recordings use different sampling rates: {sorted(fs_set)}")
    return CandidateSet(np.vstack(X_l), np.concatenate(y_l), np.array(g_l), np.concatenate(p_l),
                        fs_set.pop(), n_samples, ref)


# ----------------------------------------------------------------------------- metrics
def fhr_mae(beats_est, beats_ref, fs, step=0.5):
    t_e, h_e = fp.calculate_hr(beats_est, fs)
    t_r, h_r = fp.calculate_hr(beats_ref, fs)
    if len(t_e) < 2 or len(t_r) < 2:
        return float("nan")
    grid = np.arange(max(t_e[0], t_r[0]), min(t_e[-1], t_r[-1]), step)
    if len(grid) < 2:
        return float("nan")
    return float(np.mean(np.abs(np.interp(grid, t_e, h_e) - np.interp(grid, t_r, h_r))))


def evaluate_beats(beats, ref, fs):
    m = fp.match_beats(beats, ref, fs)
    m["FHR_err_BPM"] = fp.median_fhr_error(beats, ref, fs)
    m["FHR_MAE_BPM"] = fhr_mae(beats, ref, fs)
    return m


def choose_threshold(factory, cs, train_mask, grid=THR_GRID):
    """Pick the ML confidence threshold with an inner leave-one-recording-out loop on training data only."""
    Xtr, ytr, gtr, ptr = cs.X[train_mask], cs.y[train_mask], cs.groups[train_mask], cs.pos[train_mask]
    if len(np.unique(gtr)) < 2 or len(np.unique(ytr)) < 2:
        return 0.5
    try:
        oof = cross_val_predict(factory(), Xtr, ytr, groups=gtr, cv=LeaveOneGroupOut(),
                                method="predict_proba")[:, 1]
    except Exception as e:  # e.g. an inner fold with a single class
        warnings.warn(f"Threshold search fell back to 0.5: {e}")
        return 0.5
    best_t, best_f = 0.5, -1.0
    for t in grid:
        f = np.mean([
            fp.match_beats(
                fp.select_beats(ptr[gtr == r], oof[gtr == r], t, cs.n_samples[r], cs.fs, "ml"),
                cs.ref[r], cs.fs)["F1"]
            for r in np.unique(gtr)])
        if f >= best_f:
            best_t, best_f = float(t), float(f)
    return best_t


# ----------------------------------------------------------------------------- benchmark
def benchmark_models(cs, zoo=None, grid=THR_GRID, progress=None):
    """Evaluate every model in the zoo (and two non-ML baselines) with leave-one-recording-out.

    Returns a dict with: summary (one row per method), per_record, oof (out-of-fold probabilities
    per model, aligned with cs.y), models (final models trained on all recordings), thresholds.
    """
    zoo = zoo or fm.build_zoo()
    rec_ids = cs.rec_ids
    rows = []

    for rid in rec_ids:                                   # non-ML baselines
        te = cs.groups == rid
        for key, label in BASELINES.items():
            beats = fp.select_beats(cs.pos[te], np.zeros(int(te.sum())), 0.5,
                                    cs.n_samples[rid], cs.fs, key)
            rows.append(dict(Key=key, Model=label, Type="baseline", Record=rid, Threshold=np.nan,
                             **evaluate_beats(beats, cs.ref[rid], cs.fs)))

    oof_all, final, thresholds, extra = {}, {}, {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                   # e.g. MLP convergence notices
        for key, (name, factory) in zoo.items():
            if progress:
                progress(f"Evaluating {name} ...")
            oof = np.zeros(len(cs.y))
            ths, infer = [], []
            for rid in rec_ids:
                tr, te = cs.groups != rid, cs.groups == rid
                thr = choose_threshold(factory, cs, tr, grid)
                ths.append(thr)
                model = factory().fit(cs.X[tr], cs.y[tr])
                t0 = time.perf_counter()
                prob = model.predict_proba(cs.X[te])[:, 1]
                infer.append((time.perf_counter() - t0) * 1e6 / max(int(te.sum()), 1))  # ms per 1000 candidates
                oof[te] = prob
                beats = fp.select_beats(cs.pos[te], prob, thr, cs.n_samples[rid], cs.fs, "ml")
                rows.append(dict(Key=key, Model=name, Type="ml", Record=rid, Threshold=thr,
                                 **evaluate_beats(beats, cs.ref[rid], cs.fs)))
            both = len(np.unique(cs.y)) == 2
            t0 = time.perf_counter()
            final[key] = factory().fit(cs.X, cs.y)
            thresholds[key] = float(np.median(ths))
            extra[key] = dict(AUC=roc_auc_score(cs.y, oof) if both else np.nan,
                              AP=average_precision_score(cs.y, oof) if both else np.nan,
                              Train_s=time.perf_counter() - t0,
                              Infer_ms_per_1000=float(np.mean(infer)))
            oof_all[key] = oof

    per_record = pd.DataFrame(rows)
    return dict(summary=_summarize(per_record, extra), per_record=per_record,
                oof=oof_all, models=final, thresholds=thresholds)


def _summarize(per_record, extra):
    g = per_record.groupby(["Key", "Model", "Type"], sort=False)
    s = g[["Sensitivity", "Precision", "F1", "FHR_err_BPM", "FHR_MAE_BPM", "Threshold"]].mean()
    s["F1_std"] = g["F1"].std()
    s = s.reset_index()
    for col in ("AUC", "AP", "Train_s", "Infer_ms_per_1000"):
        s[col] = s["Key"].map(lambda k, c=col: extra.get(k, {}).get(c, np.nan))
    cols = ["Key", "Model", "Type", "Sensitivity", "Precision", "F1", "F1_std", "AUC", "AP",
            "FHR_err_BPM", "FHR_MAE_BPM", "Train_s", "Infer_ms_per_1000", "Threshold"]
    return s[cols].sort_values("F1", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------- saving
def save_run(out_dir, result, fs, rec_ids, zoo, skip=()):
    """Write trained models + benchmark tables + environment info to out_dir (replaces old *.joblib).

    Keys listed in `skip` are still benchmarked but their model file is not written
    (use skip=("random_forest",) to keep a Random Forest you trained yourself).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.joblib"):
        old.unlink()
    written = []
    for key, model in result["models"].items():
        if key in skip:
            continue
        fm.save_bundle(out_dir / f"{key}.joblib", model, key, zoo[key][0],
                       result["thresholds"][key], fs)
        written.append(key)
    result["summary"].to_csv(out_dir / "benchmark_summary.csv", index=False)
    result["per_record"].to_csv(out_dir / "benchmark_per_record.csv", index=False)
    info = fm.environment_info()
    info.update(fs=int(fs), recordings=[str(r) for r in rec_ids], models=written)
    (out_dir / "model_info.json").write_text(json.dumps(info, indent=2))
    return out_dir
