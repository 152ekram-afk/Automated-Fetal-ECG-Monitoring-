"""Automated checks on synthetic data (run with:  python -m pytest -q   or   python tests/test_pipeline.py)."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fecg import evaluation as ev  # noqa: E402
from fecg import models as fm  # noqa: E402
from fecg import pipeline as fp  # noqa: E402
from fecg import synthetic  # noqa: E402

FS = 1000


def test_load_ecg_handles_units_row_and_quotes():
    df = fp.load_ecg(synthetic.make_recording_csv(1, secs=5))
    assert set(fp.CHANNELS) <= set(df.columns)
    assert len(df) == 5 * FS


def test_direct_channel_is_optional():
    df = fp.load_ecg(synthetic.make_recording_csv(1, secs=5, direct=False))
    assert "Direct_1" not in df.columns and set(fp.ABD) <= set(df.columns)


def test_missing_columns_are_rejected():
    try:
        fp.load_ecg(b"a,b\n1,2\n3,4\n")
    except ValueError as e:
        assert "Abdomen_1" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_short_recording_is_rejected():
    try:
        fp.process_record(synthetic.make_recording_csv(1, secs=2), FS)
    except ValueError as e:
        assert "too short" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_rules_pipeline_recovers_fetal_heart_rate():
    proc = fp.process_record(synthetic.make_recording_csv(2, secs=20), FS)
    beats = fp.detect_beats(proc, "rule")
    _, hr = fp.calculate_hr(beats, FS)
    assert len(beats) > 25
    assert 120 <= np.median(hr) <= 160          # true synthetic rate is about 140 BPM


def test_progress_callback_is_called():
    msgs = []
    fp.process_record(synthetic.make_recording_csv(3, secs=5), FS, progress=msgs.append)
    assert len(msgs) >= 5


def test_refractory_rule_removes_too_close_beats():
    out = fp.apply_refractory(np.array([1000, 1100, 2000]), np.array([0.2, 0.9, 0.5]), FS)
    assert list(out) == [1100, 2000]


def test_benchmark_and_model_roundtrip():
    procs = {k: fp.process_record(v, FS) for k, v in synthetic.make_dataset(4, secs=12).items()}
    cs = ev.build_candidate_set(procs)
    zoo = fm.build_zoo()
    small = {k: zoo[k] for k in ("random_forest", "logistic_regression", "naive_bayes")}
    res = ev.benchmark_models(cs, small)
    summary = res["summary"]
    assert {"random_forest", "logistic_regression", "naive_bayes", "rule", "union"} <= set(summary["Key"])
    ml = summary[summary["Type"] == "ml"]
    assert ml["F1"].between(0, 1).all() and ml["F1"].max() > 0.5
    with tempfile.TemporaryDirectory() as d:
        ev.save_run(d, res, FS, cs.rec_ids, small)
        b = fm.load_bundle(Path(d) / "random_forest.joblib")
        prob = b["model"].predict_proba(cs.X[:10])
        assert prob.shape == (10, 2) and 0 <= b["threshold"] <= 1
        assert (Path(d) / "benchmark_summary.csv").exists() and (Path(d) / "model_info.json").exists()
    with tempfile.TemporaryDirectory() as d:          # skip= keeps a model out of the saved files
        ev.save_run(d, res, FS, cs.rec_ids, small, skip=("random_forest",))
        assert not (Path(d) / "random_forest.joblib").exists()
        assert (Path(d) / "naive_bayes.joblib").exists()


def test_pin_requirements():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "requirements.txt"
        p.write_text("streamlit\nscikit-learn\n")
        line = fm.pin_requirements(p)
        assert line.startswith("scikit-learn==") and line in p.read_text()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
