"""Synthetic mother + fetus ECG recordings, used ONLY for automated tests and for checking an install.

These signals are artificial and must never be reported as results. They mimic the layout of the
Kaggle/PhysioNet abdominal + direct fetal ECG CSV files (header row, units row, quoted names).
"""
import numpy as np
import pandas as pd


def _qrs(n, centers, width_ms, amp, fs):
    x = np.zeros(n)
    t = np.arange(n)
    s = width_ms / 1000 * fs / 2.5
    for c in centers:
        x += amp * (-(t - c) / s) * np.exp(-((t - c) ** 2) / (2 * s ** 2)) * 2.5
    return x


def make_recording_csv(seed=0, secs=20, fs=1000, direct=True, fetal_rr=0.43, maternal_rr=0.80):
    """Return CSV bytes in the same layout as the real dataset files."""
    r = np.random.default_rng(seed)
    n = int(secs * fs)
    mc = np.cumsum(r.normal(maternal_rr * fs, 0.025 * fs, int(secs / maternal_rr) + 5)).astype(int)
    fc = np.cumsum(r.normal(fetal_rr * fs, 0.015 * fs, int(secs / fetal_rr) + 5)).astype(int)
    mc, fc = mc[mc < n - int(0.3 * fs)], fc[fc < n - int(0.1 * fs)]
    mat, fet = _qrs(n, mc, 80, 1.0, fs), _qrs(n, fc, 40, 0.12, fs)
    cols = {"'Elapsed time'": np.arange(n) / fs}
    if direct:
        cols["'Direct_1'"] = fet * 8 + 0.05 * mat + 0.02 * r.normal(size=n)
    for i in range(4):
        cols[f"'Abdomen_{i + 1}'"] = (0.6 + 0.2 * i) * mat + (0.9 - 0.15 * i) * fet + 0.02 * r.normal(size=n)
    df = pd.DataFrame(cols)
    header = ",".join(df.columns)
    units = ",".join(["'seconds'"] + ["'mV'"] * (len(df.columns) - 1))
    return (header + "\n" + units + "\n" + df.to_csv(index=False, header=False)).encode()


def make_dataset(n_records=5, secs=20, fs=1000):
    """{record_id: csv_bytes} for a small synthetic dataset."""
    return {f"s{i + 1:02d}": make_recording_csv(seed=100 + i, secs=secs, fs=fs,
                                                fetal_rr=0.40 + 0.02 * i) for i in range(n_records)}
