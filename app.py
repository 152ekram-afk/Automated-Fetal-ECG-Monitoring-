"""Automated Fetal ECG Monitoring - Streamlit app.

Upload an abdominal ECG recording (CSV). The app removes the mother's ECG, finds fetal beats with
the model you choose, plots the fetal heart rate, and compares all available models.
"""
import hashlib
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fecg import models as fm  # noqa: E402
from fecg import pipeline as fp  # noqa: E402
from fecg import ui  # noqa: E402

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
LEGACY_MODEL = ROOT / "fecg_model.joblib"        # single-model file from the first version of the app

st.set_page_config(page_title="Automated Fetal ECG Monitoring", page_icon="🫀", layout="wide")


# ----------------------------------------------------------------------------- models
def _model_files():
    files = sorted(MODELS_DIR.glob("*.joblib")) if MODELS_DIR.exists() else []
    if LEGACY_MODEL.exists():
        files.append(LEGACY_MODEL)
    return files


def _signature():
    return tuple((f.name, f.stat().st_mtime_ns) for f in _model_files())


@st.cache_resource(show_spinner="Loading models...")
def load_models(signature):
    """The signature changes whenever a model file changes, so a stale result is never reused."""
    bundles, errors = {}, []
    for f in _model_files():
        try:
            b = fm.load_bundle(f)
        except Exception as e:
            errors.append(f"{f.name}: {e}")
            continue
        if b["key"] not in bundles:            # models/ wins over the legacy root file
            bundles[b["key"]] = b
    return bundles, errors


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


bundles, load_errors = load_models(_signature())
info = _read_json(MODELS_DIR / "model_info.json")
order = [k for k in fm.build_zoo() if k in bundles] + [k for k in bundles if k not in fm.build_zoo()]
labels = {k: bundles[k]["name"] for k in order}
labels["rule"] = "Rules only (no ML)"

# ----------------------------------------------------------------------------- sidebar
st.sidebar.header("1. Input")
upload = st.sidebar.file_uploader("ECG recording (.csv)", type=["csv"])
default_fs = int(bundles[order[0]]["fs"]) if order else 1000
fs_in = int(st.sidebar.number_input("Sampling rate (Hz)", min_value=100, max_value=5000,
                                    value=default_fs, step=50))

st.sidebar.header("2. Detection method")
choices = order + ["rule"]
default_idx = choices.index(fm.OUR_MODEL_KEY) if fm.OUR_MODEL_KEY in choices else 0
choice = st.sidebar.selectbox("Method", choices, index=default_idx, format_func=lambda k: labels[k])
mode = "rule" if choice == "rule" else "ml"
thr = 0.5
if mode == "ml":
    t0 = float(bundles[choice]["threshold"])
    t0 = round(min(max(round(t0 / 0.05) * 0.05, 0.05), 0.90), 2)
    thr = st.sidebar.slider("ML confidence threshold", 0.05, 0.90, t0, 0.05, key=f"thr_{choice}")
    st.sidebar.caption(fm.MODEL_NOTES.get(choice, ""))
if not order:
    st.sidebar.info("No trained models found in the `models/` folder, so only rules-only detection "
                    "is available. Run the training notebook to create them.")

st.sidebar.header("3. Display")
animations = st.sidebar.toggle("Animations", value=True)

with st.sidebar.expander("Diagnostics"):
    st.write(f"scikit-learn on this server: **{sklearn.__version__}**")
    if info.get("sklearn"):
        st.write(f"scikit-learn used for training: **{info['sklearn']}**")
    st.write("Model files found: " + (", ".join(f.name for f in _model_files()) or "none"))
    for e in load_errors:
        st.error(e)
if info.get("sklearn") and info["sklearn"] != sklearn.__version__:
    st.sidebar.warning(f"Models were trained with scikit-learn {info['sklearn']} but this server runs "
                       f"{sklearn.__version__}. If results look wrong, pin "
                       f"`scikit-learn=={info['sklearn']}` in requirements.txt.")

# ----------------------------------------------------------------------------- header
st.markdown(ui.css(animations), unsafe_allow_html=True)
st.markdown(ui.hero_html(), unsafe_allow_html=True)
st.warning("Educational prototype. It is **not** a medical device and must not be used for diagnosis "
           "or clinical decisions.")

if upload is None:
    st.info("Upload a recording in the sidebar to begin. The CSV needs the columns `Abdomen_1` ... "
            "`Abdomen_4`. If it also has `Direct_1` (the direct fetal ECG), the app compares its "
            "results against it. You can still explore the benchmark below without a file.")

# ----------------------------------------------------------------------------- benchmark (stored)
bench_path = MODELS_DIR / "benchmark_summary.csv"
bench = pd.read_csv(bench_path) if bench_path.exists() else None


def show_benchmark():
    if bench is None:
        st.info("No benchmark found yet. Run the training notebook (or `scripts/train_models.py`) and "
                "upload the generated `models/` folder.")
        return
    metric = st.radio("Rank by", ["F1", "Sensitivity", "Precision", "AUC"], horizontal=True)
    st.markdown(ui.leaderboard_html(bench, metric), unsafe_allow_html=True)

    ml = bench[bench["Type"] == "ml"].sort_values("F1", ascending=False)
    if len(ml):
        best = ml.iloc[0]
        spread = best["F1_std"] if pd.notna(best["F1_std"]) else 0.0
        tied = ml[ml["F1"] >= best["F1"] - spread]["Model"].tolist()
        msg = f"Highest mean F1: **{best['Model']}** ({best['F1']:.3f}). "
        if len(tied) > 1:
            msg += ("Models within one standard deviation of the best (spread between recordings "
                    f"= {spread:.3f}) are effectively tied with this little data: " + ", ".join(tied) + ".")
        st.info(msg)

    show = bench[["Model", "Type", "Sensitivity", "Precision", "F1", "F1_std", "AUC", "FHR_err_BPM",
                  "FHR_MAE_BPM", "Train_s", "Infer_ms_per_1000"]].round(3)
    show = show.rename(columns={"F1_std": "F1 spread", "FHR_err_BPM": "FHR error (BPM)",
                                "FHR_MAE_BPM": "FHR MAE (BPM)", "Train_s": "Train (s)",
                                "Infer_ms_per_1000": "Inference (ms / 1000 candidates)"})
    st.dataframe(show.set_index("Model"))
    st.download_button("Download benchmark (CSV)", bench.to_csv(index=False).encode(),
                       file_name="benchmark_summary.csv", mime="text/csv")
    st.caption("Leave-one-recording-out results: every recording was held out in turn, so no model was "
               "scored on data it was trained on. 'F1 spread' is the standard deviation across recordings.")


if upload is None:
    st.subheader("Model benchmark")
    show_benchmark()
    st.stop()

# ----------------------------------------------------------------------------- process the recording
raw = upload.getvalue()
key = (hashlib.md5(raw).hexdigest(), fs_in)
if st.session_state.get("proc_key") != key:
    try:
        with st.status("Analyzing the recording...", expanded=True) as status:
            proc = fp.process_record(raw, fs_in, progress=lambda msg: st.write(msg))
            status.update(label="Analysis complete", state="complete", expanded=False)
    except Exception as e:
        st.error(f"Could not process this file: {e}")
        st.stop()
    st.session_state["proc_key"], st.session_state["proc"] = key, proc
proc = st.session_state["proc"]

fs, n = proc["fs"], proc["n"]
dur = n / fs
model = bundles[choice]["model"] if mode == "ml" else None
beats = fp.detect_beats(proc, mode, thr, model)
t_hr, hr = fp.calculate_hr(beats, fs)
valid_pct = 100 * len(hr) / max(len(beats) - 1, 1)
ref = fp.trim_edges(proc["ref"], n, fs) if proc["ref"] is not None else None

# ----------------------------------------------------------------------------- headline numbers
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Duration", f"{dur:.1f} s")
c2.metric("Fetal beats found", f"{len(beats)}")
c3.metric("Median FHR", f"{np.median(hr):.0f} BPM" if len(hr) else "n/a")
c4.metric("FHR range", f"{hr.min():.0f}-{hr.max():.0f}" if len(hr) else "n/a")
c5.metric("Clean intervals", f"{valid_pct:.0f}%")
st.caption(f"Method: **{labels[choice]}**. A typical baseline fetal heart rate is roughly 110-160 BPM. "
           "'Clean intervals' is the share of beat-to-beat intervals with a plausible heart rate; "
           "a low value means detection is unreliable for this recording.")

tab_sig, tab_live, tab_hr, tab_cmp, tab_how = st.tabs(
    ["Signal and beats", "Live monitor", "Heart rate", "Model comparison", "How it works"])

# ----------------------------------------------------------------------------- tab: signal
with tab_sig:
    ca, cb, cc = st.columns(3)
    ch = ca.selectbox("Abdominal channel", fp.ABD, key="sig_ch")
    clip = cb.checkbox("Clip large artifacts for display", value=True)
    win = min(8.0, dur)
    start = 0.0
    if dur - win > 0.5:
        start = cc.slider("Window start (s)", 0.0, float(dur - win), 0.0, 0.5)
    a, b = int(start * fs), int((start + win) * fs)
    t = np.arange(n) / fs

    def disp(x):
        return np.clip(x, -8, 8) if clip else x

    filt, res = disp(proc["filtered"][ch]), disp(proc["residual"][ch])
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    ax[0].plot(t[a:b], filt[a:b], lw=0.7, color="#0f9d9a")
    mp = proc["maternal"]
    mp = mp[(mp >= a) & (mp < b)]
    ax[0].plot(mp / fs, filt[mp], "x", color="#e0457b", label="maternal beats")
    ax[0].set_title(f"{ch}: filtered signal (dominated by the mother's heartbeat)")
    ax[0].legend(loc="upper right")
    ax[1].plot(t[a:b], res[a:b], lw=0.7, color="#0f9d9a")
    ymin, ymax = ax[1].get_ylim()
    bb = beats[(beats >= a) & (beats < b)]
    ax[1].plot(bb / fs, np.full(len(bb), ymax * 0.92), "v", color="#e0457b", label="detected fetal beats")
    if ref is not None:
        rr_ = ref[(ref >= a) & (ref < b)]
        ax[1].plot(rr_ / fs, np.full(len(rr_), ymin * 0.92), "^", color="#2e7d32", label="reference (Direct_1)")
    ax[1].set_title(f"{ch}: after maternal ECG removal")
    ax[1].set_xlabel("Time (s)")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ----------------------------------------------------------------------------- tab: live monitor
with tab_live:
    st.caption("A replay of the recording as if it were being monitored live: the trace scrolls, each "
               "detected fetal beat pulses the heart, and the BPM updates from the last few beats.")
    lch = st.selectbox("Channel", fp.ABD, key="live_ch")
    payload = ui.live_payload(proc["residual"][lch], fs, beats, ref,
                              label=f"{lch} (maternal ECG removed) - {labels[choice]}")
    ui.show_frame(ui.live_monitor_html(payload, autoplay=animations), height=410)
    if dur > 60:
        st.caption("Replay shows the first 60 seconds.")

# ----------------------------------------------------------------------------- tab: heart rate
with tab_hr:
    if len(hr) < 2:
        st.info("Not enough clean beat intervals to draw a heart-rate curve for this recording.")
    else:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.axhspan(110, 160, color="grey", alpha=0.12, label="typical baseline range (110-160)")
        ax.plot(t_hr, hr, "s-", ms=3, color="#e0457b", label="estimated FHR")
        if ref is not None:
            rt, rh = fp.calculate_hr(ref, fs)
            if len(rh):
                ax.plot(rt, rh, "o-", ms=3, color="#0f9d9a", label="reference FHR (Direct_1)")
        ax.set_ylim(60, 240)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("BPM")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.download_button("Download beat intervals (CSV)", fp.interval_table(beats, fs).to_csv(index=False).encode(),
                       file_name="fetal_beat_intervals.csv", mime="text/csv")

# ----------------------------------------------------------------------------- tab: comparison
with tab_cmp:
    st.subheader("Live comparison on this recording")
    methods = [("Baseline (every candidate, no filtering)", "union", None, 0.5),
               ("Rules only (no ML)", "rule", None, 0.5)]
    methods += [(bundles[k]["name"], "ml", bundles[k]["model"], bundles[k]["threshold"]) for k in order]
    rows = []
    for name, m, mdl, th in methods:
        t0 = time.perf_counter()
        bts = fp.detect_beats(proc, m, th, mdl)
        ms = (time.perf_counter() - t0) * 1000
        _, h = fp.calculate_hr(bts, fs)
        row = {"Method": name, "Beats": len(bts),
               "Median FHR (BPM)": round(float(np.median(h)), 1) if len(h) else np.nan,
               "Clean intervals %": round(100 * len(h) / max(len(bts) - 1, 1)),
               "Time (ms)": round(ms, 1)}
        if ref is not None:
            met = fp.match_beats(bts, ref, fs)
            row.update({"Sensitivity": round(met["Sensitivity"], 3), "Precision": round(met["Precision"], 3),
                        "F1": round(met["F1"], 3),
                        "FHR error (BPM)": round(fp.median_fhr_error(bts, ref, fs), 2)})
        rows.append(row)
    cmp_df = pd.DataFrame(rows).set_index("Method")
    st.dataframe(cmp_df)

    if ref is not None:
        fig, ax = plt.subplots(figsize=(10, 0.55 * len(cmp_df) + 1.6))
        ypos = np.arange(len(cmp_df))
        h_ = 0.26
        for i, (col, colr) in enumerate((("Sensitivity", "#0f9d9a"), ("Precision", "#e0457b"), ("F1", "#0b3b4a"))):
            ax.barh(ypos + (1 - i) * h_, cmp_df[col].to_numpy(), height=h_, color=colr, label=col)
        ax.set_yticks(ypos)
        ax.set_yticklabels(cmp_df.index)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.legend(loc="lower right", frameon=False)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("If this recording was part of the training data, these scores are optimistic. "
                   "The honest comparison is the leave-one-recording-out benchmark below.")
    else:
        st.info("Add a `Direct_1` column to the file to see sensitivity, precision and F1 for each method.")
    st.download_button("Download comparison (CSV)", cmp_df.reset_index().to_csv(index=False).encode(),
                       file_name="live_comparison.csv", mime="text/csv")

    st.subheader("Benchmark on the whole dataset")
    show_benchmark()

# ----------------------------------------------------------------------------- tab: how it works
with tab_how:
    st.markdown(
        """
**Pipeline**
1. **Band-pass filter (8-45 Hz)** removes baseline drift and high-frequency noise.
2. **Maternal ECG removal:** the mother's beats are located and a matching template is subtracted
   from each one, leaving the much smaller fetal signal.
3. **Candidate peaks** are found in each of the 4 abdominal channels.
4. **Features** describe each candidate (amplitude, width, timing between peaks, distance to the
   nearest maternal beat, ...).
5. **Machine learning:** a classifier, trained on recordings where the direct fetal ECG (Direct_1)
   marks the true beats, scores each candidate as "fetal beat" or "not".
6. **Consensus:** scores are averaged across the 4 channels, and two beats closer than 300 ms are
   not allowed (the weaker one is dropped).
7. **Fetal heart rate** = 60 / (time between consecutive beats).

**Models you can compare**
        """
    )
    for k in order:
        st.markdown(f"- **{labels[k]}**: {fm.MODEL_NOTES.get(k, '')}")
    st.markdown(
        """
**How the comparison is made fair:** all models see exactly the same candidates and features, the
confidence threshold is tuned only on training recordings, and each recording is held out in turn
(leave-one-recording-out).

**Limitations:** trained on only a few short recordings, so differences between models are often
within the noise. Not validated clinically.
        """
    )
