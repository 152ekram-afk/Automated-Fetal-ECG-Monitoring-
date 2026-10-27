# Automated Fetal ECG Monitoring

Machine-learning project that extracts the **fetal heartbeat** from abdominal ECG recordings and
estimates the **fetal heart rate (FHR)**, with a Streamlit web app to explore results and compare
several standard ML models.

**Live demo:** https://fetalecg-4momkquqxwe6wsa9abrgno.streamlit.app/

> Educational prototype. It is **not** a medical device and must not be used for diagnosis or clinical decisions.

## What it does

- Removes the mother's ECG from the abdominal channels (template subtraction), so the much smaller
  fetal signal can be found.
- Finds candidate peaks in 4 abdominal channels and describes each with 24 features.
- A classifier decides which candidates are real fetal beats; scores are averaged across channels and
  physiologically impossible beats (closer than 300 ms) are dropped.
- Converts beats to a fetal heart rate and compares it with the direct fetal ECG (`Direct_1`).
- Benchmarks **9 standard classifiers** fairly (leave-one-recording-out) on sensitivity, precision, F1,
  AUC, heart-rate error, training time and inference time.

### App features
Animated interface, a **Live monitor** replay (scrolling trace, pulsing heart, live BPM), per-recording
**model comparison**, an animated benchmark leaderboard, and CSV downloads.

## How it works

```
CSV  ->  band-pass 8-45 Hz  ->  remove maternal ECG  ->  candidate peaks  ->  24 features
     ->  classifier (one of 9)  ->  average over 4 channels  ->  300 ms refractory rule  ->  FHR
```

| Model | Idea |
|---|---|
| Random Forest (ours) | many decision trees vote |
| Extra Trees | Random Forest with more randomness |
| Gradient Boosting | trees that correct each other's mistakes |
| SVM (RBF) | smooth curved decision boundary |
| Logistic Regression | simple linear model |
| k-Nearest Neighbors | the 7 most similar training examples vote |
| Decision Tree | one small tree of yes/no questions |
| Gaussian Naive Bayes | independent-feature probability model |
| Neural Network (MLP) | small two-layer network |

Plus two non-ML baselines: *every candidate* and *rules only* (at least 2 channels agree).

## Repository structure

```
app.py                      Streamlit app (entry point)
requirements.txt            dependencies (scikit-learn pinned to the training version)
fecg/
  pipeline.py               signal processing, features, beat selection, heart rate
  models.py                 the 9 models, save/load, version pinning
  evaluation.py             leave-one-recording-out benchmark
  ui.py                     animated CSS, hero, leaderboard, live monitor
  synthetic.py              artificial test signals (tests only - never report as results)
scripts/train_models.py     train + benchmark from the command line
notebooks/01_train_and_compare_models.ipynb   same, step by step (Colab friendly)
models/                     generated: trained models + benchmark tables (see models/README.md)
tests/test_pipeline.py      automated checks
.streamlit/config.toml      theme
.github/workflows/tests.yml runs the tests on every push
```

## Quick start (your computer)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 and upload a recording CSV. It needs the columns `Abdomen_1` ... `Abdomen_4`;
`Direct_1` is optional (when present, the app compares against it).

## Train and compare the models

**In Colab (easiest):** open `notebooks/01_train_and_compare_models.ipynb` (Colab -> File -> Open notebook
-> GitHub tab), edit `REPO_URL` and `DATA_DIR`, and run all cells. The last cell downloads
`fecg_models_update.zip` containing `models/` and an updated `requirements.txt`.

**Command line:**

```bash
python scripts/train_models.py --data path/to/csv_folder --pin-requirements
# keep a Random Forest you trained earlier (it is still benchmarked, just not saved):
python scripts/train_models.py --data path/to/csv_folder --skip random_forest --pin-requirements
```

The app also still reads a single `fecg_model.joblib` placed next to `app.py` (the first version of this
project). It is shown as "Random Forest (our model)" unless `models/random_forest.joblib` exists.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. At share.streamlit.io choose the repository, branch `main`, main file `app.py`, then Deploy.
3. In *Advanced settings* choose a Python version close to the one you trained with.

Every push to `main` redeploys the app automatically.

### Updating the repository from the browser
1. Open the repo on GitHub -> **Add file -> Upload files**.
2. Drag in the **contents** of the project folder (including the `fecg`, `scripts`, `tests`,
   `notebooks`, `models` folders) and commit.
3. Delete files from older versions that are no longer used (for example `fecg_pipeline.py`).

### Updating with git
```bash
git add -A
git commit -m "Update app and models"
git push
```

## Evaluation method (and its limits)

- **Leave-one-recording-out:** each recording is held out in turn, so no model is scored on data it was
  trained on. Random splits would leak information because neighbouring beats look alike.
- **Threshold tuning** uses only the training recordings (inner leave-one-recording-out).
- **Beat match:** a detection is correct if it is within 50 ms of a `Direct_1` beat.
- **Labels come from `Direct_1`**, detected with a simple algorithm; they are not human-verified.
- **Very small dataset:** with only a handful of short recordings, differences between models are often
  within the noise. The `F1_std` column shows the spread between recordings; models closer than that are
  effectively tied. Do not claim one model is better unless the gap is clearly larger.
- The Evaluation/Comparison views for an uploaded file are optimistic if that file was used for training.

## Data

Recordings are not included. The expected layout (`Direct_1`, `Abdomen_1..4`, 1000 Hz, a units row under
the header) matches the *Abdominal and Direct Fetal ECG Database* on PhysioNet and its Kaggle copies; check
the dataset page for its license and cite it in your report. Put the CSV files in a folder such as `data/`
(ignored by git).

## Tests

```bash
pip install pytest
python -m pytest -q          # or: python tests/test_pipeline.py
```

## License

MIT - see `LICENSE` (put your name in the copyright line).
