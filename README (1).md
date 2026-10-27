# models/

This folder is **generated** by the training notebook (`notebooks/01_train_and_compare_models.ipynb`)
or by `python scripts/train_models.py --data <folder>`. The app reads everything in here:

| File | What it is |
|---|---|
| `random_forest.joblib`, `svm_rbf.joblib`, ... | one trained model per file (with its tuned threshold) |
| `benchmark_summary.csv` | leave-one-recording-out results, one row per method |
| `benchmark_per_record.csv` | the same results for every held-out recording |
| `model_info.json` | scikit-learn / numpy / Python versions used for training |

Until you train, this folder is empty and the app falls back to rules-only detection.
After training, upload the **whole folder** (and the updated `requirements.txt`) to GitHub.
