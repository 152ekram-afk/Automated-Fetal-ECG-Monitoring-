"""Model zoo (standard scikit-learn classifiers), saving / loading, and version bookkeeping."""
import platform
import re
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .pipeline import FEATURE_NAMES

OUR_MODEL_KEY = "random_forest"

# One-line plain-English description of each model (shown in the app).
MODEL_NOTES = {
    "random_forest": "Many decision trees vote together. Our original model: robust, handles mixed features well.",
    "extra_trees": "Like Random Forest but with more randomness in how trees split. Often similar accuracy, faster to train.",
    "gradient_boosting": "Trees are built one after another, each fixing the mistakes of the previous ones.",
    "svm_rbf": "Finds a smooth curved boundary between fetal beats and noise. Needs scaled features.",
    "logistic_regression": "A simple linear model: a weighted sum of the features. Fast and easy to explain.",
    "knn": "Looks at the 7 most similar candidates in the training data and lets them vote.",
    "decision_tree": "One small tree of yes/no questions. Very easy to read, but less stable.",
    "naive_bayes": "Assumes features are independent and applies Bayes' rule. Very fast, often a weak baseline.",
    "mlp": "A small neural network (two hidden layers). Flexible, but needs more data to shine.",
}


def build_zoo(seed=42):
    """key -> (display name, zero-argument factory returning an untrained model)."""
    z = OrderedDict()
    z["random_forest"] = ("Random Forest (our model)", lambda: RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=3, class_weight="balanced",
        random_state=seed, n_jobs=1))
    z["extra_trees"] = ("Extra Trees", lambda: ExtraTreesClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=3, class_weight="balanced",
        random_state=seed, n_jobs=1))
    z["gradient_boosting"] = ("Gradient Boosting", lambda: GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=seed))
    z["svm_rbf"] = ("SVM (RBF kernel)", lambda: make_pipeline(
        StandardScaler(), SVC(kernel="rbf", C=1.0, class_weight="balanced",
                              probability=True, random_state=seed)))
    z["logistic_regression"] = ("Logistic Regression", lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")))
    z["knn"] = ("k-Nearest Neighbors", lambda: make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance")))
    z["decision_tree"] = ("Decision Tree", lambda: DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=seed))
    z["naive_bayes"] = ("Gaussian Naive Bayes", lambda: GaussianNB())
    z["mlp"] = ("Neural Network (MLP)", lambda: make_pipeline(
        StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3,
                                        max_iter=800, random_state=seed)))
    return z


# ----------------------------------------------------------------------------- save / load
def save_bundle(path, model, key, name, threshold, fs):
    """Save a trained model together with everything the app needs to use it."""
    bundle = {"model": model, "key": key, "name": name, "threshold": float(threshold),
              "features": list(FEATURE_NAMES), "fs": int(fs), "sklearn": sklearn.__version__}
    joblib.dump(bundle, path, compress=3)
    return Path(path)


def load_bundle(path):
    """Load a bundle; raises ValueError when it does not match the current feature set."""
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("not a valid model bundle")
    if len(bundle.get("features", [])) != len(FEATURE_NAMES):
        raise ValueError("trained with a different feature set - retrain with the current pipeline")
    bundle.setdefault("key", OUR_MODEL_KEY)
    bundle.setdefault("name", "Random Forest (our model)")
    bundle.setdefault("fs", 1000)
    return bundle


def environment_info():
    return {"sklearn": sklearn.__version__, "numpy": np.__version__,
            "python": platform.python_version()}


def pin_requirements(req_path):
    """Pin the scikit-learn line in requirements.txt to the version used for training."""
    req_path = Path(req_path)
    line = f"scikit-learn=={sklearn.__version__}"
    text = req_path.read_text() if req_path.exists() else ""
    if re.search(r"^scikit-learn\b.*$", text, flags=re.M):
        text = re.sub(r"^scikit-learn\b.*$", line, text, flags=re.M)
    else:
        text = text.rstrip("\n") + "\n" + line + "\n"
    req_path.write_text(text)
    return line
