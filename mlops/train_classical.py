"""Train TF-IDF baselines and track everything in MLflow.

    python -m mlops.train_classical                    # all models
    python -m mlops.train_classical --models logreg nb  # a subset
    python -m mlops.train_classical --tune             # + nested hyperparameter search for logreg
    python -m mlops.train_classical --limit 20000      # quick smoke run on a stratified sample
"""
import argparse
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils import estimator_html_repr
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from . import config
from .config import LABELS
from .data import load_splits, log_datasets, params_to_str
from .evaluation import compute_metrics, log_evaluation

TFIDF = dict(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=200_000,
             sublinear_tf=True, strip_accents="unicode")


def tfidf(**overrides):
    # ColumnTransformer lets the pyfunc model take a DataFrame with a "text" column
    return ColumnTransformer([("tfidf", TfidfVectorizer(**{**TFIDF, **overrides}), "text")])


MODELS = {
    "logreg": lambda: (tfidf(), LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000)),
    "linear_svc": lambda: (tfidf(), CalibratedClassifierCV(
        LinearSVC(C=0.5, class_weight="balanced"), cv=3, method="sigmoid")),
    "complement_nb": lambda: (tfidf(), ComplementNB(alpha=0.3)),
    "sgd_modified_huber": lambda: (tfidf(), SGDClassifier(
        loss="modified_huber", alpha=2e-5, class_weight="balanced", max_iter=50, random_state=42)),
    "xgboost": lambda: (tfidf(max_features=30_000), XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.15, subsample=0.8, colsample_bytree=0.5,
        tree_method="hist", eval_metric="mlogloss", n_jobs=-1, random_state=42)),
}
NEEDS_SAMPLE_WEIGHT = {"xgboost", "complement_nb"}


def top_features(pipe: Pipeline, k: int = 25) -> dict | None:
    """Most indicative n-grams per class, when the model exposes them."""
    clf = pipe.named_steps["clf"]
    names = pipe.named_steps["features"].get_feature_names_out()
    names = np.array([n.split("__", 1)[-1] for n in names])
    if hasattr(clf, "coef_"):
        weights = clf.coef_
    elif hasattr(clf, "calibrated_classifiers_"):
        weights = np.mean([c.estimator.coef_ for c in clf.calibrated_classifiers_], axis=0)
    elif hasattr(clf, "feature_log_prob_"):
        weights = clf.feature_log_prob_ - clf.feature_log_prob_.mean(0)
    elif hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
        idx = imp.argsort()[::-1][:k]
        return {"all_classes": [[names[i], float(imp[i])] for i in idx]}
    else:
        return None
    return {LABELS[c]: [[names[i], float(weights[c, i])] for i in weights[c].argsort()[::-1][:k]]
            for c in range(len(LABELS))}


def plot_top_features(feats: dict):
    fig, axes = plt.subplots(1, len(feats), figsize=(6 * len(feats), 7), squeeze=False)
    for ax, (cls, items) in zip(axes[0], feats.items()):
        words, vals = zip(*items[::-1])
        ax.barh(words, vals)
        ax.set_title(f"Top features: {cls}")
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig


def tune_logreg(splits, grid=(0.5, 1, 2, 4, 8, 16)):
    """Nested child runs: one per C value, scored on validation macro-F1."""
    X_tr, y_tr = splits["train"][["text"]], splits["train"]["label"]
    X_va, y_va = splits["val"][["text"]], splits["val"]["label"]
    features = tfidf().fit(X_tr)
    Xt_tr, Xt_va = features.transform(X_tr), features.transform(X_va)

    scores = {}
    with mlflow.start_run(run_name="tune_logreg", tags={"stage": "tuning", "model_family": "logreg"}):
        mlflow.log_params({"search": "grid", "grid_C": str(list(grid)), "objective": "val_macro_f1"})
        for C in grid:
            with mlflow.start_run(run_name=f"logreg_C={C}", nested=True, tags={"stage": "tuning-trial"}):
                t0 = time.time()
                clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(Xt_tr, y_tr)
                score = f1_score(y_va, clf.predict(Xt_va), average="macro")
                mlflow.log_param("C", C)
                mlflow.log_metrics({"val_macro_f1": score, "train_time_sec": time.time() - t0})
                scores[C] = score
                print(f"  C={C:<5} val_macro_f1={score:.4f}")
        best = max(scores, key=scores.get)
        mlflow.log_params({"best_C": best})
        mlflow.log_metric("best_val_macro_f1", scores[best])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogx(list(scores), list(scores.values()), marker="o")
        ax.set(xlabel="C", ylabel="val macro F1", title="LogReg C search")
        mlflow.log_figure(fig, "tuning_curve.png")
        plt.close(fig)
    return best


def train_one(name: str, splits: dict, builtin_eval: bool, overrides: dict | None = None):
    features, clf = MODELS[name]()
    if overrides:
        clf.set_params(**overrides)
    pipe = Pipeline([("features", features), ("clf", clf)])
    train, val, test = splits["train"], splits["val"], splits["test"]

    with mlflow.start_run(run_name=name, log_system_metrics=True,
                          tags={"model_family": name, "features": "tfidf", "stage": "baseline",
                                "framework": "scikit-learn" if name != "xgboost" else "xgboost"},
                          description=f"TF-IDF + {type(clf).__name__} on normal/abusive/threat"):
        datasets = log_datasets(splits)
        mlflow.log_params({f"tfidf_{k}": v for k, v in params_to_str(
            features.transformers[0][1].get_params()).items()})
        mlflow.log_params({f"clf_{k}": v for k, v in params_to_str(clf.get_params()).items()})
        mlflow.log_params({"classifier": type(clf).__name__,
                           "class_balancing": "sample_weight" if name in NEEDS_SAMPLE_WEIGHT else "class_weight"})

        fit_kw = {}
        if name in NEEDS_SAMPLE_WEIGHT:
            fit_kw["clf__sample_weight"] = compute_sample_weight("balanced", train["label"])
        t0 = time.time()
        pipe.fit(train[["text"]], train["label"], **fit_kw)
        train_time = time.time() - t0
        print(f"[{name}] trained in {train_time:.1f}s")

        n_features = len(pipe.named_steps["features"].get_feature_names_out())
        mlflow.log_metrics({"train_time_sec": train_time, "n_features": n_features,
                            "model_size_mb": len(pickle.dumps(pipe)) / 1e6})

        # Log the model first so evaluation metrics are linked to it (MLflow 3 LoggedModel)
        sample = test[["text"]].head(5)
        model_info = mlflow.sklearn.log_model(
            pipe, name="model", serialization_format="cloudpickle",
            signature=infer_signature(sample, pipe.predict(sample)),
            input_example=sample, params={"labels": ",".join(LABELS)},
            tags={"model_family": name},
        )
        mlflow.log_text(estimator_html_repr(pipe), "model_info/pipeline.html")
        mlflow.log_dict({"id2label": dict(enumerate(LABELS))}, "model_info/label_map.json")

        for split_name, df in [("train", train), ("val", val), ("test", test)]:
            t0 = time.time()
            proba = pipe.predict_proba(df[["text"]])
            elapsed = time.time() - t0
            if split_name == "train":  # train metrics only, to measure overfitting
                m = compute_metrics(df["label"].to_numpy(), proba)
                mlflow.log_metrics({f"train_{k}": m[k] for k in ["macro_f1", "accuracy", "log_loss"]},
                                   model_id=model_info.model_id, dataset=datasets["train"])
                continue
            metrics = log_evaluation(split_name, df["text"], df["label"].to_numpy(), proba,
                                     model_id=model_info.model_id, dataset=datasets[split_name])
            if split_name == "test":
                mlflow.log_metrics({"inference_ms_per_sample": 1000 * elapsed / len(df),
                                    "inference_samples_per_sec": len(df) / elapsed})
                print(f"[{name}] test macro_f1={metrics['test_macro_f1']:.4f} "
                      f"threat_recall={metrics['test_threat_recall']:.4f}")

        feats = top_features(pipe)
        if feats:
            mlflow.log_dict(feats, "explainability/top_features.json")
            fig = plot_top_features(feats)
            mlflow.log_figure(fig, "explainability/top_features.png")
            plt.close(fig)

        if builtin_eval:
            # MLflow's built-in evaluator adds its own metrics, per-class tables and plots
            mlflow.models.evaluate(
                model_info.model_uri, data=test[["text", "label"]], targets="label",
                model_type="classifier",
                evaluator_config={"log_model_explainability": False, "metric_prefix": "mlflow_test_",
                                  "label_list": list(range(len(LABELS)))},
                model_id=model_info.model_id,
            )
        return mlflow.active_run().info.run_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--tune", action="store_true", help="grid-search LogReg C in nested runs first")
    p.add_argument("--limit", type=int, help="stratified train sample size for quick runs")
    p.add_argument("--no-builtin-eval", action="store_true", help="skip mlflow.models.evaluate")
    args = p.parse_args()

    config.setup()
    splits = load_splits(args.limit)
    overrides = {}
    if args.tune and "logreg" in args.models:
        overrides["logreg"] = {"C": tune_logreg(splits)}
    for name in args.models:
        train_one(name, splits, builtin_eval=not args.no_builtin_eval, overrides=overrides.get(name))
    print(f"\nDone. View with:  mlflow ui --backend-store-uri {config.TRACKING_URI}")


if __name__ == "__main__":
    main()
