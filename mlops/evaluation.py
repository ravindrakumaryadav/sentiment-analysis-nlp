"""One evaluation suite for every model: metrics, plots, tables and predictions logged to the active run."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, cohen_kappa_score, confusion_matrix, f1_score, log_loss,
    matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve, top_k_accuracy_score,
)

from .config import FLAG_T, LABELS, NORMAL, THREAT, THREAT_T

SWEEP = np.round(np.arange(0.1, 0.951, 0.05), 2)
ABUSIVE = LABELS.index("abusive")


def decide(proba: np.ndarray, flag_t: float = FLAG_T, threat_t: float = THREAT_T) -> np.ndarray:
    """Production decision policy from the notebook."""
    preds = np.full(len(proba), NORMAL)
    flagged = (1 - proba[:, NORMAL]) >= flag_t
    preds[flagged] = ABUSIVE
    preds[flagged & (proba[:, THREAT] >= threat_t)] = THREAT
    return preds


def compute_metrics(y: np.ndarray, proba: np.ndarray, preds: np.ndarray | None = None) -> dict:
    preds = proba.argmax(1) if preds is None else preds
    y_onehot = np.eye(len(LABELS))[y]
    m = {
        "accuracy": accuracy_score(y, preds),
        "balanced_accuracy": balanced_accuracy_score(y, preds),
        "macro_f1": f1_score(y, preds, average="macro"),
        "weighted_f1": f1_score(y, preds, average="weighted"),
        "macro_precision": precision_score(y, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(y, preds, average="macro"),
        "mcc": matthews_corrcoef(y, preds),
        "cohen_kappa": cohen_kappa_score(y, preds),
        "log_loss": log_loss(y, proba, labels=range(len(LABELS))),
        "brier_score": float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1))),
        "roc_auc_ovr_macro": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
        "roc_auc_ovr_weighted": roc_auc_score(y, proba, multi_class="ovr", average="weighted"),
        "top2_accuracy": top_k_accuracy_score(y, proba, k=2, labels=range(len(LABELS))),
    }
    p, r, f, s = (precision_score(y, preds, average=None, zero_division=0, labels=range(len(LABELS))),
                  recall_score(y, preds, average=None, zero_division=0, labels=range(len(LABELS))),
                  f1_score(y, preds, average=None, labels=range(len(LABELS))),
                  np.bincount(y, minlength=len(LABELS)))
    for i, name in enumerate(LABELS):
        m[f"{name}_precision"] = p[i]
        m[f"{name}_recall"] = r[i]
        m[f"{name}_f1"] = f[i]
        m[f"{name}_support"] = int(s[i])
        m[f"{name}_roc_auc"] = roc_auc_score(y == i, proba[:, i])
        m[f"{name}_avg_precision"] = average_precision_score(y == i, proba[:, i])

    # Binary "harmful vs normal" view (what a moderation queue actually sees)
    harmful_true, harmful_score = y != NORMAL, 1 - proba[:, NORMAL]
    harmful_pred = preds != NORMAL
    m["harmful_recall"] = recall_score(harmful_true, harmful_pred)
    m["harmful_precision"] = precision_score(harmful_true, harmful_pred, zero_division=0)
    m["harmful_roc_auc"] = roc_auc_score(harmful_true, harmful_score)
    m["normal_wrongly_flagged"] = int((harmful_pred & ~harmful_true).sum())
    m["threat_missed"] = int(((y == THREAT) & (preds != THREAT)).sum())
    return {k: float(v) for k, v in m.items()}


def _plot_confusion(y, preds, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, norm in zip(axes, [None, "true"]):
        ConfusionMatrixDisplay(confusion_matrix(y, preds, normalize=norm, labels=range(len(LABELS))),
                               display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False,
                                                           values_format=".2f" if norm else "d")
        ax.set_title(f"{title} ({'row-normalised' if norm else 'counts'})")
    fig.tight_layout()
    return fig


def _plot_curves(y, proba):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, name in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(y == i, proba[:, i])
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y == i, proba[:, i]):.3f})")
        prec, rec, _ = precision_recall_curve(y == i, proba[:, i])
        axes[1].plot(rec, prec, label=f"{name} (AP={average_precision_score(y == i, proba[:, i]):.3f})")
        frac, mean_pred = calibration_curve(y == i, proba[:, i], n_bins=10, strategy="quantile")
        axes[2].plot(mean_pred, frac, marker="o", label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[2].plot([0, 1], [0, 1], "k--", lw=0.8)
    for ax, (t, xl, yl) in zip(axes, [("ROC (one-vs-rest)", "FPR", "TPR"),
                                      ("Precision-Recall", "Recall", "Precision"),
                                      ("Calibration", "Mean predicted prob", "Fraction positive")]):
        ax.set(title=t, xlabel=xl, ylabel=yl); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _threshold_sweep(y, proba) -> pd.DataFrame:
    rows = []
    for t in SWEEP:
        tp = proba[:, THREAT] >= t
        hp = (1 - proba[:, NORMAL]) >= t
        rows.append({
            "threshold": float(t),
            "threat_precision": precision_score(y == THREAT, tp, zero_division=0),
            "threat_recall": recall_score(y == THREAT, tp),
            "threat_f1": f1_score(y == THREAT, tp),
            "harmful_precision": precision_score(y != NORMAL, hp, zero_division=0),
            "harmful_recall": recall_score(y != NORMAL, hp),
            "harmful_f1": f1_score(y != NORMAL, hp),
            "normal_wrongly_flagged": int((hp & (y == NORMAL)).sum()),
        })
    return pd.DataFrame(rows)


def _plot_sweep(sweep: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, kind in zip(axes, ["threat", "harmful"]):
        for metric in ["precision", "recall", "f1"]:
            ax.plot(sweep["threshold"], sweep[f"{kind}_{metric}"], marker=".", label=metric)
        ax.set(title=f"{kind}: metrics vs threshold", xlabel="threshold"); ax.legend()
    fig.tight_layout()
    return fig


def _plot_proba_hist(y, proba):
    fig, axes = plt.subplots(1, len(LABELS), figsize=(15, 4))
    for i, (ax, name) in enumerate(zip(axes, LABELS)):
        for j, true_name in enumerate(LABELS):
            ax.hist(proba[y == j, i], bins=40, alpha=0.5, density=True, label=f"true={true_name}")
        ax.set_title(f"P({name})"); ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def log_evaluation(split: str, texts: pd.Series, y: np.ndarray, proba: np.ndarray,
                   model_id: str | None = None, dataset=None, max_errors: int = 300) -> dict:
    """Log the full evaluation of `proba` against `y` under `<split>_` metrics and `eval/<split>/` artifacts."""
    y = np.asarray(y)
    preds = proba.argmax(1)
    art = f"eval/{split}"

    metrics = {f"{split}_{k}": v for k, v in compute_metrics(y, proba, preds).items()}
    policy = compute_metrics(y, proba, decide(proba))
    metrics.update({f"{split}_policy_{k}": policy[k] for k in
                    ["macro_f1", "accuracy", "threat_recall", "threat_precision", "abusive_recall",
                     "abusive_precision", "harmful_recall", "harmful_precision", "normal_wrongly_flagged"]})
    mlflow.log_metrics(metrics, model_id=model_id, dataset=dataset)

    mlflow.log_text(classification_report(y, preds, target_names=LABELS, digits=4),
                    f"{art}/classification_report_argmax.txt")
    mlflow.log_text(classification_report(y, decide(proba), target_names=LABELS, digits=4),
                    f"{art}/classification_report_policy_flag{FLAG_T}_threat{THREAT_T}.txt")
    mlflow.log_dict(classification_report(y, preds, target_names=LABELS, output_dict=True),
                    f"{art}/classification_report.json")
    mlflow.log_dict({"labels": LABELS, "matrix": confusion_matrix(y, preds).tolist()},
                    f"{art}/confusion_matrix.json")

    for name, fig in [("confusion_matrix_argmax.png", _plot_confusion(y, preds, f"{split} argmax")),
                      ("confusion_matrix_policy.png", _plot_confusion(y, decide(proba), f"{split} policy")),
                      ("roc_pr_calibration.png", _plot_curves(y, proba)),
                      ("probability_distributions.png", _plot_proba_hist(y, proba))]:
        mlflow.log_figure(fig, f"{art}/{name}")
        plt.close(fig)

    sweep = _threshold_sweep(y, proba)
    mlflow.log_table(sweep, f"{art}/threshold_sweep.json")
    fig = _plot_sweep(sweep)
    mlflow.log_figure(fig, f"{art}/threshold_sweep.png")
    plt.close(fig)

    pred_df = pd.DataFrame({"text": texts.values, "true": [LABELS[i] for i in y],
                            "pred": [LABELS[i] for i in preds],
                            **{f"p_{l}": proba[:, i].round(5) for i, l in enumerate(LABELS)}})
    pred_df["confidence"] = proba.max(1)
    pred_df["correct"] = pred_df["true"] == pred_df["pred"]
    errors = (pred_df[~pred_df["correct"]].sort_values("confidence", ascending=False)
              .head(max_errors).assign(text=lambda d: d["text"].str.slice(0, 500)))
    mlflow.log_table(errors, f"{art}/top_confident_errors.json")
    mlflow.log_text(pred_df.to_csv(index=False), f"{art}/predictions.csv")
    return metrics
