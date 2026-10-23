"""Shared settings for MLflow tracking."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
TRANSFORMER_DIR = ROOT / "model"

# SQLite backend gives us the Model Registry; artifacts land in ./mlruns
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT = "toxic-comment-classification"
REGISTERED_MODEL = "toxic-comment-classifier"

LABELS = ["normal", "abusive", "threat"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
THREAT = LABEL2ID["threat"]
NORMAL = LABEL2ID["normal"]

# Decision policy used in the notebook (flag if P(not normal) >= FLAG_T, threat if P(threat) >= THREAT_T)
FLAG_T = 0.7
THREAT_T = 0.7

# Metric used to pick the champion model
SELECTION_METRIC = "test_macro_f1"


def setup():
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    mlflow.set_experiment_tags({
        "project": "sentiment-analysis-nlp",
        "task": "multiclass text classification",
        "labels": ",".join(LABELS),
        "mlflow.note.content": (
            "Classify comments as normal / abusive / threat. "
            "Classical TF-IDF baselines vs fine-tuned DistilRoBERTa."
        ),
    })
