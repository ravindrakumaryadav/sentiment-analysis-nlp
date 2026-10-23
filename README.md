# Toxic comment classification — MLflow tracking

Everything is logged to the `toxic-comment-classification` experiment in `mlflow.db` (artifacts in `mlruns/`).

```
data/processed/     train / val / test CSVs (text, category)
notebook/           data cleaning + DistilRoBERTa fine-tuning (Colab)
model/              fine-tuned DistilRoBERTa weights + trainer_state.json
mlops/              MLflow tracking code
  config.py           tracking URI, experiment, labels, decision thresholds
  data.py             split loading + dataset logging
  evaluation.py       shared evaluation suite (metrics, plots, tables)
  train_classical.py  TF-IDF baselines
  log_transformer.py  import the Colab transformer run
  register_best.py    leaderboard + Model Registry
```

```bash
python -m venv .venv
.venv\Scripts\activate                      # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # only for the transformer steps
pip install transformers accelerate

python -m mlops.train_classical --tune      # 5 TF-IDF models + nested LogReg C search
python -m mlops.log_transformer             # import the Colab DistilRoBERTa (training history)
python -m mlops.log_transformer --evaluate --log-model   # + test metrics & registrable model (needs torch)
python -m mlops.register_best               # leaderboard run + register winner as @champion

mlflow ui --backend-store-uri sqlite:///mlflow.db       # http://127.0.0.1:5000
```

## What gets tracked

| Category | Details |
|---|---|
| **Datasets** | train/val/test as MLflow datasets (digest, schema, source), class balance, text-length profile + plot |
| **Params** | every vectorizer and classifier hyperparameter; transformer architecture + `TrainingArguments` |
| **Metrics (val & test)** | accuracy, balanced accuracy, macro/weighted F1/precision/recall, MCC, Cohen's kappa, log loss, Brier, ROC-AUC (macro/weighted), top-2 accuracy |
| **Per-class metrics** | precision, recall, F1, support, ROC-AUC, average precision for normal/abusive/threat |
| **Business metrics** | harmful-vs-normal recall/precision, normals wrongly flagged, threats missed, and the notebook's decision policy (`FLAG_T`/`THREAT_T` = 0.7) |
| **Overfitting** | train macro-F1/accuracy/log loss vs val/test |
| **Ops metrics** | training time, inference ms/sample, throughput, model size, feature count, system CPU/RAM/disk (+GPU when present) |
| **Training curves** | transformer loss, learning rate, grad norm, eval metrics per step (replayed from `trainer_state.json`) |
| **Plots** | confusion matrices (counts + normalised, argmax + policy), ROC, PR, calibration, probability histograms, threshold sweep, top n-gram features, tuning curve |
| **Tables** | threshold sweep, top 300 most-confident errors, full predictions CSV, leaderboard |
| **Model** | pipeline with signature + input example, linked to its metrics (MLflow 3 LoggedModel), registered in Model Registry with `@champion` alias |
| **MLflow built-in evaluator** | `mlflow.models.evaluate` metrics (`mlflow_test_*`) and plots |
| **Tuning** | parent run with nested child run per hyperparameter value |

## Tracking future Colab training directly

In the notebook, point MLflow at Drive and let the HF Trainer log live:

```python
!pip install -q mlflow
import mlflow, os
mlflow.set_tracking_uri("sqlite:////content/drive/MyDrive/sentiment_analysis/mlflow.db")
mlflow.set_experiment("toxic-comment-classification")
os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "0"
# in TrainingArguments: report_to="mlflow", run_name="distilroberta-v2"
```

Then download `mlflow.db` + `mlruns/` from Drive, or run a shared MLflow server.
