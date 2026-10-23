"""Import the Colab-trained DistilRoBERTa (./model) into MLflow.

Always logged: architecture + training hyperparameters, the full step-by-step training history
(loss, learning rate, grad norm, eval metrics) replayed from trainer_state.json, curves, config/tokenizer.
With torch + transformers installed (`pip install torch transformers`), also:
    --evaluate      run the model on val/test and log the full evaluation suite
    --log-model     log it as an MLflow transformers model (registrable, servable)

    python -m mlops.log_transformer --evaluate --log-model
    python -m mlops.log_transformer --evaluate --limit 3000   # quicker CPU evaluation on a sample
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from mlflow.entities import Metric

from . import config
from .config import LABELS, TRANSFORMER_DIR
from .data import load_splits, log_datasets, params_to_str
from .evaluation import log_evaluation

# Hyperparameters from notebook/sentiment_analysis.ipynb (used when training_args.bin can't be unpickled)
NOTEBOOK_TRAINING_ARGS = {
    "base_model": "distilroberta-base", "learning_rate": 2e-5, "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 64, "num_train_epochs": 3, "weight_decay": 0.01, "warmup_steps": 500,
    "eval_strategy": "epoch", "save_strategy": "epoch", "metric_for_best_model": "macro_f1",
    "load_best_model_at_end": True, "fp16": True, "max_length": 256,
    "loss": "class-weighted cross entropy (n / (3 * class_count))", "hardware": "Colab GPU",
}
ARCH_KEYS = ["model_type", "architectures", "num_hidden_layers", "hidden_size", "num_attention_heads",
             "intermediate_size", "vocab_size", "max_position_embeddings", "hidden_dropout_prob",
             "attention_probs_dropout_prob", "hidden_act", "transformers_version"]
ARTIFACT_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "trainer_state.json"]


def has_torch() -> bool:
    try:
        import torch, transformers  # noqa: F401
        return True
    except ImportError:
        return False


def training_args() -> dict:
    if has_torch():
        import torch
        try:
            args = torch.load(TRANSFORMER_DIR / "training_args.bin", weights_only=False).to_dict()
            return {**{"base_model": NOTEBOOK_TRAINING_ARGS["base_model"],
                       "max_length": NOTEBOOK_TRAINING_ARGS["max_length"],
                       "loss": NOTEBOOK_TRAINING_ARGS["loss"]}, **args}
        except Exception as e:  # version mismatch etc.
            print(f"Could not read training_args.bin ({e}); using notebook values")
    return NOTEBOOK_TRAINING_ARGS


def replay_history(state: dict):
    """Log every entry of the HF Trainer log_history as step-indexed metrics, so curves appear in the UI."""
    run_id = mlflow.active_run().info.run_id
    rename = {"loss": "train_loss", "learning_rate": "learning_rate", "grad_norm": "grad_norm"}
    metrics = []
    for entry in state["log_history"]:
        step, ts = entry["step"], int(time.time() * 1000)
        for k, v in entry.items():
            if k == "step" or not isinstance(v, (int, float)):
                continue
            key = rename.get(k, k if k.startswith(("eval_", "train_", "epoch", "total_flos")) else f"train_{k}")
            metrics.append(Metric(key, float(v), ts, step))
    client = mlflow.MlflowClient()
    for i in range(0, len(metrics), 1000):
        client.log_batch(run_id, metrics=metrics[i:i + 1000])

    train = [e for e in state["log_history"] if "loss" in e]
    evals = [e for e in state["log_history"] if "eval_loss" in e]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot([e["step"] for e in train], [e["loss"] for e in train], label="train loss")
    axes[0].plot([e["step"] for e in evals], [e["eval_loss"] for e in evals], "o-", label="eval loss")
    axes[1].plot([e["step"] for e in train], [e["learning_rate"] for e in train])
    axes[2].plot([e["step"] for e in train], [e.get("grad_norm", np.nan) for e in train])
    for ax, t in zip(axes, ["Loss", "Learning rate", "Gradient norm"]):
        ax.set(title=t, xlabel="step"); ax.legend() if t == "Loss" else None
    fig.tight_layout()
    mlflow.log_figure(fig, "training/training_curves.png")
    plt.close(fig)

    if evals:
        eval_keys = [k for k in evals[0] if k.startswith("eval_") and k not in
                     ("eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second")]
        fig, ax = plt.subplots(figsize=(7, 4))
        for k in eval_keys:
            ax.plot([e["epoch"] for e in evals], [e[k] for e in evals], "o-", label=k)
        ax.set(title="Validation metrics per epoch", xlabel="epoch"); ax.legend()
        mlflow.log_figure(fig, "training/eval_metrics_per_epoch.png")
        plt.close(fig)


def predict_proba(texts, batch_size=64, max_length=256):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    out = []
    texts = list(texts)
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_length,
                  padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out.append(torch.softmax(model(**enc).logits, dim=1).cpu().numpy())
        if (i // batch_size) % 50 == 0:
            print(f"  {i + batch_size}/{len(texts)}")
    return np.concatenate(out), (tok, model)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluate", action="store_true", help="run inference on val/test (needs torch)")
    p.add_argument("--log-model", action="store_true", help="log as MLflow transformers model (needs torch)")
    p.add_argument("--log-weights", action="store_true",
                   help="without torch: copy raw model.safetensors into the run as an artifact")
    p.add_argument("--limit", type=int, help="evaluate on a stratified sample of this many rows per split")
    args = p.parse_args()

    if (args.evaluate or args.log_model) and not has_torch():
        raise SystemExit("--evaluate/--log-model need torch + transformers: pip install torch transformers")

    config.setup()
    state = json.loads((TRANSFORMER_DIR / "trainer_state.json").read_text())
    hf_config = json.loads((TRANSFORMER_DIR / "config.json").read_text())

    with mlflow.start_run(run_name="distilroberta-finetuned",
                          tags={"model_family": "distilroberta", "features": "subword-transformer",
                                "stage": "finetuned", "framework": "transformers",
                                "trained_on": "Google Colab GPU", "imported_from": str(TRANSFORMER_DIR)},
                          description="DistilRoBERTa fine-tuned with class-weighted loss (imported from Colab)."):
        mlflow.log_params({f"arch_{k}": v for k, v in params_to_str(
            {k: hf_config.get(k) for k in ARCH_KEYS}).items()})
        mlflow.log_params({f"train_{k}": v for k, v in params_to_str(training_args()).items()
                           if not k.startswith(("hub_", "push_to_hub", "logging_dir", "output_dir"))})
        mlflow.log_params({"trainer_global_step": state["global_step"], "trainer_epochs": state["epoch"],
                           "trainer_best_checkpoint": state["best_model_checkpoint"],
                           "trainer_best_metric": state["best_metric"],
                           "id2label": json.dumps(hf_config["id2label"])})
        mlflow.log_metric("total_flos", state["total_flos"])
        replay_history(state)
        for f in ARTIFACT_FILES:
            mlflow.log_artifact(str(TRANSFORMER_DIR / f), "hf_files")
        size = (TRANSFORMER_DIR / "model.safetensors").stat().st_size
        mlflow.log_metric("model_size_mb", size / 1e6)
        if args.log_weights and not args.log_model:
            mlflow.log_artifact(str(TRANSFORMER_DIR / "model.safetensors"), "hf_files")

        model_id = None
        if args.evaluate or args.log_model:
            splits = load_splits()
            if args.limit:
                splits = {k: v.groupby("label", group_keys=False)
                               .apply(lambda g: g.sample(max(1, args.limit * len(g) // len(v)), random_state=42))
                               .reset_index(drop=True) for k, v in splits.items()}
            datasets = log_datasets(splits)

            if args.log_model:
                from transformers import pipeline as hf_pipeline
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                clf = hf_pipeline("text-classification",
                                  model=AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR),
                                  tokenizer=AutoTokenizer.from_pretrained(TRANSFORMER_DIR), top_k=None)
                import torch
                import transformers
                info = mlflow.transformers.log_model(
                    clf, name="model", task="text-classification",
                    # explicit, since auto-detection wrongly expects torchvision
                    pip_requirements=[f"torch=={torch.__version__.split('+')[0]}",
                                      f"transformers=={transformers.__version__}", "accelerate"],
                    input_example=["I will find you", "thanks for the edit"])
                model_id = info.model_id

            if args.evaluate:
                for split in ("val", "test"):
                    df = splits[split]
                    print(f"Scoring {split} ({len(df)} rows)...")
                    t0 = time.time()
                    proba, _ = predict_proba(df["text"])
                    elapsed = time.time() - t0
                    m = log_evaluation(split, df["text"], df["label"].to_numpy(), proba,
                                       model_id=model_id, dataset=datasets[split])
                    if split == "test":
                        mlflow.log_metrics({"inference_ms_per_sample": 1000 * elapsed / len(df),
                                            "inference_samples_per_sec": len(df) / elapsed})
                        print(f"test macro_f1={m['test_macro_f1']:.4f} threat_recall={m['test_threat_recall']:.4f}")
                mlflow.set_tag("eval_sample_limit", str(args.limit or "full"))
        else:
            print("Logged training history only. Install torch + transformers and pass --evaluate for test metrics.")


if __name__ == "__main__":
    main()
