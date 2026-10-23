"""Compare all tracked models, log a leaderboard run, and register the best one with the `champion` alias.

    python -m mlops.register_best
    python -m mlops.register_best --metric test_threat_recall
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from . import config

COLUMNS = ["test_macro_f1", "test_weighted_f1", "test_accuracy", "test_roc_auc_ovr_macro", "test_log_loss",
           "test_threat_recall", "test_threat_precision", "test_abusive_f1", "test_policy_macro_f1",
           "test_policy_threat_recall", "test_normal_wrongly_flagged", "inference_ms_per_sample",
           "model_size_mb", "train_time_sec"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metric", default=config.SELECTION_METRIC)
    args = p.parse_args()

    config.setup()
    runs = mlflow.search_runs(
        experiment_names=[config.EXPERIMENT],
        filter_string="attributes.status = 'FINISHED'",
    )
    # Tag filters don't support IN, so select model runs here
    runs = runs[runs.get("tags.stage", pd.Series(index=runs.index, dtype=str)).isin(["baseline", "finetuned"])]
    runs = runs[runs.get(f"metrics.{args.metric}").notna()] if f"metrics.{args.metric}" in runs else runs.iloc[0:0]
    if runs.empty:
        raise SystemExit(f"No finished runs with metric {args.metric}. Train some models first.")

    # Keep only the latest run per model family
    runs = runs.sort_values("start_time", ascending=False).drop_duplicates("tags.model_family")
    ascending = "loss" in args.metric
    runs = runs.sort_values(f"metrics.{args.metric}", ascending=ascending)
    board = runs[["run_id", "tags.model_family"] + [f"metrics.{c}" for c in COLUMNS if f"metrics.{c}" in runs]]
    board.columns = [c.replace("metrics.", "").replace("tags.", "") for c in board.columns]
    print(board.drop(columns="run_id").round(4).to_string(index=False))

    exp_id = runs.iloc[0]["experiment_id"]

    def logged_model(run_id):
        found = mlflow.search_logged_models(experiment_ids=[exp_id], output_format="list",
                                            filter_string=f"source_run_id = '{run_id}'")
        return found[0] if found else None

    registrable = [(r, m) for r in board.itertuples() if (m := logged_model(r.run_id))]
    if not registrable:
        raise SystemExit("No run has a logged model to register.")
    best, best_model = registrable[0]

    with mlflow.start_run(run_name="model-comparison", tags={"stage": "comparison"}):
        mlflow.log_param("selection_metric", args.metric)
        mlflow.log_param("winner", best.model_family)
        mlflow.log_table(board, "leaderboard.json")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, m in zip(axes, ["test_macro_f1", "test_threat_recall", "inference_ms_per_sample"]):
            if m in board:
                ax.barh(board["model_family"], board[m]); ax.set_title(m); ax.invert_yaxis()
        fig.tight_layout()
        mlflow.log_figure(fig, "leaderboard.png")
        plt.close(fig)

    client = mlflow.MlflowClient()
    mv = mlflow.register_model(f"models:/{best_model.model_id}", config.REGISTERED_MODEL,
                               tags={"model_family": best.model_family, "selection_metric": args.metric})
    client.update_registered_model(config.REGISTERED_MODEL,
                                   description="Comment classifier: normal / abusive / threat.")
    client.update_model_version(config.REGISTERED_MODEL, mv.version,
                                description=f"{best.model_family}, {args.metric}={getattr(best, args.metric):.4f}")
    client.set_registered_model_alias(config.REGISTERED_MODEL, "champion", mv.version)
    print(f"\nRegistered {config.REGISTERED_MODEL} v{mv.version} ({best.model_family}) as @champion")
    print(f"Load it with: mlflow.pyfunc.load_model('models:/{config.REGISTERED_MODEL}@champion')")


if __name__ == "__main__":
    main()
