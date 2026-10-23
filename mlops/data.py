"""Loading data splits and logging them (as MLflow datasets + a profile) to the active run."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from .config import DATA_DIR, LABEL2ID, LABELS


def load_split(name: str, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{name}.csv").dropna(subset=["text"]).reset_index(drop=True)
    df["label"] = df["category"].map(LABEL2ID).astype(int)
    if limit:
        # Stratified subsample so every class survives
        df = (df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), max(1, limit * len(g) // len(df))), random_state=42))
                .reset_index(drop=True))
    return df


def load_splits(limit: int | None = None) -> dict[str, pd.DataFrame]:
    return {s: load_split(s, limit if s == "train" else (limit // 3 if limit else None))
            for s in ("train", "val", "test")}


def to_mlflow_dataset(df: pd.DataFrame, split: str):
    return mlflow.data.from_pandas(
        df[["text", "label"]], source=str(DATA_DIR / f"{split}.csv"),
        targets="label", name=f"toxic-{split}",
    )


def log_datasets(splits: dict[str, pd.DataFrame]) -> dict:
    """Log each split as a run input, plus a profile (class balance, text length) as params/artifacts."""
    contexts = {"train": "training", "val": "validation", "test": "testing"}
    datasets = {}
    profile = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for split, df in splits.items():
        ds = to_mlflow_dataset(df, split)
        mlflow.log_input(ds, context=contexts[split])
        datasets[split] = ds

        counts = df["category"].value_counts().reindex(LABELS, fill_value=0)
        lengths = df["text"].str.split().str.len()
        profile[split] = {
            "rows": len(df),
            "class_counts": counts.to_dict(),
            "class_ratio": (counts / len(df)).round(4).to_dict(),
            "words_mean": round(float(lengths.mean()), 1),
            "words_median": float(lengths.median()),
            "words_p95": float(lengths.quantile(0.95)),
            "duplicates": int(df["text"].duplicated().sum()),
        }
        mlflow.log_param(f"data_{split}_rows", len(df))
        axes[0].bar([f"{split}\n{l}" for l in LABELS], counts.values)
        axes[1].hist(lengths.clip(upper=400), bins=60, alpha=0.5, label=split)

    mlflow.log_param("data_dir", str(DATA_DIR))
    axes[0].set_title("Class distribution"); axes[0].tick_params(axis="x", labelsize=7)
    axes[1].set_title("Words per comment (clipped at 400)"); axes[1].legend()
    fig.tight_layout()
    mlflow.log_figure(fig, "data/data_profile.png")
    plt.close(fig)
    mlflow.log_dict(profile, "data/data_profile.json")
    return datasets


def params_to_str(params: dict) -> dict:
    """MLflow params must be short strings; drop objects and truncate long values."""
    out = {}
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool, type(None), tuple, list)):
            s = json.dumps(v) if isinstance(v, (tuple, list)) else str(v)
            out[k] = s[:500]
    return out
