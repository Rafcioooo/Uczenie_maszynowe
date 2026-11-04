"""Logika eksperymentów MLP zgodna z instrukcją laboratorium."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import config


METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def load_data(
    path: Path, target_column: str, drop_duplicates: bool
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Wczytuje i sprawdza dane, nie wykonując skalowania."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Nie znaleziono pliku CSV: {path}\n"
            "Sprawdź DATA_PATH w config.py albo umieść plik w katalogu "
            "'uploads' projektu."
        )
    data = pd.read_csv(path)
    if target_column not in data.columns:
        raise ValueError(f"Brak kolumny docelowej {target_column!r}.")

    duplicate_count = int(data.duplicated().sum())
    if drop_duplicates:
        data = data.drop_duplicates().reset_index(drop=True)

    if data.isna().any().any():
        missing = int(data.isna().sum().sum())
        raise ValueError(f"Zbiór zawiera {missing} brakujących wartości.")

    X = data.drop(columns=target_column)
    y = data[target_column]
    non_numeric = X.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric:
        raise ValueError(f"Cechy nienumeryczne wymagają kodowania: {non_numeric}")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Cechy zawierają wartości NaN lub nieskończone.")
    if y.nunique() < 2:
        raise ValueError("Do klasyfikacji potrzebne są co najmniej dwie klasy.")

    class_counts = y.value_counts().sort_index()
    info = {
        "source": str(path),
        "rows": int(len(data)),
        "features": int(X.shape[1]),
        "classes": {str(key): int(value) for key, value in class_counts.items()},
        "duplicates_found": duplicate_count,
        "duplicates_removed": duplicate_count if drop_duplicates else 0,
    }
    return X, y, info


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    n_repeats: int,
    random_seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generuje wszystkie podziały jeden raz, aby warianty były porównywalne."""
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_seed,
    )
    return list(cv.split(X, y))


def params_for_variant(variant: config.Variant) -> dict[str, Any]:
    params = dict(config.BASE_MLP_PARAMS)
    params.update(variant.changes)
    return params


def _validate_training_params(params: dict[str, Any]) -> None:
    if params["solver"] not in {"adam", "sgd"}:
        raise ValueError(
            "Uczenie epokami przez partial_fit obsługuje solvery 'adam' i 'sgd'. "
            f"Otrzymano: {params['solver']!r}."
        )


def run_experiment(
    experiment_name: str,
    variants: Iterable[config.Variant],
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_splits: int,
    epochs: int,
    model_seed_base: int,
) -> pd.DataFrame:
    """Uruchamia warianty na identycznych splitach i zwraca wynik długi."""
    records: list[dict[str, Any]] = []
    classes = np.unique(y)

    for variant in variants:
        params = params_for_variant(variant)
        _validate_training_params(params)
        print(f"\n[{experiment_name}] wariant: {variant.name}")

        for split_id, (train_idx, val_idx) in enumerate(splits):
            repeat_id = split_id // n_splits
            fold_id = split_id % n_splits
            model_seed = model_seed_base + split_id

            # Scaler zna wyłącznie dane treningowe bieżącego foldu.
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X.iloc[train_idx])
            X_val = scaler.transform(X.iloc[val_idx])
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            # Każdy wariant na tym samym splicie ma identyczne ziarno startowe.
            model = MLPClassifier(**params, random_state=model_seed)
            cumulative_fit_seconds = 0.0

            for epoch in range(1, epochs + 1):
                started = time.perf_counter()
                if epoch == 1:
                    model.partial_fit(X_train, y_train, classes=classes)
                else:
                    model.partial_fit(X_train, y_train)
                epoch_seconds = time.perf_counter() - started
                cumulative_fit_seconds += epoch_seconds

                prediction = model.predict(X_val)
                records.append(
                    {
                        "experiment": experiment_name,
                        "variant": variant.name,
                        "repeat": repeat_id,
                        "fold": fold_id,
                        "split_id": split_id,
                        "epoch": epoch,
                        "model_seed": model_seed,
                        "accuracy": accuracy_score(y_val, prediction),
                        "balanced_accuracy": balanced_accuracy_score(
                            y_val, prediction
                        ),
                        "macro_f1": f1_score(
                            y_val, prediction, average="macro", zero_division=0
                        ),
                        "epoch_fit_seconds": epoch_seconds,
                        "cumulative_fit_seconds": cumulative_fit_seconds,
                    }
                )

        print(
            f"  ukończono {len(splits)} podziałów × {epochs} epok",
            flush=True,
        )

    return pd.DataFrame.from_records(records)


def aggregate_results(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Najpierw uśrednia foldy, następnie liczy statystyki z powtórzeń."""
    repeat_scores = (
        results.groupby(
            ["experiment", "variant", "repeat", "epoch"], as_index=False
        )[[*METRICS, "cumulative_fit_seconds"]]
        .mean()
        .sort_values(["experiment", "variant", "repeat", "epoch"])
    )

    summary = (
        repeat_scores.groupby(
            ["experiment", "variant", "epoch"], as_index=False
        )
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            fit_seconds_mean=("cumulative_fit_seconds", "mean"),
            fit_seconds_std=("cumulative_fit_seconds", "std"),
        )
        .fillna(0.0)
    )
    return repeat_scores, summary


def save_plots(
    repeat_scores: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    show_plots: bool,
    save_plots: bool,
) -> None:
    """Tworzy i wyświetla wykresy, opcjonalnie zapisując kopie PNG."""
    plot_dir = output_dir / "plots"
    if save_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    for experiment in summary["experiment"].unique():
        experiment_summary = summary[summary["experiment"] == experiment]
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for variant in experiment_summary["variant"].unique():
                values = experiment_summary[
                    experiment_summary["variant"] == variant
                ].sort_values("epoch")
                epochs = values["epoch"].to_numpy()
                mean = values[f"{metric}_mean"].to_numpy()
                std = values[f"{metric}_std"].to_numpy()
                ax.plot(epochs, mean, label=variant, linewidth=2)
                ax.fill_between(epochs, mean - std, mean + std, alpha=0.15)

            ax.set(
                title=f"{experiment}: {metric} (średnia ± SD z powtórzeń)",
                xlabel="Epoka",
                ylabel=metric,
                ylim=(0.0, 1.0),
            )
            ax.set_yticks(np.arange(0.0, 1.01, 0.1))
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            if save_plots:
                fig.savefig(
                    plot_dir / f"{experiment}_{metric}_curve.png", dpi=160
                )

        final_epoch = int(
            repeat_scores.loc[
                repeat_scores["experiment"] == experiment, "epoch"
            ].max()
        )
        final_values = repeat_scores[
            (repeat_scores["experiment"] == experiment)
            & (repeat_scores["epoch"] == final_epoch)
        ]
        variant_order = [
            name
            for name in config.EXPERIMENTS[experiment]
            if name.name in set(final_values["variant"])
        ]
        labels = [item.name for item in variant_order]
        values = [
            final_values.loc[
                final_values["variant"] == label, "accuracy"
            ].to_numpy()
            for label in labels
        ]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.boxplot(values, tick_labels=labels)
        ax.set(
            title=f"{experiment}: AccuracyCV po {final_epoch} epokach",
            xlabel="Wariant",
            ylabel="AccuracyCV",
            ylim=(0.0, 1.0),
        )
        ax.set_yticks(np.arange(0.0, 1.01, 0.1))
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        if save_plots:
            fig.savefig(
                plot_dir / f"{experiment}_accuracy_boxplot.png", dpi=160
            )

    if show_plots:
        print(
            "\nWyświetlam wykresy. Zamknij okna wykresów, aby zakończyć program."
        )
        plt.show()

    plt.close("all")


def save_split_indices(
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_splits: int,
    output_path: Path,
) -> None:
    """Zapisuje indeksy użytych podziałów do pełnej reprodukcji badania."""
    payload = [
        {
            "split_id": split_id,
            "repeat": split_id // n_splits,
            "fold": split_id % n_splits,
            "train_indices": train_idx.tolist(),
            "validation_indices": val_idx.tolist(),
        }
        for split_id, (train_idx, val_idx) in enumerate(splits)
    ]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def evaluate_selected_model_once(
    summary: pd.DataFrame,
    experiment_variants: dict[str, tuple[config.Variant, ...]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    epochs: int,
    selection_metric: str,
) -> dict[str, Any]:
    """Wybiera konfigurację wyłącznie z CV i jeden raz ocenia ją na teście."""
    if selection_metric not in METRICS:
        raise ValueError(f"Nieznana metryka wyboru: {selection_metric}")

    final_epoch = int(summary["epoch"].max())
    finalists = summary[summary["epoch"] == final_epoch]
    best = finalists.loc[finalists[f"{selection_metric}_mean"].idxmax()]
    experiment = str(best["experiment"])
    variant_name = str(best["variant"])
    variant = next(
        item
        for item in experiment_variants[experiment]
        if item.name == variant_name
    )
    params = params_for_variant(variant)

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(X_train)
    scaled_test = scaler.transform(X_test)
    model = MLPClassifier(**params, random_state=config.MODEL_SEED_BASE)
    classes = np.unique(y_train)
    for epoch in range(epochs):
        if epoch == 0:
            model.partial_fit(scaled_train, y_train, classes=classes)
        else:
            model.partial_fit(scaled_train, y_train)

    prediction = model.predict(scaled_test)
    return {
        "selected_experiment": experiment,
        "selected_variant": variant_name,
        "selection_metric": selection_metric,
        "cv_score": float(best[f"{selection_metric}_mean"]),
        "epochs": epochs,
        "parameters": params,
        "test_accuracy": float(accuracy_score(y_test, prediction)),
        "test_balanced_accuracy": float(
            balanced_accuracy_score(y_test, prediction)
        ),
        "test_macro_f1": float(
            f1_score(y_test, prediction, average="macro", zero_division=0)
        ),
        "test_rows": int(len(y_test)),
    }


def serializable_experiments(
    selected: dict[str, tuple[config.Variant, ...]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        experiment: [asdict(variant) for variant in variants]
        for experiment, variants in selected.items()
    }

