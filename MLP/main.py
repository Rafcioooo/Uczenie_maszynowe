"""Uruchomienie laboratorium MLP.

Najprościej:
    python main.py

Szybki test poprawności:
    python main.py --epochs 2 --splits 2 --repeats 1

Wybrany eksperyment:
    python main.py --experiments activations --epochs 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import config
from mlp_lab import (
    aggregate_results,
    evaluate_selected_model_once,
    load_data,
    make_splits,
    run_experiment,
    save_plots,
    save_split_indices,
    serializable_experiments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Powtarzana walidacja krzyżowa modeli MLP."
    )
    parser.add_argument("--data", type=Path, default=config.DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--splits", type=int, default=config.CV_SPLITS)
    parser.add_argument("--repeats", type=int, default=config.CV_REPEATS)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=tuple(config.EXPERIMENTS),
        default=list(config.EXPERIMENTS),
        help="Eksperymenty do wykonania (domyślnie wszystkie).",
    )
    parser.add_argument(
        "--drop-duplicates",
        action=argparse.BooleanOptionalAction,
        default=config.DROP_DUPLICATES,
        help="Usuń identyczne rekordy przed podziałem.",
    )
    parser.add_argument(
        "--independent-test",
        action=argparse.BooleanOptionalAction,
        default=config.USE_INDEPENDENT_TEST,
        help="Odłóż test i oceń na nim automatycznie wybrany model.",
    )
    parser.add_argument(
        "--show-plots",
        action=argparse.BooleanOptionalAction,
        default=config.SHOW_PLOTS,
        help="Wyświetl wykresy podczas uruchomienia.",
    )
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=config.SAVE_PLOTS,
        help="Zapisz dodatkowo kopie wykresów jako pliki PNG.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs musi być dodatnie.")
    if args.splits < 2:
        raise ValueError("--splits musi wynosić co najmniej 2.")
    if args.repeats < 1:
        raise ValueError("--repeats musi być dodatnie.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X, y, data_info = load_data(
        args.data, config.TARGET_COLUMN, args.drop_duplicates
    )
    print("Dane:", json.dumps(data_info, ensure_ascii=False))
    if data_info["duplicates_found"] and not args.drop_duplicates:
        print(
            "UWAGA: zachowano duplikaty. Ich kopie mogą znaleźć się po obu "
            "stronach podziału i zawyżyć wynik; opisz tę decyzję w raporcie."
        )

    X_test = y_test = None
    if args.independent_test:
        X, X_test, y, y_test = train_test_split(
            X,
            y,
            test_size=config.TEST_SIZE,
            stratify=y,
            random_state=config.RANDOM_SEED,
        )
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        print(
            f"Odłożono niezależny test: train={len(y)}, test={len(y_test)}."
        )

    selected_experiments = {
        name: config.EXPERIMENTS[name] for name in args.experiments
    }
    splits = make_splits(
        X,
        y,
        n_splits=args.splits,
        n_repeats=args.repeats,
        random_seed=config.RANDOM_SEED,
    )
    save_split_indices(
        splits, args.splits, args.output_dir / "split_indices.json"
    )

    result_parts: list[pd.DataFrame] = []
    for experiment_name, variants in selected_experiments.items():
        part = run_experiment(
            experiment_name=experiment_name,
            variants=variants,
            X=X,
            y=y,
            splits=splits,
            n_splits=args.splits,
            epochs=args.epochs,
            model_seed_base=config.MODEL_SEED_BASE,
        )
        result_parts.append(part)
        # Checkpoint po każdym eksperymencie.
        pd.concat(result_parts, ignore_index=True).to_csv(
            args.output_dir / "fold_epoch_results.csv", index=False
        )

    results = pd.concat(result_parts, ignore_index=True)
    repeat_scores, summary = aggregate_results(results)
    repeat_scores.to_csv(
        args.output_dir / "repeat_epoch_scores.csv", index=False
    )
    summary.to_csv(args.output_dir / "summary_by_epoch.csv", index=False)

    final_summary = summary[
        summary["epoch"] == args.epochs
    ].sort_values(["experiment", "accuracy_mean"], ascending=[True, False])
    final_summary.to_csv(
        args.output_dir / "final_summary.csv", index=False
    )
    save_plots(
        repeat_scores,
        summary,
        args.output_dir,
        show_plots=args.show_plots,
        save_plots=args.save_plots,
    )

    metadata = {
        "data": data_info,
        "cv": {
            "splits": args.splits,
            "repeats": args.repeats,
            "total_splits": len(splits),
            "epochs": args.epochs,
            "random_seed": config.RANDOM_SEED,
            "model_seed_base": config.MODEL_SEED_BASE,
        },
        "standardization": "StandardScaler fitted separately on each train fold",
        "independent_test": args.independent_test,
        "test_size": config.TEST_SIZE if args.independent_test else None,
        "show_plots": args.show_plots,
        "save_plots": args.save_plots,
        "base_mlp_params": config.BASE_MLP_PARAMS,
        "experiments": serializable_experiments(selected_experiments),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.independent_test:
        assert X_test is not None and y_test is not None
        test_result = evaluate_selected_model_once(
            summary=summary,
            experiment_variants=selected_experiments,
            X_train=X,
            y_train=y,
            X_test=X_test,
            y_test=y_test,
            epochs=args.epochs,
            selection_metric=config.SELECTION_METRIC,
        )
        (args.output_dir / "independent_test_result.json").write_text(
            json.dumps(test_result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("\nTest końcowy:", json.dumps(test_result, ensure_ascii=False))

    print("\nWyniki końcowej epoki:")
    print(final_summary.to_string(index=False))
    print(f"\nPliki zapisano w: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

