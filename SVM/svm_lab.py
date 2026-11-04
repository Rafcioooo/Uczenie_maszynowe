"""Laboratorium 2: Algorytm wektorów wspierających (SVM).

Badanie zdolności uogólniania klasyfikatora SVC (scikit-learn) za pomocą
powtarzanej, stratyfikowanej k-krotnej walidacji krzyżowej (n x k).

Uruchomienie:
    python svm_lab.py                         # zbiór Spambase w katalogu projektu
    python svm_lab.py --test-size 0.2         # + zadanie dodatkowe (zbiór testowy)
    python svm_lab.py --csv dane.csv --target klasa   # inny zbiór danych
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Parametry ustalone PRZED rozpoczęciem badań (podawane w raporcie)
# ---------------------------------------------------------------------------
K_FOLDS = 5
N_REPEATS = 10
RANDOM_STATE = 42
CACHE_SIZE = 200  # MB

# Eksperyment 1: porównanie jąder przy ustalonych parametrach
C_FIXED = 1.0
GAMMA_FIXED = 0.05
POLY_DEGREES = [2, 3, 4]
POLY_COEF0 = 1.0  # wymagane, aby SVC odpowiadało jądru (u.v + 1)^d z wykładu

# Eksperyment 2: wpływ C (jądro rbf, gamma = GAMMA_FIXED)
C_VALUES = [0.1, 1, 10, 100, 1000, 10000, 100000]

# Eksperyment 3: wpływ gamma (jądro rbf, C = C_FIXED)
GAMMA_VALUES = [0.001, 0.005, 0.01, 0.05, 0.2, 1, 2.5, 5, 10, 50, 100]

# Eksperyment 4 (opcjonalny): siatka (C, gamma) dla jądra rbf
GRID_C = [0.1, 1, 10, 100, 1000]
GRID_GAMMA = GAMMA_VALUES

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "wyniki"


def resolve_default_csv() -> Path:
    """Wybiera istniejący plik danych w projekcie.

    W repozytorium stosowany jest plik CSV w katalogu głównym, a nie w podkatalogu data/.
    """
    candidates = [
        BASE_DIR / "data" / "spambase.csv",
        BASE_DIR / "G2_S10__a09_spambase__v03.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_CSV = resolve_default_csv()
DEFAULT_TARGET = "target"
SPAMBASE_LABELS = {0: "nie-spam", 1: "spam"}


# ---------------------------------------------------------------------------
# Dane
# ---------------------------------------------------------------------------
def load_data(csv: str, target: str) -> tuple[pd.DataFrame, pd.Series, str]:
    df = pd.read_csv(csv)
    if target not in df.columns:
        raise SystemExit(f"Podaj poprawną kolumnę klasy --target (dostępne: {list(df.columns)})")
    df = df.dropna(subset=[target])
    y = df[target]
    if Path(csv).name == Path(DEFAULT_CSV).name:
        y = y.map(SPAMBASE_LABELS)
    return df.drop(columns=[target]), y, Path(csv).name


def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    """Wszystkie operacje zależne od danych (imputacja, kodowanie, skalowanie).

    Obiekt jest dopasowywany od nowa w każdym foldzie, wyłącznie na części uczącej.
    """
    num_steps: list = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), numeric)]
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def describe_data(X: pd.DataFrame, y: pd.Series, name: str, numeric, categorical) -> str:
    lines = [
        f"Zbiór danych: {name}",
        f"Liczba rekordów: {len(X)}",
        f"Liczba cech: {X.shape[1]} (ilościowe: {len(numeric)}, kategoryczne: {len(categorical)})",
        f"Liczba klas: {y.nunique()}",
        "Rozkład klas:",
    ]
    for cls, cnt in y.value_counts().sort_index().items():
        lines.append(f"  {cls}: {cnt} ({cnt / len(y):.1%})")
    lines.append(f"Braki danych: {int(X.isna().sum().sum())}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Konfiguracje i walidacja krzyżowa
# ---------------------------------------------------------------------------
@dataclass
class Config:
    name: str
    params: dict
    scale: bool = True


@dataclass
class CVResult:
    config: Config
    acc: np.ndarray  # (n, k)
    f1: np.ndarray  # (n, k)
    n_sv: np.ndarray  # (n, k)
    fit_time: np.ndarray  # (n, k)
    acc_cv: np.ndarray = field(init=False)  # (n,) średnia z k foldów w każdym powtórzeniu
    sv_cv: np.ndarray = field(init=False)

    def __post_init__(self):
        self.acc_cv = self.acc.mean(axis=1)
        self.sv_cv = self.n_sv.mean(axis=1)

    def summary(self) -> dict:
        return {
            "konfiguracja": self.config.name,
            **{k: v for k, v in self.config.params.items() if k != "cache_size"},
            "standaryzacja": self.config.scale,
            "AccCV_srednia": self.acc_cv.mean(),
            "AccCV_std": self.acc_cv.std(ddof=1),
            "AccCV_min": self.acc_cv.min(),
            "AccCV_max": self.acc_cv.max(),
            "F1_macro": self.f1.mean(axis=1).mean(),
            "nSV_srednia": self.sv_cv.mean(),
            "nSV_std": self.sv_cv.std(ddof=1),
            "czas_uczenia_ms": 1000 * self.fit_time.mean(),
        }


class FoldCache:
    """Przetworzone dane dla każdego z n*k podziałów.

    Indeksy podziałów generowane są raz; dla danego sposobu przetwarzania
    (ze standaryzacją / bez) dane foldu przygotowywane są tylko raz i
    współdzielone przez wszystkie konfiguracje.
    """

    def __init__(self, X: pd.DataFrame, y: np.ndarray, splits, numeric, categorical):
        self.X, self.y, self.splits = X, y, splits
        self.numeric, self.categorical = numeric, categorical
        self._cache: dict[bool, list] = {}

    def folds(self, scale: bool):
        if scale not in self._cache:
            prepared = []
            for train_idx, val_idx in self.splits:
                X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                pre = make_preprocessor(self.numeric, self.categorical, scale)
                pre.fit(X_train)
                prepared.append(
                    (pre.transform(X_train), pre.transform(X_val), self.y[train_idx], self.y[val_idx])
                )
            self._cache[scale] = prepared
        return self._cache[scale]


def evaluate(config: Config, cache: FoldCache, k: int, n: int) -> CVResult:
    acc, f1, n_sv, times = (np.empty(n * k) for _ in range(4))
    for i, (X_tr, X_val, y_tr, y_val) in enumerate(cache.folds(config.scale)):
        model = SVC(cache_size=CACHE_SIZE, **config.params)  # nowy model dla każdego podziału
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_tr, y_tr)
        times[i] = time.perf_counter() - t0
        y_pred = model.predict(X_val)
        acc[i] = accuracy_score(y_val, y_pred)
        f1[i] = f1_score(y_val, y_pred, average="macro")
        n_sv[i] = len(model.support_)
    # RepeatedStratifiedKFold zwraca podziały kolejno: powtórzenie 1 (k foldów), powtórzenie 2, ...
    shape = (n, k)
    return CVResult(config, acc.reshape(shape), f1.reshape(shape), n_sv.reshape(shape), times.reshape(shape))


def run_experiment(configs: list[Config], cache: FoldCache, k: int, n: int, title: str) -> list[CVResult]:
    print(f"\n=== {title} ===")
    results = []
    for cfg in configs:
        r = evaluate(cfg, cache, k, n)
        results.append(r)
        print(
            f"  {cfg.name:<28s} AccCV = {r.acc_cv.mean():.4f} ± {r.acc_cv.std(ddof=1):.4f}   "
            f"nSV = {r.sv_cv.mean():7.1f}"
        )
    return results


# ---------------------------------------------------------------------------
# Wykresy
# ---------------------------------------------------------------------------
def plot_experiment(results: list[CVResult], title: str, xlabel: str, fname: str,
                    x_values=None, log_x: bool = False) -> None:
    labels = [r.config.name for r in results]
    x = np.arange(len(results)) if x_values is None else np.asarray(x_values, dtype=float)
    acc_mean = [r.acc_cv.mean() for r in results]
    acc_std = [r.acc_cv.std(ddof=1) for r in results]
    sv_mean = [r.sv_cv.mean() for r in results]
    sv_std = [r.sv_cv.std(ddof=1) for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=13)

    ax = axes[0]
    ax.errorbar(x, acc_mean, yerr=acc_std, marker="o", capsize=4)
    ax.set_title("Średnia AccuracyCV ± odchylenie std (z n powtórzeń)")
    ax.set_ylabel("AccuracyCV")

    ax = axes[1]
    ax.boxplot([r.acc_cv for r in results], positions=np.arange(len(results)))
    ax.set_xticks(np.arange(len(results)), labels, rotation=45, ha="right")
    ax.set_title("Boxploty AccuracyCV (n wartości na wariant)")
    ax.set_ylabel("AccuracyCV")

    ax = axes[2]
    ax.errorbar(x, sv_mean, yerr=sv_std, marker="s", color="tab:red", capsize=4)
    ax.set_title("Liczba wektorów wspierających (średnia z k foldów)")
    ax.set_ylabel("liczba SV")

    for ax in (axes[0], axes[2]):
        if log_x:
            ax.set_xscale("log")
        else:
            ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_xlabel(xlabel)
    axes[1].set_xlabel(xlabel)
    for ax in axes:
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=130)
    plt.close(fig)


def plot_grid(grid: dict[tuple[float, float], CVResult]) -> None:
    acc = np.array([[grid[(c, g)].acc_cv.mean() for g in GRID_GAMMA] for c in GRID_C])
    std = np.array([[grid[(c, g)].acc_cv.std(ddof=1) for g in GRID_GAMMA] for c in GRID_C])

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    ax = axes[0]
    for i, c in enumerate(GRID_C):
        ax.errorbar(GRID_GAMMA, acc[i], yerr=std[i], marker="o", capsize=3, label=f"C = {c:g}")
    ax.set_xscale("log")
    ax.set_xlabel("γ")
    ax.set_ylabel("AccuracyCV")
    ax.set_title("AccuracyCV = f(γ) dla kilku wartości C (jądro rbf)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    im = ax.imshow(acc, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(GRID_GAMMA)), [f"{g:g}" for g in GRID_GAMMA])
    ax.set_yticks(range(len(GRID_C)), [f"{c:g}" for c in GRID_C])
    ax.set_xlabel("γ")
    ax.set_ylabel("C")
    ax.set_title("Mapa średniej AccuracyCV dla siatki (C, γ)")
    for i in range(len(GRID_C)):
        for j in range(len(GRID_GAMMA)):
            ax.text(j, i, f"{acc[i, j]:.3f}", ha="center", va="center",
                    color="white" if acc[i, j] < acc.max() - 0.1 else "black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "exp4_siatka_C_gamma.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Program główny
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Laboratorium SVM: badanie zdolności uogólniania SVC")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="ścieżka do pliku CSV (domyślnie: plik Spambase w katalogu projektu)")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="nazwa kolumny z etykietą klasy")
    parser.add_argument("--k", type=int, default=K_FOLDS, help="liczba foldów k")
    parser.add_argument("--n", type=int, default=N_REPEATS, help="liczba powtórzeń n")
    parser.add_argument("--test-size", type=float, default=0.0,
                        help="część danych na niezależny zbiór testowy, np. 0.2 (zadanie dodatkowe)")
    parser.add_argument("--no-grid", action="store_true", help="pomiń siatkę (C, γ)")
    args = parser.parse_args()
    k, n = args.k, args.n

    OUT_DIR.mkdir(exist_ok=True)
    X, y_raw, name = load_data(args.csv, args.target)
    numeric, categorical = split_feature_types(X)
    y = y_raw.astype(str).to_numpy()

    print(describe_data(X, y_raw, name, numeric, categorical))
    if categorical:
        print(f"Cechy kategoryczne (kodowanie one-hot): {categorical}")

    X_test = y_test = None
    if args.test_size > 0:
        X, X_test, y, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE
        )
        X, X_test = X.reset_index(drop=True), X_test.reset_index(drop=True)
        print(f"\nPodział hold-out: uczący {len(X)} / testowy {len(X_test)} rekordów "
              f"(walidacja krzyżowa wyłącznie na zbiorze uczącym)")

    # Indeksy wszystkich n x k podziałów generowane są jeden raz
    cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=n, random_state=RANDOM_STATE)
    splits = list(cv.split(X, y))
    cache = FoldCache(X, y, splits, numeric, categorical)
    print(f"\nProtokół: RepeatedStratifiedKFold, k = {k}, n = {n}, random_state = {RANDOM_STATE} "
          f"-> {len(splits)} podziałów wspólnych dla wszystkich konfiguracji")

    all_results: list[tuple[str, CVResult]] = []

    # --- Eksperyment 0: wpływ standaryzacji -------------------------------------
    exp0 = run_experiment(
        [
            Config("rbf bez standaryzacji", {"kernel": "rbf", "C": C_FIXED, "gamma": GAMMA_FIXED}, scale=False),
            Config("rbf ze standaryzacją", {"kernel": "rbf", "C": C_FIXED, "gamma": GAMMA_FIXED}, scale=True),
            # Bez skalowania wartości jądra wielomianowego są rzędu 1e18 i solver
            # się nie zbiega - stąd limit iteracji (ostrzeżenia są wyciszone).
            Config("poly d=3 bez standaryzacji",
                   {"kernel": "poly", "C": C_FIXED, "degree": 3, "gamma": GAMMA_FIXED, "coef0": POLY_COEF0,
                    "max_iter": 100_000},
                   scale=False),
            Config("poly d=3 ze standaryzacją",
                   {"kernel": "poly", "C": C_FIXED, "degree": 3, "gamma": GAMMA_FIXED, "coef0": POLY_COEF0},
                   scale=True),
        ],
        cache, k, n,
        f"Eksperyment 0: wpływ standaryzacji (C = {C_FIXED}, γ = {GAMMA_FIXED})",
    )
    plot_experiment(exp0, f"Wpływ standaryzacji cech (C = {C_FIXED:g}, γ = {GAMMA_FIXED:g})",
                    "wariant", "exp0_standaryzacja.png")
    all_results += [("0: standaryzacja", r) for r in exp0]

    # --- Eksperyment 1: jądro wielomianowe vs gaussowskie ----------------------
    exp1_cfg = [Config("linear", {"kernel": "linear", "C": C_FIXED})]
    exp1_cfg += [
        Config(f"poly d={d}", {"kernel": "poly", "C": C_FIXED, "degree": d,
                               "gamma": GAMMA_FIXED, "coef0": POLY_COEF0})
        for d in POLY_DEGREES
    ]
    exp1_cfg.append(Config("rbf", {"kernel": "rbf", "C": C_FIXED, "gamma": GAMMA_FIXED}))
    exp1 = run_experiment(
        exp1_cfg, cache, k, n,
        f"Eksperyment 1: porównanie jąder (C = {C_FIXED}, γ = {GAMMA_FIXED}, coef0 = {POLY_COEF0})",
    )
    plot_experiment(exp1, f"Porównanie jąder: C = {C_FIXED:g}, γ = {GAMMA_FIXED:g}, "
                          f"poly: coef0 = {POLY_COEF0:g}, degree ∈ {POLY_DEGREES}",
                    "jądro", "exp1_jadra.png")
    all_results += [("1: jądra", r) for r in exp1]

    # --- Eksperyment 2: wpływ C -------------------------------------------------
    exp2 = run_experiment(
        [Config(f"rbf C={c:g}", {"kernel": "rbf", "C": c, "gamma": GAMMA_FIXED}) for c in C_VALUES],
        cache, k, n, f"Eksperyment 2: wpływ C (rbf, γ = {GAMMA_FIXED} stałe)",
    )
    plot_experiment(exp2, f"Wpływ parametru C (jądro rbf, γ = {GAMMA_FIXED:g} stałe)",
                    "C", "exp2_wplyw_C.png", x_values=C_VALUES, log_x=True)
    all_results += [("2: wpływ C", r) for r in exp2]

    # --- Eksperyment 3: wpływ gamma ---------------------------------------------
    exp3 = run_experiment(
        [Config(f"rbf γ={g:g}", {"kernel": "rbf", "C": C_FIXED, "gamma": g}) for g in GAMMA_VALUES],
        cache, k, n, f"Eksperyment 3: wpływ γ (rbf, C = {C_FIXED} stałe)",
    )
    plot_experiment(exp3, f"Wpływ parametru γ (jądro rbf, C = {C_FIXED:g} stałe)",
                    "γ", "exp3_wplyw_gamma.png", x_values=GAMMA_VALUES, log_x=True)
    all_results += [("3: wpływ γ", r) for r in exp3]

    # --- Eksperyment 4 (opcjonalny): siatka (C, gamma) --------------------------
    if not args.no_grid:
        grid_cfg = [Config(f"rbf C={c:g} γ={g:g}", {"kernel": "rbf", "C": c, "gamma": g})
                    for c in GRID_C for g in GRID_GAMMA]
        print("\n=== Eksperyment 4: siatka (C, γ) dla jądra rbf ===")
        grid = {}
        for cfg in grid_cfg:
            grid[(cfg.params["C"], cfg.params["gamma"])] = evaluate(cfg, cache, k, n)
        plot_grid(grid)
        all_results += [("4: siatka", r) for r in grid.values()]
        best_grid = max(grid.values(), key=lambda r: r.acc_cv.mean())
        print(f"  najlepsza para: {best_grid.config.name}  AccCV = {best_grid.acc_cv.mean():.4f}")

    # --- Tabela podsumowująca ---------------------------------------------------
    table = pd.DataFrame([{"eksperyment": e, **r.summary()} for e, r in all_results])
    table.to_csv(OUT_DIR / "tabela_wynikow.csv", index=False, float_format="%.4f")
    per_repeat = pd.DataFrame(
        {f"{e} | {r.config.name}": r.acc_cv for e, r in all_results},
        index=pd.Index(range(1, n + 1), name="powtórzenie"),
    )
    per_repeat.to_csv(OUT_DIR / "accuracy_cv_powtorzenia.csv", float_format="%.4f")

    # --- Wybór modelu ------------------------------------------------------------
    # Kryterium: najwyższa średnia AccuracyCV; przy remisie (różnica < 1e-4)
    # mniejsza liczba wektorów wspierających (prostszy model).
    candidates = [r for _, r in all_results if r.config.scale]
    best_acc = max(r.acc_cv.mean() for r in candidates)
    best = min((r for r in candidates if r.acc_cv.mean() >= best_acc - 1e-4), key=lambda r: r.sv_cv.mean())
    print("\n=== Wybór modelu ===")
    print(f"  {best.config.name}: {best.config.params}")
    print(f"  AccCV = {best.acc_cv.mean():.4f} ± {best.acc_cv.std(ddof=1):.4f}, nSV = {best.sv_cv.mean():.1f}")

    # --- Zadanie dodatkowe: niezależny zbiór testowy -----------------------------
    if X_test is not None:
        pre = make_preprocessor(numeric, categorical, best.config.scale)
        pre.fit(X)
        model = SVC(cache_size=CACHE_SIZE, **best.config.params)
        model.fit(pre.transform(X), y)
        y_pred = model.predict(pre.transform(X_test))
        print("\n=== Ocena końcowa na niezależnym zbiorze testowym (jednokrotna) ===")
        print(f"  Accuracy = {accuracy_score(y_test, y_pred):.4f}, nSV = {len(model.support_)}")
        print(classification_report(y_test, y_pred, digits=4))
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        print("Macierz pomyłek (wiersze: klasa rzeczywista, kolumny: przewidziana):")
        print(pd.DataFrame(cm, index=model.classes_, columns=model.classes_))
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot(ax=ax, colorbar=False)
        ax.set_title(f"Zbiór testowy: {best.config.name}")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "macierz_pomylek_test.png", dpi=130)
        plt.close(fig)

    print(f"\nWykresy i tabele zapisano w katalogu: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
