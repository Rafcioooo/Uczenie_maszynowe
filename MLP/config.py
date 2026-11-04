"""Centralna konfiguracja laboratorium MLP.

W typowym użyciu wystarczy edytować ten plik. Parametry podane w wierszu
poleceń (zob. ``python main.py --help``) tymczasowo nadpisują część ustawień.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Dane i odtwarzalność
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent / "G2_S10__a09_spambase__v03.csv"
TARGET_COLUMN = "target"
OUTPUT_DIR = Path("results")

# Usunięcie duplikatów zmieni liczbę rekordów z 1533 na 1451.
# False zachowuje oryginalny zbiór otrzymany do laboratorium.
DROP_DUPLICATES = False

RANDOM_SEED = 42
MODEL_SEED_BASE = 10_000


# ---------------------------------------------------------------------------
# Protokół oceny
# ---------------------------------------------------------------------------
CV_SPLITS = 5
CV_REPEATS = 10
EPOCHS = 50

# Domyślnie wykresy pojawiają się w oknach/Plot Viewerze po obliczeniach.
# Zapis PNG można niezależnie włączyć, np. na potrzeby raportu.
SHOW_PLOTS = True
SAVE_PLOTS = False

# Po ustawieniu True test zostanie odłożony PRZED walidacją krzyżową.
# Po eksperymentach najlepsza z faktycznie sprawdzonych konfiguracji zostanie
# ponownie nauczona na całym train i oceniona na teście dokładnie raz.
USE_INDEPENDENT_TEST = False
TEST_SIZE = 0.20

# Metryka wybierająca najlepszy model. Dostępne: accuracy,
# balanced_accuracy, macro_f1.
SELECTION_METRIC = "accuracy"


# ---------------------------------------------------------------------------
# Parametry wspólne dla wszystkich sieci
# ---------------------------------------------------------------------------
BASE_MLP_PARAMS: dict[str, Any] = {
    "hidden_layer_sizes": (20, 10),
    "activation": "relu",
    "solver": "adam",
    "batch_size": "auto",
    "learning_rate_init": 0.001,
    "alpha": 0.0001,
    "shuffle": True,
    # Parametry istotne dla SGD; inne solvery je ignorują.
    "momentum": 0.9,
    "nesterovs_momentum": True,
}


@dataclass(frozen=True)
class Variant:
    """Jeden wariant; ``changes`` nadpisuje parametry bazowe."""

    name: str
    changes: dict[str, Any] = field(default_factory=dict)


# Każdy eksperyment zmienia tylko jeden element konfiguracji.
# Nazwy z tego słownika można przekazać przez --experiments.
EXPERIMENTS: dict[str, tuple[Variant, ...]] = {
    "activations": (
        Variant("identity", {"activation": "identity"}),
        Variant("logistic", {"activation": "logistic"}),
        Variant("tanh", {"activation": "tanh"}),
        Variant("relu", {"activation": "relu"}),
    ),
    "solvers": (
        Variant("adam", {"solver": "adam"}),
        Variant(
            "sgd_momentum",
            {
                "solver": "sgd",
                "momentum": 0.9,
                "nesterovs_momentum": True,
            },
        ),
        Variant(
            "sgd_plain",
            {
                "solver": "sgd",
                "momentum": 0.0,
                "nesterovs_momentum": False,
            },
        ),
    ),
    "architectures": (
        Variant("(10,)", {"hidden_layer_sizes": (10,)}),
        Variant("(20,)", {"hidden_layer_sizes": (20,)}),
        Variant("(20, 10)", {"hidden_layer_sizes": (20, 10)}),
        Variant("(20, 20)", {"hidden_layer_sizes": (20, 20)}),
    ),
}

