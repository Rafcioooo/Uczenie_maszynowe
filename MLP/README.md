# Laboratorium MLP — klasyfikacja Spambase

Kod realizuje porównanie funkcji aktywacji, solverów i architektur sieci MLP
przy użyciu powtarzanej, stratyfikowanej walidacji krzyżowej.

## Uruchomienie w Visual Studio Code

1. Otwórz ten katalog w VS Code.
2. Otwórz terminal (`Terminal` → `New Terminal`).
3. Utwórz i aktywuj środowisko:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   Linux/macOS:

   ```bash
   source .venv/bin/activate
   ```

4. Zainstaluj biblioteki:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Wybierz interpreter `.venv` w VS Code i uruchom:

   ```bash
   python main.py
   ```

## Sterowanie eksperymentami

Najważniejsze wartości są opisane w `config.py`:

- `CV_SPLITS`, `CV_REPEATS`, `EPOCHS` — protokół walidacji;
- `DROP_DUPLICATES` — sposób obsługi duplikatów;
- `USE_INDEPENDENT_TEST`, `TEST_SIZE` — opcjonalny test końcowy;
- `SHOW_PLOTS`, `SAVE_PLOTS` — wyświetlanie i opcjonalny zapis wykresów;
- `BASE_MLP_PARAMS` — parametry niezmieniane w eksperymentach;
- `EXPERIMENTS` — warianty aktywacji, solverów i architektur.

W dostarczonym zbiorze znaleziono 82 nadmiarowe duplikaty. Domyślne
`DROP_DUPLICATES = False` zachowuje wszystkie 1533 rekordy, ale kopie mogą
trafić do różnych foldów i zawyżyć ocenę. Należy świadomie wybrać wariant i
opisać go w raporcie.

Parametry jednego uruchomienia można zmienić bez edycji pliku:

```bash
# Krótki test, czy środowisko działa
python main.py --epochs 2 --splits 2 --repeats 1

# Tylko funkcje aktywacji, 100 epok
python main.py --experiments activations --epochs 100

# Solvery i architektury
python main.py --experiments solvers architectures

# Usunięcie duplikatów
python main.py --drop-duplicates

# Odłożenie testu przed CV i jednorazowa ocena wybranego modelu
python main.py --independent-test

# Bez okien wykresów, ale z zapisem PNG
python main.py --no-show-plots --save-plots
```

Pełna lista:

```bash
python main.py --help
```

## Organizacja obliczeń

- Wszystkie warianty dostają te same, wygenerowane jeden raz podziały.
- Na każdym splicie warianty otrzymują to samo ziarno inicjalizacji modelu.
- `StandardScaler` jest dopasowywany wyłącznie na części treningowej foldu.
- Dla każdej konfiguracji i każdego foldu powstaje nowy model.
- Jedno wywołanie `partial_fit` odpowiada jednej epoce.
- Wyniki foldów są najpierw uśredniane w obrębie powtórzenia. Dopiero z
  wartości uzyskanych w kolejnych powtórzeniach obliczane są średnia i
  odchylenie standardowe.

## Wyniki

Katalog `results/` zawiera:

- `fold_epoch_results.csv` — surowe metryki każdego foldu i epoki;
- `repeat_epoch_scores.csv` — średnie z foldów dla każdego powtórzenia;
- `summary_by_epoch.csv` — średnia i SD z powtórzeń;
- `final_summary.csv` — tabela dla ostatniej epoki;
- `split_indices.json` — użyte indeksy train/validation;
- `run_metadata.json` — pełna konfiguracja uruchomienia;
- `plots/` — opcjonalne kopie PNG, gdy włączono `--save-plots`;
- `independent_test_result.json` — tylko przy włączonym teście niezależnym.

Domyślnie po zakończeniu obliczeń wykresy są wyświetlane przez Matplotlib
razem z uruchomionym programem i nie są zapisywane jako PNG. Program kończy
się po zamknięciu okien wykresów. W VS Code mogą one pojawić się w osobnych
oknach albo w Plot Viewerze, zależnie od konfiguracji interpretera.

Katalog jest nadpisywany przy następnym uruchomieniu z tą samą ścieżką.
Do zachowania kilku badań użyj np.:

```bash
python main.py --output-dir results/100_epok
```

## Uwaga o teście końcowym

Bez `--independent-test` wyniki służą do porównania konfiguracji, ale nie są
niezależnym końcowym testem wybranego modelu. Przy włączonej opcji test jest
odkładany przed CV, a kod używa go dopiero po wybraniu konfiguracji na
podstawie wyników walidacji.
