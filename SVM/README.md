# Laboratorium 2 – algorytm SVM (zbiór Spambase)

Badanie zdolności uogólniania klasyfikatora SVM (`sklearn.svm.SVC`) na zbiorze
`data/spambase.csv` (1533 wiadomości, 56 cech liczbowych, klasa `target`: 0 = nie-spam, 1 = spam)
z użyciem powtarzanej, stratyfikowanej k-krotnej walidacji krzyżowej (domyślnie k = 5, n = 10).
Wszystkie konfiguracje są oceniane na tych samych 50 podziałach, a standaryzacja jest dopasowywana
w każdym foldzie wyłącznie na części uczącej.

Skrypt wykonuje następujące eksperymenty:

0. wpływ standaryzacji cech,
1. porównanie jąder: liniowe, wielomianowe (degree 2/3/4, gamma = 0.05, coef0 = 1) i rbf przy C = 1, γ = 0.05,
2. wpływ C przy stałym γ = 0.05 (rbf),
3. wpływ γ przy stałym C = 1 (rbf),
4. (opcjonalnie) siatkę (C, γ),

a następnie wybiera model i, z flagą `--test-size`, ocenia go jednokrotnie na niezależnym zbiorze testowym.

## Uruchomienie

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python svm_lab.py                    # Spambase, walidacja krzyżowa na całym zbiorze
python svm_lab.py --test-size 0.2    # + zadanie dodatkowe: zbiór testowy 20%
python svm_lab.py --csv inne.csv --target klasa   # inny zbiór
```

Pełne uruchomienie trwa ok. 3 minuty (`--no-grid` pomija siatkę i skraca czas).
Wyniki (wykresy PNG oraz tabele CSV) trafiają do katalogu `wyniki/`.
