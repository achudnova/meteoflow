import pandas as pd
import sys
import config
from rich.console import Console


def split_data(data_featured: pd.DataFrame, console: Console):
    print("Definiere Feature- und Zielspalten...")

    features_cols = [
        col
        for col in data_featured.columns
        if col not in config.TARGET_COLUMNS + config.ORIGINAL_TARGET_BASE_COLUMNS
    ]
    target_cols_present = [
        col for col in config.TARGET_COLUMNS if col in data_featured.columns
    ]

    if not target_cols_present:
        console.print(
            "[red] Fehler: keine der Zielvariablen konnte erstellt werden. [/red]"
        )
        sys.exit(1)
    if not features_cols:
        console.print("[red] Fehler: keine Feature-Spalten gefunden. [/red]")
        sys.exit(1)
    console.print(f"   Gefundene Features: {len(features_cols)}")
    console.print(f"   Gefundene Targets: {target_cols_present}")

    X = data_featured[features_cols]
    y = data_featured[target_cols_present]

    total_samples = len(data_featured)
    test_days = config.TEST_PERIOD_DAYS

    console.print(
        f"\nFühre chronologischen Split durch (Testset-Größe: {test_days} Tage)..."
    )

    if total_samples <= test_days:
        console.print(
            f"[red] Nicht genügend Daten ({len(data_featured)} Zeilen) für einen sinnvollen Train/Test-Split mit {config.TEST_PERIOD_DAYS} Testtagen vorhanden. [/red]"
        )
        sys.exit(1)

    split_index = total_samples - test_days

    try:
        split_date = data_featured.index[split_index]
    except IndexError:
        console.print(
            f"[red] Fehler: Ungültiger Split-Index {split_index} (Gesamt: {total_samples}, Testtage: {test_days}).[/red]"
        )
        sys.exit(1)

    # Daten aufteilen
    X_train = X[X.index < split_date]
    X_test = X[X.index >= split_date]
    y_train = y[y.index < split_date]
    y_test = y[y.index >= split_date]

    # Überprüfung des Split-Verhältnisses
    total_samples_after_engineering = X_train.shape[0] + X_test.shape[0]

    train_percentage = (X_train.shape[0] / total_samples_after_engineering) * 100
    test_percentage = (X_test.shape[0] / total_samples_after_engineering) * 100
    
    console.print(f"   Trainingsdaten: {X_train.shape[0]} Samples ({X_train.index.min().date()} bis {X_train.index.max().date()})")
    console.print(f"   Testdaten: {X_test.shape[0]} Samples ({X_test.index.min().date()} bis {X_test.index.max().date()})")
    console.print(f"   Anzahl Features: {X_train.shape[1]}")
    console.print(f"   Zielvariablen: {target_cols_present}")
    console.print(f"\n   Überprüfung des Split-Verhältnisses:")
    console.print(f"     Trainings-Anteil: [bold magenta]{train_percentage:.2f}%[/bold magenta]")
    console.print(f"     Test-Anteil:      [bold magenta]{test_percentage:.2f}%[/bold magenta]")
    console.print("[green]   ✔️ Train/Test Split abgeschlossen.[/green]")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        features_cols,
        target_cols_present,
        split_date,
        train_percentage,
        test_percentage,
    )
