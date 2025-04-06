import os
import sys
import config
import pandas as pd

from data_collection import get_weather_data
from eda import start_eda
from data_preprocessing import preprocess_data
from feature_engineering import engineer_feautures
from model_training import train_models
from model_evaluation import evaluate_model
from prediction import predict_next_day
# from .model_manager import load_model

from rich.console import Console
from rich.panel import Panel

console = Console()

def main():
    console.rule("[bold blue]⛅ Wettervorhersage für Berlin ⛅[/bold blue]")
    
    # ----- 1. Datenerfassung -----
    console.rule("\n[cyan]1. Datenerfassung[/cyan]")
    try:
        data = get_weather_data(
            location=config.LOCATION,
            start_date=config.START_DATE,
            end_date=config.END_DATE,
            required_columns=config.REQUIRED_COLUMNS,
            essential_columns=config.ESSENTIAL_COLS,
        )
    except Exception as e:
        console.print(f"[red] Ein Fehler ist aufgetreten: [/red] {e}")
        sys.exit(1)

    # ----- 2. Explorative Datenanalyse (EDA) -----
    console.rule("[cyan]1.5 Explorative Datenanalyse (EDA)[/cyan]")
    start_eda(data, plot_columns=config.EDA_PLOT_COLUMNS, save_dir=config.EDA_PLOT_DIR)

    # ----- 3. Datenvorverarbeitung -----
    console.rule("[cyan]2. Datenvorverarbeitung[/cyan]")
    preprocess_data(data)

    # ----- 4. Feature Engineering -----
    console.rule("[cyan]3. Feature Engineering[/cyan]")
    data_featured = engineer_feautures(
        data=data,
        target_cols=config.TARGET_COLUMNS,
        target_base_cols=config.ORIGINAL_TARGET_BASE_COLUMNS,
        lag_days=config.LAG_DAYS,
    )

    if data is None or data.empty:
        console.print(f"[red] Nach dem Feature Engineering sind keine Daten mehr verfügbar [/red]")
        sys.exit(1)

    # ----- 5. Train/Test Split -----
    console.rule("[cyan]4. Train/Test Split[/cyan]")
    # Feature Spalten definieren
    features_cols = [
        col
        for col in data_featured.columns
        if col not in config.TARGET_COLUMNS + config.ORIGINAL_TARGET_BASE_COLUMNS
    ]
    target_cols_present = [
        col for col in config.TARGET_COLUMNS if col in data_featured.columns
    ]  # Nur die tatsächlich erstellten Targets

    if not target_cols_present:
        console.print("[red] Fehler: keine der Zielvariablen konnte erstellt werden. [/red]")
        sys.exit(1)
    if not features_cols:
        print("[red] Fehler: keine Feature-Spalten gefunden. [/red]")
        sys.exit(1)

    X = data_featured[features_cols]
    y = data_featured[target_cols_present]  # nur existierende Targets verwenden

    if len(data_featured) <= config.TEST_PERIOD_DAYS:
        print(
            f"[red] Nicht genügend Daten ({len(data_featured)} Zeilen) für einen sinnvollen Train/Test-Split mit {config.TEST_PERIOD_DAYS} Testtagen vorhanden. [/red]"
        )
        print("Workflow wird abgebrochen.")
        sys.exit(1)

    split_index = len(data_featured) - config.TEST_PERIOD_DAYS
    split_date = data_featured.index[split_index]

    X_train = X[X.index < split_date]
    X_test = X[X.index >= split_date]
    y_train = y[y.index < split_date]
    y_test = y[y.index >= split_date]

    print(X_train.shape)
    print(f"Split-Datum: {split_date.date()}")
    print(
        f"Trainingsdaten: {X_train.shape[0]} Samples ({X_train.index.min().date()} bis {X_train.index.max().date()})"
    )
    print(
        f"Testdaten: {X_test.shape[0]} Samples ({X_test.index.min().date()} bis {X_test.index.max().date()})"
    )
    print(f"Anzahl Features: {X_train.shape[1]}")
    print(
        f"Zielvariablen: {target_cols_present}"
    )  # Zeige die tatsächlichen Zielvariablen an

    # Überprüfung des Split-Verhältnisses
    total_samples_after_engineering = X_train.shape[0] + X_test.shape[0]

    train_percentage = (X_train.shape[0] / total_samples_after_engineering) * 100
    test_percentage = (X_test.shape[0] / total_samples_after_engineering) * 100

    print(f"\nÜberprüfung des Split-Verhältnisses:")
    print(
        f"  Gesamte Samples nach Feature Engineering: {total_samples_after_engineering}"
    )
    print(f"  Trainings-Anteil: {train_percentage:.2f}%")
    print(f"  Test-Anteil:      {test_percentage:.2f}%")

    console.print("[green] Train/Test Split abgeschlossen. [/green]")

    # ----- 6. Modelltraining -----
    console.rule("[cyan]5. Modelltraining[/cyan]")
    trained_models = train_models(
        X_train,
        y_train,
        config.RF_PARAMETER,
        config.XGB_PARAMETER,
        config.MODEL_SAVE_DIR,
    )

    # ----- 7. Modellbewertung -----
    console.rule("[cyan]6. Modellbewertung[/cyan]")
    evaluate_model(
        models=trained_models,
        X_test=X_test,
        y_test=y_test,
        target_cols=target_cols_present,
        save_dir=config.EDA_PLOT_DIR
    )

    # ----- 8. Vorhersage für den nächsten Tag -----
    console.rule("[cyan]7. Vorhersage für den nächsten Tag[/cyan]")
    last_available_data_row = data_featured.iloc[-1:]
    predict_next_day(
        models=trained_models,
        last_available_data_row=last_available_data_row,
        features_cols=features_cols,  # Die Liste der Feature-Namen
        target_cols=target_cols_present,  # Die Liste der Ziel-Namen
    )

    console.rule("[bold blue]🎉 Wettervorhersage Workflow Abgeschlossen 🎉[/bold blue]")


if __name__ == "__main__":
    main()
