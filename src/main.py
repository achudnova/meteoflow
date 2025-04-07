import os
import sys
import config
import pandas as pd

from data_collection import find_stations, get_data_for_stations
from eda import start_eda
from data_preprocessing import preprocess_data
from feature_engineering import engineer_features
from data_splitting import split_data
from model_training import train_models
from model_evaluation import evaluate_model, create_temperature_time_series
from prediction import predict_next_day
from interpolation import idw_interpolate, get_station_data, DEFAULT_IDW_POWER

from rich.console import Console
from rich.panel import Panel

console = Console()

def main():
    console.rule("[bold purple4]⛅ Wettervorhersage für Berlin ⛅[/bold purple4]")
    
    # ----- 1. Datenerfassung -----
    
    console.rule("\n[orange1]1. Stationssuche & Datenerfassung[/orange1]")
    try:
        # Finde relevante Stations-IDs
        station_ids = find_stations(console=console)
        if not station_ids:
             console.print("[bold red]Keine Stationen gefunden. Breche ab.[/bold red]")
             sys.exit(1)

        # Lade Daten für diese Stationen
        # WICHTIG: Definiere Start/Ende für den historischen Abruf
        # Wir verwenden die Config-Werte wie zuvor für den langen Zeitraum
        end_dt_hist = config.END_DATE # datetime Objekt
        start_dt_hist = config.START_DATE # datetime Objekt

        all_station_data_dict = get_data_for_stations(
            station_ids=station_ids,
            start_date=start_dt_hist,
            end_date=end_dt_hist,
            required_columns=config.REQUIRED_COLUMNS,
            essential_columns=config.ESSENTIAL_COLS,
            console=console
        )

        if not all_station_data_dict:
             console.print("[bold red]Keine Daten für relevante Stationen geladen. Breche ab.[/bold red]")
             sys.exit(1)

    except Exception as e:
        console.print(f"[red] Ein Fehler bei Datenerfassung/Stationssuche ist aufgetreten: [/red] {e}")
        console.print_exception(show_locals=False)
        sys.exit(1)
    
    # ----- Interpolation -----
    console.rule("[orange1]1.5 Räumliche Interpolation (IDW)[/orange1]")
    try:
        # 1. Metadaten für gefundene Stationen holen
        console.print("   Hole Metadaten für Interpolation...")
        station_metadata = get_station_data(list(all_station_data_dict.keys()), console=console) # Nur für die, wo Daten geladen wurden
        if not station_metadata:
             console.print("[bold red]FEHLER: Konnte keine Metadaten für Interpolation laden. Breche ab.[/bold red]")
             sys.exit(1)

        # 2. IDW durchführen
        # Definiere, welche Variablen interpoliert werden sollen (aus Config)
        vars_to_interpolate = config.REQUIRED_COLUMNS # Annahme: alle benötigten Spalten

        berlin_interpolated_df = idw_interpolate(
            all_station_data=all_station_data_dict,
            station_metadata=station_metadata,
            target_lat=config.TARGET_LAT, # Ziel-Koordinaten aus Config
            target_lon=config.TARGET_LON,
            variables=vars_to_interpolate,
            console=console,
            power=DEFAULT_IDW_POWER # Potenz aus interpolation.py oder Config
        )

        if berlin_interpolated_df is None:
             console.print("[bold red]FEHLER: IDW-Interpolation fehlgeschlagen. Breche ab.[/bold red]")
             sys.exit(1)

        # Zeige Infos zum Ergebnis
        console.print("   Beispiel der interpolierten Daten für Berlin:")
        console.print(berlin_interpolated_df.head())
        console.print(berlin_interpolated_df.info())


    except Exception as e:
        console.print(f"[red] Ein Fehler bei der Interpolation ist aufgetreten: [/red] {e}")
        console.print_exception(show_locals=False)
        sys.exit(1)

    
    # ----- 2. Explorative Datenanalyse (EDA) -----
    console.rule("[orange1]1.5 Explorative Datenanalyse (EDA)[/orange1]")
    start_eda(berlin_interpolated_df, plot_columns=config.EDA_PLOT_COLUMNS, save_dir=config.EDA_PLOT_DIR)

    # ----- 3. Datenvorverarbeitung -----
    console.rule("[orange1]2. Datenvorverarbeitung[/orange1]")
    data_processed = preprocess_data(berlin_interpolated_df)
    if data_processed is None: sys.exit(1)

    # ----- 4. Feature Engineering -----
    console.rule("[orange1]3. Feature Engineering[/orange1]")
    data_featured = engineer_features(
        data=data_processed,
        target_cols=config.TARGET_COLUMNS,
        target_base_cols=config.ORIGINAL_TARGET_BASE_COLUMNS,
        lag_days=config.LAG_DAYS,
    )

    if data_featured is None or data_featured.empty:
        console.print(f"[red] Nach dem Feature Engineering sind keine Daten mehr verfügbar [/red]")
        sys.exit(1)

    # ----- 5. Train/Test Split -----
    console.rule("[orange1]4. Train/Test Split[/orange1]")
    try:
        X_train, X_test, y_train, y_test, features_cols, target_cols_present, split_date, train_percentage, test_percentage = split_data(
            data_featured, console
        )
        
    except Exception as e:
        console.print(f"[red] Fehler während des Train/Test Splits: {e} [/red]")
        sys.exit(1)


    # ----- 6. Modelltraining -----
    console.rule("[orange1]5. Modelltraining[/orange1]")
    trained_models = train_models(
        X_train,
        y_train,
        config.RF_PARAMETER,
        config.XGB_PARAMETER,
        config.MODEL_SAVE_DIR,
    )

    # ----- 7. Modellbewertung -----
    console.rule("[orange1]6. Modellbewertung[/orange1]")
    evaluate_model(
        models=trained_models,
        X_test=X_test,
        y_test=y_test,
        target_cols=target_cols_present,
        save_dir=config.EDA_PLOT_DIR
    )
    console.rule("[orange1]6.5 Temperatur-Zeitreihe erstellen[/orange1]")
    # Finde den Index der Temperaturspalte
    temp_target_idx = -1
    for i, col in enumerate(target_cols_present):
        if "tavg" in col:
            temp_target_idx = i
            break
    
    if temp_target_idx != -1:
        console.print("[green]Erstelle Temperatur-Zeitreihe...[/green]")
        create_temperature_time_series(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=trained_models,
            target_col_idx=temp_target_idx,
            target_cols=target_cols_present,
            save_dir=config.EDA_PLOT_DIR
        )
    else:
        console.print("[yellow]Keine Temperaturspalte gefunden. Überspringe Temperatur-Zeitreihe.[/yellow]")

    # ----- 8. Vorhersage für den nächsten Tag -----
    console.rule("[reverse green]7. Vorhersage für den nächsten Tag[/reverse green]")
    last_available_data_row = data_featured.iloc[-1:]
    
    console.print("\n[bold yellow]--- DEBUG: Features für Vorhersage aus main.py ---[/bold yellow]")
    console.print(f"Letzter Datenpunkt Index (main.py): {last_available_data_row.index[0]}")
    
    console.print("Feature-Werte (main.py):")
    console.print(last_available_data_row[features_cols].iloc[0].to_dict()) # Zeige Werte als Dictionary
    console.print("[bold yellow]-----------------------------------------------------[/bold yellow]\n")
    
    predict_next_day(
        models=trained_models,
        last_available_data_row=last_available_data_row,
        features_cols=features_cols,  # Die Liste der Feature-Namen
        target_cols=target_cols_present,  # Die Liste der Ziel-Namen
    )
    
    console.print("\n[bold blue]🎉 Wettervorhersage Workflow Abgeschlossen 🎉[/bold blue]")


if __name__ == "__main__":
    main()
