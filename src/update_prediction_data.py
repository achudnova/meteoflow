import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime, timedelta

# --- Konfiguration und nötige Imports aus deinem Projekt ---
import config # Dein config.py
from data_collection import get_weather_data # Deine Funktion
from feature_engineering import engineer_features # Deine Funktion
from model_manager import load_model # Deine Funktion zum Laden
from rich.console import Console

console = Console()

def get_latest_features():
    """Holt aktuelle Daten und erstellt Features für die morgige Vorhersage."""
    console.print("[cyan]Hole aktuelle Daten für Feature-Erstellung...[/cyan]")
    # --- Daten holen (nur die letzten paar Tage reichen!) ---
    # Passe das Startdatum an, sodass nur genug Daten für Lags geholt werden
    fetch_end = datetime.now() + timedelta(days=1) # Sicherheitspuffer
    fetch_start = fetch_end - timedelta(days=config.LAG_DAYS + 5) # Nur Lags + Puffer holen
    try:
        # WICHTIG: Stelle sicher, dass get_weather_data keine Plots erzeugt etc.
        # Eventuell eine vereinfachte Version oder Parameter hinzufügen.
        # Hier gehen wir davon aus, dass es nur die Daten holt.
        # Console wird hier nicht übergeben, da get_weather_data sie evtl. nicht erwartet
        data_raw = get_weather_data(
            location=config.LOCATION,
            start_date=fetch_start,
            end_date=fetch_end,
            required_columns=config.REQUIRED_COLUMNS,
            essential_columns=config.ESSENTIAL_COLS
        )
        console.print("[green]   ✔️ Rohdaten für Features geholt.[/green]")

        # --- Minimales Preprocessing (nur was für Features nötig ist) ---
        # Meist nur fillna, wenn aktuellste Daten fehlen
        data_raw.ffill(inplace=True)
        data_raw.bfill(inplace=True)
        if data_raw.isnull().sum().sum() > 0:
             console.print("[yellow]   Warnung: Immer noch NaNs nach ffill/bfill in Rohdaten.[/yellow]")
             data_raw.dropna(inplace=True) # Letzte Rettung

        # --- Feature Engineering (wie in main.py, aber auf kürzeren Daten) ---
        # WICHTIG: Stelle sicher, dass engineer_features nur die Features baut
        # und keine Plots etc. macht. Console hier nicht übergeben.
        data_featured = engineer_features(
            data=data_raw,
            target_cols=config.TARGET_COLUMNS,
            target_base_cols=config.ORIGINAL_TARGET_BASE_COLUMNS,
            lag_days=config.LAG_DAYS
        )

        if data_featured is None or data_featured.empty:
            console.print("[red]   FEHLER: Keine Features nach Engineering.[/red]")
            return None, None

        # Extrahiere die letzte Zeile für die Vorhersage
        last_row = data_featured.iloc[-1:]
        features_cols = [col for col in data_featured.columns if col not in config.TARGET_COLUMNS + config.ORIGINAL_TARGET_BASE_COLUMNS]
        features_for_prediction = last_row[features_cols]

        # Überprüfe auf NaNs in der finalen Feature-Zeile
        if features_for_prediction.isnull().sum().sum() > 0:
            console.print("[red]   FEHLER: NaNs in finalen Features für Vorhersage.[/red]")
            console.print(features_for_prediction.isnull().sum())
            return None, None

        console.print("[green]   ✔️ Features für Vorhersage extrahiert.[/green]")
        return features_for_prediction, features_cols

    except Exception as e:
        console.print(f"[red]   FEHLER beim Holen/Erstellen der Features: {e}[/red]")
        return None, None


def run_prediction_and_save():
    """Lädt Modelle, macht Vorhersage und speichert als JSON."""
    console.rule("[bold blue]Starte tägliches Vorhersage-Update[/bold blue]")

    # --- Modelle laden ---
    console.print("[cyan]Lade Modelle...[/cyan]")
    rf_model_path = os.path.join(config.MODEL_SAVE_DIR, 'random_forest_model.joblib')
    xgb_model_path = os.path.join(config.MODEL_SAVE_DIR, 'xgboost_model.joblib')

    # Verwende die load_model Funktion, übergebe die Konsole
    rf_model = load_model(rf_model_path, console)
    xgb_model = load_model(xgb_model_path, console)

    if rf_model is None or xgb_model is None:
        console.print("[red]FEHLER: Mindestens ein Modell konnte nicht geladen werden. Abbruch.[/red]")
        sys.exit(1)
    console.print("[green]   ✔️ Modelle geladen.[/green]")
    models = {'rf': rf_model, 'xgb': xgb_model}

    # --- Features holen ---
    features_for_prediction, features_cols = get_latest_features()
    if features_for_prediction is None:
        console.print("[red]FEHLER: Features konnten nicht erstellt werden. Abbruch.[/red]")
        sys.exit(1)

    # --- Vorhersage ---
    console.print("[cyan]Mache Vorhersage...[/cyan]")
    predictions_output = {}
    prediction_date = features_for_prediction.index.max() + timedelta(days=1)
    target_cols = config.TARGET_COLUMNS # Aus config holen

    try:
        tavg_idx = target_cols.index('tavg_target') if 'tavg_target' in target_cols else -1
        wspd_idx = target_cols.index('wspd_target') if 'wspd_target' in target_cols else -1
    except ValueError:
        console.print("[red]FEHLER: Zielspalten nicht in config.TARGET_COLUMNS gefunden.[/red]")
        sys.exit(1)

    for model_name, model in models.items():
        try:
            prediction = model.predict(features_for_prediction)
            if prediction.ndim == 1:
                prediction = prediction.reshape(1, -1)

            temp_pred = prediction[0, tavg_idx] if tavg_idx != -1 and tavg_idx < prediction.shape[1] else None
            wind_pred = prediction[0, wspd_idx] if wspd_idx != -1 and wspd_idx < prediction.shape[1] else None

            predictions_output[model_name] = {
                'temp': temp_pred,
                'wspd': wind_pred
            }
        except Exception as e:
            console.print(f"[yellow]   Warnung bei Vorhersage mit {model_name}: {e}[/yellow]")
            predictions_output[model_name] = {'temp': None, 'wspd': None} # Fehler markieren

    console.print("[green]   ✔️ Vorhersage abgeschlossen.[/green]")

    # --- Daten für JSON aufbereiten ---
    output_data = {
        "forecast_date": prediction_date.strftime("%Y-%m-%d"),
        "rf_temp_c": predictions_output.get('rf', {}).get('temp'),
        "rf_wspd_kmh": predictions_output.get('rf', {}).get('wspd'),
        "xgb_temp_c": predictions_output.get('xgb', {}).get('temp'),
        "xgb_wspd_kmh": predictions_output.get('xgb', {}).get('wspd'),
        "generated_at": datetime.now().isoformat() # Zeitstempel der Erstellung
    }

    # --- JSON Speichern ---
    # WICHTIG: Der Pfad muss relativ zum Repository-Stamm sein!
    json_filepath = "prediction.json" # Speichert im Stammverzeichnis
    console.print(f"[cyan]Speichere Vorhersage in '{json_filepath}'...[/cyan]")
    try:
        with open(json_filepath, 'w') as f:
            json.dump(output_data, f, indent=2, default=lambda x: round(x, 1) if isinstance(x, float) else None) # Runden & None für Fehler
        console.print(f"[green]   ✔️ Vorhersage erfolgreich in '{json_filepath}' gespeichert.[/green]")
    except Exception as e:
        console.print(f"[red]   FEHLER beim Speichern der JSON-Datei: {e}[/red]")
        sys.exit(1)

    console.rule("[bold blue]Update abgeschlossen[/bold blue]")

if __name__ == "__main__":
    # Stelle sicher, dass MODEL_SAVE_DIR in config existiert und korrekt ist
    if not hasattr(config, 'MODEL_SAVE_DIR') or not os.path.isdir(config.MODEL_SAVE_DIR):
         # Versuche, relativ zum Skriptpfad zu finden
         script_dir = os.path.dirname(__file__)
         potential_model_dir = os.path.abspath(os.path.join(script_dir, '..', 'saved_models')) # Annahme: saved_models liegt parallel zu src
         if os.path.isdir(potential_model_dir):
             config.MODEL_SAVE_DIR = potential_model_dir
             console.print(f"[yellow]Nutze gefundenen Modellordner: {config.MODEL_SAVE_DIR}[/yellow]")
         else:
             console.print(f"[red]FEHLER: Modell-Verzeichnis '{config.MODEL_SAVE_DIR}' nicht gefunden oder nicht konfiguriert.[/red]")
             sys.exit(1)

    run_prediction_and_save()