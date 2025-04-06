# src/update_prediction_data.py
import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime, timedelta, date

import config
from data_collection import get_weather_data
from feature_engineering import engineer_features
from model_manager import load_model
from rich.console import Console

console = Console()

def get_latest_features_for_tomorrow():
    """Holt die neuesten Daten und erstellt Features für die morgige Vorhersage."""
    console.print("[cyan]Hole neueste Daten für Feature-Erstellung...[/cyan]")
    # Daten bis HEUTE holen, damit der letzte Feature-Tag GESTERN ist
    # (oder vorgestern, wenn heute noch nicht verfügbar/verarbeitet)
    fetch_end_date = date.today() + timedelta(days=1) # Ende ist morgen früh
    fetch_start_date = fetch_end_date - timedelta(days=config.LAG_DAYS + 10) # Etwas mehr Puffer holen

    # --- Umwandlung in datetime für get_weather_data ---
    end_dt = datetime(fetch_end_date.year, fetch_end_date.month, fetch_end_date.day, 0, 0, 0)
    start_dt = datetime(fetch_start_date.year, fetch_start_date.month, fetch_start_date.day, 0, 0, 0)
    fetch_end_dt_inclusive = end_dt - timedelta(seconds=1) # Bis 23:59:59 des Vortages holen

    console.print(f"   Hole Rohdaten von {start_dt.strftime('%Y-%m-%d')} bis {fetch_end_dt_inclusive.strftime('%Y-%m-%d')}")

    try:
        data_raw = get_weather_data(
            location=config.LOCATION,
            start_date=start_dt,
            end_date=fetch_end_dt_inclusive,
            required_columns=config.REQUIRED_COLUMNS,
            essential_columns=config.ESSENTIAL_COLS
        )
        if data_raw.empty:
             console.print("[red]   FEHLER: Keine Rohdaten erhalten.[/red]")
             return None, None
        console.print("[green]   ✔️ Rohdaten geholt.[/green]")

        # --- Preprocessing ---
        data_raw.ffill(inplace=True)
        data_raw.bfill(inplace=True)
        if data_raw.isnull().sum().sum() > 0:
             data_raw.dropna(inplace=True)
             if data_raw.empty:
                  console.print("[red]   FEHLER: Keine Daten nach dropna.[/red]")
                  return None, None

        # --- Feature Engineering ---
        data_featured = engineer_features(
            data=data_raw,
            target_cols=config.TARGET_COLUMNS,
            target_base_cols=config.ORIGINAL_TARGET_BASE_COLUMNS,
            lag_days=config.LAG_DAYS
        )
        if data_featured is None or data_featured.empty:
            console.print("[red]   FEHLER: Keine Features nach Engineering.[/red]")
            return None, None

        # --- Letzte Zeile holen (Basis für Vorhersage) ---
        last_row = data_featured.iloc[-1:]
        last_data_date = last_row.index.max().date()
        console.print(f"   Letzter Feature-Tag (Basis für Vorhersage): [cyan]{last_data_date}[/cyan].")

        # --- Features extrahieren ---
        features_cols = [col for col in data_featured.columns if col not in config.TARGET_COLUMNS + config.ORIGINAL_TARGET_BASE_COLUMNS]
        features_for_prediction = last_row[features_cols]

        # --- Finale NaN-Prüfung ---
        if features_for_prediction.isnull().sum().sum() > 0:
            console.print("[red]   FEHLER: NaNs in finalen Features für Vorhersage.[/red]")
            console.print(features_for_prediction.isnull().sum())
            return None, None

        console.print("[green]   ✔️ Features für Vorhersage extrahiert.[/green]")
        # WICHTIG: Wir geben hier auch das Datum der Features zurück
        return features_for_prediction, features_cols, last_data_date

    except Exception as e:
        console.print(f"[red]   FEHLER beim Holen/Erstellen der Features: {e}[/red]")
        console.print_exception(show_locals=False)
        return None, None, None


def run_prediction_and_save():
    """Lädt Modelle, macht Vorhersage für MORGEN basierend auf letzten Features und speichert als JSON."""
    console.rule("[bold blue]Starte tägliches Vorhersage-Update[/bold blue]")

    # --- Modelle laden ---
    console.print("\n[cyan]Lade Modelle...[/cyan]")
    rf_model_path = os.path.join(config.MODEL_SAVE_DIR, 'rf_model.joblib')
    xgb_model_path = os.path.join(config.MODEL_SAVE_DIR, 'xgb_model.joblib')
    rf_model = load_model(rf_model_path, console)
    xgb_model = load_model(xgb_model_path, console)
    if rf_model is None or xgb_model is None:
        console.print("[red]FEHLER: Mindestens ein Modell konnte nicht geladen werden. Abbruch.[/red]")
        sys.exit(1)
    console.print("[green]   ✔️ Modelle geladen.[/green]")
    models = {'rf': rf_model, 'xgb': xgb_model}

    # --- Neueste Features holen ---
    console.print("\n[cyan]Hole neueste verfügbare Features...[/cyan]")
    # Funktion gibt jetzt auch das Datum der Features zurück
    features_for_prediction, features_cols, last_feature_date = get_latest_features_for_tomorrow()
    if features_for_prediction is None:
        console.print("[red]FEHLER: Features konnten nicht erstellt werden. Abbruch.[/red]")
        sys.exit(1)

    # --- Vorhersage machen ---
    # Das Zieldatum ist der Tag NACH dem Datum der Features
    prediction_target_date = last_feature_date + timedelta(days=1)
    console.print(f"\n[cyan]Mache Vorhersage für: [bold green]{prediction_target_date}[/bold green] (basierend auf Daten vom {last_feature_date})[/cyan]")

    predictions_output = {}
    target_cols = config.TARGET_COLUMNS
    # --- Vorhersage-Schleife ---
    for model_name, model in models.items():
         console.print(f"--- Verarbeite Vorhersage für: [bold]{model_name}[/bold] ---")
         temp_processed: float | None = None
         wind_processed: float | None = None
         try:
             features_np = features_for_prediction.to_numpy()
             prediction = model.predict(features_np)
             if prediction.ndim == 1: prediction = prediction.reshape(1, -1)
             tavg_idx = target_cols.index('tavg_target') if 'tavg_target' in target_cols else -1
             wspd_idx = target_cols.index('wspd_target') if 'wspd_target' in target_cols else -1
             temp_raw = prediction[0, tavg_idx] if tavg_idx != -1 and tavg_idx < prediction.shape[1] else None
             wind_raw = prediction[0, wspd_idx] if wspd_idx != -1 and wspd_idx < prediction.shape[1] else None
             if temp_raw is not None:
                 if isinstance(temp_raw, (float, np.floating)) and np.isnan(temp_raw): temp_processed = None
                 else: temp_processed = float(temp_raw)
             if wind_raw is not None:
                 if isinstance(wind_raw, (float, np.floating)) and np.isnan(wind_raw): wind_processed = None
                 else: wind_processed = float(wind_raw)
             predictions_output[model_name] = {'temp': temp_processed,'wspd': wind_processed}
             console.print(f"Verarbeitete Werte ({model_name}): temp={temp_processed}, wind={wind_processed}")
         except Exception as e:
             console.print(f"[bold red]   FEHLER bei Vorhersage mit {model_name}: {e}[/bold red]")
             predictions_output[model_name] = {'temp': None, 'wspd': None}

    console.print("[green]   ✔️ Vorhersage-Loop abgeschlossen.[/green]")

    # --- Daten für JSON aufbereiten ---
    output_data = {
        "forecast_date": prediction_target_date.strftime("%Y-%m-%d"), # Tag nach den Features
        "rf_temp_c": predictions_output.get('rf', {}).get('temp'),
        "rf_wspd_kmh": predictions_output.get('rf', {}).get('wspd'),
        "xgb_temp_c": predictions_output.get('xgb', {}).get('temp'),
        "xgb_wspd_kmh": predictions_output.get('xgb', {}).get('wspd'),
        "generated_at": datetime.now().isoformat()
    }
    console.print("\n[bold]Finale Daten für JSON:[/bold]")
    console.print(output_data)

    # --- JSON Speichern ---
    json_filepath = "prediction.json"
    console.print(f"\n[cyan]Speichere Vorhersage in '{json_filepath}'...[/cyan]")
    try:
        with open(json_filepath, 'w') as f:
            json.dump(output_data, f, indent=2, default=lambda x: round(x, 1) if isinstance(x, (float, int)) else None)
        console.print(f"[green]   ✔️ Vorhersage erfolgreich in '{json_filepath}' gespeichert.[/green]")
    except Exception as e:
        console.print(f"[red]   FEHLER beim Speichern der JSON-Datei: {e}[/red]")
        sys.exit(1)

    console.rule("[bold blue]Update abgeschlossen[/bold blue]")


if __name__ == "__main__":
    # --- Modell-Verzeichnis-Check (bleibt gleich) ---
    if not hasattr(config, 'MODEL_SAVE_DIR') or not os.path.isdir(config.MODEL_SAVE_DIR):
         script_dir = os.path.dirname(__file__)
         potential_model_dir = os.path.abspath(os.path.join(script_dir, '..', 'saved_models'))
         if os.path.isdir(potential_model_dir):
             config.MODEL_SAVE_DIR = potential_model_dir
             console.print(f"[yellow]Nutze gefundenen Modellordner: {config.MODEL_SAVE_DIR}[/yellow]")
         else:
             abs_model_dir = getattr(config, 'MODEL_SAVE_DIR', 'saved_models')
             console.print(f"[red]FEHLER: Modell-Verzeichnis nicht gefunden. Erwartet unter '{abs_model_dir}' oder relativ als '../saved_models'[/red]")
             sys.exit(1)

    run_prediction_and_save()