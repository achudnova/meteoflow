# src/update_prediction_data.py
import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime, timedelta, date # date hinzufügen

# --- Konfiguration und nötige Imports aus deinem Projekt ---
import config
from data_collection import get_weather_data
from feature_engineering import engineer_features
from model_manager import load_model
from rich.console import Console

console = Console()

def get_features_for_date(target_prediction_date: date):
    """Holt Daten bis zum Vortag und erstellt Features für die Vorhersage des target_prediction_date."""
    console.print(f"[cyan]Hole Daten für Vorhersage vom {target_prediction_date}...[/cyan]")

    # Daten nur bis zum ENDE DES VORTAGES holen!
    fetch_end_date = target_prediction_date # datetime(target_prediction_date.year, target_prediction_date.month, target_prediction_date.day)
    fetch_start_date = fetch_end_date - timedelta(days=config.LAG_DAYS + 5)

    console.print(f"   Hole Rohdaten von {fetch_start_date.strftime('%Y-%m-%d')} bis {(fetch_end_date - timedelta(days=1)).strftime('%Y-%m-%d')}")

    try:
        data_raw = get_weather_data(
            location=config.LOCATION,
            start_date=fetch_start_date,
            end_date=fetch_end_date - timedelta(days=1), # Nur bis Ende des Vortages!
            required_columns=config.REQUIRED_COLUMNS,
            essential_columns=config.ESSENTIAL_COLS
        )
        if data_raw.empty:
             console.print("[red]   FEHLER: Keine Rohdaten für den benötigten Zeitraum erhalten.[/red]")
             return None, None
        console.print("[green]   ✔️ Rohdaten geholt.[/green]")

        # --- Minimales Preprocessing ---
        data_raw.ffill(inplace=True)
        data_raw.bfill(inplace=True)
        if data_raw.isnull().sum().sum() > 0:
             console.print("[yellow]   Warnung: Immer noch NaNs nach ffill/bfill in Rohdaten.[/yellow]")
             data_raw.dropna(inplace=True)
             if data_raw.empty:
                  console.print("[red]   FEHLER: Keine Daten mehr nach dropna.[/red]")
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

        # Extrahiere die LETZTE Zeile (entspricht dem Vortag der Vorhersage)
        last_row = data_featured.iloc[-1:]

        # --- Stelle sicher, dass die letzte Zeile wirklich der Vortag ist ---
        last_data_date = last_row.index.max().date()
        expected_last_date = target_prediction_date - timedelta(days=1)
        if last_data_date != expected_last_date:
             console.print(f"[red]   FEHLER: Letzter Feature-Tag ({last_data_date}) entspricht nicht dem erwarteten Vortag ({expected_last_date}). Datenlücke?[/red]")
             return None, None
        console.print(f"   Features basieren auf Daten vom [cyan]{last_data_date}[/cyan].")

        features_cols = [col for col in data_featured.columns if col not in config.TARGET_COLUMNS + config.ORIGINAL_TARGET_BASE_COLUMNS]
        features_for_prediction = last_row[features_cols]

        if features_for_prediction.isnull().sum().sum() > 0:
            console.print("[red]   FEHLER: NaNs in finalen Features für Vorhersage.[/red]")
            console.print(features_for_prediction.isnull().sum())
            return None, None

        console.print("[green]   ✔️ Features für Vorhersage extrahiert.[/green]")
        return features_for_prediction, features_cols

    except Exception as e:
        console.print(f"[red]   FEHLER beim Holen/Erstellen der Features: {e}[/red]")
        console.print_exception(show_locals=False) # Zeige Traceback
        return None, None


def run_prediction_and_save():
    """Lädt Modelle, macht Vorhersage für HEUTE und speichert als JSON."""
    console.rule("[bold blue]Starte tägliches Vorhersage-Update[/bold blue]")

    # --- Das Zieldatum für die Vorhersage ist HEUTE ---
    # (Wenn die Action um 07:01 UTC läuft, ist 'heute' der Kalendertag dieses Zeitpunkts)
    prediction_date_target = date.today() # Holt das aktuelle Datum des Servers (UTC)
    console.print(f"Zieldatum der Vorhersage: [bold green]{prediction_date_target}[/bold green]")

    # --- Modelle laden (bleibt gleich) ---
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

    # --- Features für den VORTAG holen ---
    console.print("\n[cyan]Hole Features basierend auf Daten bis gestern...[/cyan]")
    features_for_prediction, features_cols = get_features_for_date(prediction_date_target)
    if features_for_prediction is None:
        console.print("[red]FEHLER: Features konnten nicht erstellt werden. Abbruch.[/red]")
        sys.exit(1)

    # --- Vorhersage machen (bleibt gleich, aber basiert auf Vortages-Features) ---
    console.print("\n[cyan]Mache Vorhersage für heute...[/cyan]")
    predictions_output = {}
    target_cols = config.TARGET_COLUMNS
    # ... (Schleife für Vorhersage mit try-except, Konvertierung etc. wie zuvor) ...
    # ... (Stelle sicher, dass die letzte Debugging-Version hier eingefügt ist) ...
    for model_name, model in models.items():
        # ... (Code aus deiner letzten funktionierenden Version) ...
        # Wichtig: Der *Inhalt* der Vorhersageschleife bleibt gleich
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
    #---------------------------------------------------------------------

    console.print("[green]   ✔️ Vorhersage-Loop abgeschlossen.[/green]")

    # --- Daten für JSON aufbereiten ---
    output_data = {
        # Das Datum im JSON ist jetzt das Zieldatum (heute)
        "forecast_date": prediction_date_target.strftime("%Y-%m-%d"),
        "rf_temp_c": predictions_output.get('rf', {}).get('temp'),
        "rf_wspd_kmh": predictions_output.get('rf', {}).get('wspd'),
        "xgb_temp_c": predictions_output.get('xgb', {}).get('temp'),
        "xgb_wspd_kmh": predictions_output.get('xgb', {}).get('wspd'),
        "generated_at": datetime.now().isoformat()
    }
    console.print("\n[bold]Finale Daten für JSON:[/bold]")
    console.print(output_data)

    # --- JSON Speichern (bleibt gleich) ---
    json_filepath = "prediction.json"
    console.print(f"\n[cyan]Speichere Vorhersage in '{json_filepath}'...[/cyan]")
    # ... (try-except für json.dump wie zuvor) ...
    try:
        with open(json_filepath, 'w') as f:
            json.dump(output_data, f, indent=2, default=lambda x: round(x, 1) if isinstance(x, (float, int)) else None)
        console.print(f"[green]   ✔️ Vorhersage erfolgreich in '{json_filepath}' gespeichert.[/green]")
    except Exception as e:
        console.print(f"[red]   FEHLER beim Speichern der JSON-Datei: {e}[/red]")
        sys.exit(1)

    console.rule("[bold blue]Update abgeschlossen[/bold blue]")

if __name__ == "__main__":
    # ... (Modell-Verzeichnis-Check wie zuvor) ...
    run_prediction_and_save()