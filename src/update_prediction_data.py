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
    rf_model_path = os.path.join(config.MODEL_SAVE_DIR, 'rf_model.joblib')
    xgb_model_path = os.path.join(config.MODEL_SAVE_DIR, 'xgb_model.joblib')

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

    console.print("[cyan]Mache Vorhersage...[/cyan]")
    predictions_output = {} # Dictionary zum Sammeln der *verarbeiteten* Vorhersagen
    prediction_date = features_for_prediction.index.max() + timedelta(days=1)
    target_cols = config.TARGET_COLUMNS

    try:
        tavg_idx = target_cols.index('tavg_target') if 'tavg_target' in target_cols else -1
        wspd_idx = target_cols.index('wspd_target') if 'wspd_target' in target_cols else -1
    except ValueError:
        console.print("[red]FEHLER: Zielspalten nicht in config.TARGET_COLUMNS gefunden.[/red]")
        sys.exit(1)

    for model_name, model in models.items():
        console.print(f"--- Verarbeite Vorhersage für: [bold]{model_name}[/bold] ---")
        temp_processed: float | None = None # Variable für verarbeitete Temperatur
        wind_processed: float | None = None # Variable für verarbeiteten Wind

        try:
            features_np = features_for_prediction.to_numpy()
            console.print(f"Input Features shape für {model_name}: {features_np.shape}")

            prediction = model.predict(features_np)
            console.print(f"Roh-Vorhersage ({model_name}): {prediction}")
            console.print(f"Typ der Roh-Vorhersage ({model_name}): {type(prediction)}")
            console.print(f"Shape der Roh-Vorhersage ({model_name}): {getattr(prediction, 'shape', 'N/A')}")

            if prediction.ndim == 1:
                prediction = prediction.reshape(1, -1)
                console.print(f"Reshaped Vorhersage ({model_name}): {prediction}")

            console.print(f"Indices für {model_name}: tavg_idx={tavg_idx}, wspd_idx={wspd_idx}, prediction.shape={prediction.shape}")

            # Werte extrahieren
            temp_raw = prediction[0, tavg_idx] if tavg_idx != -1 and tavg_idx < prediction.shape[1] else None
            wind_raw = prediction[0, wspd_idx] if wspd_idx != -1 and wspd_idx < prediction.shape[1] else None

            console.print(f"Extrahierte Roh-Werte ({model_name}): temp={temp_raw} (Typ: {type(temp_raw)}), wind={wind_raw} (Typ: {type(wind_raw)})")

            # --- HIER die explizite Umwandlung und NaN-Prüfung ---
            if temp_raw is not None:
                if isinstance(temp_raw, (float, np.floating)) and np.isnan(temp_raw):
                     console.print(f"[yellow]WARNUNG ({model_name}): Temperaturvorhersage ist NaN![/yellow]")
                     temp_processed = None # NaN wird zu None
                else:
                     try:
                         # In Standard Python float umwandeln
                         temp_processed = float(temp_raw)
                     except (ValueError, TypeError):
                          console.print(f"[yellow]WARNUNG ({model_name}): Konnte Temperaturwert '{temp_raw}' nicht in float umwandeln.[/yellow]")
                          temp_processed = None

            if wind_raw is not None:
                 if isinstance(wind_raw, (float, np.floating)) and np.isnan(wind_raw):
                      console.print(f"[yellow]WARNUNG ({model_name}): Windvorhersage ist NaN![/yellow]")
                      wind_processed = None # NaN wird zu None
                 else:
                      try:
                          # In Standard Python float umwandeln
                          wind_processed = float(wind_raw)
                      except (ValueError, TypeError):
                           console.print(f"[yellow]WARNUNG ({model_name}): Konnte Windwert '{wind_raw}' nicht in float umwandeln.[/yellow]")
                           wind_processed = None

            # Speichere die verarbeiteten Werte
            predictions_output[model_name] = {
                'temp': temp_processed,
                'wspd': wind_processed
            }
            console.print(f"Verarbeitete Werte ({model_name}): temp={temp_processed}, wind={wind_processed}")

        except Exception as e:
            console.print(f"[bold red]   FEHLER bei Vorhersage mit {model_name}: {e}[/bold red]")
            console.print_exception(show_locals=True) # Mehr Details zum Fehler
            predictions_output[model_name] = {'temp': None, 'wspd': None}

    console.print("[green]   ✔️ Vorhersage-Loop abgeschlossen.[/green]") # Geändert

    # --- Daten für JSON aufbereiten (holt jetzt die verarbeiteten Werte) ---
    output_data = {
        "forecast_date": prediction_date.strftime("%Y-%m-%d"),
        "rf_temp_c": predictions_output.get('rf', {}).get('temp'),
        "rf_wspd_kmh": predictions_output.get('rf', {}).get('wspd'),
        "xgb_temp_c": predictions_output.get('xgb', {}).get('temp'),
        "xgb_wspd_kmh": predictions_output.get('xgb', {}).get('wspd'),
        "generated_at": datetime.now().isoformat()
    }

    # --- FINALES LOGGING vor dem Speichern ---
    console.print("\n[bold]Finale Daten für JSON:[/bold]")
    console.print(output_data)
    console.print(f"Typ von xgb_temp_c: {type(output_data['xgb_temp_c'])}")
    console.print(f"Typ von xgb_wspd_kmh: {type(output_data['xgb_wspd_kmh'])}")
    # -----------------------------------------

    # --- JSON Speichern ---
    json_filepath = "prediction.json"
    console.print(f"\n[cyan]Speichere Vorhersage in '{json_filepath}'...[/cyan]")
    try:
        with open(json_filepath, 'w') as f:
            # Die default lambda ist jetzt weniger kritisch, kann aber bleiben
            json.dump(output_data, f, indent=2, default=lambda x: round(x, 1) if isinstance(x, (float, int)) else None)
        console.print(f"[green]   ✔️ Vorhersage erfolgreich in '{json_filepath}' gespeichert.[/green]")
    except Exception as e:
        console.print(f"[red]   FEHLER beim Speichern der JSON-Datei: {e}[/red]")
        console.print_exception(show_locals=True) # Mehr Details zum Speicherfehler
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