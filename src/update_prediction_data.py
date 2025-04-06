# src/update_prediction_data.py
import pandas as pd
import numpy as np
import sys
import os
import json
# Importiere datetime UND date
from datetime import datetime, timedelta, date

# --- Konfiguration und nötige Imports ---
import config
from data_collection import get_weather_data # Stelle sicher, dass der Pfad/Import stimmt
from feature_engineering import engineer_features # Stelle sicher, dass der Pfad/Import stimmt
from model_manager import load_model # Stelle sicher, dass der Pfad/Import stimmt
from rich.console import Console

console = Console()

def get_features_for_date(target_prediction_date: date):
    """Holt Daten bis zum Vortag und erstellt Features für die Vorhersage des target_prediction_date."""
    console.print(f"[cyan]Hole Daten für Vorhersage vom {target_prediction_date}...[/cyan]")

    # --- DATUM IN DATETIME UMWANDELN ---
    end_dt = datetime(target_prediction_date.year, target_prediction_date.month, target_prediction_date.day, 0, 0, 0)
    start_dt = end_dt - timedelta(days=config.LAG_DAYS + 5)
    fetch_end_dt_inclusive = end_dt - timedelta(seconds=1) # 23:59:59 des Vortages

    console.print(f"   Hole Rohdaten von {start_dt.strftime('%Y-%m-%d')} bis {fetch_end_dt_inclusive.strftime('%Y-%m-%d')}")

    try:
        # --- ÜBERGIB DATETIME-OBJEKTE ---
        data_raw = get_weather_data(
            location=config.LOCATION,
            start_date=start_dt,
            end_date=fetch_end_dt_inclusive,
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
            # Console hier nicht übergeben, wenn engineer_features es nicht erwartet
        )

        if data_featured is None or data_featured.empty:
            console.print("[red]   FEHLER: Keine Features nach Engineering.[/red]")
            return None, None

        # --- Letzte Zeile holen und prüfen ---
        last_row = data_featured.iloc[-1:]
        last_data_date = last_row.index.max().date()
        expected_last_date = target_prediction_date - timedelta(days=1) # Erwartet wird der Vortag von heute
        if last_data_date != expected_last_date:
             # Diese Warnung ist wichtig, sie zeigt an, dass wir nicht die Features vom Vortag haben
             console.print(f"[yellow]   WARNUNG: Letzter Feature-Tag ({last_data_date}) != erwarteter Vortag ({expected_last_date}). Datenlücke oder Effekt von Feature Engineering? Versuche trotzdem Vorhersage.[/yellow]")
             # NICHT ABBRECHEN HIER, WIR WOLLEN DEN DEBUG SEHEN
             # Wenn hier abgebrochen wird, gibt es keine Features zum Debuggen

        # --- Features extrahieren ---
        features_cols = [col for col in data_featured.columns if col not in config.TARGET_COLUMNS + config.ORIGINAL_TARGET_BASE_COLUMNS]
        features_for_prediction = last_row[features_cols]

        # --- Finale NaN-Prüfung ---
        if features_for_prediction.isnull().sum().sum() > 0:
            console.print("[red]   FEHLER: NaNs in finalen Features für Vorhersage.[/red]")
            console.print(features_for_prediction.isnull().sum())
            return None, None

        console.print(f"   Features basieren auf Daten vom [cyan]{last_data_date}[/cyan].")
        console.print("[green]   ✔️ Features für Vorhersage extrahiert.[/green]")
        # Gibt die Features zurück, auch wenn das Datum nicht dem erwarteten Vortag entspricht
        return features_for_prediction, features_cols

    except Exception as e:
        console.print(f"[red]   FEHLER beim Holen/Erstellen der Features: {e}[/red]")
        console.print_exception(show_locals=False)
        return None, None


def run_prediction_and_save():
    """Lädt Modelle, macht Vorhersage und speichert als JSON."""
    console.rule("[bold blue]Starte tägliches Vorhersage-Update[/bold blue]")

    # --- Heutiges Datum bestimmen ---
    today_date = date.today() # Holt das aktuelle Datum des Servers (UTC)
    console.print(f"Heutiges Datum (Basis für Feature-Abruf): [bold cyan]{today_date}[/bold cyan]")

    # --- Modelle laden ---
    console.print("\n[cyan]Lade Modelle...[/cyan]")
    # Stelle sicher, dass die Dateinamen mit denen beim Speichern übereinstimmen
    rf_model_path = os.path.join(config.MODEL_SAVE_DIR, 'rf_model.joblib')
    xgb_model_path = os.path.join(config.MODEL_SAVE_DIR, 'xgb_model.joblib')
    rf_model = load_model(rf_model_path, console)
    xgb_model = load_model(xgb_model_path, console)
    if rf_model is None or xgb_model is None:
        console.print("[red]FEHLER: Mindestens ein Modell konnte nicht geladen werden. Abbruch.[/red]")
        sys.exit(1)
    console.print("[green]   ✔️ Modelle geladen.[/green]")
    models = {'rf': rf_model, 'xgb': xgb_model}

    # --- Features holen (basierend auf Daten bis GESTERN) ---
    console.print("\n[cyan]Hole Features basierend auf Daten bis gestern...[/cyan]")
    features_for_prediction, features_cols = get_features_for_date(today_date)
    if features_for_prediction is None:
        console.print("[red]FEHLER: Features konnten nicht erstellt werden. Abbruch.[/red]")
        sys.exit(1)

    # --- Bestimme das tatsächliche Datum der verwendeten Features ---
    last_feature_date = features_for_prediction.index.max().date()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++
    # HIER IST DER DEBUG-CODE EINGEFÜGT:
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++
    console.print("\n[bold yellow]-- DEBUG: Features für Vorhersage aus Action --[/bold yellow]")
    # Das Datum, für das die Vorhersage gemacht wird (Tag nach den Features)
    target_forecast_date = last_feature_date + timedelta(days=1)
    console.print(f"Tatsächliches Vorhersagedatum (Action): {target_forecast_date}")
    console.print(f"Datum der verwendeten Features (Action): {last_feature_date}")
    console.print("Feature-Werte (Action):")
    try:
        # Zeige Werte als Dictionary für bessere Lesbarkeit
        console.print(features_for_prediction.iloc[0].to_dict())
    except Exception as e:
        console.print(f"[red]Fehler beim Anzeigen der Feature-Werte: {e}[/red]")
    console.print("[bold yellow]-------------------------------------------------[/bold yellow]\n")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ENDE DEBUG-CODE
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++


    # --- Vorhersage machen ---
    # Das Zieldatum der Vorhersage ist EIN Tag nach dem Datum der Features
    actual_prediction_target_date = last_feature_date + timedelta(days=1)
    console.print(f"\n[cyan]Mache Vorhersage für: [bold green]{actual_prediction_target_date}[/bold green][/cyan]")

    predictions_output = {}
    target_cols = config.TARGET_COLUMNS
    # --- Vorhersage-Schleife (wie in deiner letzten funktionierenden Version) ---
    for model_name, model in models.items():
         console.print(f"--- Verarbeite Vorhersage für: [bold]{model_name}[/bold] ---")
         temp_processed: float | None = None
         wind_processed: float | None = None
         try:
             # Verwende to_numpy() um sicherzustellen, dass es ein Array ist
             features_np = features_for_prediction.to_numpy()
             # Prüfe auf korrekte Feature-Anzahl (optional, aber gut zum Debuggen)
             expected_features = len(features_cols)
             if features_np.shape[1] != expected_features:
                 console.print(f"[bold red]FEHLER ({model_name}): Falsche Feature-Anzahl! Erwartet {expected_features}, bekommen {features_np.shape[1]}[/bold red]")
                 raise ValueError("Falsche Feature-Anzahl")

             prediction = model.predict(features_np)

             if prediction.ndim == 1: prediction = prediction.reshape(1, -1)

             # Indizes holen
             tavg_idx = target_cols.index('tavg_target') if 'tavg_target' in target_cols else -1
             wspd_idx = target_cols.index('wspd_target') if 'wspd_target' in target_cols else -1

             # Rohwerte extrahieren
             temp_raw = prediction[0, tavg_idx] if tavg_idx != -1 and tavg_idx < prediction.shape[1] else None
             wind_raw = prediction[0, wspd_idx] if wspd_idx != -1 and wspd_idx < prediction.shape[1] else None

             # Verarbeiten (NaN-Check und float-Konvertierung)
             if temp_raw is not None:
                 if isinstance(temp_raw, (float, np.floating)) and np.isnan(temp_raw): temp_processed = None
                 else: temp_processed = float(temp_raw)
             if wind_raw is not None:
                 if isinstance(wind_raw, (float, np.floating)) and np.isnan(wind_raw): wind_processed = None
                 else: wind_processed = float(wind_raw)

             # Speichern
             predictions_output[model_name] = {'temp': temp_processed,'wspd': wind_processed}
             console.print(f"Verarbeitete Werte ({model_name}): temp={temp_processed}, wind={wind_processed}")

         except Exception as e:
             console.print(f"[bold red]   FEHLER bei Vorhersage mit {model_name}: {e}[/bold red]")
             console.print_exception(show_locals=False) # Zeige Traceback für Fehler
             predictions_output[model_name] = {'temp': None, 'wspd': None}

    console.print("[green]   ✔️ Vorhersage-Loop abgeschlossen.[/green]")

    # --- Daten für JSON aufbereiten ---
    output_data = {
        # Verwende das tatsächliche Vorhersagedatum (Tag nach den Features)
        "forecast_date": actual_prediction_target_date.strftime("%Y-%m-%d"),
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
        console.print_exception(show_locals=False) # Zeige Traceback
        sys.exit(1)

    console.rule("[bold blue]Update abgeschlossen[/bold blue]")


if __name__ == "__main__":
    # --- Modell-Verzeichnis-Check ---
    if not hasattr(config, 'MODEL_SAVE_DIR') or not os.path.isdir(config.MODEL_SAVE_DIR):
         script_dir = os.path.dirname(__file__)
         # Gehe eine Ebene höher von 'src' zu 'meteoflow' und dann in 'saved_models'
         potential_model_dir = os.path.abspath(os.path.join(script_dir, '..', 'saved_models'))
         if os.path.isdir(potential_model_dir):
             # WICHTIG: Weise den korrekten Pfad config zu, falls er nicht existiert
             config.MODEL_SAVE_DIR = potential_model_dir
             console.print(f"[yellow]Nutze gefundenen Modellordner: {config.MODEL_SAVE_DIR}[/yellow]")
         else:
             # Versuche absoluten Pfad, falls konfiguriert aber nicht gefunden
             abs_model_dir = getattr(config, 'MODEL_SAVE_DIR', 'saved_models') # Default, falls nicht gesetzt
             console.print(f"[red]FEHLER: Modell-Verzeichnis nicht gefunden. Erwartet unter '{abs_model_dir}' oder relativ als '../saved_models'[/red]")
             sys.exit(1)

    run_prediction_and_save()