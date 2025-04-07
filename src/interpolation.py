import pandas as pd
import numpy as np
from rich.console import Console
from tqdm import tqdm
from meteostat import Stations
import sys

# ... (Import für Distanzfunktion) ...
from geo_utils import haversine_distance
# ODER
# from haversine import haversine

import config

DEFAULT_IDW_POWER = 2
MIN_STATIONS_FOR_IDW = 1

def get_station_data(station_ids: list, console: Console) -> dict: # Konsole hinzugefügt
    """ Holt Metadaten (Koordinaten) für die gegebenen Stations-IDs. """
    console.print("   Lade Stationsinventar...") # Nachricht geändert
    try:
        # Lade das *gesamte* Stationsinventar
        all_stations_df = Stations().fetch()
        console.print("      ✔️ Stationsinventar geladen.")

        # Filtere das DataFrame nach den benötigten IDs
        stations_meta_filtered = all_stations_df[all_stations_df.index.isin(station_ids)]

        if stations_meta_filtered.empty:
            console.print(f"[red]FEHLER: Keine Metadaten für die spezifischen IDs {station_ids} im Inventar gefunden.[/red]")
            return {}

        # Erstelle das Dictionary: station_id -> (latitude, longitude)
        metadata_dict = {
            idx: (row['latitude'], row['longitude'])
            for idx, row in stations_meta_filtered.iterrows()
            if pd.notna(row['latitude']) and pd.notna(row['longitude']) # Nur Stationen mit Koordinaten
        }

        # Prüfe, ob für alle angefragten IDs Metadaten gefunden wurden
        found_ids = set(metadata_dict.keys())
        missing_ids = set(station_ids) - found_ids
        if missing_ids:
            console.print(f"[yellow]WARNUNG: Keine Metadaten (Koordinaten) gefunden für Station(en): {list(missing_ids)}[/yellow]")

        if not metadata_dict:
             console.print(f"[red]FEHLER: Keine gültigen Metadaten für die angefragten Stationen gefunden.[/red]")
             return {}

        console.print(f"      ✔️ Metadaten für {len(metadata_dict)} Station(en) extrahiert.")
        return metadata_dict

    except Exception as e:
        console.print(f"[red]FEHLER beim Holen/Filtern der Stationsmetadaten: {e}[/red]")
        console.print_exception(show_locals=False)
        return {}

def idw_interpolate(
    all_station_data: dict[str, pd.DataFrame],
    station_metadata: dict[str, tuple[float, float]],
    target_lat: float,
    target_lon: float,
    variables: list[str],
    console: Console,
    power: int = DEFAULT_IDW_POWER,
) -> pd.DataFrame | None:
    
    console.print(f"\n[cyan]Starte IDW-Interpolation für {variables} (p={power})...[/cyan]")
    if not all_station_data or not station_metadata:
        console.print("[red]FEHLER: Keine Stationsdaten oder Metadaten für Interpolation vorhanden.[/red]")
        return None
    
    reference_index = None
    all_indices = []
    for df in all_station_data.values():
        all_indices.append(df.index)
    if not all_indices: return None
    # Finde den frühesten Start und das späteste Ende
    min_date = min(idx.min() for idx in all_indices)
    max_date = max(idx.max() for idx in all_indices)
    reference_index = pd.date_range(start=min_date, end=max_date, freq='D') # Tagesfrequenz annehmen
    console.print(f"   Interpoliere für Zeitraum: {min_date.date()} bis {max_date.date()}")
    
    interpolated_data = {}
    
    distances = {}
    valid_station_ids = list(station_metadata.keys()) # IDs der Stationen mit Metadaten
    for station_id in valid_station_ids:
        lat, lon = station_metadata[station_id]
        distances[station_id] = haversine_distance(lat, lon, target_lat, target_lon)
        
    for var in variables:
        console.print(f"   Interpoliere Variable: [magenta]{var}[/magenta]...")
        interpolated_values = []
        num_missing_days = 0

        # Iteriere über jeden Tag im Referenzzeitraum (mit Fortschrittsbalken)
        for target_date in tqdm(reference_index, desc=f"Interpolating {var}", unit="day", file=sys.stdout): # Zeige Fortschritt in Konsole
            weighted_sum = 0.0
            sum_of_weights = 0.0
            stations_with_value = 0

            # Iteriere über die verfügbaren Stationen
            for station_id, station_df in all_station_data.items():
                if station_id not in distances: continue # Überspringe, falls keine Distanz/Metadaten

                # Hole Wert der Station für diesen Tag (falls vorhanden und nicht NaN)
                if target_date in station_df.index and var in station_df.columns:
                    value = station_df.loc[target_date, var]
                    if pd.notna(value):
                        dist = distances[station_id]
                        # Vermeide Division durch Null, wenn Distanz sehr klein ist
                        if dist < 0.001: # Wenn Station quasi am Zielpunkt liegt
                            # Nimm direkt diesen Wert (oder behandle speziell)
                            weighted_sum = value
                            sum_of_weights = 1.0
                            stations_with_value = 999 # Markierung für direkten Treffer
                            break # Keine weitere Berechnung nötig für diesen Tag

                        weight = 1.0 / (dist ** power)
                        weighted_sum += weight * value
                        sum_of_weights += weight
                        stations_with_value += 1

            # Berechne den interpolierten Wert für diesen Tag
            interpolated_value = np.nan # Standardwert, falls nichts gefunden
            if stations_with_value >= MIN_STATIONS_FOR_IDW:
                if sum_of_weights > 0:
                    interpolated_value = weighted_sum / sum_of_weights
                elif stations_with_value == 999: # Direkter Treffer
                     interpolated_value = weighted_sum # War schon der Wert selbst

            if np.isnan(interpolated_value):
                num_missing_days += 1

            interpolated_values.append(interpolated_value)

        interpolated_data[var] = interpolated_values
        if num_missing_days > 0:
            console.print(f"     [yellow]Warnung: Für {var} konnten an {num_missing_days} Tagen keine Werte interpoliert werden (zu wenige Stationen?).[/yellow]")

    # Erstelle den finalen DataFrame
    try:
        result_df = pd.DataFrame(interpolated_data, index=reference_index)
        result_df.index.name = 'time' # Index benennen
        console.print("[green]   ✔️ IDW-Interpolation abgeschlossen.[/green]")
        return result_df
    except Exception as e:
        console.print(f"[red]FEHLER beim Erstellen des finalen interpolierten DataFrames: {e}[/red]")
        return None