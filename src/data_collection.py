import pandas as pd
from datetime import datetime
from meteostat import Point, Daily, Stations
from config import TARGET_LAT, TARGET_LON, SEARCH_RADIUS_KM, MAX_NEARBY_STATIONS
from rich.console import Console


def find_stations(console: Console) -> list:
    print(
        f"Suche nach Wetterstationen im Umkreis von {SEARCH_RADIUS_KM} km um Berlin..."
    )
    stations = Stations()

    try:
        nearby_stations_df = stations.nearby(
            TARGET_LAT, TARGET_LON, SEARCH_RADIUS_KM * 1000
        )
        nearby_stations_df = nearby_stations_df.fetch(limit=MAX_NEARBY_STATIONS * 2)

        if nearby_stations_df.empty:
            print("FEHLER: Keine Stationen im angegebenen Radius gefunden.")
            return []

        relevant_stations = nearby_stations_df[
            nearby_stations_df["daily_end"].isnull()
            | (
                nearby_stations_df["daily_end"]
                >= datetime.now() - pd.Timedelta(days=3 * 365)
            )
        ]
        
        if relevant_stations.empty:
                print("WARNUNG: Keine kürzlich aktiven Stationen gefunden. Nehme die nächstgelegenen.")
                relevant_stations = nearby_stations_df
        
        final_station_ids = relevant_stations.sort_values('distance').head(MAX_NEARBY_STATIONS).index.tolist()

        console.print(f"   Gefundene relevante Stations-IDs (bis zu {MAX_NEARBY_STATIONS}): [magenta]{final_station_ids}[/magenta]")
        console.print("   Details der Top Stationen:")
        console.print(relevant_stations.sort_values('distance').head(MAX_NEARBY_STATIONS)[['name', 'country', 'distance']])
        
        return final_station_ids

    except Exception as e:
        console.print(f"[red]FEHLER bei der Stationssuche: {e}[/red]")
        return []

def get_data_for_stations(
    station_ids: list,
    start_date: datetime,
    end_date: datetime,
    required_columns: list,
    essential_columns: list, # Behalten wir für spätere Checks
    console: Console
) -> dict[str, pd.DataFrame]:
    """Ruft tägliche Wetterdaten für eine Liste von Stations-IDs ab."""
    console.print(f"\n[cyan]Lade Daten für {len(station_ids)} Station(en) vom {start_date.strftime('%Y-%m-%d')} bis {end_date.strftime('%Y-%m-%d')}...[/cyan]")
    all_station_data = {}
    successful_stations = []

    for station_id in station_ids:
        console.print(f"   Versuche Station [bold]{station_id}[/bold]...")
        try:
            data_request = Daily(station_id, start_date, end_date)
            station_df = data_request.fetch()

            if station_df.empty:
                 console.print(f"     [yellow]Keine Daten für Station {station_id} im Zeitraum.[/yellow]")
                 continue # Nächste Station

            # --- Spaltenprüfung pro Station (Optional, aber gut) ---
            available_cols = [col for col in required_columns if col in station_df.columns]
            missing_essential = [col for col in essential_columns if col not in station_df.columns]

            if missing_essential:
                 console.print(f"     [yellow]WARNUNG: Essentielle Spalten {missing_essential} fehlen für Station {station_id}. Überspringe.[/yellow]")
                 continue # Nächste Station

            # Wähle nur benötigte, verfügbare Spalten aus
            station_df_filtered = station_df[available_cols].copy()

            # Füge zum Ergebnis hinzu
            all_station_data[station_id] = station_df_filtered
            successful_stations.append(station_id)
            console.print(f"     [green]Daten für {station_id} ({len(station_df_filtered)} Einträge) geladen.[/green]")

        except Exception as e:
            console.print(f"     [red]Fehler beim Laden/Verarbeiten für Station {station_id}: {e}[/red]")

    console.print(f"\nDaten erfolgreich geladen für {len(successful_stations)} von {len(station_ids)} angefragten Stationen: {successful_stations}")
    return all_station_data


def get_weather_data(
    location: Point,
    start_date: datetime,
    end_date: datetime,
    required_columns: list,
    essential_columns: list,
) -> pd.DataFrame:

    try:
        data_raw_request = Daily(location, start_date, end_date)
        data_raw = data_raw_request.fetch()
        print(f"Daten von {start_date.date()} bis {end_date.date()} für Berlin abgerufen.")
        print(f"Anzahl der Roh-Datensätze: {len(data_raw)}")
        print("Verfügbare Spalten in Rohdaten:", data_raw.columns.to_list())

        # Überprüfen, ob alle erforderlichen Spalten vorhanden sind
        available_columns = [col for col in required_columns if col in data_raw.columns]

        # Herausfinden, welche Spalten fehlen
        missing_columns = [col for col in required_columns if col not in data_raw.columns]

        missing_essential_columns = [col for col in essential_columns if col not in available_columns]

        # Warning ausgeben, wenn Spalten fehlen
        if missing_columns:
            print(f"\nWarnung: Folgende erwartete Spalten fehlen und werden ignoriert: {missing_columns}")

        # wenn essentielle Spalten fehlen, dann wird eine Exception ausgegeben
        if missing_essential_columns:
            raise ValueError(f"Fehlende essentielle Spalten: {missing_essential_columns}. Abbruch der Datenabfrage.")

        # DataFrame erstellen, der nur die Spalten enthält, die wir benötigen und die auch vorhanden sind
        data = data_raw[available_columns].copy()
        print(f"\nVerwende folgende Spalten für die Analyse: {available_columns}")

        if data.empty:
            raise ValueError("Keine Daten nach Spaltenauswahl verfügbar.")
        print("Datenerfassung abgeschlossen.")

        # DataFrame an main.py zurückgeben
        return data

    # falls bei der Abfrage ein Fehler auftritt
    except Exception as e:
        print(f"\nFehler beim Abrufen oder grundlegenden Verarbeiten der Rohdaten: {e}")
        raise
