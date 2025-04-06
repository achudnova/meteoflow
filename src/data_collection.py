import pandas as pd
from datetime import datetime
from meteostat import Point, Daily


# Ruft Wetterdaten für Berlin ab und wählt benötigte Spalten aus
# Die Funktion bekommt alle Infos von außen -> config.py
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
        