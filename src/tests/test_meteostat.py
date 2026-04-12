from datetime import datetime
from meteostat import Daily

weatherstations = {
    'Berlin / Tempelhof': '10384',
    'Berlin / Tegel': '10382',
    'Berlin / Schönefeld': '10385',
    'Berlin / Dahlem': '10381',
    'Berlin / Alexanderplatz': '10389'
}

# The timeframe we want to check (March 10 to today)
start = datetime(2026, 4, 10)
end = datetime.now()  # 2026-04-12

for station_name, station_id in weatherstations.items():
    print(f"\nDaten für Station: {station_name} (ID: {station_id})")
    print(f"Hole Daten von {start.date()} bis {end.date()}...")
    df = Daily(station_id, start, end).fetch()

    if df.empty:
        print("  -> Keine Daten gefunden!")
    else:
        cols = [c for c in ['tavg', 'tmin', 'tmax', 'prcp'] if c in df.columns]
        print(f"  -> Letzter verfügbarer Datenpunkt: {df.index.max().date()}")
        print("  -> Die letzten 5 Zeilen:")
        print(df[cols].tail())