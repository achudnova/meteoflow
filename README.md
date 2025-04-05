# METEOFLOW

### data_collection.py

> Die Funktion `get_weather_data` nimmt einige Argumente entgegen (...) und es wird erwartet, dass sie ein pandas DataFrame-Objekt mit den abgerufenen und ausgewählten Daten zurückgibt

Funktion: `get_weather_data(location: Point, start_date: datetime, end_date: datetime, required_cols: list, essential_cols: list) -> pd.DataFrame:` - ruft Wetterdaten für einen gegebenen Standort und Zeitraum ab und wählt benötigte Spalten aus

Args:
- location: meteostat Point Objekt des Standorts; kommt aus config.LOCATION
- start_date: Startdatum für den Datenabruf; von wann? -> kommt aus config.START_DATE
- end_date: Enddatum für den Datenabruf; bis wann? -> kommt aus config.END_DATE
- required_cols: Liste der gewünschten Spaltennamen; was wollen wir? -> kommt aus config.REQUIRED_COLUMNS
- essential_cols: Liste der Spaltennamen, die zwingend vorhanden sein müssen; kommt aus config.ESSENTIAL_COLUMNS

Returns:
- -> pd.DataFrame ist ein Typ-Hinweis (Type Hint) für den Rückgabewert einer Funktion in Python

### eda.py

> die Funktion führt eine explorative Datenanalyse dür den gegebenen DataFrame durch

Args:
- data: der Pandas-DataFrame (aus data_collection.py -> get_weather_data())
- plot_columns: Liste der Spaltennamen für die Zeitreihenplots
- save_dir: das Verzeichnis, in dem die Plots gespeichert werden sollen