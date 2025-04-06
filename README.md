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

### data_preprocessing.py

> die Funktion bereinigt die Daten und gibt den DataFrame zurück

Args:
- data: Pandas DataFrame mit den bereinigten Daten

### feature_engineering.py

> die Funktion erstellt Zielvariablen, Lag-Features und zeitbasierte Features.

Args:
- data: vorverarbeiteter Pandas DataFrame
- target_cols: Liste der Namen für die Zielvariablen (`tavg_target`, `wspd_target`)
- target_base_cols: Liste der Originalspalten, aus denen Targets erstellt werden (`tavg`, `wspd`)
- lag_days: Anzahl der Tage für Lag-Features
  
Returns:
- Pandas DataFrame mit den neuen Features
- oder None, wenn nicht genügend Daten für Lags vorhanden sind

### main.py (Train/Test Split)



### model_training.py


### model_manager.py


### model_evaluation.py

> Wie genau sind die Vorhersagen im Vergleich zur Realität, wenn das Modell auf Daten trifft, die es noch nie gesehen hat?"

