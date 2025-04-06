import pandas as pd

def engineer_feautures(data: pd.DataFrame, target_cols: list, target_base_cols: list, lag_days: int) -> pd.DataFrame:
    data = data.copy()
    
    # sicherstellen, dass Zielspalten und Basisspalten übereinstimmen
    if len(target_cols) != len(target_base_cols):
        raise ValueError("target_cols und target_base_cols müssen die gleiche Länge haben.")
    
    # Zielspalten erstellen
    print("Erstelle Zielspalten...")
    for target, base in zip(target_cols, target_base_cols):
        if base in data.columns:
            data[target] = data[base].shift(-1) # Zielwert ist der Wert des nächsten Tages
        else:
            raise ValueError(f"Basisspalte '{base}' für Zielvariable '{target}' nicht gefunden.")
    
    # Spalten für Lag Features definieren
    all_feature_cols = [col for col in data.columns if col in data.columns]
    print(f"\nErstelle Lag-Features für Spalten: {all_feature_cols}")
    
    # Lag Features erstellen
    for col in all_feature_cols:
        for i in range(1, lag_days + 1):
            data[f'{col}_lag_{i}'] = data[col].shift(i)

    # Zeitbasierte Features
    print("\nErstelle zeitbasierte Features: Monat, Tag des Jahres, Wochentag...")
    data['month'] = data.index.month
    data['dayofyear'] = data.index.dayofyear
    data['weekday'] = data.index.weekday
    
    # Entfernen von Zeilen mit NaN-Werten, die durch shift() entstanden sind
    initial_rows = len(data)
    data.dropna(inplace=True)
    rows_dropped = initial_rows - len(data)
    print(f"\n{rows_dropped} Zeilen mit NaN-Werten entfernt.")
    
    if data.empty:
        raise ValueError("\nNach Feature Engineering sind keine Daten mehr verfügbar.")
        return None # es sind keine Daten vorhanden

    print("\nDaten nach Feature Engineering (erste paar Zeilen):")
    print(data.head())
    print("\nDimensionen der aufbereiteten Daten:", data.shape)
    print("\nFeature Engineering abgeschlossen.")
    return data