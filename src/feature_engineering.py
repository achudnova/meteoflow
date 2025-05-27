import pandas as pd

def engineer_features(data: pd.DataFrame, target_cols: list, target_base_cols: list, lag_days: int) -> pd.DataFrame:
    data = data.copy()
    
    # sicherstellen, dass Zielspalten und Basisspalten übereinstimmen
    if len(target_cols) != len(target_base_cols):
        raise ValueError("target_cols und target_base_cols müssen die gleiche Länge haben.")
    
    demo_col = target_base_cols[0]
    print(f"\n--- Demonstration: Effekt von .shift(-1) für '{demo_col}' ---")
    print("Vorher (erste 5 Zeilen):")
    print(data[[demo_col]].head(5))
    
    # Zielspalten erstellen
    print("Erstelle Zielspalten...")
    for target, base in zip(target_cols, target_base_cols):
        if base in data.columns:
            data[target] = data[base].shift(-1) # Zielwert ist der Wert des nächsten Tages
        else:
            raise ValueError(f"Basisspalte '{base}' für Zielvariable '{target}' nicht gefunden.")
    
    # shift(-1)
    print(f"\nNachher (erste 5 Zeilen von '{demo_col}'):")
    print(data[[demo_col, target_cols[0]]].head(5))
    print("-------------------------------------------------------------\n")
    
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
    with pd.option_context(
        'display.max_columns', 13,      # 13 Spalten anzeigen
        'display.width', 2000,            
        'display.max_colwidth', None,     
        'display.max_rows', 10            # bis zu 10 Zeilen anzeigen
    ):
        print(data.head(10))
    print("\nDimensionen der aufbereiteten Daten:", data.shape)
    print("\nFeature Engineering abgeschlossen.")
    return data