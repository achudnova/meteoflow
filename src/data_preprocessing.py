import pandas as pd
import numpy as np
from rich.console import Console

def preprocess_data(data: pd.DataFrame, console: Console) -> pd.DataFrame:
    print("Überpüfung auf fehlende Werte (vor Imputation)")
    print(data.isnull().sum())
    
    # Kopie des DataFrames erstellen, um Originaldaten nicht zu ändern
    data_copy = data.copy()
    
    # Strategie für fehlende Werte: Vorwärtsfüllen (fill forward), dann rückwärts
    if data_copy.isnull().sum().sum() > 0:
        print("\nFülle fehlende Werte mit ffill und bfill...")
        data_copy.ffill(inplace=True)
        # Falls nach ffill immer noch NaNs am Anfang vorhanden sind, füllen wir mit bfill
        data_copy.bfill(inplace=True)
        
        print("\nÜberprüfung auf fehlende Werte (nach Imputation):")
        print(data_copy.isnull().sum())
    
    return data_copy