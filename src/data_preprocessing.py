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
    
    console.print("\n   Behandle potenzielle Ausreißer durch Clipping (IQR * 1.5)...")
    numeric_cols = data_copy.select_dtypes(include=np.number).columns
    clipped_count = {} # Zähle, wie viele Werte pro Spalte geändert wurden

    for col in numeric_cols:
        Q1 = data_copy[col].quantile(0.25)
        Q3 = data_copy[col].quantile(0.75)
        IQR = Q3 - Q1

        # Definiere Grenzen (robust gegen NaN im IQR)
        if pd.notna(IQR) and IQR > 0: # Nur clippen, wenn IQR gültig ist
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Zähle Werte vor dem Clipping
            lower_outliers_before = (data_copy[col] < lower_bound).sum()
            upper_outliers_before = (data_copy[col] > upper_bound).sum()
            total_outliers_before = lower_outliers_before + upper_outliers_before

            if total_outliers_before > 0:
                # Wende Clipping an
                data_copy[col] = data_copy[col].clip(lower=lower_bound, upper=upper_bound)
                clipped_count[col] = total_outliers_before # Speichere Anzahl geänderter Werte
                console.print(f"      Variable '{col}': {total_outliers_before} Werte auf Grenzen [{lower_bound:.2f}, {upper_bound:.2f}] gesetzt.")
        else:
            # Wenn IQR Null ist (konstante Spalte) oder NaN, nicht clippen
             if pd.notna(IQR) and IQR == 0:
                 console.print(f"      Variable '{col}': Übersprungen (konstanter Wert).")
             # Fall pd.isna(IQR) sollte durch NaN-Handling vorher unwahrscheinlich sein


    if clipped_count:
        console.print("      ✔️ Ausreißerbehandlung (Clipping) abgeschlossen.")
    else:
        console.print("      Keine Ausreißer zum Clippen gefunden.")


    console.print("\n[green]✔️ Datenvorverarbeitung abgeschlossen.[/green]")
    
    return data_copy