import pandas as pd
import numpy as np
from rich.console import Console
from scipy.stats import mstats

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
    
    # console.print("\n   Behandle potenzielle Ausreißer durch Clipping (IQR * 1.5)...")
    # numeric_cols = data_copy.select_dtypes(include=np.number).columns
    # clipped_count = {} # Zähle, wie viele Werte pro Spalte geändert wurden

    # for col in numeric_cols:
    #     Q1 = data_copy[col].quantile(0.25)
    #     Q3 = data_copy[col].quantile(0.75)
    #     IQR = Q3 - Q1

    #     # Definiere Grenzen (robust gegen NaN im IQR)
    #     if pd.notna(IQR) and IQR > 0: # Nur clippen, wenn IQR gültig ist
    #         lower_bound = Q1 - 1.5 * IQR
    #         upper_bound = Q3 + 1.5 * IQR

    #         # Zähle Werte vor dem Clipping
    #         lower_outliers_before = (data_copy[col] < lower_bound).sum()
    #         upper_outliers_before = (data_copy[col] > upper_bound).sum()
    #         total_outliers_before = lower_outliers_before + upper_outliers_before

    #         if total_outliers_before > 0:
    #             # Wende Clipping an
    #             data_copy[col] = data_copy[col].clip(lower=lower_bound, upper=upper_bound)
    #             clipped_count[col] = total_outliers_before # Speichere Anzahl geänderter Werte
    #             console.print(f"      Variable '{col}': {total_outliers_before} Werte auf Grenzen [{lower_bound:.2f}, {upper_bound:.2f}] gesetzt.")
    #     else:
    #         # Wenn IQR Null ist (konstante Spalte) oder NaN, nicht clippen
    #          if pd.notna(IQR) and IQR == 0:
    #              console.print(f"      Variable '{col}': Übersprungen (konstanter Wert).")
    #          # Fall pd.isna(IQR) sollte durch NaN-Handling vorher unwahrscheinlich sein


    # if clipped_count:
    #     console.print("      ✔️ Ausreißerbehandlung (Clipping) abgeschlossen.")
    # else:
    #     console.print("      Keine Ausreißer zum Clippen gefunden.")


    # console.print("\n[green]✔️ Datenvorverarbeitung abgeschlossen.[/green]")
    
    console.print("\n   Behandle potenzielle Ausreißer durch Winsorizing (5%/95% Perzentil)...")
    numeric_cols = data_copy.select_dtypes(include=np.number).columns
    winsorized_info = {} # Speichere Infos über geänderte Spalten

    for col in numeric_cols:
        # Ignoriere Spalten, die fast konstant sind oder nur NaNs enthalten nach Vorverarbeitung
        if data_copy[col].nunique() < 2 or data_copy[col].isnull().all():
            console.print(f"      Variable '{col}': Übersprungen (konstant oder nur NaNs).")
            continue

        # Kopiere die Originaldaten der Spalte für den Vergleich
        original_col_data = data_copy[col].copy()

        # Wende Winsorizing an
        # limits=(0.05, 0.05) bedeutet: untere 5% auf 5. Perzentil, obere 5% auf 95. Perzentil
        # nan_policy='omit' ignoriert NaNs bei der Perzentilberechnung (sollte nach Schritt 1 nicht mehr nötig sein)
        try:
            winsorized_data = mstats.winsorize(data_copy[col].dropna(), limits=(0.05, 0.05))

            # Zähle, wie viele Werte sich geändert haben
            # Wir müssen die Indizes angleichen, da dropna() Indizes entfernt haben könnte
            original_valid = original_col_data.dropna()
            if len(original_valid) == len(winsorized_data): # Nur vergleichen, wenn Längen passen
                 num_changed = np.sum(original_valid.values != winsorized_data) # Vergleiche NumPy-Arrays
            else:
                 num_changed = -1 # Kann nicht direkt verglichen werden (sollte nicht passieren)

            # Weise die winsorisierten Daten zurück zu (berücksichtige ursprüngliche NaNs)
            # Fülle die winsorisierten Werte zurück in den ursprünglichen Index
            data_copy.loc[original_valid.index, col] = winsorized_data

            if num_changed > 0:
                p5 = np.percentile(original_valid, 5)
                p95 = np.percentile(original_valid, 95)
                winsorized_info[col] = num_changed
                console.print(f"      Variable '{col}': {num_changed} Werte auf Grenzen [{p5:.2f}, {p95:.2f}] (5./95. Perzentil) gesetzt.")
            elif num_changed == 0:
                console.print(f"      Variable '{col}': Keine Werte durch Winsorizing geändert.")
            else: # num_changed == -1
                 console.print(f"      Variable '{col}': Winsorizing angewendet, aber Vergleich der Änderungen nicht möglich (Längenunterschied).")


        except Exception as e:
             console.print(f"      [red]FEHLER[/red] beim Winsorizing für Variable '{col}': {e}")


    if winsorized_info:
        console.print("      ✔️ Ausreißerbehandlung (Winsorizing) abgeschlossen.")
    else:
        console.print("      Keine Werte durch Winsorizing geändert oder Fehler aufgetreten.")

    console.print("\n[green]✔️ Datenvorverarbeitung abgeschlossen.[/green]")
    
    return data_copy