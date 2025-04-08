import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from rich.table import Table 
from rich.console import Console

from plot_manager import save_plot

def start_eda(data: pd.DataFrame, plot_columns: list, save_dir: str, console: Console):
    if data.empty:
        print("DataFrame ist leer. EDA kann nicht durchgeführt werden.")
        return

    # Zeige die ersten und letzten Zeilen des Datensatzes
    print("\nErste 5 Zeilen der ausgewählten Daten:")
    print(data.head())
    print("\nLetzte 5 Zeilen der ausgewählten Daten:")
    print(data.tail())

    # Informationen über den DataFrame
    print("\nInformationen über den DataFrame (Typen, Nicht-Null-Werte):")
    data.info()

    # Deskriptive Statistiken
    print("\nDeskriptive Statistiken:")
    print(data.describe())

    # Überprüfen auf fehlende Werte
    print("\nFehlende Werte pro Spalte (vor der Datenverarbeitung):")
    missing_values = data.isnull().sum()
    missing_values_filtered = missing_values[missing_values > 0]
    if not missing_values_filtered.empty:
        print(missing_values_filtered)
    else:
        print("Keine fehlenden Werte gefunden.")

    # Überprüfen auf Duplikate
    duplicates = data.duplicated().sum()
    if duplicates > 0:
        print(f"\nAnzahl der Duplikate: {duplicates}")
    else:
        print("\nKeine Duplikate gefunden.")

    # TODO: Überprüfen auf Ausreißer
    print("\nÜberprüfung auf Ausreißer:")
    console.print("\n[cyan]Quantifizierung potenzieller Ausreißer (IQR-Methode)[/cyan]")
    outlier_table = Table(title="Potenzielle Ausreißer pro Variable (IQR * 1.5)")
    outlier_table.add_column("Variable", style="dim", width=12)
    outlier_table.add_column("Anzahl Ausreißer", justify="right")
    outlier_table.add_column("% Ausreißer", justify="right")
    outlier_table.add_column("Untere Grenze", justify="right")
    outlier_table.add_column("Obere Grenze", justify="right")

    total_rows = len(data)
    numeric_cols = data.select_dtypes(include=np.number).columns

    if total_rows > 0:
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Finde Ausreißer
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            num_outliers = len(outliers)
            percentage_outliers = (num_outliers / total_rows) * 100

            # Füge zur Tabelle hinzu, nur wenn Ausreißer gefunden wurden
            if num_outliers > 0:
                 outlier_table.add_row(
                     col,
                     str(num_outliers),
                     f"{percentage_outliers:.2f}%",
                     f"{lower_bound:.2f}",
                     f"{upper_bound:.2f}"
                 )

        if outlier_table.row_count > 0:
            console.print(outlier_table)
        else:
             console.print("   Keine potenziellen Ausreißer nach der IQR-Methode gefunden.")
    else:
         console.print("   Keine Daten zur Ausreißeranalyse vorhanden.")

    # Visualisierung der fehlenden Werte
    if data.isnull().sum().sum() > 0:
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
        plt.title("Muster der fehlenden Werte")
        save_plot('fehlende_werte.png', save_dir)
    else:
        print("\nKeine fehlenden Werte zum Visualisieren.")

    # Verteilung der einzelnen Merkmale/Variablen (Histogramme)
    print("\nVisualisierung der Verteilungen der Variablen:")
    try:
        data.hist(bins=30, figsize=(15, 10), layout=(-1, 3))
        plt.suptitle("Histogramme der Wettervariablen")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_plot('histogramm.png', save_dir)
    except Exception as e:
        print(f"Fehler beim Erstellen der Histogramme: {e}")

    # Zeitreihen-Plots für Schlüsselvariablen
    print("\nVisualisierung der Zeitreihen:")
    valid_plot_cols = [col for col in plot_columns if col in data.columns]
    if valid_plot_cols:
        num_plots = len(valid_plot_cols)
        plt.figure(figsize=(15, 3 * num_plots))
        for i, col in enumerate(valid_plot_cols):
            plt.subplot(num_plots, 1, i + 1)
            plt.plot(data.index, data[col], label=col)
            plt.title(f"Zeitlicher Verlauf von {col}", fontsize=10)
            plt.legend(loc="upper right")
            plt.grid(True, linestyle="--", alpha=0.6)
        plt.xlabel("Datum")
        plt.tight_layout()
        save_plot('zeitreihen_plots.png', save_dir)
    else:
        print("Keine der spezifizierten Spalten für Zeitreihen-Plots gefunden.")

    # Korrelationsmatrix (Beziehung zwischen den Variablen)
    print("\nKorrelationsmatrix:")
    if len(data.columns) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = data.corr()
        sns.heatmap(
            correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
        )
        plt.title("Korrelationsmatrix der Wettervariablen")
        save_plot('korrelationsmatrix.png', save_dir)
        print("\nKorrelationsmatrix:")
        print(correlation_matrix)
    else:
        print("Nicht genügend Spalten für Korrelationsmatrix vorhanden.")

    # Boxplots zur Erkennung von Ausreißern
    print("\nBoxplots zur Visualisierung von Verteilungen und Ausreißern:")
    if not data.empty:
        plt.figure(figsize=(15, 8))
        num_cols = len(data.columns)
        rows = 2 if num_cols > 1 else 1
        cols_per_row = (num_cols + rows - 1) // rows  # Ceiling division
        for i, col in enumerate(data.columns):
            plt.subplot(rows, cols_per_row, i + 1)
            sns.boxplot(y=data[col])
            plt.title(col, fontsize=10)
            plt.ylabel("")  # Y-Label entfernen für Kompaktheit
        plt.suptitle("Boxplots der Wettervariablen")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_plot('boxplots.png', save_dir)
    else:
        print("Keine Daten für Boxplots.")

    print("\nEDA abgeschlossen.")