import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from plot_manager import save_plot


def evaluate_model(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    target_cols: list,
    # plot_target_col: str,
    save_dir: str,
):
    results = {}
    # predictions = {}

    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        # predictions[model_name] = y_pred
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            print(f"FEHLER bei Vorhersage mit {model_name}: {e}. Überspringe Modell.")
            continue

        # sicherstellen, dass y_pred 2D ist
        if y_pred.ndim == 1 and len(target_cols) == 1:
            y_pred = y_pred.reshape(-1, 1)
        elif y_pred.ndim == 1 and len(target_cols) > 1:
            print(
                f"WARNUNG: {model_name} gab eine 1D-Ausgabe zurück, obwohl mehrere Targets erwartet wurden. Überspringe Metrikberechnung."
            )
            continue  # Nächstes Modell

        metrics = {}

        for i, target in enumerate(target_cols):
            if not (i < y_test.shape[1] and i < y_pred.shape[1]):
                print(
                    f"Warnung: Index {i} für Zielvariable '{target}' ungültig für {model_name}. Überspringe Metriken & Plot."
                )
                continue

            true_values = y_test.iloc[:, i]
            pred_values = y_pred[:, i]

            # Metriken berechnen
            try:
                mae = mean_absolute_error(true_values, pred_values)
                rmse = np.sqrt(mean_squared_error(true_values, pred_values))
                r2 = r2_score(true_values, pred_values)
                metrics[target] = {"MAE": mae, "RMSE": rmse, "R2": r2}
                print(f"  {target}:")
                print(f"    MAE:  {mae:.2f}")
                print(f"    RMSE: {rmse:.2f}")
                print(f"    R²:   {r2:.2f}")
            except Exception as e:
                print(f"FEHLER bei Metrikberechnung für {model_name} - {target}: {e}")

            print(f"\nErstelle Plot für {model_name} - {target}...")
            try:
                plt.figure(figsize=(15, 6))
                plt.plot(
                    y_test.index,
                    true_values,
                    label=f"Tatsächlich ({target})",
                    alpha=0.7,
                    marker=".",
                    linestyle="None",
                )
                plt.plot(
                    y_test.index,
                    pred_values,
                    label=f"{model_name} Vorhersage",
                    linestyle="-",
                )
                plt.title(
                    f"Vorhersage vs. Tatsächlich: {model_name} - {target} (Testset)"
                )
                plt.xlabel("Datum")
                y_label_base = target.split("_target")[0]  # Basisnamen extrahieren
                unit = ""
                if (
                    "tavg" in y_label_base
                    or "tmin" in y_label_base
                    or "tmax" in y_label_base
                ):
                    unit = " (°C)"
                elif "wspd" in y_label_base:
                    unit = " (km/h)"
                elif "prcp" in y_label_base:
                    unit = " (mm)"  # Annahme für Niederschlag
                elif "pres" in y_label_base:
                    unit = " (hPa)"  # Annahme für Druck
                plt.ylabel(f"{y_label_base.capitalize()}{unit}")
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                filename = f"evaluation_{model_name}_{target}.png"
                save_plot(filename, save_dir)
            except Exception as e:
                print(
                    f"FEHLER beim Erstellen/Speichern des Plots für {model_name} - {target}: {e}"
                )
                plt.close("all")

        results[model_name] = metrics  # Speichere Metriken für das Modell

    print("Modellbewertung abgeschlossen.")

def create_temperature_time_series(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    models: dict,
    target_col_idx: int,
    target_cols: list,
    save_dir: str,
):
    """
    Erstellt eine Zeitreihen-Grafik der Durchschnittstemperatur mit drei Farben:
    - Schwarz für Trainingsdaten
    - Blau für Testdaten
    - Rot für Vorhersagen
    
    Args:
        X_train: Trainings-Features
        y_train: Trainings-Targets
        X_test: Test-Features
        y_test: Test-Targets
        models: Dictionary mit trainierten Modellen
        target_col_idx: Index der Zielspalte (Temperatur) in den Target-Arrays
        target_cols: Liste der Zielspalten-Namen
        save_dir: Verzeichnis zum Speichern der Plots
    """
    if target_col_idx >= len(target_cols):
        print(f"FEHLER: Ungültiger Target-Index {target_col_idx} für {target_cols}")
        return
    
    target_col = target_cols[target_col_idx]
    if "tavg" not in target_col and "tmin" not in target_col and "tmax" not in target_col:
        print(f"WARNUNG: Die ausgewählte Zielspalte '{target_col}' scheint keine Temperaturspalte zu sein.")
    
    # Extrahiere die tatsächlichen Werte
    y_train_values = y_train.iloc[:, target_col_idx]
    y_test_values = y_test.iloc[:, target_col_idx]
    
    for model_name, model in models.items():
        try:
            # Vorhersagen für Testdaten
            y_pred_test = model.predict(X_test)
            if y_pred_test.ndim == 1:
                y_pred_test = y_pred_test.reshape(-1, 1)
            
            # Vorhersagen für Trainingsdaten (für Vergleichszwecke)
            y_pred_train = model.predict(X_train)
            if y_pred_train.ndim == 1:
                y_pred_train = y_pred_train.reshape(-1, 1)
            
            # Erstelle den Plot
            plt.figure(figsize=(15, 8))
            
            # Trainingsdaten (schwarz)
            plt.plot(
                y_train.index, 
                y_train_values, 
                'k-', 
                label='Trainingsdaten', 
                alpha=0.7
            )
            
            # Testdaten (blau)
            plt.plot(
                y_test.index, 
                y_test_values, 
                'b-', 
                label='Testdaten', 
                alpha=0.7
            )
            
            # Vorhersagen (rot)
            plt.plot(
                y_test.index, 
                y_pred_test[:, target_col_idx], 
                'r-', 
                label=f'{model_name} Vorhersage', 
                alpha=0.9
            )
            
            # Beschriftungen und Layout
            plt.title(f'Durchschnittstemperatur Zeitreihe: {model_name} - {target_col}')
            plt.xlabel('Zeit')
            plt.ylabel('Temperatur (°C)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            # Speichern des Plots
            filename = f"temperature_time_series_{model_name}_{target_col}.png"
            save_plot(filename, save_dir)
        except Exception as e:
            print(f"FEHLER beim Erstellen/Speichern des Plots für {model_name} - {target_col}: {e}")
            plt.close("all")