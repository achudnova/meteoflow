import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    target_cols: list,
    plot_target_col: str,
):
    results = {}
    predictions = {}

    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        y_pred = model.predict(X_test)
        predictions[model_name] = y_pred
        metrics = {}

        # sicherstellen, dass y_pred 2D ist
        if y_pred.ndim == 1 and len(target_cols) == 1:
            y_pred = y_pred.reshape(-1, 1)
        elif y_pred.ndim == 1 and len(target_cols) > 1:
            print(
                f"WARNUNG: {model_name} gab eine 1D-Ausgabe zurück, obwohl mehrere Targets erwartet wurden. Überspringe Metrikberechnung."
            )
            continue  # Nächstes Modell

        # Metriken berechnen
        for i, target in enumerate(target_cols):
            if i < y_test.shape[1] and i < y_pred.shape[1]:
                true_values = y_test.iloc[:, i]
                pred_values = y_pred[:, i]

                mae = mean_absolute_error(true_values, pred_values)
                rmse = np.sqrt(mean_squared_error(true_values, pred_values))
                r2 = r2_score(true_values, pred_values)
                metrics[target] = {"MAE": mae, "RMSE": rmse, "R2": r2}
                print(f"  {target}:")
                print(f"    MAE:  {mae:.2f}")
                print(f"    RMSE: {rmse:.2f}")
                print(f"    R²:   {r2:.2f}")
            else:
                 print(f"Warnung: Index {i} für Zielvariable '{target}' außerhalb des gültigen Bereichs für y_test oder y_pred von {model_name}.")
        results[model_name] = metrics
        
        # Visualisierung der Vorhersagen vs. tatsächliche Werte (für eine ausgewählte Zielvariable)
        try:
            if plot_target_col not in target_cols:
                raise ValueError(f"Die zu plottende Zielvariable '{plot_target_col}' ist nicht in target_cols enthalten.")

            plot_target_index = target_cols.index(plot_target_col)

            plt.figure(figsize=(15, 6))
            # Stelle sicher, dass die Spalte im Testset existiert
            if plot_target_index < y_test.shape[1]:
                plt.plot(y_test.index, y_test.iloc[:, plot_target_index], label=f'Tatsächlich ({plot_target_col})', alpha=0.7, marker='.', linestyle='None')
            else:
                print(f"Warnung: Plot-Zielvariable '{plot_target_col}' (Index {plot_target_index}) nicht im y_test DataFrame gefunden.")

            # Plot für jedes Modell hinzufügen
            for model_name, y_pred in predictions.items():
                # Sicherstellen, dass y_pred existiert und die richtige Dimension hat
                if y_pred is not None and y_pred.ndim == 2 and plot_target_index < y_pred.shape[1]:
                    plt.plot(y_test.index, y_pred[:, plot_target_index], label=f'{model_name} Vorhersage', linestyle='--' if 'rf' in model_name.lower() else ':')
                elif y_pred is not None:
                    print(f"Warnung: Vorhersage-Daten für '{model_name}' für Plot ungeeignet (ndim={y_pred.ndim}, shape={y_pred.shape}, index={plot_target_index}).")


            plt.title(f'Vorhersage vs. Tatsächliche Werte für {plot_target_col} (Testset)')
            plt.xlabel('Datum')
            plt.ylabel(plot_target_col.split('_')[0].capitalize()) # Versuch, eine Einheit zu bekommen
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()

        except ValueError as ve:
            print(f"Fehler bei der Vorbereitung des Evaluationsplots: {ve}")
        except IndexError:
            print("Indexfehler beim Zugriff auf Vorhersagedaten für den Plot. Überprüfe die Dimensionen und Indizes.")
        except Exception as e:
            print(f"Allgemeiner Fehler beim Erstellen des Evaluationsplots: {e}")


        print("Modellbewertung abgeschlossen.")
