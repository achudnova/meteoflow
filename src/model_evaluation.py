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
