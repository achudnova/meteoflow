import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel

import config
from plot_manager import save_plot


def evaluate_arima_on_test_set(
    y_train: pd.Series,
    y_test: pd.Series,
    target_col_name: str,
    order: tuple,
    console: Console,
    save_dir: str
) -> tuple[dict | None, pd.Series | None]:
    pass

    console.print(f"\n--- Evaluiere ARIMA({order}) für: [bold magenta]{target_col_name}[/bold magenta] ---")
    history = y_train.astype(float)

    try:
        # --- Training ---
        with console.status(f"[bold green]Trainiere ARIMA({order}) für {target_col_name}...[/]", spinner="dots"):
            model = ARIMA(history, order=order)
            model_fit = model.fit()
        console.print(f"   ✔️ ARIMA Training abgeschlossen.")

        # --- Vorhersage ---
        console.print(f"   Mache Vorhersage für Testzeitraum ({len(y_test)} Schritte)...")
        start_index = len(history)
        end_index = len(history) + len(y_test) - 1
        y_pred_arima = model_fit.predict(start=start_index, end=end_index)
        # Stelle sicher, dass der Index passt (manchmal gibt predict einen RangeIndex zurück)
        if not isinstance(y_pred_arima.index, pd.DatetimeIndex) and isinstance(y_test.index, pd.DatetimeIndex):
             y_pred_arima.index = y_test.index

        # --- Metriken ---
        true_values = y_test
        mae = mean_absolute_error(true_values, y_pred_arima)
        rmse = np.sqrt(mean_squared_error(true_values, y_pred_arima))
        r2 = r2_score(true_values, y_pred_arima)
        metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}

        metrics_str = f"    MAE:  [yellow]{mae:.2f}[/yellow]\n"
        metrics_str += f"    RMSE: [yellow]{rmse:.2f}[/yellow]\n"
        metrics_str += f"    R²:   [green]{r2:.2f}[/green]"
        console.print(Panel(metrics_str, title=f"ARIMA Metriken für {target_col_name}", border_style="magenta"))

        # --- Plot erstellen und speichern ---
        console.print(f"   Erstelle ARIMA Vergleichsplot...")
        try:
            plt.figure(figsize=(15, 6))
            plt.plot(y_test.index, y_test, label=f'Tatsächlich ({target_col_name})', alpha=0.7, marker='.', linestyle='None')
            plt.plot(y_pred_arima.index, y_pred_arima, label=f'ARIMA{order} Vorhersage', linestyle='--')
            # ... (Restliche Plot-Formatierung: Titel, Labels, Legende, Grid) ...
            plt.title(f'ARIMA Vorhersage vs. Tatsächlich: {target_col_name} (Testset)')
            plt.xlabel('Datum')
            y_label_base = target_col_name.split('_target')[0]
            unit = " (°C)" if "t" in y_label_base else (" (km/h)" if "wspd" in y_label_base else "")
            plt.ylabel(f"{y_label_base.capitalize()}{unit}")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()

            plot_filename = f'arima_evaluation_{target_col_name}.png'
            save_plot(plot_filename, save_dir) # Deine Speicherfunktion nutzen
        except Exception as plot_e:
             console.print(f"[bold red]   FEHLER beim Erstellen des ARIMA Plots: {plot_e}[/bold red]")
             plt.close('all')

        return metrics, y_pred_arima

    except Exception as e:
        console.print(f"[bold red]   FEHLER bei ARIMA Training/Evaluierung für {target_col_name}: {e}[/bold red]")
        console.print_exception(show_locals=False)
        return None, None


def predict_next_day_arima(
    y_full_history: pd.Series,
    target_col_name: str,
    order: tuple,
    console: Console
) -> float | None:
    
    console.print(f"\n--- Finales ARIMA({order}) für: [bold magenta]{target_col_name}[/bold magenta] ---")
    history = y_full_history.astype(float)

    try:
        with console.status(f"[bold green]Trainiere finales ARIMA({order}) für {target_col_name}...[/]", spinner="dots"):
            final_model = ARIMA(history, order=order)
            final_model_fit = final_model.fit()
        console.print(f"   ✔️ Finales ARIMA Training abgeschlossen.")

        forecast_result = final_model_fit.forecast(steps=1)
        next_day_pred = forecast_result.iloc[0]
        console.print(f"   ARIMA Vorhersage ({target_col_name}): [yellow]{next_day_pred:.2f}[/yellow]")
        return float(next_day_pred) # Explizit als float zurückgeben

    except Exception as e:
        console.print(f"[bold red]   FEHLER beim finalen ARIMA Training/Vorhersage für {target_col_name}: {e}[/bold red]")
        console.print_exception(show_locals=False)
        return None