import pandas as pd
import numpy as np
# Importiere SARIMAX statt ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
import warnings # Um statsmodels-Warnungen ggf. zu unterdrücken

# Importiere deine Hilfsfunktionen und Config
import config # Für SARIMA Parameter, falls gewünscht
from plot_manager import save_plot # Stelle sicher, dass der Importpfad stimmt

# --- Konstanten für SARIMA (Beispiel) ---
# p, d, q: Nicht-saisonale Ordnung (wie bei ARIMA)
# DEFAULT_ORDER = (2, 1, 1) # Beispiel, muss optimiert werden
# P, D, Q: Saisonale Ordnung
# s: Saisonale Periodenlänge (365 für jährliche Saison bei Tagesdaten)
# DEFAULT_SEASONAL_ORDER = (1, 1, 0, 365) # Beispiel, muss optimiert werden! s=365 ist sehr rechenintensiv!

def evaluate_sarima_on_test_set(
    y_train: pd.Series,
    y_test: pd.Series,
    target_col_name: str,
    order: tuple,
    seasonal_order: tuple, # Zusätzlicher Parameter für SARIMA
    console: Console,
    save_dir: str
) -> tuple[dict | None, pd.Series | None]:
    """
    Trainiert ein SARIMA-Modell auf y_train, evaluiert auf y_test und erstellt einen Plot.

    Args:
        y_train: Trainings-Zeitreihe.
        y_test: Test-Zeitreihe.
        target_col_name: Name der Zielvariable.
        order: Die nicht-saisonale ARIMA (p,d,q) Ordnung.
        seasonal_order: Die saisonale (P,D,Q,s) Ordnung.
        console: Rich Console Objekt.
        save_dir: Verzeichnis zum Speichern des Plots.

    Returns:
        Ein Tuple: (metrics_dict, y_pred_sarima) oder (None, None) bei Fehler.
    """
    model_str = f"SARIMA({order})({seasonal_order})" # Für Logs/Titel
    console.print(f"\n--- Evaluiere {model_str} für: [bold magenta]{target_col_name}[/bold magenta] ---")
    history = y_train.astype(float)

    # Sicherstellen, dass der Index ein DatetimeIndex mit Frequenz ist (wichtig für SARIMAX)
    # Versuche, eine Tagesfrequenz zu setzen, falls nicht vorhanden
    if not isinstance(history.index, pd.DatetimeIndex):
         console.print("[yellow]Warnung: Trainingsdaten haben keinen DatetimeIndex. Versuche Umwandlung.[/yellow]")
         try:
             history.index = pd.to_datetime(history.index)
         except Exception:
              console.print("[red]FEHLER: Konnte Index nicht in DatetimeIndex umwandeln.[/red]")
              return None, None
    if history.index.freq is None:
        console.print("[yellow]Warnung: DatetimeIndex hat keine Frequenz. Setze auf 'D' (Tage).[/yellow]")
        history = history.asfreq('D')
        # Fülle evtl. entstandene Lücken (optional, je nach Daten)
        history = history.ffill().bfill()


    try:
        # --- Training ---
        # Unterdrücke Konvergenz-Warnungen von statsmodels, die oft auftreten
        warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels')
        warnings.filterwarnings("ignore", category=RuntimeWarning, module='statsmodels')

        with console.status(f"[bold green]Trainiere {model_str} für {target_col_name}...[/]", spinner="dots"):
            # Verwende SARIMAX statt ARIMA
            model = SARIMAX(history,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False, # Oft besser bei komplexen Modellen
                            enforce_invertibility=False)
            model_fit = model.fit(disp=False) # disp=False unterdrückt Konvergenz-Infos
        warnings.resetwarnings() # Warnungen wieder aktivieren
        console.print(f"   ✔️ {model_str} Training abgeschlossen.")

        # --- Vorhersage ---
        console.print(f"   Mache Vorhersage für Testzeitraum ({len(y_test)} Schritte)...")
        start_index = len(history)
        end_index = len(history) + len(y_test) - 1
        # Verwende predict - bei SARIMAX ist der Index oft korrekt
        y_pred_sarima = model_fit.predict(start=start_index, end=end_index)

        # Index-Anpassung (wie bei ARIMA)
        if not isinstance(y_pred_sarima.index, pd.DatetimeIndex) and isinstance(y_test.index, pd.DatetimeIndex):
             y_pred_sarima.index = y_test.index

        # --- Metriken ---
        true_values = y_test
        mae = mean_absolute_error(true_values, y_pred_sarima)
        rmse = np.sqrt(mean_squared_error(true_values, y_pred_sarima))
        r2 = r2_score(true_values, y_pred_sarima)
        metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}

        metrics_str = f"    MAE:  [yellow]{mae:.2f}[/yellow]\n"
        metrics_str += f"    RMSE: [yellow]{rmse:.2f}[/yellow]\n"
        metrics_str += f"    R²:   [green]{r2:.2f}[/green]"
        console.print(Panel(metrics_str, title=f"{model_str} Metriken für {target_col_name}", border_style="magenta"))

        # --- Plot erstellen und speichern ---
        console.print(f"   Erstelle {model_str} Vergleichsplot...")
        try:
            plt.figure(figsize=(15, 6))
            plt.plot(y_test.index, y_test, label=f'Tatsächlich ({target_col_name})', alpha=0.7, marker='.', linestyle='None')
            plt.plot(y_pred_sarima.index, y_pred_sarima, label=f'{model_str} Vorhersage', linestyle='--')
            # ... (Restliche Plot-Formatierung) ...
            plt.title(f'{model_str} Vorhersage vs. Tatsächlich: {target_col_name} (Testset)')
            plt.xlabel('Datum')
            y_label_base = target_col_name.split('_target')[0]
            unit = " (°C)" if "t" in y_label_base else (" (km/h)" if "wspd" in y_label_base else "")
            plt.ylabel(f"{y_label_base.capitalize()}{unit}")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()

            plot_filename = f'sarima_evaluation_{target_col_name}.png' # Angepasster Name
            save_plot(plot_filename, save_dir)
        except Exception as plot_e:
             console.print(f"[bold red]   FEHLER beim Erstellen des {model_str} Plots: {plot_e}[/bold red]")
             plt.close('all')

        return metrics, y_pred_sarima

    except Exception as e:
        console.print(f"[bold red]   FEHLER bei {model_str} Training/Evaluierung für {target_col_name}: {e}[/bold red]")
        console.print_exception(show_locals=False)
        return None, None


def predict_next_day_sarima(
    y_full_history: pd.Series,
    target_col_name: str,
    order: tuple,
    seasonal_order: tuple, # Zusätzlicher Parameter
    console: Console
) -> float | None:
    """
    Trainiert ein finales SARIMA-Modell und gibt die Vorhersage zurück.
    """
    model_str = f"SARIMA({order})({seasonal_order})"
    console.print(f"\n--- Finales {model_str} für: [bold magenta]{target_col_name}[/bold magenta] ---")
    history = y_full_history.astype(float)

    # Index prüfen und Frequenz setzen (wie oben)
    if not isinstance(history.index, pd.DatetimeIndex):
         try: history.index = pd.to_datetime(history.index)
         except Exception: return None # Fehler bei Indexumwandlung
    if history.index.freq is None:
        history = history.asfreq('D').ffill().bfill()

    try:
        warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels')
        warnings.filterwarnings("ignore", category=RuntimeWarning, module='statsmodels')
        with console.status(f"[bold green]Trainiere finales {model_str} für {target_col_name}...[/]", spinner="dots"):
            final_model = SARIMAX(history,
                                  order=order,
                                  seasonal_order=seasonal_order,
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
            final_model_fit = final_model.fit(disp=False)
        warnings.resetwarnings()
        console.print(f"   ✔️ Finales {model_str} Training abgeschlossen.")

        # Verwende forecast statt predict für Vorhersage nach Trainingsende
        forecast_result = final_model_fit.forecast(steps=1)
        next_day_pred = forecast_result.iloc[0]
        console.print(f"   {model_str} Vorhersage ({target_col_name}): [yellow]{next_day_pred:.2f}[/yellow]")
        return float(next_day_pred)

    except Exception as e:
        console.print(f"[bold red]   FEHLER beim finalen {model_str} Training/Vorhersage für {target_col_name}: {e}[/bold red]")
        console.print_exception(show_locals=False)
        return None