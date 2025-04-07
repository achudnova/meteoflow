import os
import sys
import config
import pandas as pd

from data_collection import get_weather_data
from eda import start_eda
from data_preprocessing import preprocess_data
from feature_engineering import engineer_features
from data_splitting import split_data
from model_training import train_models
from model_evaluation import evaluate_model
from prediction import predict_next_day
from arima_pipeline import evaluate_arima_on_test_set, predict_next_day_arima
from sarima_pipeline import evaluate_sarima_on_test_set, predict_next_day_sarima
# from .model_manager import load_model

from rich.console import Console
from rich.panel import Panel

console = Console()

def main():
    console.rule("[bold purple4]⛅ Wettervorhersage für Berlin ⛅[/bold purple4]")
    
    # ----- 1. Datenerfassung -----
    console.rule("[orange1]1. Datenerfassung[/orange1]")
    try:
        data = get_weather_data(
            location=config.LOCATION,
            start_date=config.START_DATE,
            end_date=config.END_DATE,
            required_columns=config.REQUIRED_COLUMNS,
            essential_columns=config.ESSENTIAL_COLS,
        )
    except Exception as e:
        console.print(f"[red] Ein Fehler ist aufgetreten: [/red] {e}")
        sys.exit(1)

    # ----- 2. Explorative Datenanalyse (EDA) -----
    console.rule("[orange1]1.5 Explorative Datenanalyse (EDA)[/orange1]")
    start_eda(data, plot_columns=config.EDA_PLOT_COLUMNS, save_dir=config.EDA_PLOT_DIR)

    # ----- 3. Datenvorverarbeitung -----
    console.rule("[orange1]2. Datenvorverarbeitung[/orange1]")
    data_processed = preprocess_data(data)
    if data_processed is None: # Prüfen ob erfolgreich
        console.print("[red] Fehler während der Datenvorverarbeitung, Abbruch! [/red]")
        sys.exit(1)

    # ----- 4. Feature Engineering -----
    console.rule("[orange1]3. Feature Engineering[/orange1]")
    data_featured = engineer_features(
        data=data_processed,
        target_cols=config.TARGET_COLUMNS,
        target_base_cols=config.ORIGINAL_TARGET_BASE_COLUMNS,
        lag_days=config.LAG_DAYS,
    )

    if data is None or data.empty:
        console.print(f"[red] Nach dem Feature Engineering sind keine Daten mehr verfügbar [/red]")
        sys.exit(1)

    # ----- 5. Train/Test Split -----
    console.rule("[orange1]4. Train/Test Split[/orange1]")
    try:
        X_train, X_test, y_train, y_test, features_cols, target_cols_present, split_date, train_percentage, test_percentage = split_data(
            data_featured, console
        )
        
    except Exception as e:
        console.print(f"[red] Fehler während des Train/Test Splits: {e} [/red]")
        sys.exit(1)


    # ----- 6. Modelltraining -----
    console.rule("[orange1]5. Modelltraining[/orange1]")
    trained_models = train_models(
        X_train,
        y_train,
        config.RF_PARAMETER,
        config.XGB_PARAMETER,
        config.MODEL_SAVE_DIR,
    )
    
    # console.rule("[cyan]5.5 ARIMA Evaluierung[/cyan]")
    # all_arima_results = {}
    # for target_col in target_cols_present:
    #      metrics, _ = evaluate_arima_on_test_set( # Vorhersagen werden nicht direkt gebraucht
    #          y_train=y_train[target_col],
    #          y_test=y_test[target_col],
    #          target_col_name=target_col,
    #          order=config.DEFAULT_ARIMA_ORDER, 
    #          console=console,
    #          save_dir=config.EDA_PLOT_DIR 
    #      )
    #      all_arima_results[target_col] = metrics
    
    console.rule("[cyan]5.5 SARIMA Evaluierung[/cyan]")
    all_sarima_results = {} # Umbenannt
    for target_col in target_cols_present:
         # Rufe die SARIMA-Funktion auf
         metrics, _ = evaluate_sarima_on_test_set(
             y_train=y_train[target_col],
             y_test=y_test[target_col],
             target_col_name=target_col,
             order=config.DEFAULT_ORDER, # Nicht-saisonal
             seasonal_order=config.DEFAULT_SEASONAL_ORDER, # Saisonal
             console=console,
             save_dir=config.EDA_PLOT_DIR # Oder EVAL_PLOTS_DIR
         )
         all_sarima_results[target_col] = metrics

    # ----- 7. Modellbewertung -----
    console.rule("[orange1]6. Modellbewertung[/orange1]")
    evaluate_model(
        models=trained_models,
        X_test=X_test,
        y_test=y_test,
        target_cols=target_cols_present,
        save_dir=config.EDA_PLOT_DIR
    )

    # ----- 8. Vorhersage für den nächsten Tag -----
    console.rule("[reverse green]7. Vorhersage für den nächsten Tag[/reverse green]")
    last_available_data_row = data_featured.iloc[-1:]
    
    console.print("\n[bold yellow]--- DEBUG: Features für Vorhersage aus main.py ---[/bold yellow]")
    console.print(f"Letzter Datenpunkt Index (main.py): {last_available_data_row.index[0]}")
    
    console.print("Feature-Werte (main.py):")
    console.print(last_available_data_row[features_cols].iloc[0].to_dict()) # Zeige Werte als Dictionary
    console.print("[bold yellow]-----------------------------------------------------[/bold yellow]\n")
    
    predict_next_day(
        models=trained_models,
        last_available_data_row=last_available_data_row,
        features_cols=features_cols,  # Die Liste der Feature-Namen
        target_cols=target_cols_present,  # Die Liste der Ziel-Namen
    )
    
    # console.rule("[cyan]7.5 ARIMA Vorhersage für nächsten Tag[/cyan]")
    # arima_next_day_predictions = {}
    # next_day_date = y_test.index.max() + pd.Timedelta(days=1)
    # panel_content = ""
    
    # for target_col in target_cols_present:
    #     # Baue die gesamte Historie für das finale Training
    #     full_history = pd.concat([y_train[target_col], y_test[target_col]])
    #     pred_value = predict_next_day_arima(
    #         y_full_history=full_history,
    #         target_col_name=target_col,
    #         order=config.DEFAULT_ARIMA_ORDER, # Verwende Default oder aus Config
    #         console=console
    #     )
    #     arima_next_day_predictions[target_col] = pred_value

    #     # Füge zum Panel-Inhalt hinzu
    #     label = "Temperatur" if "tavg" in target_col else "Windgeschw."
    #     unit = "°C" if "tavg" in target_col else "km/h"
    #     style = "yellow" if "tavg" in target_col else "cyan"
    #     panel_content += f"  Vorhergesagte {label}: [bold {style}]{pred_value:.1f}{unit}[/bold {style}]\n" if pred_value is not None else f"  {label}-Vorhersage: N/A\n"

    # # Gib das Panel aus
    # console.print(Panel(panel_content.strip(), title=f"ARIMA Vorhersage für {next_day_date.date()}", border_style="magenta", expand=False, padding=(1, 2)))
    console.rule("[cyan]7.5 SARIMA Vorhersage für nächsten Tag[/cyan]")
    sarima_next_day_predictions = {} # Umbenannt
    next_day_date = y_test.index.max() + pd.Timedelta(days=1)
    panel_content = ""

    for target_col in target_cols_present:
        full_history = pd.concat([y_train[target_col], y_test[target_col]])
        # Rufe die SARIMA-Funktion auf
        pred_value = predict_next_day_sarima(
            y_full_history=full_history,
            target_col_name=target_col,
            order=config.DEFAULT_ORDER, # Nicht-saisonal
            seasonal_order=config.DEFAULT_SEASONAL_ORDER, # Saisonal
            console=console
        )
        sarima_next_day_predictions[target_col] = pred_value

        # Füge zum Panel hinzu (wie zuvor)
        label = "Temperatur" if "tavg" in target_col else "Windgeschw."
        unit = "°C" if "tavg" in target_col else "km/h"
        style = "yellow" if "tavg" in target_col else "cyan"
        panel_content += f"  Vorhergesagte {label}: [bold {style}]{pred_value:.1f}{unit}[/bold {style}]\n" if pred_value is not None else f"  {label}-Vorhersage: N/A\n"

    # Gib das Panel aus
    console.print(Panel(panel_content.strip(), title=f"SARIMA Vorhersage für {next_day_date.date()}", border_style="magenta", expand=False, padding=(1, 2)))
    
    console.print("\n[bold blue]🎉 Wettervorhersage Workflow Abgeschlossen 🎉[/bold blue]")


if __name__ == "__main__":
    main()
