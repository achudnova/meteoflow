import os
import sys
import config
import pandas as pd

from data_collection import get_weather_data
from eda import start_eda
from data_preprocessing import preprocess_data
from feature_engineering import engineer_feautures
from model_training import train_models
from model_manager import load_model

def main():
    # ----- 1. Datenerfassung -----
    print("1. Datenerfassung")
    try:
        data = get_weather_data(
            location=config.LOCATION,
            start_date=config.START_DATE,
            end_date=config.END_DATE,
            required_columns=config.REQUIRED_COLUMNS,
            essential_columns=config.ESSENTIAL_COLS,
        )
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        sys.exit(1)

    # ----- 2. Explorative Datenanalyse (EDA) -----
    print("\n2. Explorative Datenanalyse (EDA)")
    start_eda(data, plot_columns=config.EDA_PLOT_COLUMNS, save_dir=config.EDA_PLOT_DIR)

    # ----- 3. Datenvorverarbeitung -----
    print("\n3. Datenvorverarbeitung")
    preprocess_data(data)

    # ----- 4. Feature Engineering -----
    print("\n4. Feature Engineering")
    data_featured = engineer_feautures(
        data=data,
        target_cols=config.TARGET_COLUMNS,
        target_base_cols=config.ORIGINAL_TARGET_BASE_COLUMNS,
        lag_days=config.LAG_DAYS,
    )

    if data is None or data.empty:
        print("Nach dem Feature Engineering sind keine Daten mehr verfügbar.")
        sys.exit(1)

    # ----- 5. Train/Test Split -----
    print("\n5. Train/Test Split")
    # Feature Spalten definieren
    features_cols = [
        col
        for col in data_featured.columns
        if col not in config.TARGET_COLUMNS + config.ORIGINAL_TARGET_BASE_COLUMNS
    ]
    target_cols_present = [
        col for col in config.TARGET_COLUMNS if col in data_featured.columns
    ]  # Nur die tatsächlich erstellten Targets

    if not target_cols_present:
        print("Fehler: keine der Zielvariablen konnte erstellt werden.")
        sys.exit(1)
    if not features_cols:
        print("Fehler: keine Feature-Spalten gefunden.")
        sys.exit(1)

    X = data_featured[features_cols]
    y = data_featured[target_cols_present]  # nur existierende Targets verwenden

    if len(data_featured) <= config.TEST_PERIOD_DAYS:
        print(
            f"Nicht genügend Daten ({len(data_featured)} Zeilen) für einen sinnvollen Train/Test-Split mit {config.TEST_PERIOD_DAYS} Testtagen vorhanden."
        )
        print("Workflow wird abgebrochen.")
        sys.exit(1)

    split_index = len(data_featured) - config.TEST_PERIOD_DAYS
    split_date = data_featured.index[split_index]

    X_train = X[X.index < split_date]
    X_test = X[X.index >= split_date]
    y_train = y[y.index < split_date]
    y_test = y[y.index >= split_date]

    print(X_train.shape)
    print(f"Split-Datum: {split_date.date()}")
    print(f"Trainingsdaten: {X_train.shape[0]} Samples ({X_train.index.min().date()} bis {X_train.index.max().date()})")
    print(f"Testdaten: {X_test.shape[0]} Samples ({X_test.index.min().date()} bis {X_test.index.max().date()})")
    print(f"Anzahl Features: {X_train.shape[1]}")
    print(f"Zielvariablen: {target_cols_present}") # Zeige die tatsächlichen Zielvariablen an
    
     # Überprüfung des Split-Verhältnisses
    total_samples_after_engineering = X_train.shape[0] + X_test.shape[0]

    train_percentage = (X_train.shape[0] / total_samples_after_engineering) * 100
    test_percentage = (X_test.shape[0] / total_samples_after_engineering) * 100

    print(f"\nÜberprüfung des Split-Verhältnisses:")
    print(f"  Gesamte Samples nach Feature Engineering: {total_samples_after_engineering}")
    print(f"  Trainings-Anteil: {train_percentage:.2f}%")
    print(f"  Test-Anteil:      {test_percentage:.2f}%")
    
    print("Train/Test Split abgeschlossen.")
    
    # ----- 6. Modelltraining -----
    print("\n6. Modelltraining")
    train_models(X_train, y_train, config.RF_PARAMETER, config.XGB_PARAMETER, config.MODEL_SAVE_DIR)
    
if __name__ == "__main__":
    main()


# # 6. Modellbewertung
# #-------------------------------------------------------------------------------
# print("\n6. Modellbewertung")
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Vorhersagen auf dem Testset
# y_pred_rf = rf_model.predict(X_test)
# y_pred_xgb = xgb_model.predict(X_test)

# # Metriken berechnen
# results = {}
# models = {'RandomForest': y_pred_rf, 'XGBoost': y_pred_xgb}

# for model_name, y_pred in models.items():
#     print(f"\n--- {model_name} ---")
#     metrics = {}
#     for i, target in enumerate(target_cols):
#         true_values = y_test.iloc[:, i]
#         pred_values = y_pred[:, i]

#         mae = mean_absolute_error(true_values, pred_values)
#         rmse = np.sqrt(mean_squared_error(true_values, pred_values))
#         r2 = r2_score(true_values, pred_values)
#         metrics[target] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
#         print(f"  {target}:")
#         print(f"    MAE:  {mae:.2f}")
#         print(f"    RMSE: {rmse:.2f}")
#         print(f"    R²:   {r2:.2f}")
#     results[model_name] = metrics

# # Visualisierung der Vorhersagen vs. tatsächliche Werte (Temperatur)
# try:
#     tavg_target_index = target_cols.index('tavg_target')
#     plt.figure(figsize=(15, 6))
#     plt.plot(y_test.index, y_test.iloc[:, tavg_target_index], label='Tatsächliche Temperatur', alpha=0.7, marker='.', linestyle='None') # Punkte für Ist-Werte
#     plt.plot(y_test.index, y_pred_rf[:, tavg_target_index], label='RF Vorhersage', linestyle='--')
#     plt.plot(y_test.index, y_pred_xgb[:, tavg_target_index], label='XGBoost Vorhersage', linestyle=':')
#     plt.title('Temperaturvorhersage vs. Tatsächliche Werte (Testset)')
#     plt.xlabel('Datum')
#     plt.ylabel('Temperatur (°C)')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.tight_layout()
#     plt.show()
# except ValueError:
#     print("Temperatur ('tavg_target') nicht in den Zielvariablen gefunden, Plot wird übersprungen.")
# except IndexError:
#      print("Indexfehler beim Zugriff auf Vorhersagedaten für den Plot. Überprüfe die Dimensionen.")


# # 7. Vorhersage für den nächsten Tag
# #-------------------------------------------------------------------------------
# print("\n7. Vorhersage für den nächsten Tag")

# # Nimm die letzte verfügbare Zeile aus den *aufbereiteten Daten*
# last_available_data_row = data.iloc[-1:]

# # Extrahiere die Features für die Vorhersage
# features_for_prediction = last_available_data_row[features_cols]

# print("\nFeatures für die Vorhersage von morgen (basierend auf Daten vom {}):".format(last_available_data_row.index[0].date()))

# if features_for_prediction.isnull().sum().sum() > 0:
#     print("\nWarnung: Fehlende Werte in den Features für die Vorhersage entdeckt!")

# # Vorhersage treffen
# prediction_rf = rf_model.predict(features_for_prediction)
# prediction_xgb = xgb_model.predict(features_for_prediction)

# # Datum für die Vorhersage (morgen)
# prediction_date = data.index.max() + timedelta(days=1)

# print(f"\nVorhersage für {prediction_date.date()}:")

# # Finde Indizes der Zielvariablen dynamisch
# try:
#     tavg_idx = target_cols.index('tavg_target')
#     wspd_idx = target_cols.index('wspd_target')

#     print("--- RandomForest ---")
#     print(f"  Vorhergesagte Temperatur: {prediction_rf[0, tavg_idx]:.1f} °C")
#     print(f"  Vorhergesagte Windgeschwindigkeit: {prediction_rf[0, wspd_idx]:.1f} km/h")

#     print("--- XGBoost ---")
#     print(f"  Vorhergesagte Temperatur: {prediction_xgb[0, tavg_idx]:.1f} °C")
#     print(f"  Vorhergesagte Windgeschwindigkeit: {prediction_xgb[0, wspd_idx]:.1f} km/h")

# except ValueError as ve:
#     print(f"Fehler beim Extrahieren der Vorhersagewerte: {ve}")
# except IndexError:
#      print("Indexfehler beim Zugriff auf Vorhersagedaten. Überprüfe die Dimensionen der Vorhersage-Arrays.")


# print("\nProjekt abgeschlossen.")
