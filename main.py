import os
import sys
import config
import pandas as pd

from data_collection import get_weather_data
from eda import start_eda
from data_preprocessing import preprocess_data
from feature_engineering import engineer_feautures

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
    data = engineer_feautures(
        data=data,
        target_cols=config.TARGET_COLUMNS,
        target_base_cols=config.ORIGINAL_TARGET_BASE_COLUMNS,
        lag_days=config.LAG_DAYS
    )
    
    if data is None or data.empty:
        print("Nach dem Feature Engineering sind keine Daten mehr verfügbar.")
        sys.exit(1)


if __name__ == "__main__":
    main()

# # 2. Datenvorverarbeitung
# #-------------------------------------------------------------------------------
# print("\n2. Datenvorverarbeitung")
# print("Überprüfung auf fehlende Werte (vor Imputation - sollte mit EDA übereinstimmen):")
# print(data.isnull().sum())

# # Strategie für fehlende Werte: Vorwärtsfüllen (fill forward)
# print("\nFülle fehlende Werte mit ffill und bfill...")
# data.ffill(inplace=True)
# # Falls nach ffill immer noch NaNs am Anfang vorhanden sind, füllen wir mit bfill
# data.bfill(inplace=True)

# print("\nÜberprüfung auf fehlende Werte (nach Imputation):")
# print(data.isnull().sum())

# if data.isnull().sum().sum() > 0:
#     print("Warnung: Es gibt immer noch fehlende Werte nach ffill/bfill.")
#     print("Entferne verbleibende Zeilen mit NaNs...")
#     data.dropna(inplace=True)


# # 3. Feature Engineering
# #-------------------------------------------------------------------------------
# # (Rest des Codes bleibt unverändert wie in der vorherigen Version)
# print("\n3. Feature Engineering")

# # Zielvariablen erstellen: Wetterbedingungen des *nächsten* Tages
# data['tavg_target'] = data['tavg'].shift(-1)
# data['wspd_target'] = data['wspd'].shift(-1)

# # Spalten für Lag Features definieren (alle verfügbaren Input-Spalten)
# lag_feature_cols = data.columns.tolist()
# lag_feature_cols = [col for col in lag_feature_cols if not col.endswith('_target')]

# print(f"Erstelle Lag-Features für Spalten: {lag_feature_cols}")

# # Lag Features erstellen
# for col in lag_feature_cols:
#     for i in range(1, 4): # Wetter der letzten 3 Tage als Features
#         data[f'{col}_lag_{i}'] = data[col].shift(i)

# # Zeitbasierte Features
# data['month'] = data.index.month
# data['dayofyear'] = data.index.dayofyear
# data['weekday'] = data.index.weekday

# # Entfernen von Zeilen mit NaN-Werten, die durch shift() entstanden sind
# data.dropna(inplace=True)

# print("Daten nach Feature Engineering (erste paar Zeilen):")
# print(data.head())
# print("\nDimensionen der aufbereiteten Daten:", data.shape)


# # 4. Train/Test Split (Chronologisch)
# #-------------------------------------------------------------------------------
# print("\n4. Train/Test Split")

# target_cols = ['tavg_target', 'wspd_target']
# original_target_base_cols = ['tavg', 'wspd']
# features_cols = [col for col in data.columns if col not in target_cols + original_target_base_cols]

# X = data[features_cols]
# y = data[target_cols]

# test_period_days = 2 * 365
# if len(data) > test_period_days:
#     split_index = len(data) - test_period_days
#     split_date = data.index[split_index]

#     X_train = X[X.index < split_date]
#     X_test = X[X.index >= split_date]
#     y_train = y[y.index < split_date]
#     y_test = y[y.index >= split_date]

#     print(f"Split-Datum: {split_date.date()}")
#     print(f"Trainingsdaten: {X_train.shape[0]} Samples ({X_train.index.min().date()} bis {X_train.index.max().date()})")
#     print(f"Testdaten: {X_test.shape[0]} Samples ({X_test.index.min().date()} bis {X_test.index.max().date()})")
#     print(f"Anzahl Features: {X_train.shape[1]}")
# else:
#     print("Nicht genügend Daten für einen sinnvollen Train/Test-Split vorhanden.")
#     exit()


# # 5. Modelltraining
# #-------------------------------------------------------------------------------
# print("\n5. Modelltraining")
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor

# # --- Random Forest ---
# print("Training RandomForestRegressor...")
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, min_samples_split=5)
# rf_model.fit(X_train, y_train)
# print("RandomForestRegressor trainiert.")

# # --- XGBoost ---
# print("Training XGBoostRegressor...")
# xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1, learning_rate=0.1, max_depth=7)
# xgb_model.fit(X_train, y_train)
# print("XGBoostRegressor trainiert.")


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
