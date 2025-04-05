import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn for advanced plots

# 1. Datenerfassung
#-------------------------------------------------------------------------------
print("1. Datenerfassung")
from meteostat import Point, Daily

# Standort: Berlin (ungefähre Koordinaten)
latitude = 52.5200
longitude = 13.4050
altitude = 34 # Höhe in Metern (ungefähr)
berlin = Point(latitude, longitude, altitude)

# Zeitraum für historische Daten (z.B. letzte 10 Jahre)
end = datetime.now()
# Vorsichtshalber Startdatum etwas weiter zurück, falls die letzten Tage fehlen
start = end - timedelta(days=10*365 + 5)

# Versuche, Daten abzurufen
try:
    data_raw = Daily(berlin, start, end)
    data_raw = data_raw.fetch()
    print(f"Daten von {start.date()} bis {end.date()} für Berlin abgerufen.")
    print(f"Anzahl der Roh-Datensätze: {len(data_raw)}")
    print("Verfügbare Spalten in Rohdaten:")
    print(data_raw.columns)
    # print("Erste paar Zeilen der Rohdaten:") # Wird später in EDA gezeigt
    # print(data_raw.head()) # Wird später in EDA gezeigt

    # --- Spaltenauswahl basierend auf Verfügbarkeit ---
    # Definiere die Spalten, die wir *wollen* und die *verfügbar sein müssen*
    # 'rhum' wurde im vorherigen Schritt als oft fehlend identifiziert
    required_cols = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres']
    # Potenzielle zusätzliche Features, falls vorhanden (nur zur Info hier)
    potential_feature_cols = ['snow', 'wdir', 'wpgt', 'tsun']

    # Überprüfe, ob die erforderlichen Spalten vorhanden sind
    available_cols = [col for col in required_cols if col in data_raw.columns]
    missing_cols = [col for col in required_cols if col not in data_raw.columns]

    if missing_cols:
        print(f"\nWarnung: Folgende erwartete Spalten fehlen und werden ignoriert: {missing_cols}")
        if 'tavg' not in available_cols or 'wspd' not in available_cols:
             raise ValueError("Essentielle Spalten 'tavg' oder 'wspd' fehlen.")

    # Wähle die tatsächlich verfügbaren, benötigten Spalten aus
    # Wir verwenden diese 'data' DataFrame für die EDA und die weitere Verarbeitung
    data = data_raw[available_cols].copy()
    print(f"\nVerwende folgende Spalten für die Analyse: {available_cols}")

    # Check ob überhaupt Daten vorhanden sind
    if data.empty:
        raise ValueError("Keine Daten nach Spaltenauswahl verfügbar.")


except Exception as e:
    print(f"\nFehler beim Abrufen oder grundlegenden Verarbeiten der Rohdaten: {e}")
    print("Stelle sicher, dass 'meteostat' installiert ist und eine Internetverbindung besteht.")
    exit() # Beende das Skript, wenn keine Daten vorhanden sind


# 1.5. Explorative Datenanalyse (EDA)
#-------------------------------------------------------------------------------
print("\n1.5. Explorative Datenanalyse (EDA)")

# Zeige die ersten und letzten paar Zeilen
print("\nErste 5 Zeilen der ausgewählten Daten:")
print(data.head())
print("\nLetzte 5 Zeilen der ausgewählten Daten:")
print(data.tail())

# Grundlegende Informationen über den DataFrame
print("\nDateninformationen (Typen, Nicht-Null-Werte):")
data.info()

# Deskriptive Statistiken für numerische Spalten
print("\nDeskriptive Statistiken:")
print(data.describe())

# Überprüfung auf fehlende Werte (vor der Vorverarbeitung)
print("\nFehlende Werte pro Spalte (vor der Vorverarbeitung):")
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0]) # Zeige nur Spalten mit fehlenden Werten

# Visualisierung der fehlenden Werte (Mustererkennung)
if data.isnull().sum().sum() > 0: # Nur plotten, wenn es fehlende Werte gibt
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title('Muster der fehlenden Werte')
    plt.show()
else:
    print("\nKeine fehlenden Werte in den ausgewählten Spalten gefunden.")

# Verteilung der einzelnen Variablen (Histogramme)
print("\nVisualisierung der Verteilungen der Variablen:")
data.hist(bins=30, figsize=(15, 10), layout=(-1, 3)) # Layout anpassen je nach Spaltenanzahl
plt.suptitle('Histogramme der Wettervariablen')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Platz für den Suptitel lassen
plt.show()

# Zeitreihen-Plots für Schlüsselvariablen
print("\nVisualisierung der Zeitreihen:")
plot_cols = ['tavg', 'wspd', 'prcp', 'pres'] # Wähle relevante Spalten zum Plotten
num_plots = len(plot_cols)
plt.figure(figsize=(15, 3 * num_plots))
for i, col in enumerate(plot_cols):
    if col in data.columns: # Nur plotten, wenn Spalte vorhanden ist
        plt.subplot(num_plots, 1, i + 1)
        plt.plot(data.index, data[col], label=col)
        plt.title(f'Zeitlicher Verlauf von {col}', fontsize=10)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel('Datum')
plt.tight_layout()
plt.show()

# Korrelationsmatrix (Beziehungen zwischen Variablen)
print("\nVisualisierung der Korrelationen zwischen Variablen:")
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Korrelationsmatrix der Wettervariablen')
plt.show()
print("\nKorrelationsmatrix:")
print(correlation_matrix)

# Boxplots zur Erkennung von Ausreißern (optional, aber nützlich)
print("\nBoxplots zur Visualisierung von Verteilungen und Ausreißern:")
plt.figure(figsize=(15, 8))
num_cols = len(data.columns)
for i, col in enumerate(data.columns):
    plt.subplot(2, (num_cols + 1) // 2, i + 1) # Erstellt ein Grid mit 2 Zeilen
    sns.boxplot(y=data[col])
    plt.title(col, fontsize=10)
    plt.ylabel('') # Y-Label entfernen für Kompaktheit
plt.suptitle('Boxplots der Wettervariablen')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\nEDA abgeschlossen.")

# --- Ab hier beginnt der ursprüngliche Ablauf ---

# 2. Datenvorverarbeitung
#-------------------------------------------------------------------------------
print("\n2. Datenvorverarbeitung")
print("Überprüfung auf fehlende Werte (vor Imputation - sollte mit EDA übereinstimmen):")
print(data.isnull().sum())

# Strategie für fehlende Werte: Vorwärtsfüllen (fill forward)
print("\nFülle fehlende Werte mit ffill und bfill...")
data.ffill(inplace=True)
# Falls nach ffill immer noch NaNs am Anfang vorhanden sind, füllen wir mit bfill
data.bfill(inplace=True)

print("\nÜberprüfung auf fehlende Werte (nach Imputation):")
print(data.isnull().sum())

if data.isnull().sum().sum() > 0:
    print("Warnung: Es gibt immer noch fehlende Werte nach ffill/bfill.")
    print("Entferne verbleibende Zeilen mit NaNs...")
    data.dropna(inplace=True)


# 3. Feature Engineering
#-------------------------------------------------------------------------------
# (Rest des Codes bleibt unverändert wie in der vorherigen Version)
print("\n3. Feature Engineering")

# Zielvariablen erstellen: Wetterbedingungen des *nächsten* Tages
data['tavg_target'] = data['tavg'].shift(-1)
data['wspd_target'] = data['wspd'].shift(-1)

# Spalten für Lag Features definieren (alle verfügbaren Input-Spalten)
lag_feature_cols = data.columns.tolist()
lag_feature_cols = [col for col in lag_feature_cols if not col.endswith('_target')]

print(f"Erstelle Lag-Features für Spalten: {lag_feature_cols}")

# Lag Features erstellen
for col in lag_feature_cols:
    for i in range(1, 4): # Wetter der letzten 3 Tage als Features
        data[f'{col}_lag_{i}'] = data[col].shift(i)

# Zeitbasierte Features
data['month'] = data.index.month
data['dayofyear'] = data.index.dayofyear
data['weekday'] = data.index.weekday

# Entfernen von Zeilen mit NaN-Werten, die durch shift() entstanden sind
data.dropna(inplace=True)

print("Daten nach Feature Engineering (erste paar Zeilen):")
print(data.head())
print("\nDimensionen der aufbereiteten Daten:", data.shape)


# 4. Train/Test Split (Chronologisch)
#-------------------------------------------------------------------------------
print("\n4. Train/Test Split")

target_cols = ['tavg_target', 'wspd_target']
original_target_base_cols = ['tavg', 'wspd']
features_cols = [col for col in data.columns if col not in target_cols + original_target_base_cols]

X = data[features_cols]
y = data[target_cols]

test_period_days = 2 * 365
if len(data) > test_period_days:
    split_index = len(data) - test_period_days
    split_date = data.index[split_index]

    X_train = X[X.index < split_date]
    X_test = X[X.index >= split_date]
    y_train = y[y.index < split_date]
    y_test = y[y.index >= split_date]

    print(f"Split-Datum: {split_date.date()}")
    print(f"Trainingsdaten: {X_train.shape[0]} Samples ({X_train.index.min().date()} bis {X_train.index.max().date()})")
    print(f"Testdaten: {X_test.shape[0]} Samples ({X_test.index.min().date()} bis {X_test.index.max().date()})")
    print(f"Anzahl Features: {X_train.shape[1]}")
else:
    print("Nicht genügend Daten für einen sinnvollen Train/Test-Split vorhanden.")
    exit()


# 5. Modelltraining
#-------------------------------------------------------------------------------
print("\n5. Modelltraining")
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# --- Random Forest ---
print("Training RandomForestRegressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, min_samples_split=5)
rf_model.fit(X_train, y_train)
print("RandomForestRegressor trainiert.")

# --- XGBoost ---
print("Training XGBoostRegressor...")
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1, learning_rate=0.1, max_depth=7)
xgb_model.fit(X_train, y_train)
print("XGBoostRegressor trainiert.")


# 6. Modellbewertung
#-------------------------------------------------------------------------------
print("\n6. Modellbewertung")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Vorhersagen auf dem Testset
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# Metriken berechnen
results = {}
models = {'RandomForest': y_pred_rf, 'XGBoost': y_pred_xgb}

for model_name, y_pred in models.items():
    print(f"\n--- {model_name} ---")
    metrics = {}
    for i, target in enumerate(target_cols):
        true_values = y_test.iloc[:, i]
        pred_values = y_pred[:, i]

        mae = mean_absolute_error(true_values, pred_values)
        rmse = np.sqrt(mean_squared_error(true_values, pred_values))
        r2 = r2_score(true_values, pred_values)
        metrics[target] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"  {target}:")
        print(f"    MAE:  {mae:.2f}")
        print(f"    RMSE: {rmse:.2f}")
        print(f"    R²:   {r2:.2f}")
    results[model_name] = metrics

# Visualisierung der Vorhersagen vs. tatsächliche Werte (Temperatur)
try:
    tavg_target_index = target_cols.index('tavg_target')
    plt.figure(figsize=(15, 6))
    plt.plot(y_test.index, y_test.iloc[:, tavg_target_index], label='Tatsächliche Temperatur', alpha=0.7, marker='.', linestyle='None') # Punkte für Ist-Werte
    plt.plot(y_test.index, y_pred_rf[:, tavg_target_index], label='RF Vorhersage', linestyle='--')
    plt.plot(y_test.index, y_pred_xgb[:, tavg_target_index], label='XGBoost Vorhersage', linestyle=':')
    plt.title('Temperaturvorhersage vs. Tatsächliche Werte (Testset)')
    plt.xlabel('Datum')
    plt.ylabel('Temperatur (°C)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
except ValueError:
    print("Temperatur ('tavg_target') nicht in den Zielvariablen gefunden, Plot wird übersprungen.")
except IndexError:
     print("Indexfehler beim Zugriff auf Vorhersagedaten für den Plot. Überprüfe die Dimensionen.")


# 7. Vorhersage für den nächsten Tag
#-------------------------------------------------------------------------------
print("\n7. Vorhersage für den nächsten Tag")

# Nimm die letzte verfügbare Zeile aus den *aufbereiteten Daten*
last_available_data_row = data.iloc[-1:]

# Extrahiere die Features für die Vorhersage
features_for_prediction = last_available_data_row[features_cols]

print("\nFeatures für die Vorhersage von morgen (basierend auf Daten vom {}):".format(last_available_data_row.index[0].date()))

if features_for_prediction.isnull().sum().sum() > 0:
    print("\nWarnung: Fehlende Werte in den Features für die Vorhersage entdeckt!")

# Vorhersage treffen
prediction_rf = rf_model.predict(features_for_prediction)
prediction_xgb = xgb_model.predict(features_for_prediction)

# Datum für die Vorhersage (morgen)
prediction_date = data.index.max() + timedelta(days=1)

print(f"\nVorhersage für {prediction_date.date()}:")

# Finde Indizes der Zielvariablen dynamisch
try:
    tavg_idx = target_cols.index('tavg_target')
    wspd_idx = target_cols.index('wspd_target')

    print("--- RandomForest ---")
    print(f"  Vorhergesagte Temperatur: {prediction_rf[0, tavg_idx]:.1f} °C")
    print(f"  Vorhergesagte Windgeschwindigkeit: {prediction_rf[0, wspd_idx]:.1f} km/h")

    print("--- XGBoost ---")
    print(f"  Vorhergesagte Temperatur: {prediction_xgb[0, tavg_idx]:.1f} °C")
    print(f"  Vorhergesagte Windgeschwindigkeit: {prediction_xgb[0, wspd_idx]:.1f} km/h")

except ValueError as ve:
    print(f"Fehler beim Extrahieren der Vorhersagewerte: {ve}")
except IndexError:
     print("Indexfehler beim Zugriff auf Vorhersagedaten. Überprüfe die Dimensionen der Vorhersage-Arrays.")


print("\nProjekt abgeschlossen.")