from datetime import datetime, timedelta
from meteostat import Point
import os

# ----- Standort: Berlin -----
TARGET_LAT = 52.5200
TARGET_LON = 13.4050
SEARCH_RADIUS_KM = 30
MAX_NEARBY_STATIONS = 4

LATITUDE = 52.5200
LONGITUDE = 13.4050
ALTITUDE = 34 # Höhe in Metern 
LOCATION = Point(LATITUDE, LONGITUDE, ALTITUDE)

# ----- Zeitraum für historische Wetterdaten -----
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(20*365 + 5)  # 20 Jahre + 5 Tage Puffer

# ----- Spaltenauswahl -----
# diese Spalten sind für die Analyse erforderlich
REQUIRED_COLUMNS = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres']

# Potenzielle zusätzliche Features, falls vorhanden (nur zur Info hier)
POTENTIAL_COLUMNS = ['snow', 'wdir', 'wpgt', 'tsun']

ESSENTIAL_COLS = ['tavg', 'wspd']

# ----- Feature Engineering -----
TARGET_COLUMNS = ['tavg_target', 'wspd_target']
ORIGINAL_TARGET_BASE_COLUMNS = ['tavg', 'wspd'] # Originalspalten, die zu Targets werden
LAG_DAYS = 5 # Anzahl der Lag-Tage

# ----- Train/Test Daten -----
TEST_PERIOD_DAYS = 4 * 365 # Tage für den Testdatensatz

# ----- Modellparameter -----
RANDOM_STATE = 42

# Random Forest
RF_PARAMETER = {
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
    'n_jobs': 1,
    'max_depth': 15,
    'min_samples_split': 5
}

# XGBoost
XGB_PARAMETER = {
    'objective': 'reg:squarederror',
    'n_estimators': 1000,
    'random_state': RANDOM_STATE,
    'n_jobs': 1,
    'learning_rate': 0.03,
    'max_depth': 3
}

# TODO: LightGBM
# LGBM = {
#     'objective': 'regression',
#     'n_estimators': 100,
#     'random_state': RANDOM_STATE,
#     'n_jobs': 1,
#     'learning_rate': 0.1,
#     'max_depth': 7
# }

# ----- Plotting -----
EDA_PLOT_COLUMNS = ['tavg', 'wspd', 'prcp', 'pres'] # Spalten für Zeitreihenplots
EVAL_PLOT_TARGET_COLUMN = 'tavg_target' # Zielspalte für Evaluierungsplots

# ----- Pfade -----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

EDA_PLOT_DIR = os.path.join(_PROJECT_ROOT, "plots")
MODEL_SAVE_DIR = os.path.join(_PROJECT_ROOT, "saved_models")