import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def train_models(X_train: pd.DataFrame, y_train: pd.DataFrame, rf_parameter: dict, xgb_parameter: dict) -> dict:
    models = {}
    
    # ----- Random Forest -----
    print("Training RandomForestRegressor...")
    rf_model = RandomForestRegressor(**rf_parameter)
    rf_model.fit(X_train, y_train)
    models['rf'] = rf_model
    print("RandomForestRegressor trainiert.")
    
    # ----- XGBoost -----
    print("Training XGBoostRegressor...")
    xgb_model = XGBRegressor(**xgb_parameter)
    xgb_model.fit(X_train, y_train)
    models['xgb'] = xgb_model
    print("XGBoostRegressor trainiert.")
    
    print("Modelltraining abgeschlossen.")
    
    return models