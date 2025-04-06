import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import os

from model_manager import save_model

def train_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    rf_parameter: dict,
    xgb_parameter: dict,
    save_dir: str,
) -> dict:
    models = {}

    # ----- Random Forest -----
    print("Training RandomForestRegressor...")
    rf_model = RandomForestRegressor(**rf_parameter)
    rf_model.fit(X_train, y_train)
    models["rf"] = rf_model
    print("\nRandomForestRegressor trainiert.")
    
    if save_dir:
        rf = os.path.join(save_dir, 'rf_model.joblib')
        save_model(rf_model, rf) 
    else:
        print("\nKein Speicherverzeichnis angegeben, RandomForest-Modell wird nicht gespeichert.")
    
    # ----- XGBoost -----
    print("Training XGBoostRegressor...")
    xgb_model = XGBRegressor(**xgb_parameter)
    xgb_model.fit(X_train, y_train)
    models["xgb"] = xgb_model
    print("\nXGBoostRegressor trainiert.")
    
    if save_dir:
        xgb = os.path.join(save_dir, 'xgb_model.joblib')
        save_model(xgb_model, xgb) 
    else:
        print("\nKein Speicherverzeichnis angegeben, RandomForest-Modell wird nicht gespeichert.")

    print("\nModelltraining abgeschlossen!")

    return models
