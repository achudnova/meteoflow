# prediction.py
import pandas as pd
from datetime import timedelta


def predict_next_day(
    models: dict,
    last_available_data_row: pd.DataFrame,
    features_cols: list,
    target_cols: list,
):
    """
    Macht eine Vorhersage für den nächsten Tag basierend auf den letzten verfügbaren Daten.

    Args:
        models: Dictionary mit den trainierten Modellen {'rf': rf_model, 'xgb': xgb_model}.
        last_available_data_row: Ein DataFrame mit der letzten Zeile der aufbereiteten Daten.
        features_cols: Liste der Feature-Namen, die das Modell erwartet.
        target_cols: Liste der Namen der Zielvariablen.
    """

    if last_available_data_row.empty:
        print("Keine Daten für die Vorhersage verfügbar.")
        return

    # Extrahiere die Features für die Vorhersage
    # Sicherstellen, dass alle benötigten Feature-Spalten vorhanden sind
    missing_features = [
        col for col in features_cols if col not in last_available_data_row.columns
    ]
    if missing_features:
        print(f"FEHLER: Fehlende Features für die Vorhersage: {missing_features}")
        return

    features_for_prediction = last_available_data_row[features_cols]

    last_data_date = last_available_data_row.index[0].date()
    print(
        f"\nFeatures für die Vorhersage von morgen (basierend auf Daten vom {last_data_date}):"
    )
    # print(features_for_prediction.iloc[0]) # Kann sehr lang sein

    if features_for_prediction.isnull().sum().sum() > 0:
        print("\nWarnung: Fehlende Werte in den Features für die Vorhersage entdeckt!")
        print(features_for_prediction.isnull().sum())
        # Hier könnte man versuchen, die Lücken zu füllen, aber das ist riskant
        print("Vorhersage wird übersprungen wegen fehlender Feature-Werte.")
        return

    # Datum für die Vorhersage (morgen)
    prediction_date = last_available_data_row.index.max() + timedelta(days=1)
    print(f"\nVorhersage für {prediction_date.date()}:")

    # Finde Indizes der Zielvariablen dynamisch
    try:
        tavg_idx = (
            target_cols.index("tavg_target") if "tavg_target" in target_cols else -1
        )
        wspd_idx = (
            target_cols.index("wspd_target") if "wspd_target" in target_cols else -1
        )
    except ValueError:
        print(
            "Fehler: Zielspalten 'tavg_target' oder 'wspd_target' nicht in der target_cols Liste gefunden."
        )
        tavg_idx, wspd_idx = -1, -1  # Sicherstellen, dass Indizes ungültig sind

    for model_name, model in models.items():
        try:
            prediction = model.predict(features_for_prediction)
            # Sicherstellen, dass prediction 2D ist
            if prediction.ndim == 1:
                prediction = prediction.reshape(1, -1)

            print(f"--- {model_name} ---")
            if tavg_idx != -1 and tavg_idx < prediction.shape[1]:
                print(f"  Vorhergesagte Temperatur: {prediction[0, tavg_idx]:.1f} °C")
            else:
                print("  Temperaturvorhersage nicht verfügbar/gefunden.")

            if wspd_idx != -1 and wspd_idx < prediction.shape[1]:
                print(
                    f"  Vorhergesagte Windgeschwindigkeit: {prediction[0, wspd_idx]:.1f} km/h"
                )
            else:
                print("  Windgeschwindigkeitsvorhersage nicht verfügbar/gefunden.")

        except ValueError as ve:
            print(f"Fehler bei der Vorhersage mit {model_name}: {ve}")
        except IndexError:
            print(
                f"Indexfehler beim Zugriff auf Vorhersagedaten von {model_name}. Überprüfe die Dimensionen des Vorhersage-Arrays (shape: {prediction.shape}) und die Zielspalten-Indizes (tavg_idx={tavg_idx}, wspd_idx={wspd_idx})."
            )
        except Exception as e:
            print(f"Allgemeiner Fehler bei der Vorhersage mit {model_name}: {e}")

    print("\nVorhersage abgeschlossen.")
