import os
import joblib
import sys

from rich.console import Console

def save_model(model, filepath: str, console: Console) -> bool:
    if not filepath:
        console.print("\nFehler: Dateipfad ist leer.", style="bold red")
        return False
    try:
        save_dir = os.path.dirname(filepath)
        
        # Verzeichnis erstellen, falls es nicht existiert
        if save_dir and not os.path.exists(save_dir):
            console.print(f"\nErstelle Verzeichnis für ML-Modelle: {save_dir}")
            os.makedirs(save_dir, exist_ok=True)
            
        # Modelle speichern
        joblib.dump(model, filepath)
        console.print(f"\nModell erfolgreich gespeichert: {filepath}")
        return True
    
    except OSError as e:
        console.print(f"\nFehler: Konnte Verzeichnis für Modell nicht erstellen unter '{save_dir}': {e}")
        return False
    except Exception as e:
        console.print(f"\nFehler beim Speichern des Modells unter {filepath}: {e}")
        return False

def load_model(filepath: str, console: Console):
    if not filepath:
        console.print("\nFehler: kein Dateipfad zum Laden des Modells angegeben.")
        return None

    if not os.path.exists(filepath):
        console.print(f"\nModelldatei nicht gefunden unter {filepath}. Überspringe Laden.")
        return None
    
    try:
        model = joblib.load(filepath)
        console.print(f"\nModell erfolgreich geladen von: {filepath}")
        return model
    except Exception as e:
        console.print(f"\nFehler beim Laden des Modells von {filepath}: {e}")
        return None