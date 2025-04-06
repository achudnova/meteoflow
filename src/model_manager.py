import os
import joblib
import sys

from rich.console import Console

def save_model(model, filepath: str) -> bool:
    if not filepath:
        print("\nFehler: Dateipfad ist leer.")
        return False
    try:
        save_dir = os.path.dirname(filepath)
        
        # Verzeichnis erstellen, falls es nicht existiert
        if save_dir and not os.path.exists(save_dir):
            print(f"\nErstelle Verzeichnis für ML-Modelle: {save_dir}")
            os.makedirs(save_dir, exist_ok=True)
            
        # Modelle speichern
        joblib.dump(model, filepath)
        print(f"\nModell erfolgreich gespeichert: {filepath}")
        return True
    
    except OSError as e:
        print(f"\nFehler: Konnte Verzeichnis für Modell nicht erstellen unter '{save_dir}': {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"\nFehler beim Speichern des Modells unter {filepath}: {e}", file=sys.stderr)
        return False

def load_model(filepath: str, console: Console):
    if not filepath:
        console.print("\nFehler: kein Dateipfad zum Laden des Modells angegeben.", file=sys.stderr)
        return None

    if not os.path.exists(filepath):
        console.print(f"\nModelldatei nicht gefunden unter {filepath}. Überspringe Laden.", file=sys.stderr)
        return None
    
    try:
        model = joblib.load(filepath)
        console.print(f"\nModell erfolgreich geladen von: {filepath}")
        return model
    except Exception as e:
        console.print(f"\nFehler beim Laden des Modells von {filepath}: {e}", file=sys.stderr)
        return None