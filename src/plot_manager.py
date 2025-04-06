import matplotlib.pyplot as plt
import os
import sys

def save_plot(filename: str, save_dir: str):
    if not save_dir:
        print("Kein Speicherverzeichnis angegeben. Plot wird nicht gespeichert", file=sys.stderr)
        try:
            plt.close(plt.gcf())
        except Exception:
            pass
        return
    
    try:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
        
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Plot erfolgreich gespeichert: {filepath}")
    
    except OSError as e:
        print(f"Fehler: konnte Verzeichnis '{save_dir}' nicht erstellen oder darauf zugreifen: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Fehler beim Speichern des Plots: {e}", file=sys.stderr)
    finally:
        try:
            plt.close(plt.gcf())
        except Exception:
            pass