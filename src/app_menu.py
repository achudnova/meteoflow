import os
import sys
import subprocess
from rich.console import Console

console = Console()

MODEL_SAVE_DIR = "../saved_models"
predict_script = "update_prediction_data.py"


def main_menu():
    while True:
        console.print("\n[bold cyan]MeteoFlow App Menü[/bold cyan]")
        console.print("1. Modell trainieren")
        console.print("2. Gespeicherte Modelle für die Vorhersage benutzen")
        console.print("3. Programm beenden")
        choice = input("\nBitte wähle eine Option (1-3): ")

        if choice == "1":
            main_script_path = "main.py"
            console.print("\n[green]Starte Training...[/green]\n")
            try:
                subprocess.run([sys.executable, main_script_path], check=False)
            except Exception as e:
                console.print(f"[red]Fehler beim Ausführen von main.py: {e}[/red]")
        elif choice == "2":
            console.print(f"\n[green]Starte Vorhersage")
            try:
                subprocess.run([sys.executable, predict_script], check=False)
            except Exception as e:
                console.print(f"[red]Fehler beim Ausführen von {predict_script}: {e}[/red]")
        elif choice == "3":
            console.print("[bold magenta]Programm wird beendet.[/bold magenta]")
            sys.exit(0)
        else:
            console.print("[red]Ungültige Eingabe. Bitte wähle 1, 2 oder 3.[/red]")

if __name__ == "__main__":
    main_menu()