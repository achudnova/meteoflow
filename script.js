document.addEventListener('DOMContentLoaded', function() {
    // Hole die HTML-Elemente, die wir aktualisieren wollen
    const forecastDiv = document.getElementById('forecast');
    const lastUpdateSpan = document.getElementById('last-update');

    // Pfad zur JSON-Datei (liegt im selben Verzeichnis wie index.html)
    const dataUrl = 'prediction.json';

    // Füge einen Zeitstempel zur URL hinzu, um Browser-Caching zu umgehen
    // Das stellt sicher, dass wir immer die neueste Version der Datei anfordern
    const uniqueUrl = `${dataUrl}?t=${new Date().getTime()}`;

    console.log(`Versuche Daten zu laden von: ${uniqueUrl}`); // Log zum Debuggen

    // Starte den Fetch-Vorgang, um die JSON-Datei zu laden
    fetch(uniqueUrl)
        .then(response => {
            // Überprüfe zuerst, ob die Anfrage erfolgreich war (HTTP-Status 200-299)
            if (!response.ok) {
                // Wenn nicht OK, werfe einen Fehler mit dem HTTP-Status
                throw new Error(`HTTP Fehler! Status: ${response.status} - ${response.statusText}`);
            }
            // Wenn OK, parse den Body der Antwort als JSON
            return response.json();
        })
        .then(data => {
            // Daten wurden erfolgreich als JSON geparst
            console.log("Rohdaten erfolgreich geladen:", data); // Log zum Debuggen

            // Überprüfe, ob die erwarteten Daten vorhanden sind
            if (!data || !data.forecast_date) {
                throw new Error("Vorhersagedaten sind ungültig, leer oder das Datum fehlt.");
            }

            // Funktion zum sicheren Formatieren der Werte (gibt 'N/A' zurück bei null/undefined)
            const formatValue = (value, unit = '') => {
                if (value !== null && value !== undefined) {
                    // Prüfe, ob es eine Zahl ist, bevor toFixed aufgerufen wird
                    if (typeof value === 'number') {
                         // Runde auf eine Nachkommastelle und füge Einheit hinzu
                        return `${value.toFixed(1)}${unit}`;
                    } else {
                        // Wenn es kein null/undefined ist, aber keine Zahl, gib es so aus
                        return `${value}${unit}`;
                    }
                }
                return 'N/A'; // Gib 'N/A' zurück für null oder undefined
            };

            // Formatiere die einzelnen Vorhersagewerte sicher
            const rfTemp = formatValue(data.rf_temp_c, '°C');
            const rfWind = formatValue(data.rf_wspd_kmh, ' km/h');
            const xgbTemp = formatValue(data.xgb_temp_c, '°C');
            const xgbWind = formatValue(data.xgb_wspd_kmh, ' km/h');

            // Erstelle das HTML, um die formatierten Daten anzuzeigen
            forecastDiv.innerHTML = `
                <h2>Vorhersage für: ${data.forecast_date}</h2>
                <div class="prediction-box">
                    <h3>Random Forest</h3>
                    <p>Temperatur: <strong>${rfTemp}</strong></p>
                    <p>Windgeschw.: <strong>${rfWind}</strong></p>
                </div>
                <div class="prediction-box">
                    <h3>XGBoost</h3>
                    <p>Temperatur: <strong>${xgbTemp}</strong></p>
                    <p>Windgeschw.: <strong>${xgbWind}</strong></p>
                </div>
            `;

            // Aktualisiere den Zeitstempel des letzten Updates
            try {
                // Versuche, das Datum aus dem ISO-Format zu parsen
                const updateDate = new Date(data.generated_at);
                // Formatiere es lesbar für deutsche Spracheinstellungen
                lastUpdateSpan.textContent = updateDate.toLocaleString('de-DE', {
                    timeZone: 'Europe/Berlin',
                    dateStyle: 'medium', // z.B. 06.04.2025
                    timeStyle: 'medium'  // z.B. 16:43:25
                });
            } catch (e) {
                // Falls das Datumsformat ungültig ist
                console.error("Fehler beim Parsen des Update-Datums:", e);
                lastUpdateSpan.textContent = 'Unbekannt';
            }
        })
        .catch(error => {
            // Fängt Fehler vom fetch() oder aus den .then()-Blöcken ab
            console.error('Fehler im Fetch-Vorgang oder bei der Datenverarbeitung:', error);

            // Zeige eine Fehlermeldung auf der Webseite an
            forecastDiv.innerHTML = `<p class="error">Fehler: Vorhersage konnte nicht geladen oder verarbeitet werden. (${error.message})</p>`;
            lastUpdateSpan.textContent = 'Fehler';
        });
});