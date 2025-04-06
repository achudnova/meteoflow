document.addEventListener('DOMContentLoaded', function() {
    const forecastDiv = document.getElementById('forecast');
    const lastUpdateSpan = document.getElementById('last-update');

    // Pfad zur JSON-Datei (relativ zur index.html)
    const dataUrl = 'prediction.json';

    fetch(dataUrl)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP Fehler! Status: ${response.status}`);
            }
            // Cache umgehen, um sicherzustellen, dass wir die neueste Datei bekommen
            // Fügen Sie einen zufälligen Query-Parameter hinzu
            const uniqueUrl = `${dataUrl}?t=${new Date().getTime()}`;
            return fetch(uniqueUrl);
         })
        .then(response => response.json())
        .then(data => {
            // Daten erfolgreich geladen, HTML aktualisieren
            if (!data || !data.forecast_date) {
                throw new Error("Vorhersagedaten sind ungültig oder leer.");
            }

            // Formatieren der Ausgabe
            const rfTemp = data.rf_temp_c !== null ? `${data.rf_temp_c.toFixed(1)}°C` : 'N/A';
            const rfWind = data.rf_wspd_kmh !== null ? `${data.rf_wspd_kmh.toFixed(1)} km/h` : 'N/A';
            const xgbTemp = data.xgb_temp_c !== null ? `${data.xgb_temp_c.toFixed(1)}°C` : 'N/A';
            const xgbWind = data.xgb_wspd_kmh !== null ? `${data.xgb_wspd_kmh.toFixed(1)} km/h` : 'N/A';

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

            // Letztes Update formatieren
            try {
                const updateDate = new Date(data.generated_at);
                lastUpdateSpan.textContent = updateDate.toLocaleString('de-DE');
            } catch (e) {
                lastUpdateSpan.textContent = 'Unbekannt';
            }
        })
        .catch(error => {
            console.error('Fehler beim Laden der Vorhersage:', error);
            forecastDiv.innerHTML = '<p class="error">Fehler: Vorhersage konnte nicht geladen werden.</p>';
            lastUpdateSpan.textContent = 'Fehler';
        });
});