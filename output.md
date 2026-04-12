──────────────────────────────────────────────────── ⛅ Wettervorhersage für Berlin ⛅ ─────────────────────────────────────────────────────
────────────────────────────────────────────────────  1. Stationssuche & Datenerfassung ────────────────────────────────────────────────────
Suche nach Wetterstationen im Umkreis von 30 km um Berlin...
   Gefundene relevante Stations-IDs (bis zu 4): ['10384', 'ETNB0', '10381', 'D0420']
   Details der Stationen:
                         name country      distance
id                                                 
10384      Berlin / Tempelhof      DE   5936.348752
ETNB0  Berlin / Reinickendorf      DE   7913.333565
10381         Berlin / Dahlem      DE   9255.188364
D0420          Berlin-Marzahn      DE  10825.095000

Lade Daten für 4 Station(en) vom 2006-04-12 bis 2026-04-12...
   Versuche Station 10384...
     Daten für 10384 (7280 Einträge) geladen.
   Versuche Station ETNB0...
     Keine Daten für Station ETNB0 im Zeitraum.
   Versuche Station 10381...
     Daten für 10381 (7280 Einträge) geladen.
   Versuche Station D0420...
     Daten für D0420 (6785 Einträge) geladen.

Daten erfolgreich geladen für 3 von 4 angefragten Stationen: ['10384', '10381', 'D0420']
──────────────────────────────────────────────────── 1.5 Räumliche Interpolation (IDW) ─────────────────────────────────────────────────────
   Hole Metadaten für Interpolation...
   Lade Stationsinventar...
      ✔️ Stationsinventar geladen.
      ✔️ Metadaten für 3 Station(en) extrahiert.

Starte IDW-Interpolation für ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres'] (p=2)...
   Interpoliere für Zeitraum: 2006-04-13 bis 2026-03-18
   Interpoliere Variable: tavg...

Interpolating tavg:   0%|          | 0/7280 [00:00<?, ?day/s]
Interpolating tavg:  50%|█████     | 3662/7280 [00:00<00:00, 36608.86day/s]
Interpolating tavg: 100%|██████████| 7280/7280 [00:00<00:00, 36961.91day/s]
   Interpoliere Variable: tmin...

Interpolating tmin:   0%|          | 0/7280 [00:00<?, ?day/s]
Interpolating tmin:  47%|████▋     | 3420/7280 [00:00<00:00, 34191.55day/s]
Interpolating tmin:  98%|█████████▊| 7146/7280 [00:00<00:00, 35991.46day/s]
Interpolating tmin: 100%|██████████| 7280/7280 [00:00<00:00, 35599.69day/s]
   Interpoliere Variable: tmax...

Interpolating tmax:   0%|          | 0/7280 [00:00<?, ?day/s]
Interpolating tmax:  50%|█████     | 3646/7280 [00:00<00:00, 36455.34day/s]
Interpolating tmax: 100%|██████████| 7280/7280 [00:00<00:00, 36714.63day/s]
   Interpoliere Variable: prcp...

Interpolating prcp:   0%|          | 0/7280 [00:00<?, ?day/s]
Interpolating prcp:  49%|████▊     | 3541/7280 [00:00<00:00, 35404.63day/s]
Interpolating prcp:  99%|█████████▊| 7182/7280 [00:00<00:00, 35992.37day/s]
Interpolating prcp: 100%|██████████| 7280/7280 [00:00<00:00, 35827.76day/s]
     Warnung: Für prcp konnten an 7 Tagen keine Werte interpoliert werden (zu wenige Stationen?).
   Interpoliere Variable: wspd...

Interpolating wspd:   0%|          | 0/7280 [00:00<?, ?day/s]
Interpolating wspd:  50%|█████     | 3672/7280 [00:00<00:00, 36713.47day/s]
Interpolating wspd: 100%|██████████| 7280/7280 [00:00<00:00, 36652.85day/s]
   Interpoliere Variable: pres...

Interpolating pres:   0%|          | 0/7280 [00:00<?, ?day/s]
Interpolating pres:  50%|█████     | 3652/7280 [00:00<00:00, 36514.64day/s]
Interpolating pres: 100%|██████████| 7280/7280 [00:00<00:00, 36753.70day/s]
   ✔️ IDW-Interpolation abgeschlossen.
   Beispiel der interpolierten Daten für Berlin:
                 tavg      tmin       tmax      prcp  wspd         pres
time                                                                   
2006-04-13   7.670851  4.525109  11.116594  2.458297  17.6  1006.983840
2006-04-14   7.441703  4.654257  11.316594  3.717029  21.2  1006.642138
2006-04-15   9.070851  3.854257  13.945743  5.142138  11.2  1013.912989
2006-04-16  10.700000  7.458297  15.474891  6.650217  15.1  1004.742138
2006-04-17   9.812554  7.421069  13.258297  0.641703  14.0  1009.442138
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 7280 entries, 2006-04-13 to 2026-03-18
Freq: D
Data columns (total 6 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   tavg    7280 non-null   float64
 1   tmin    7280 non-null   float64
 2   tmax    7280 non-null   float64
 3   prcp    7273 non-null   float64
 4   wspd    7280 non-null   float64
 5   pres    7280 non-null   float64
dtypes: float64(6)
memory usage: 398.1 KB
None
──────────────────────────────────────────────────── 2. Explorative Datenanalyse (EDA) ─────────────────────────────────────────────────────

Erste 5 Zeilen der ausgewählten Daten:
                 tavg      tmin       tmax      prcp  wspd         pres
time                                                                   
2006-04-13   7.670851  4.525109  11.116594  2.458297  17.6  1006.983840
2006-04-14   7.441703  4.654257  11.316594  3.717029  21.2  1006.642138
2006-04-15   9.070851  3.854257  13.945743  5.142138  11.2  1013.912989
2006-04-16  10.700000  7.458297  15.474891  6.650217  15.1  1004.742138
2006-04-17   9.812554  7.421069  13.258297  0.641703  14.0  1009.442138

Letzte 5 Zeilen der ausgewählten Daten:
                tavg      tmin       tmax  prcp       wspd         pres
time                                                                   
2026-03-14  7.195960  4.462772   9.570851   NaN  11.978931  1006.912554
2026-03-15  6.141703  2.808514  10.016594   NaN  12.320634  1009.100000
2026-03-16  5.500000  3.695960   8.074891   NaN  13.291486  1008.258297
2026-03-17  4.362337  2.129149   6.882971   NaN  15.337228  1012.545743
2026-03-18  5.149783  2.133188   8.624674   NaN  14.758297  1016.029149

Informationen über den DataFrame (Typen, Nicht-Null-Werte):
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 7280 entries, 2006-04-13 to 2026-03-18
Freq: D
Data columns (total 6 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   tavg    7280 non-null   float64
 1   tmin    7280 non-null   float64
 2   tmax    7280 non-null   float64
 3   prcp    7273 non-null   float64
 4   wspd    7280 non-null   float64
 5   pres    7280 non-null   float64
dtypes: float64(6)
memory usage: 398.1 KB

Deskriptive Statistiken:
              tavg         tmin         tmax         prcp         wspd         pres
count  7280.000000  7280.000000  7280.000000  7273.000000  7280.000000  7280.000000
mean     10.857104     6.663547    15.043643     1.556704    12.067673  1014.794896
std       7.660480     6.682377     9.017416     3.787821     4.444572     8.966932
min     -14.787072   -19.457655   -11.099675     0.000000     2.700000   973.534058
25%       5.102163     1.703322     7.973148     0.000000     8.867679  1009.371286
50%      10.792785     6.598335    15.072526     0.017565    11.358477  1014.963206
75%      17.062144    12.081723    22.305795     1.403330    14.400000  1020.443374
max      30.224584    23.493339    38.213483   104.072715    35.907794  1044.867246

Fehlende Werte pro Spalte (vor der Datenverarbeitung):
prcp    7
dtype: int64

Keine Duplikate gefunden.

Überprüfung auf Ausreißer:

Quantifizierung potenzieller Ausreißer (IQR-Methode)
                 Potenzielle Ausreißer pro Variable (IQR * 1.5)                 
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Variable     ┃ Anzahl Ausreißer ┃ % Ausreißer ┃ Untere Grenze ┃ Obere Grenze ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ tavg         │                2 │       0.03% │        -12.84 │        35.00 │
│ tmin         │               17 │       0.23% │        -13.86 │        27.65 │
│ prcp         │             1016 │      13.96% │         -2.10 │         3.51 │
│ wspd         │              190 │       2.61% │          0.57 │        22.70 │
│ pres         │              158 │       2.17% │        992.76 │      1037.05 │
└──────────────┴──────────────────┴─────────────┴───────────────┴──────────────┘
Plot erfolgreich gespeichert: /home/alina/Documents/projects/meteoflow/plots/fehlende_werte.png


Visualisierung der Verteilungen der Variablen:
Plot erfolgreich gespeichert: /home/alina/Documents/projects/meteoflow/plots/histogramm.png


Visualisierung der Zeitreihen:
Plot erfolgreich gespeichert: /home/alina/Documents/projects/meteoflow/plots/zeitreihen_plots.png


Korrelationsmatrix:
Plot erfolgreich gespeichert: /home/alina/Documents/projects/meteoflow/plots/korrelationsmatrix.png


Korrelationsmatrix:
          tavg      tmin      tmax      prcp      wspd      pres
tavg  1.000000  0.964980  0.984269  0.067538 -0.153030 -0.069194
tmin  0.964980  1.000000  0.913742  0.127739 -0.089361 -0.124978
tmax  0.984269  0.913742  1.000000  0.038388 -0.189571 -0.035209
prcp  0.067538  0.127739  0.038388  1.000000  0.137511 -0.273835
wspd -0.153030 -0.089361 -0.189571  0.137511  1.000000 -0.283961
pres -0.069194 -0.124978 -0.035209 -0.273835 -0.283961  1.000000

Boxplots zur Visualisierung von Verteilungen und Ausreißern:
Plot erfolgreich gespeichert: /home/alina/Documents/projects/meteoflow/plots/boxplots.png


EDA abgeschlossen.
───────────────────────────────────────────────────────── 3. Datenvorverarbeitung ──────────────────────────────────────────────────────────
Überpüfung auf fehlende Werte (vor Imputation)
tavg    0
tmin    0
tmax    0
prcp    7
wspd    0
pres    0
dtype: int64

Fülle fehlende Werte mit ffill und bfill...

Überprüfung auf fehlende Werte (nach Imputation):
tavg    0
tmin    0
tmax    0
prcp    0
wspd    0
pres    0
dtype: int64

   Behandle potenzielle Ausreißer durch Winsorizing (5%/95% Perzentil)...
      Variable 'tavg': 728 Werte auf Grenzen [-1.42, 22.79] (5./95. Perzentil) gesetzt.
      Variable 'tmin': 728 Werte auf Grenzen [-4.15, 16.94] (5./95. Perzentil) gesetzt.
      Variable 'tmax': 728 Werte auf Grenzen [0.78, 29.31] (5./95. Perzentil) gesetzt.
      Variable 'prcp': 364 Werte auf Grenzen [0.00, 8.01] (5./95. Perzentil) gesetzt.
      Variable 'wspd': 728 Werte auf Grenzen [6.30, 20.20] (5./95. Perzentil) gesetzt.
      Variable 'pres': 727 Werte auf Grenzen [999.51, 1029.16] (5./95. Perzentil) gesetzt.
      ✔️ Ausreißerbehandlung (Winsorizing) abgeschlossen.

✔️ Datenvorverarbeitung abgeschlossen.
────────────────────────────────────────────────────────── 4. Feature Engineering ──────────────────────────────────────────────────────────

--- Demonstration: Effekt von .shift(-1) für 'tavg' ---
Vorher (erste 5 Zeilen):
                 tavg
time                 
2006-04-13   7.670851
2006-04-14   7.441703
2006-04-15   9.070851
2006-04-16  10.700000
2006-04-17   9.812554
Erstelle Zielspalten...

Nachher (erste 5 Zeilen von 'tavg'):
                 tavg  tavg_target
time                              
2006-04-13   7.670851     7.441703
2006-04-14   7.441703     9.070851
2006-04-15   9.070851    10.700000
2006-04-16  10.700000     9.812554
2006-04-17   9.812554    10.083406
-------------------------------------------------------------


Erstelle Lag-Features für Spalten: ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres', 'tavg_target', 'wspd_target']

Erstelle zeitbasierte Features: Monat, Tag des Jahres, Wochentag...

6 Zeilen mit NaN-Werten entfernt.

Daten nach Feature Engineering (erste paar Zeilen):
                 tavg      tmin       tmax      prcp  wspd         pres  ...  wspd_target_lag_3  wspd_target_lag_4  wspd_target_lag_5  month  dayofyear  weekday
time                                                                     ...                                                                                    
2006-04-18  10.083406  5.991920  14.770851  0.370851  15.8  1011.842138  ...               15.1               11.2               20.2      4        108        1
2006-04-19  11.008514  5.887880  15.333188  0.000000   8.3  1014.942138  ...               14.0               15.1               11.2      4        109        2
2006-04-20  12.683406  6.175326  18.400000  0.000000  10.4  1015.071286  ...               15.8               14.0               15.1      4        110        3
2006-04-21  13.900000  6.849783  20.404040  1.908080   9.0  1016.283840  ...                8.3               15.8               14.0      4        111        4
2006-04-22  11.700000  9.312554  14.700000  3.000000   7.2  1015.012989  ...               10.4                8.3               15.8      4        112        5
2006-04-23  11.700000  8.316594  15.504040  0.100000   8.3  1014.971286  ...                9.0               10.4                8.3      4        113        6
2006-04-24  10.841703  7.112554  14.474891  0.000000   7.6  1019.512989  ...                7.2                9.0               10.4      4        114        0
2006-04-25  14.183406  5.300000  21.500000  0.000000   7.6  1018.671286  ...                8.3                7.2                9.0      4        115        1
2006-04-26  16.458297  9.570851  22.278931  1.037228   8.3  1016.271286  ...                7.6                8.3                7.2      4        116        2
2006-04-27  13.641703  9.229149  16.112554  0.870851  11.2  1016.971286  ...                7.6                7.6                8.3      4        117        3

[10 rows x 51 columns]

Dimensionen der aufbereiteten Daten: (7274, 51)

Feature Engineering abgeschlossen.
─────────────────────────────────────────────────────────── 5. Train/Test Split ────────────────────────────────────────────────────────────
Definiere Feature- und Zielspalten...
   Gefundene Features: 47
   Gefundene Targets: ['tavg_target', 'wspd_target']

Führe chronologischen Split durch (Testset-Größe: 1460 Tage)...
   Trainingsdaten: 5814 Samples (2006-04-18 bis 2022-03-18)
   Testdaten: 1460 Samples (2022-03-19 bis 2026-03-17)
   Anzahl Features: 47
   Zielvariablen: ['tavg_target', 'wspd_target']

   Überprüfung des Split-Verhältnisses:
     Trainings-Anteil: 79.93%
     Test-Anteil:      20.07%
   ✔️ Train/Test Split abgeschlossen.

Trainingsdaten (erste 3 und letzte 3 Zeilen):
                tmin       tmax      prcp         pres  tavg_lag_1  tavg_lag_2  ...  wspd_target_lag_5  month  dayofyear  weekday  tavg_target  wspd_target
time                                                                            ...                                                                        
2006-04-18  5.991920  14.770851  0.370851  1011.842138    9.812554   10.700000  ...               20.2      4        108        1    11.008514          8.3
2006-04-19  5.887880  15.333188  0.000000  1014.942138   10.083406    9.812554  ...               11.2      4        109        2    12.683406         10.4
2006-04-20  6.175326  18.400000  0.000000  1015.071286   11.008514   10.083406  ...               15.1      4        110        3    13.900000          9.0

[3 rows x 49 columns]
...
                tmin       tmax  prcp         pres  tavg_lag_1  tavg_lag_2  ...  wspd_target_lag_5  month  dayofyear  weekday  tavg_target  wspd_target
time                                                                        ...                                                                        
2022-03-16  2.874469  11.385014   0.0  1027.451028    8.592459    7.527914  ...          14.970542      3         75        2     7.634378    11.421541
2022-03-17  3.934378  11.764120   0.0  1029.163206    6.615507    8.592459  ...          14.499532      3         76        3     8.689651    12.609168
2022-03-18  5.333627  13.506139   0.0  1029.163206    7.634378    6.615507  ...          13.014623      3         77        4     6.396115    17.255068

[3 rows x 49 columns]

Testdaten (erste 3 und letzte 3 Zeilen):
                tmin       tmax  prcp         pres  tavg_lag_1  tavg_lag_2  ...  wspd_target_lag_5  month  dayofyear  weekday  tavg_target  wspd_target
time                                                                        ...                                                                        
2022-03-19  1.400196  10.960986   0.0  1029.163206    8.689651    7.634378  ...           7.424551      3         78        5     6.582435    17.216800
2022-03-20  0.441593  12.748057   0.0  1029.163206    6.396115    8.689651  ...          10.719292      3         79        6     7.414234    12.896728
2022-03-21  0.928665  14.036206   0.0  1029.163206    6.582435    6.396115  ...          11.421541      3         80        0     8.581914     8.439403

[3 rows x 49 columns]
...
                tmin       tmax      prcp         pres  tavg_lag_1  tavg_lag_2  ...  wspd_target_lag_5  month  dayofyear  weekday  tavg_target  wspd_target
time                                                                            ...                                                                        
2026-03-15  2.808514  10.016594  2.754747  1009.100000    7.195960    9.483406  ...          12.468269      3         74        6     5.500000    13.291486
2026-03-16  3.695960   8.074891  2.754747  1008.258297    6.141703    7.195960  ...          12.508080      3         75        0     4.362337    15.337228
2026-03-17  2.129149   6.882971  2.754747  1012.545743    5.500000    6.141703  ...          15.849348      3         76        1     5.149783    14.758297

[3 rows x 49 columns]
──────────────────────────────────────────────────────────── 6. Modelltraining ─────────────────────────────────────────────────────────────
Training RandomForestRegressor...

RandomForestRegressor trainiert.

Modell erfolgreich gespeichert: /home/alina/Documents/projects/meteoflow/saved_models/rf_model.joblib
Training XGBoostRegressor...

XGBoostRegressor trainiert.

Modell erfolgreich gespeichert: /home/alina/Documents/projects/meteoflow/saved_models/xgb_model.joblib

Modelltraining abgeschlossen!
──────────────────────────────────────────────────────────── 7. Modellbewertung ────────────────────────────────────────────────────────────

--- rf ---
  tavg_target:
    MAE:  1.45
    RMSE: 1.86
    R²:   0.93

Erstelle Plot für rf - tavg_target...
Plot erfolgreich gespeichert: /home/alina/Documents/projects/meteoflow/plots/evaluation_rf_tavg_target.png

  wspd_target:
    MAE:  2.67
    RMSE: 3.29
    R²:   0.27

Erstelle Plot für rf - wspd_target...
Plot erfolgreich gespeichert: /home/alina/Documents/projects/meteoflow/plots/evaluation_rf_wspd_target.png


--- xgb ---
  tavg_target:
    MAE:  1.40
    RMSE: 1.81
    R²:   0.94

Erstelle Plot für xgb - tavg_target...
Plot erfolgreich gespeichert: /home/alina/Documents/projects/meteoflow/plots/evaluation_xgb_tavg_target.png

  wspd_target:
    MAE:  2.63
    RMSE: 3.24
    R²:   0.29

Erstelle Plot für xgb - wspd_target...
Plot erfolgreich gespeichert: /home/alina/Documents/projects/meteoflow/plots/evaluation_xgb_wspd_target.png

Modellbewertung abgeschlossen.
────────────────────────────────────────────────────── Temperatur-Zeitreihe erstellen ──────────────────────────────────────────────────────
Erstelle Temperatur-Zeitreihe...
Plot erfolgreich gespeichert: /home/alina/Documents/projects/meteoflow/plots/temperature_time_series_rf_tavg_target.png

Plot erfolgreich gespeichert: /home/alina/Documents/projects/meteoflow/plots/temperature_time_series_xgb_tavg_target.png

──────────────────────────────────────────────────── 8. Vorhersage für den nächsten Tag ────────────────────────────────────────────────────

--- DEBUG: Features für Vorhersage aus main.py ---
Letzter Datenpunkt Index (main.py): 2026-03-17 00:00:00
Feature-Werte (main.py):
{
    'tmin': 2.129148553681292,
    'tmax': 6.8829710736258445,
    'prcp': 2.7547466060969783,
    'pres': 1012.5457427684065,
    'tavg_lag_1': 5.5,
    'tavg_lag_2': 6.141702892637416,
    'tavg_lag_3': 7.195960124230955,
    'tavg_lag_4': 9.483405785274831,
    'tavg_lag_5': 10.054257231593539,
    'tmin_lag_1': 3.6959601242309548,
    'tmin_lag_2': 2.8085144631870786,
    'tmin_lag_3': 4.462771694780618,
    'tmin_lag_4': 5.93766301686837,
    'tmin_lag_5': 7.425108677912247,
    'tmax_lag_1': 8.074891322087753,
    'tmax_lag_2': 10.016594214725169,
    'tmax_lag_3': 9.570851446318708,
    'tmax_lag_4': 13.770851446318709,
    'tmax_lag_5': 13.312554338956124,
    'prcp_lag_1': 2.7547466060969783,
    'prcp_lag_2': 2.7547466060969783,
    'prcp_lag_3': 2.7547466060969783,
    'prcp_lag_4': 2.7547466060969783,
    'prcp_lag_5': 2.7547466060969783,
    'wspd_lag_1': 13.291485536812921,
    'wspd_lag_2': 12.320634090494213,
    'wspd_lag_3': 11.978931197856797,
    'wspd_lag_4': 15.849347932526518,
    'wspd_lag_5': 12.50807975153809,
    'pres_lag_1': 1008.2582971073626,
    'pres_lag_2': 1009.1,
    'pres_lag_3': 1006.9125543389563,
    'pres_lag_4': 1008.5125543389561,
    'pres_lag_5': 1018.0417028926375,
    'tavg_target_lag_1': 4.362336983131629,
    'tavg_target_lag_2': 5.5,
    'tavg_target_lag_3': 6.141702892637416,
    'tavg_target_lag_4': 7.195960124230955,
    'tavg_target_lag_5': 9.483405785274831,
    'wspd_target_lag_1': 15.337228305219384,
    'wspd_target_lag_2': 13.291485536812921,
    'wspd_target_lag_3': 12.320634090494213,
    'wspd_target_lag_4': 11.978931197856797,
    'wspd_target_lag_5': 15.849347932526518,
    'month': 3.0,
    'dayofyear': 76.0,
    'weekday': 1.0
}
-----------------------------------------------------


Features für die Vorhersage von morgen (basierend auf Daten vom 2026-03-17):

Vorhersage für 2026-03-18:

--- rf ---
  Vorhergesagte Temperatur: 5.2 °C
  Vorhergesagte Windgeschwindigkeit: 14.9 km/h

--- xgb ---
  Vorhergesagte Temperatur: 4.7 °C
  Vorhergesagte Windgeschwindigkeit: 14.3 km/h

Vorhersage abgeschlossen.

🎉 Wettervorhersage Workflow Abgeschlossen 🎉