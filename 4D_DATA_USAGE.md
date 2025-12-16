# 4D Radar Data Processing - Dokumentation

## Überblick

Das Programm unterstützt jetzt sowohl **3D-Daten** (RadarCube) als auch **4D-Daten** (ADC-Cube) für Radarverarbeitung.

## Datenformate

### 3D-Daten (RadarCube)
- **Format**: `(Samples, Chirps, Virtual_Channels)`
- **Verwendung**: Bereits vorverarbeitete Daten mit virtuellen MIMO-Kanälen
- **Beispiel**: `MRR_CornField/3D/RadarCube*.npy`

### 4D-Daten (ADC-Cube)
- **Format**: `(Samples, Chirps, Rx_Channels, Tx_Channels)`
- **Verwendung**: Rohe ADC-Daten mit separaten Tx- und Rx-Kanälen
- **Beispiel**: `USRR_Dynamic10m/4D/ADCCube*.npy`

## Verarbeitungsablauf

### Tasks 1-4: Identisch für 3D und 4D

Bei 4D-Daten werden diese automatisch in eine 3D-Repräsentation konvertiert:
- **Konversion**: `(Samples, Chirps, Rx, Tx)` → `(Samples, Chirps, Rx*Tx)`
- Die Virtual_Channels werden durch Umformen (reshape) der 4D-Struktur erstellt
- Alle bisherigen Berechnungen funktionieren damit identisch

**Task 1**: Daten laden, Windowing, 2D-FFT
**Task 2**: Key-Parameter berechnen, Range-Doppler Maps
**Task 3**: CFAR-Detektion
**Task 4**: Winkelschätzung und 3D-Lokalisierung

### Task 5: 4D-spezifische Verarbeitung

**Nur verfügbar wenn `Dim_4=True` und 4D-Daten vorhanden sind!**

#### 5.1 Strukturanalyse
- Zeigt die 4D-Dimensionen an
- Gibt Speichergröße und Datentyp aus

#### 5.2 Tx-Kanal-Analyse
- Verarbeitet jeden Tx-Kanal separat
- Summiert über alle Rx-Kanäle pro Tx
- Führt 2D-FFT für jeden Tx-Kanal durch
- Visualisiert alle Tx-Kanäle im Vergleich

#### 5.3 Rx-Kanal-Analyse
- Verarbeitet jeden Rx-Kanal separat
- Summiert über alle Tx-Kanäle pro Rx
- Führt 2D-FFT für jeden Rx-Kanal durch
- Visualisiert eine Auswahl von Rx-Kanälen

## Verwendung

### Mit 4D-Daten (USRR):

```python
from radar import Radar
from main import USRR_Dynamic_Config

# Erstelle Radar-Objekt mit 4D-Daten
radar = Radar(*USRR_Dynamic_Config(), num=8, use_tk=True, output_print=True, Dim_4=True)

# Führe alle Tasks aus
radar.Task_Step_1()  # Laden und FFT (nutzt 3D-Repräsentation)
radar.Task_Step_2()  # Range-Doppler Maps
radar.Task_Step_3()  # CFAR-Detektion
radar.Task_Step_4()  # 3D-Lokalisierung
radar.Task_Step_5_4D_Processing()  # 4D-spezifische Analyse
```

### Mit 3D-Daten (MRR):

```python
from radar import Radar
from main import MRR_CornField_Config

# Erstelle Radar-Objekt mit 3D-Daten
radar = Radar(*MRR_CornField_Config(), num=8, use_tk=True, output_print=True, Dim_4=False)

# Führe Tasks 1-4 aus (Task 5 wird automatisch übersprungen)
radar.Task_Step_1()
radar.Task_Step_2()
radar.Task_Step_3()
radar.Task_Step_4()
radar.Task_Step_5_4D_Processing()  # Wird übersprungen mit Warnung
```

## Konfiguration

In `main.py` gibt es zwei Konfigurationsfunktionen:

### USRR_Dynamic_Config (4D)
```python
radar_file_3D = "./RadarCube/USRR_Dynamic10m/3D/RadarCube"
radar_file_4D = "./RadarCube/USRR_Dynamic10m/4D/ADCCube"  # 4D-Pfad gesetzt
```

### MRR_CornField_Config (3D)
```python
radar_file_3D = "./RadarCube/MRR_CornField/3D/RadarCube"
radar_file_4D = ""  # Kein 4D-Pfad
```

## Wichtige Hinweise

1. **Automatische Konversion**: 4D-Daten werden für Tasks 1-4 automatisch zu 3D konvertiert
2. **Keine Doppelverarbeitung**: Die 3D-Repräsentation wird nur einmal erstellt
3. **Speichereffizienz**: Die originalen 4D-Daten bleiben erhalten für Task 5
4. **Kompatibilität**: Alle bestehenden Funktionen arbeiten transparent mit beiden Formaten
5. **Fehlerbehandlung**: Bei `Dim_4=True` ohne 4D-Datei wird das Programm gestoppt

## Vorteile der 4D-Verarbeitung

- **Flexibilität**: Tx- und Rx-Kanäle können separat analysiert werden
- **Beamforming**: Erweiterte Beamforming-Techniken möglich
- **Diagnose**: Kanalspezifische Fehlerdiagnose
- **DOA**: Verbesserte Direction-of-Arrival Schätzung durch separate Kanalverarbeitung
- **MIMO**: Optimierte MIMO-Verarbeitung durch Zugriff auf Einzelkanäle
- **Elevation-Schätzung**: Vollständige 3D-Lokalisierung mit Azimut UND Elevation

## Erweiterte Winkelschätzung für 4D-Daten

### Azimut- und Elevation-Berechnung

Bei 4D-Daten wird für jede CFAR-Detektion automatisch berechnet:

#### **Azimut-Winkel** (horizontal, links-rechts)
- Verwendet Antennen mit y=0 (horizontale Antennenreihe)
- FFT über Azimut-Antennen
- CFAR-Detektion im Winkelspektrum
- Visualisierung mit CFAR-Threshold

#### **Elevation-Winkel** (vertikal, oben-unten)
- Verwendet Antennen mit x=0 (vertikale Antennenreihe)
- FFT über Elevation-Antennen
- CFAR-Detektion im Winkelspektrum
- Separate Visualisierung mit CFAR-Threshold

### CFAR-Plots für Winkelschätzung

Für jede Detektion werden automatisch generiert:

1. **Azimut-Spektrum Plot**
   - Blaue Linie: Power-Spektrum über Azimut-Winkel
   - Grüne gestrichelte Linie: CFAR-Threshold
   - Orange Punkte: Alle CFAR-Detektionen über Threshold
   - Roter Punkt + Linie: Geschätzter Peak-Winkel
   - X-Achse: Winkel in Grad (FOV-begrenzt)
   - Y-Achse: Leistung in dB

2. **Elevation-Spektrum Plot** (nur für 4D-Daten)
   - Gleiche Darstellung wie Azimut
   - Zeigt vertikale Winkelauflösung
   - Ermöglicht vollständige 3D-Lokalisierung

### Konfiguration der Winkelschätzung

In `radar.py` können folgende Parameter angepasst werden:

```python
# Methode wählen: 'cfar' oder 'max'
self.angle_detection_method = 'cfar'

# CFAR-Parameter für Winkelschätzung
self.cfar_1d_train_cells = 10   # Anzahl Trainingszellen
self.cfar_1d_guard_cells = 4    # Anzahl Guard-Zellen
self.cfar_1d_threshold_factor = 3.0  # Threshold-Faktor
```

## 3D-Visualisierung mit Elevation

Die 3D-Objektlokalisierung (`Task_Step_4()`) nutzt beide Winkel:

**Kartesische Koordinaten:**
```python
x = r * cos(elevation) * sin(azimut)      # Lateral (links-rechts)
y = r * cos(elevation) * cos(azimut)      # Longitudinal (vor-zurück)
z = r * sin(elevation)                     # Vertikal (oben-unten)
```

**Bei 3D-Daten:** Elevation = 0° (alle Objekte in horizontaler Ebene)

**Bei 4D-Daten:** Elevation wird geschätzt → echte 3D-Lokalisierung

## Erweiterungsmöglichkeiten

Die 4D-Struktur ermöglicht zukünftige Erweiterungen wie:
- Adaptives Beamforming
- Kanalseparation für Interferenzunterdrückung  
- Tx/Rx-spezifische Kalibrierung
- Erweiterte MIMO-Algorithmen
- Time-Division Multiplexing (TDM) Analyse
- Verbessertes 2D-DOA (Direction of Arrival) mit MUSIC oder ESPRIT
- Höhenprofilierung von Objekten
