# Task 6: 4D Angle Estimation with Elevation - Dokumentation

## Übersicht

Task 6 erweitert Task 4 um vollständige **Elevation-Schätzung** für 4D-Radardaten. Während Task 4 nur Azimuth-Winkel (horizontale Richtung) berechnet, ermittelt Task 6 zusätzlich den Elevation-Winkel (vertikale Richtung) für eine vollständige 3D-Lokalisierung von Objekten.

## Hauptunterschiede Task 4 vs Task 6

| Aspekt | Task 4 (3D-Daten) | Task 6 (4D-Daten) |
|--------|-------------------|-------------------|
| **Datenquelle** | RadarCube 3D (Samples, Chirps, Channels) | ADCCube 4D (Samples, Chirps, Rx, Tx) |
| **Winkelschätzung** | Nur Azimuth (horizontal) | Azimuth + Elevation (vertikal) |
| **Antennenfilterung** | y=0 (Azimuth-Antennen) | y=0 (Azimuth) + x=0 (Elevation) |
| **3D-Position** | z=0 (alle Objekte in Ebene) | z≠0 (echte Höheninformation) |
| **Visualisierung** | 2D-Ebene im 3D-Raum | Volle 3D-Lokalisierung |

## Implementierte Methoden

### 1. `Task_Step_6()`
**Hauptmethode** - Koordiniert die gesamte 4D-Verarbeitung

```python
def Task_Step_6(self):
    # 6.1 Berechne Wellenlänge und Antennenabstände (Azimuth + Elevation)
    self._calculate_antenna_parameters_4d()
    
    # 6.2 Erzeuge virtuelle Antennen-Array (MIMO)
    self._create_virtual_array_4d()
    
    # 6.3 Winkelschätzung mit Elevation
    self.detections_4d = self._estimate_angles_with_elevation()
    
    # 6.4 4D-Visualisierung
    self._plot_4d_detections()
```

**Voraussetzungen:**
- `self.Dim_4 = True`
- `self.ADCCube_4D` muss geladen sein
- CFAR-Detektionen müssen existieren (`self.cfar_detections`)

---

### 2. `_calculate_antenna_parameters_4d()`
**Berechnet erweiterte Antennenparameter für 4D-Daten**

**Berechnet:**
- `self.d_azimuth`: Antennenabstand in Azimuth-Richtung (x-Achse)
- `self.d_elevation`: Antennenabstand in Elevation-Richtung (y-Achse)
- `self.FOV_azimuth_min/max`: Field-of-View horizontal (±90° typisch)
- `self.FOV_elevation_min/max`: Field-of-View vertikal (±90° typisch)

**Formel für FOV:**
```
max_angle = arcsin(λ / (2 * d_antenna))
```

**Ausgabe:**
```
- Wellenlänge λ = 3.95 mm
- Anzahl virtuelle Antennen: 192
- Azimuth Antennenabstand: 1.975 mm
- Elevation Antennenabstand: 1.975 mm
- Azimuth Field of View: -90.0° bis 90.0°
- Elevation Field of View: -90.0° bis 90.0°
```

---

### 3. `_create_virtual_array_4d()`
**Bestätigt virtuelles MIMO-Array**

Das virtuelle Array wurde bereits in `load_radar_cube()` erstellt:
```python
self.AntennaArray = self.ADCCube_4D.reshape(num_samples, num_chirps, num_rx * num_tx)
```

Diese Methode protokolliert nur die Array-Eigenschaften.

---

### 4. `_estimate_angles_with_elevation()` ⭐
**Kernmethode - Schätzt Azimuth UND Elevation für jede CFAR-Detektion**

#### Ablauf für jede Detektion (r_bin, d_bin):

##### A) **Azimuth-Schätzung** (wie Task 4)
```python
1. Filtere Azimuth-Antennen (y=0):
   ind_az = np.where(self.AntennaPositions[:, 1] == 0)
   AzimuthAntennaOnly = indices_az

2. Extrahiere Signal:
   x_azimuth = antenna_signal[AzimuthAntennaOnly]

3. FFT mit Zero-Padding:
   angle_fft_az = np.fft.fft(x_azimuth * hann_window, n=512)
   angle_spectrum_az = |fftshift(angle_fft_az)|

4. Winkelachse berechnen:
   sin(θ) = (k / (N/2)) * (λ / (2*d))
   θ = arcsin(sin(θ))

5. Peak-Finding (CFAR oder Max):
   - CFAR: Berechne 1D-Threshold, finde Peaks
   - Max: Suche globales Maximum
   
   → estimated_azimuth [Grad]
```

##### B) **Elevation-Schätzung** (NEU!)
```python
1. Filtere Elevation-Antennen (x=0):
   ind_el = np.where(self.AntennaPositions[:, 0] == 0)
   ElevationAntennaOnly = indices_el

2. Extrahiere Signal:
   x_elevation = antenna_signal[ElevationAntennaOnly]

3. FFT mit Zero-Padding:
   angle_fft_el = np.fft.fft(x_elevation * hann_window, n=512)
   angle_spectrum_el = |fftshift(angle_fft_el)|

4. Winkelachse berechnen:
   sin(φ) = (k / (N/2)) * (λ / (2*d))
   φ = arcsin(sin(φ))

5. Peak-Finding (CFAR oder Max):
   - CFAR: Berechne 1D-Threshold, finde Peaks
   - Max: Suche globales Maximum
   
   → estimated_elevation [Grad]
```

#### Gespeicherte Detektionsdaten:
```python
detection_dict = {
    'range_m': 12.34,                # Entfernung [m]
    'velocity_m_s': 5.67,            # Geschwindigkeit [m/s]
    'azimuth_deg': 15.2,             # Horizontal-Winkel [°]
    'elevation_deg': 3.5,            # Vertikal-Winkel [°] ← NEU
    'magnitude': 0.156,              # Signalstärke
    'r_bin': 45,                     # Range-Bin
    'd_bin': 78,                     # Doppler-Bin
    'power_db': 85.2,                # Azimuth-Leistung [dB]
    'elevation_power_db': 82.1       # Elevation-Leistung [dB] ← NEU
}
```

#### Visualisierung pro Detektion:
1. **Azimuth-Spektrum Plot:**
   - X-Achse: Azimuth-Winkel [-90° bis +90°]
   - Y-Achse: Leistung [dB]
   - Zeigt CFAR-Threshold (falls aktiviert)
   - Markiert Peak (roter Punkt)

2. **Elevation-Spektrum Plot:**
   - X-Achse: Elevation-Winkel [-90° bis +90°]
   - Y-Achse: Leistung [dB]
   - Zeigt CFAR-Threshold (falls aktiviert)
   - Markiert Peak (roter Punkt)

---

### 5. `_plot_4d_detections()`
**4D-Visualisierung mit vollständiger 3D-Position**

#### Kartesische Transformation:
```python
Gegeben (Polarkoordinaten):
- r: Range [m]
- az: Azimuth [rad]
- el: Elevation [rad]

Berechnet (Kartesische Koordinaten):
- x = r × cos(el) × sin(az)   # Lateral (links-rechts)
- y = r × cos(el) × cos(az)   # Longitudinal (vor-zurück)
- z = r × sin(el)             # Vertikal (oben-unten) ← NEU!
```

#### 3D-Scatter-Plot Elemente:

1. **Detektierte Objekte:**
   - Punktgröße: Proportional zur Signalstärke
   - Farbe: Magnitude (hot colormap)
   - Position: (x, y, z) mit echter Elevation

2. **Radar Field-of-View:**
   - **Horizontale Ebene** (z=0, cyan): Azimuth-Kegel
   - **Vertikale Ebene** (x=0, grün): Elevation-Kegel ← NEU!

3. **Mittellinie:**
   - Gestrichelte schwarze Linie bei 0° Azimuth/Elevation

4. **Reichweitenkreise:**
   - Konzentrische Kreise bei 25%, 50%, 75%, 100% max_range

5. **Achsengrenzen:**
   ```python
   x: [-x_max, +x_max]  # Basierend auf Azimuth-FOV
   y: [0, R_max]        # Longitudinal vorwärts
   z: [-z_max, +z_max]  # Basierend auf Elevation-FOV
   ```

---

## Mathematische Grundlagen

### Winkelauflösung

**Theoretische Winkelauflösung:**
```
Δθ = λ / (N * d)
```
- λ: Wellenlänge
- N: Anzahl Antennen
- d: Antennenabstand

**Bei d = λ/2 und N = 12:**
```
Δθ = λ / (12 * λ/2) = 2 / 12 ≈ 9.5°
```

### Field of View (FOV)

**Maximaler Winkel ohne Aliasing:**
```
θ_max = arcsin(λ / (2*d))
```

**Bei d = λ/2:**
```
θ_max = arcsin(1) = 90°
→ FOV: -90° bis +90°
```

### Räumliche Mehrdeutigkeit

**Eindeutigkeitsbereich:**
```
|sin(θ)| ≤ 1
→ -90° ≤ θ ≤ +90°
```

Bei größerem Antennenabstand (d > λ/2) entstehen Grating Lobes!

---

## Anwendungsbeispiel

### Szenario: Auto-Radar erkennt Fußgänger auf Brücke

**3D-Daten (Task 4, ohne Elevation):**
```
Range: 15.0 m
Azimuth: 10°
Elevation: 0° (angenommen)

Position: 
x = 15.0 × cos(0°) × sin(10°) = 2.6 m
y = 15.0 × cos(0°) × cos(10°) = 14.8 m
z = 15.0 × sin(0°) = 0.0 m

→ Fußgänger erscheint auf Straßenebene
→ Kollisionswarnung!
```

**4D-Daten (Task 6, mit Elevation):**
```
Range: 15.0 m
Azimuth: 10°
Elevation: 12° (gemessen)

Position: 
x = 15.0 × cos(12°) × sin(10°) = 2.5 m
y = 15.0 × cos(12°) × cos(10°) = 14.4 m
z = 15.0 × sin(12°) = 3.1 m

→ Fußgänger ist 3.1 m über Straßenebene
→ Keine Kollisionsgefahr!
```

---

## Parameter-Konfiguration

### CFAR-Parameter für Winkelschätzung

In `__init__()` des Radar-Objekts:

```python
# Winkelschätzung Methode
self.angle_detection_method = 'cfar'  # oder 'max'

# 1D-CFAR Parameter für Azimuth/Elevation
self.cfar_1d_train_cells = 10      # Trainingszellen
self.cfar_1d_guard_cells = 4       # Guardzellen
self.cfar_1d_threshold_factor = 3.0  # Schwellwert-Faktor
```

### FFT-Parameter

```python
nfft_az = 512   # Zero-Padding für Azimuth
nfft_el = 512   # Zero-Padding für Elevation
window = 'hann'  # Fensterfunktion
```

---

## Fehlerbehandlung

### Prüfung auf 4D-Daten

```python
if not self.Dim_4 or self.ADCCube_4D is None:
    self._log("Task 6 übersprungen: Keine 4D-Daten verfügbar.")
    return
```

### Prüfung auf Elevation-Antennen

```python
if len(ElevationAntennaOnly) > 0:
    # Elevation-Schätzung durchführen
else:
    estimated_elevation = 0.0
```

Fallback: Wenn keine Elevation-Antennen vorhanden → `elevation_deg = 0.0`

---

## Performance-Hinweise

### Rechenaufwand

**Task 4 (nur Azimuth):**
- 1 FFT pro Detektion (512 Punkte)
- 1 Plot pro Detektion

**Task 6 (Azimuth + Elevation):**
- 2 FFTs pro Detektion (je 512 Punkte)
- 2 Plots pro Detektion

**Typische Laufzeit:**
- 10 Detektionen: ~5-10 Sekunden
- Hauptanteil: Plotting (matplotlib)

### Optimierungsmöglichkeiten

1. **Plot-Unterdrückung für viele Detektionen:**
   ```python
   if len(det_indices) > 20:
       plot_enabled = False
   ```

2. **Reduzierte FFT-Größe:**
   ```python
   nfft = 256  # Statt 512
   ```

3. **Parallele Verarbeitung:**
   ```python
   from concurrent.futures import ThreadPoolExecutor
   ```

---

## Ausgabe-Log

### Konsolen-Ausgabe (Beispiel)

```
Task 6: 4D Angle Estimation with Elevation (Processing file ADCCube0.npy)

 - Task 6.1:
	- Wellenlänge λ = 3.95 mm
	- Anzahl virtuelle Antennen: 192
	- Azimuth Antennenabstand: 1.975 mm
	- Elevation Antennenabstand: 1.975 mm
	- Azimuth Field of View: -90.0° bis 90.0°
	- Elevation Field of View: -90.0° bis 90.0°

 - Task 6.2:
	- Virtuelles Array bereits erstellt: 12 Tx × 16 Rx = 192 virtuelle Antennen

 - Task 6.3:
	- Winkelschätzung mit Elevation für detektierte Objekte
	- Anzahl Range-Doppler-Bins mit Detektionen: 3
	- Azimuth-Antennen: 48 von 192
	- Elevation-Antennen: 32 von 192
	- Antennenabstand Azimuth: 1.975 mm
	- Antennenabstand Elevation: 1.975 mm
	- Wellenlänge: 3.950 mm
	- Verarbeite 3 Detektionen...
	  Detektion #1: Range=12.34m, Vel=5.67m/s, Azimut=15.2°, Power=85.3dB, Elevation=3.5°, El-Power=82.1dB
	  Detektion #2: Range=8.91m, Vel=-2.34m/s, Azimut=-8.7°, Power=78.9dB, Elevation=1.2°, El-Power=75.4dB
	  Detektion #3: Range=20.15m, Vel=0.12m/s, Azimut=0.3°, Power=72.5dB, Elevation=-0.8°, El-Power=70.2dB

	- Winkelschätzung mit Elevation abgeschlossen
	- Anzahl gefundener Objekte: 3

 - Task 6.4:
	- 4D-Visualisierung der Objekte mit Elevation
```

---

## Zusammenfassung

### ✅ Was Task 6 leistet:

1. **Vollständige 3D-Lokalisierung** von Objekten (x, y, z)
2. **Azimuth-Schätzung** (horizontal, wie Task 4)
3. **Elevation-Schätzung** (vertikal, NEU)
4. **CFAR-basierte Peak-Detection** in Winkelspektren
5. **Erweiterte Visualisierung** mit 3D-FOV-Kegeln
6. **Kompatibilität** mit Task 4 (beide können verwendet werden)

### 🔧 Voraussetzungen:

- 4D-ADC-Daten verfügbar (`ADCCube_4D`)
- Antennen-Positionsdaten geladen (`AntennaPositions`)
- CFAR-Detektionen berechnet (`cfar_detections`)

### 📊 Output:

- **2 Plots pro Detektion:** Azimuth + Elevation Spektrum
- **1 finaler 3D-Plot:** Alle Objekte mit echter Höhe
- **Detailliertes Log:** Alle Winkel und Leistungswerte

---

## Verwendung

### In main.py oder Skript:

```python
# Radar-Objekt mit 4D-Daten initialisieren
radar = Radar(
    radar_file_3D="path/to/RadarCube",
    radar_file_4D="path/to/ADCCube",
    Dim_4=True,  # Wichtig!
    # ... weitere Parameter
)

# Tasks ausführen
radar.load_radar_cube()           # Task 1
radar.Task_Step_2()               # Task 2: FFT
radar.Task_Step_3()               # Task 3: CFAR

# Option A: Nur Azimuth (3D-Daten)
radar.Task_Step_4()

# Option B: Azimuth + Elevation (4D-Daten)
radar.Task_Step_6()
```

---

## Technische Details

### Antennen-Array Struktur

**AntennaPositions.npy:**
```
Shape: (192, 2)  # 192 virtuelle Antennen, 2D-Positionen
Columns: [x, y]  # x: Azimuth, y: Elevation
Units: Normalisiert (Vielfache von λ/2)
```

**Filterung:**
- **Azimuth:** `y == 0` → Alle Antennen auf horizontaler Linie
- **Elevation:** `x == 0` → Alle Antennen auf vertikaler Linie

### FFT-Winkelachse

**Berechnung:**
```python
k_bins = np.arange(N_FFT) - N_FFT / 2
sin_theta = (k_bins / (N_FFT/2)) * (λ / (2*d))
sin_theta = np.clip(sin_theta, -1, 1)  # Verhindere arcsin-Fehler
theta_rad = np.arcsin(sin_theta)
theta_deg = np.degrees(theta_rad)
```

**Eigenschaften:**
- Nicht-lineare Achse (arcsin)
- Hohe Auflösung bei kleinen Winkeln
- Niedrige Auflösung bei ±90°

---

**Autor:** GitHub Copilot  
**Datum:** 2025-12-15  
**Version:** 1.0
