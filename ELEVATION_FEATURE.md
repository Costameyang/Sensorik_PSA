# Elevation-Schätzung für 4D-Radardaten - Übersicht

## Neu implementierte Funktionalität

### 1. Automatische Elevation-Schätzung bei 4D-Daten

**In `_estimate_angles_for_detections()`:**

```
Für jede CFAR-Detektion (Range-Doppler-Bin):
├── AZIMUT-Schätzung (wie bisher)
│   ├── Filter Antennen mit y=0 (horizontal)
│   ├── FFT über Azimut-Antennen (mit Zero-Padding)
│   ├── CFAR-Detektion im Winkelspektrum
│   ├── Peak-Finding → Azimut-Winkel
│   └── Plot: Azimut-Spektrum mit CFAR-Threshold
│
└── ELEVATION-Schätzung (NEU für 4D)
    ├── Filter Antennen mit x=0 (vertikal)
    ├── FFT über Elevation-Antennen (mit Zero-Padding)
    ├── CFAR-Detektion im Winkelspektrum
    ├── Peak-Finding → Elevation-Winkel
    └── Plot: Elevation-Spektrum mit CFAR-Threshold
```

### 2. Erweiterte Visualisierungen

#### A) Azimut-Spektrum Plot (für alle Daten)
```
┌─────────────────────────────────────────────┐
│  Azimut-Spektrum - Detektion #1             │
│  Range: 12.34m, Velocity: 5.67 m/s          │
│  Azimut: 15.2°, Elevation: 3.5°             │ (4D)
├─────────────────────────────────────────────┤
│                                             │
│    │                                        │
│  P │        ╱╲                             │
│  o │       ╱  ╲    ○ CFAR-Detektionen      │
│  w │   ○ ╱    ╲○                           │
│  e │----╱------╲---- CFAR-Threshold        │
│  r │  ╱        ╲                           │
│    │ ╱          ╲                          │
│    └──────────────────────────────────────→│
│           Azimut-Winkel [°]                 │
│                                             │
│  ● Peak bei 15.2°                          │
└─────────────────────────────────────────────┘
```

#### B) Elevation-Spektrum Plot (NEU, nur 4D)
```
┌─────────────────────────────────────────────┐
│  Elevation-Spektrum - Detektion #1          │
│  Range: 12.34m, Velocity: 5.67 m/s          │
│  Elevation: 3.5°, Azimut: 15.2°             │
├─────────────────────────────────────────────┤
│                                             │
│    │                                        │
│  P │      ╱╲                                │
│  o │     ╱  ╲    ○ CFAR-Detektionen        │
│  w │  ○ ╱    ╲○                            │
│  e │---╱------╲---- CFAR-Threshold         │
│  r │  ╱        ╲                           │
│    │ ╱          ╲                          │
│    └──────────────────────────────────────→│
│         Elevation-Winkel [°]                │
│                                             │
│  ● Peak bei 3.5°                           │
└─────────────────────────────────────────────┘
```

### 3. Vollständige 3D-Lokalisierung

**Kartesische Transformation:**

```
Eingabe (Polarkoordinaten):
- Range (r): Entfernung zum Objekt
- Azimut (az): Horizontaler Winkel
- Elevation (el): Vertikaler Winkel

Ausgabe (Kartesische Koordinaten):
- x = r × cos(el) × sin(az)   [lateral, links-rechts]
- y = r × cos(el) × cos(az)   [longitudinal, vorwärts-rückwärts]
- z = r × sin(el)             [vertikal, oben-unten]
```

**Unterschied 3D vs 4D:**
- **3D-Daten**: el = 0° → alle Objekte in horizontaler Ebene (z = 0)
- **4D-Daten**: el geschätzt → echte Höheninformation

### 4. Ausgaben und Logging

**Console/Tkinter-Log für jede Detektion:**
```
Detektion #1: Range=12.34m, Vel=5.67m/s, Azimut=15.1°, Power=85.2dB, Elevation=3.5°, El-Power=82.1dB
```

**Gespeicherte Daten pro Detektion:**
```python
detection_dict = {
    'range_m': 12.34,
    'velocity_m_s': 5.67,
    'azimuth_deg': 15.1,
    'elevation_deg': 3.5,           # NEU
    'magnitude': 0.156,
    'r_bin': 45,
    'd_bin': 78,
    'power_db': 85.2,
    'elevation_power_db': 82.1,     # NEU
    # ... weitere Elevation-Daten für Debugging
}
```

## Anwendungsbeispiel

### Szenario: Auto-Radar erkennt Fußgänger auf Brücke

**3D-Daten (ohne Elevation):**
```
Position: x=2.5m, y=15.0m, z=0.0m
→ Fußgänger wird als auf Straßenebene erkannt
```

**4D-Daten (mit Elevation):**
```
Position: x=2.5m, y=15.0m, z=3.2m
→ Fußgänger wird korrekt als auf erhöhter Brücke erkannt
→ Keine Kollisionsgefahr!
```

## Technische Details

### Antennen-Filterung

**Azimut (horizontal):**
```python
# Filtere Antennen mit y=0 (alle auf horizontaler Linie)
ind = np.where(self.AntennaPositions[:, 1] == 0)
AzimuthAntennaOnly = indices
```

**Elevation (vertikal):**
```python
# Filtere Antennen mit x=0 (alle auf vertikaler Linie)
ind_el = np.where(self.AntennaPositions[:, 0] == 0)
ElevationAntennaOnly = indices_el
```

### CFAR-Parameter für Winkelschätzung

```python
# In radar.py __init__
self.angle_detection_method = 'cfar'  # oder 'max'
self.cfar_1d_train_cells = 10
self.cfar_1d_guard_cells = 4
self.cfar_1d_threshold_factor = 3.0
```

## Visualisierungs-Workflow

```
Task 4: Winkelschätzung & 3D-Lokalisierung
├── Für jede CFAR-Detektion:
│   ├── [Plot 1] Azimut-Spektrum mit CFAR
│   ├── [Plot 2] Elevation-Spektrum mit CFAR (4D)
│   └── Log-Ausgabe der Winkel
├── [Plot 3] 3D-Scatter: Alle Objekte im Raum
└── Zusammenfassung: Anzahl gefundener Objekte
```

## Performance-Hinweise

- **FFT-Größe**: 512 Bins (Zero-Padding) für höhere Winkelauflösung
- **Windowing**: Hann-Fenster zur Seitenkeulenunterdrückung
- **CFAR**: Reduziert False Alarms in Winkelspektren
- **Plots**: Pro Detektion 1-2 Plots (je nach 3D/4D)

## Fehlerbehandlung

```python
# Elevation nur wenn 4D-Daten vorhanden
if self.Dim_4 and len(ElevationAntennaOnly) > 0:
    # Elevation-Schätzung durchführen
else:
    # elevation_deg = 0.0 (Standard)
```

## Zusammenfassung

✅ **Azimut**: Immer geschätzt (3D + 4D)
✅ **Elevation**: Nur bei 4D-Daten geschätzt
✅ **CFAR-Plots**: Für beide Winkel automatisch generiert
✅ **3D-Lokalisierung**: Nutzt beide Winkel für kartesische Koordinaten
✅ **Kompatibilität**: 3D-Daten funktionieren weiterhin (elevation=0°)
