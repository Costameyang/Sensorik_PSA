"""
Testskript: Überprüft Elevation-Antennen-Auswahl, führt FFTs aus,
zeigt Debug-Informationen und plottet Azimuth- & Elevation-Spektren.

Ausführen:
    python dev_tests/test_task5_elevation.py

Die erzeugte Grafik wird als `test_task5_elevation.png` im Projektordner gespeichert.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

# Lokale Importe
from main import USRR_Dynamic_Config
from radar import Radar


def main():
    # Konfiguration (USRR Beispiel)
    cfg = USRR_Dynamic_Config()

    # Erzeuge Radar-Objekt (verwende num=0 - passe an falls nötig)
    radar = Radar(*cfg, num=0, use_tk=False, output_print=True, Dim_4=True)

    print("\n--- Lade 4D-ADC-Cube und erstelle virtuelles Array ---")
    radar.load_radar_cube()

    # Debug: Formen und Antennenpositionen
    if hasattr(radar, 'ADCCube_4D') and radar.ADCCube_4D is not None:
        print(f"ADCCube_4D shape: {radar.ADCCube_4D.shape}")
    print(f"AntennaArray (3D) shape: {radar.AntennaArray.shape}")
    print(f"AntennaPositions shape: {radar.AntennaPositions.shape}")
    print(f"AntennaPositions (first 12):\n{radar.AntennaPositions[:12]}\n")

    # Kontrolle: X- und Y-Range
    x_min, x_max = np.min(radar.AntennaPositions[:,0]), np.max(radar.AntennaPositions[:,0])
    y_min, y_max = np.min(radar.AntennaPositions[:,1]), np.max(radar.AntennaPositions[:,1])
    print(f"X-Range: {x_min:.6f} .. {x_max:.6f} (m)")
    print(f"Y-Range: {y_min:.6f} .. {y_max:.6f} (m)")

    # Bestimme Azimuth- und Elevation-Antennen indices wie in Task 5
    azimuth_ind = np.where(radar.AntennaPositions[:, 1] == 0)[0]
    azimuth_positions = radar.AntennaPositions[azimuth_ind, 0]
    val_a, idx_a = np.unique(azimuth_positions, return_index=True)
    AzimuthAntennaOnly = azimuth_ind[idx_a]

    elevation_ind = np.where(radar.AntennaPositions[:, 0] == 0)[0]
    elevation_positions = radar.AntennaPositions[elevation_ind, 1]
    val_e, idx_e = np.unique(elevation_positions, return_index=True)
    ElevationAntennaOnly = elevation_ind[idx_e]

    print(f"Azimuth antenna indices (y==0), count: {len(AzimuthAntennaOnly)}\n{AzimuthAntennaOnly}")
    print(f"Azimuth antenna x-positions: {radar.AntennaPositions[AzimuthAntennaOnly,0]}\n")

    print(f"Elevation antenna indices (x==0), count: {len(ElevationAntennaOnly)}\n{ElevationAntennaOnly}")
    print(f"Elevation antenna y-positions: {radar.AntennaPositions[ElevationAntennaOnly,1]}\n")

    # Einfache Plausibilitätsprüfungen
    if len(ElevationAntennaOnly) < 2:
        print("\n[WARNUNG] Es wurden weniger als 2 Elevation-Antennen gefunden. Die Elevation-FFT ergibt dann wahrscheinlich eine konstante Linie.")

    # Berechne Range/Doppler-FFT (analog zu Task_Step_2: nutze alle Channels summiert)
    sum_channels = np.sum(radar.AntennaArray, axis=2)
    # Berechne Schlüsselparameter (range_max, vel_max, etc.) bevor Achsen verwendet werden
    try:
        radar.calculation_of_key_parameter()
    except Exception as e:
        print(f"Warnung: calculation_of_key_parameter() schlug fehl: {e}")

    # Optional: aktualisiere Antennenparameter falls nötig
    try:
        radar._calculate_antenna_parameters()
    except Exception:
        # nicht kritisch; nur wenn Funktion vorhanden
        pass

    radar.apply_window(sum_channels, plot_window=False)
    radar.perform_2d_fft(first_time=False)
    # Erzeuge 4D-FFT (Range/Doppler über alle Rx/Tx), damit Task 5 den 4D-Pfad nutzen kann
    try:
        radar.perform_4d_fft()
        print(f"Performed 4D FFT, shape: {radar.fft_4d_shifted.shape}")
    except Exception as e:
        print(f"Warnung: perform_4d_fft() schlug fehl: {e}")

    # Debug: Rx/Tx Positionsschätzungen, falls vorhanden
    if getattr(radar, 'Rx_positions_est', None) is not None and getattr(radar, 'Tx_positions_est', None) is not None:
        print(f"Rx_positions_est shape: {radar.Rx_positions_est.shape}")
        print(radar.Rx_positions_est)
        print(f"Tx_positions_est shape: {radar.Tx_positions_est.shape}")
        print(radar.Tx_positions_est)
        # Zeige mittlere Abstände
        try:
            rx_dx = np.mean(np.diff(np.sort(radar.Rx_positions_est[:,0])))
            tx_dy = np.mean(np.diff(np.sort(radar.Tx_positions_est[:,1])))
            print(f"Estimated rx dx: {rx_dx:.6f}, tx dy: {tx_dy:.6f}")
        except Exception:
            pass
    else:
        print("Keine Rx/Tx-Positionsschätzungen verfügbar (VirtualPositions nicht kompatibel)")

    # Wähle den stärksten Peak in Range-Doppler als Untersuchungs-Bin
    mag = np.abs(radar.fft_shifted)
    r_bin, d_bin = np.unravel_index(np.argmax(mag), mag.shape)
    print(f"Ausgewählter Range-Doppler-Bin: r_bin={r_bin}, d_bin={d_bin}")

    # Achsen
    range_axis = np.linspace(0, radar.range_max, radar.num_samples)
    velocity_axis = np.linspace(-radar.vel_max, radar.vel_max, radar.num_chirps)
    print(f"Range (m) at r_bin: {range_axis[r_bin]:.3f}, Velocity (m/s) at d_bin: {velocity_axis[d_bin]:.3f}")

    # Antennensignal für diesen Bin
    antenna_signal = radar.AntennaArray[r_bin, d_bin, :]
    print(f"Antenna signal length (num channels): {antenna_signal.size}")

    # FFT-Parameter
    nfft = 512
    d_antenna = radar.wavelength / 2
    # Wenn 4D-FFT vorhanden ist, verwende Rx x Tx Matrix und mache 2D-Angular-FFT
    if getattr(radar, 'fft_4d_shifted', None) is not None:
        try:
            ant_mat = radar.fft_4d_shifted[r_bin, d_bin, :, :]
            print(f"Using ant_mat shape (Rx,Tx): {ant_mat.shape}")
            # 2D-FFT für Winkelraum
            nfft_x = max(64, ant_mat.shape[0] * 4)
            nfft_y = max(64, ant_mat.shape[1] * 4)
            ang2d = np.fft.fftshift(np.fft.fft2(ant_mat, s=(nfft_x, nfft_y)))
            ang2d_mag = np.abs(ang2d)

            # 1D-Schnitte (Summation über Achsen)
            az_1d = np.sum(ang2d_mag, axis=1)
            el_1d = np.sum(ang2d_mag, axis=0)

            # Verwende geschätzte Abstände falls vorhanden
            d_x = d_y = d_antenna
            if getattr(radar, 'Rx_positions_est', None) is not None:
                try:
                    dx_candidates = np.diff(np.sort(radar.Rx_positions_est[:,0]))
                    dx_pos = dx_candidates[dx_candidates > 1e-9]
                    if dx_pos.size > 0:
                        d_x = float(np.mean(dx_pos))
                except Exception:
                    pass
                # Candidate for vertical spacing from Rx positions (in case Rx rows vary in y)
                try:
                    dy_rx = np.diff(np.sort(radar.Rx_positions_est[:,1]))
                    dy_rx_pos = dy_rx[dy_rx > 1e-9]
                    d_y_rx = float(np.mean(dy_rx_pos)) if dy_rx_pos.size>0 else 0.0
                except Exception:
                    d_y_rx = 0.0
            else:
                d_y_rx = 0.0

            if getattr(radar, 'Tx_positions_est', None) is not None:
                try:
                    dy_tx = np.diff(np.sort(radar.Tx_positions_est[:,1]))
                    dy_tx_pos = dy_tx[dy_tx > 1e-9]
                    d_y_tx = float(np.mean(dy_tx_pos)) if dy_tx_pos.size>0 else 0.0
                except Exception:
                    d_y_tx = 0.0
            else:
                d_y_tx = 0.0

            # Wähle robusten d_y (bevorzugt Rx-Vert-Abstand, sonst Tx-Vert, sonst fallback)
            if d_y_rx > 1e-6:
                d_y = d_y_rx
            elif d_y_tx > 1e-6:
                d_y = d_y_tx
            else:
                d_y = d_antenna

            # Winkelachsen
            k_az = np.arange(ang2d_mag.shape[0]) - ang2d_mag.shape[0]/2
            theta_az = np.degrees(np.arcsin(np.clip((k_az / (ang2d_mag.shape[0]/2)) * (radar.wavelength/(2*d_x)), -1, 1)))
            k_el = np.arange(ang2d_mag.shape[1]) - ang2d_mag.shape[1]/2
            theta_el = np.degrees(np.arcsin(np.clip((k_el / (ang2d_mag.shape[1]/2)) * (radar.wavelength/(2*d_y)), -1, 1)))

            power_az_db = 20*np.log10(az_1d + 1e-12)
            power_el_db = 20*np.log10(el_1d + 1e-12)

            print(f"Azimuth theta range: {theta_az.min():.2f} .. {theta_az.max():.2f} deg")
            print(f"Azimuth power (dB) min/max: {power_az_db.min():.2f}/{power_az_db.max():.2f}")
            print(f"Elevation theta range: {theta_el.min():.2f} .. {theta_el.max():.2f} deg")
            print(f"Elevation power (dB) min/max: {power_el_db.min():.2f}/{power_el_db.max():.2f}")
        except Exception as e:
            print(f"Fehler bei 4D angular processing: {e}")
            # Fallback: alte Methode
            ant_mat = None
    else:
        ant_mat = None

    # Falls kein 4D-Angular-Matrixpfad benutzt wurde, fall back auf virtual-channel analysis
    if ant_mat is None:
        # --- Azimuth ---
        x_az = antenna_signal[AzimuthAntennaOnly]
        print(f"x_az shape: {x_az.shape}")
        win_az = np.hanning(len(x_az)) if len(x_az) > 0 else np.array([1.0])
        az_fft = np.fft.fftshift(np.fft.fft(x_az * win_az, n=nfft))
        az_spectrum = np.abs(az_fft)
        k_az = np.arange(nfft) - nfft/2
        sin_theta_az = (k_az / (nfft/2)) * (radar.wavelength / (2 * d_antenna))
        sin_theta_az = np.clip(sin_theta_az, -1, 1)
        theta_az = np.degrees(np.arcsin(sin_theta_az))
        power_az_db = 10*np.log10(az_spectrum**2 + 1e-10)

        print(f"Azimuth theta range: {theta_az.min():.2f} .. {theta_az.max():.2f} deg")
        print(f"Azimuth power (dB) min/max: {power_az_db.min():.2f}/{power_az_db.max():.2f}")

        # --- Elevation ---
        x_el = antenna_signal[ElevationAntennaOnly]
        print(f"x_el shape: {x_el.shape}")
        win_el = np.hanning(len(x_el)) if len(x_el) > 0 else np.array([1.0])
        el_fft = np.fft.fftshift(np.fft.fft(x_el * win_el, n=nfft))
        el_spectrum = np.abs(el_fft)
        k_el = np.arange(nfft) - nfft/2
        sin_theta_el = (k_el / (nfft/2)) * (radar.wavelength / (2 * d_antenna))
        sin_theta_el = np.clip(sin_theta_el, -1, 1)
        theta_el = np.degrees(np.arcsin(sin_theta_el))
        power_el_db = 10*np.log10(el_spectrum**2 + 1e-10)

        print(f"Elevation theta range: {theta_el.min():.2f} .. {theta_el.max():.2f} deg")
        print(f"Elevation power (dB) min/max: {power_el_db.min():.2f}/{power_el_db.max():.2f}")

    # Plotten
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))

    ax1.plot(theta_az, power_az_db, '-b')
    ax1.set_title('Azimuth-Spektrum (horizontal, y=0)')
    ax1.set_xlabel('Azimuth [deg]')
    ax1.set_ylabel('Power [dB]')
    ax1.grid(True)

    ax2.plot(theta_el, power_el_db, '-g')
    ax2.set_title('Elevation-Spektrum (vertikal, x=0)')
    ax2.set_xlabel('Elevation [deg]')
    ax2.set_ylabel('Power [dB]')
    ax2.grid(True)

    fig.suptitle(f"Test Task5 Elevation Check - Bin r={r_bin}, d={d_bin}")
    plt.tight_layout(rect=[0,0,1,0.95])

    out_path = Path('test_task5_elevation.png')
    plt.savefig(out_path)
    print(f"Plot gespeichert: {out_path.resolve()}")
    plt.show()

    # Einfache Assertions / Checks
    if len(ElevationAntennaOnly) < 2:
        print('\n[ERgebnis] Elevation-Antennenanzahl < 2 -> Vermutlich falsche Auswahl der Elevation-Dimension')
    else:
        print('\n[ERgebnis] Elevation-Antennen vorhanden -> Prüfe ob Elevation-Spektrum sinnvoll variiert (siehe Plot)')


if __name__ == '__main__':
    main()
