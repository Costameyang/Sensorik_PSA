# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import get_window

# Task 1.1 Function to load radar cube data
def load_radar_cube(file_path):
    """Load radar cube data from .npy file"""
    return np.load(file_path)

# Helper function to normalize window names
def _normalize_window_name(name):
    """Map common aliases to scipy/get_window accepted names."""
    if not isinstance(name, str):
        return name
    n = name.lower()
    if n == "hanning":
        return "hann"
    if n == "rect":
        return "boxcar"
    return n


def _plot_shape_of_data(radar_file, AntennaArray):
    # Plotting Shape of Data
    num_samples, num_chirps, num_channels = AntennaArray.shape
    print(f"Loaded {radar_file.name}\n - Number of Samples: {num_samples}\n - Number of Chirps: {num_chirps}\n - Number of Channels: {num_channels}\n\n")

    print(f"Shape of AntennaArray: {AntennaArray.shape}")
    print(f"num samples: {num_samples}")
    print(f"Antenna Array: {AntennaArray[0,0,0]}")
    
    return num_samples, num_chirps, num_channels

def _select_single_channel(AntennaArray, channel=0):
    # Choose frame 0 by default (modify if you want to loop frames)
    if AntennaArray.ndim == 3:
        # assume shape (range, antenna, frames) or (range, angle, frames)
        single_channel = AntennaArray[:, :, channel]
    elif AntennaArray.ndim == 2:
        single_channel = AntennaArray
    else:
        raise ValueError("Unsupported radar_data dimensions: expected 2D or 3D array")
    return single_channel


def apply_window(single_channel, window_name, num_samples, num_chirps):
    # Task 1.2 Apply different windowing functions and process
    if isinstance(window_name, (list, tuple)):
        if len(window_name) == 0:
            print("Warnung: Keine Fenstertypen angegeben -> verwende 'boxcar'.")
            wr_name = wd_name = "boxcar"
        else:
            wr_name = _normalize_window_name(window_name[0])
            wd_name = _normalize_window_name(window_name[1]) if len(window_name) > 1 else wr_name
    else:
        wr_name = wd_name = _normalize_window_name(window_name)

    try:
        # get_window ist flexibler als die einzelnen np.funktionen
        window_range = get_window(wr_name, num_samples)
        window_doppler = get_window(wd_name, num_chirps)
        
    except (ValueError, TypeError) as e:
        print(f"Warnung: Fenstertypen '{wr_name}', '{wd_name}' nicht erkannt oder Parameter falsch: {e}. Verwende 'boxcar' als Fallback.")
        window_range = np.ones(num_samples)
        window_doppler = np.ones(num_chirps)
        # continue  # don't skip entire file; proceed with rectangular window
    
    # 2. 2D-Fenster erstellen
    window_2d = np.outer(window_range, window_doppler)

    # 3. Fenster anwenden
    windowed_data = single_channel * window_2d

    return windowed_data, wr_name, wd_name



def perform_2d_fft(windowed_data):   
    # Führe die 2D-FFT durch
    fft_result_2d = np.fft.fft2(windowed_data)
    return fft_result_2d



def plot_fft_result(fft_result_2d, radar_file, channel):
    # Task 1.4 Plot FFT Ergebnis
    # 1. Verschiebe den Nullfrequenzpunkt (wie bisher)
    fft_shifted = np.fft.fftshift(fft_result_2d)

    # 2. Berechne die absolute Amplitude in dB (wie bisher)
    magnitude_db = 20 * np.log10(np.abs(fft_shifted) + 1e-10)

    # 3. NEU: Normalisieren auf den Peak (0 dB)
    # Finde den maximalen Wert und ziehe ihn von allen Werten ab.
    normalized_db = magnitude_db - np.max(magnitude_db)

    # 4. Plotte die normalisierten Daten
    plt.figure(figsize=(10, 7))
    
    # Verwende 'normalized_db' für den Plot
    # Setze vmax=0 (der Peak) und vmin=-50 (um das Rauschen abzuschneiden)
    plt.imshow(normalized_db, aspect='auto', cmap='seismic', vmin=-50, vmax=0)
    plt.colorbar()
    
    # Dynamischer Titel
    plot_title = f'Task 1: Range/Doppler Heat Map over channel {channel} \nData File: {radar_file.name}'
    plt.title(plot_title)
    plt.xlabel('Doppler Index (Bins)')
    plt.ylabel('Range Index (Bins)')
    
    # Speichern oder anzeigen
    # plt.savefig(f"step1_{radar_file.stem}_{window_name}_norm.png")
    plt.show()



def Task_Step_1(radar_file, window_types):
    # Print wht File is being processed
    print(f"\nProcessing {radar_file.name}")
    

    # TASK 1.1 Load radar cube
    AntennaArray = load_radar_cube(radar_file)

    # Plot Shape of Data
    num_samples, num_chirps, num_channels = _plot_shape_of_data(radar_file, AntennaArray)

    # Select single channel for processing (e.g., channel 0)
    channel = 0
    single_channel = _select_single_channel(AntennaArray, channel)


    # TASK 1.2 Apply different windowing functions and process
    windowed_data, wr_name, wd_name = apply_window(single_channel, window_types, num_samples, num_chirps)


    # TASK 1.3 Perform 2D-FFT    
    fft_result_2d = perform_2d_fft(windowed_data)


    # TASK 1.4 Plot FFT Result
    plot_fft_result(fft_result_2d, radar_file, channel)

    return fft_result_2d, channel, num_samples, num_chirps, num_channels





def _define_konstants_and_parameters(num_samples, num_chirps):
    # --- STEP 2: Parameter-Definition ---
        # Konstanten
        c = 299792458  # Lichtgeschwindigkeit in m/s
        
        # Konfigurationsparameter (in SI-Einheiten)
        f_center = 78.26375e9   # Center Frequency (Hz)
        B = 2.5275e9          # Radar Bandwidth (Hz)
        f_sampling = 8e6      # Sampling Rate (Hz)
        Nc = num_samples      # N samples per chirp (256)
        N = num_chirps        # N Chirps per CPI (128)
        tc = 55e-6            # Chirp time / interval (s)
        TX_CHANNELS = 12      # Aus "TDM-MIMO (12 Tx..."
        RX_CHANNELS = 16      # Aus "TDM-MIMO (...16 Rx)"
        
        return c, f_center, B, f_sampling, Nc, N, tc, TX_CHANNELS, RX_CHANNELS

def _print_key_parameters(chirp_slope, range_res, range_max, vel_res, vel_max):
    print("--- Calculated Key Parameters (Step 2) ---")
    print(f"  Chirp Slope: {chirp_slope:.2e} Hz/s")
    print(f"  Range Resolution: {range_res:.2f} m")
    print(f"  Max Range: {range_max:.2f} m")
    print(f"  Velocity Resolution: {vel_res:.4f} m/s")
    print(f"  Max Velocity: {vel_max:.3f} m/s (Range: +/- {vel_max:.3f} m/s)")
    print("-------------------------------------------\n")


def calculation_of_key_parameter(num_samples, num_chirps):
    # --- STEP 2: Parameter-Definition ---
    c, f_center, B, f_sampling, Nc, N, tc, TX_CHANNELS, RX_CHANNELS = _define_konstants_and_parameters(num_samples, num_chirps)
        

    # STEP 2: Task 2.1 Parameterberechnung ---

    # 1. Chirp Slope
    chirp_slope = B / ((1/f_sampling)*Nc) 
    
    # 2. Range-Resolution
    range_res = c / (2 * B)

    # 3. Max Range
    range_max = (f_sampling * c) / (2 * chirp_slope)
    
    # 4. Max Velocity
    vel_max = c / (2 * f_center * tc * TX_CHANNELS)
    
    # 5. Velocity Resolution
    vel_res = c / (2 * f_center * tc * N * TX_CHANNELS)  
        
    _print_key_parameters(chirp_slope, range_res, range_max, vel_res, vel_max)

    return chirp_slope, range_res, range_max, vel_res, vel_max



def plot_fft_results(radar_file, fft_result_2d , channel, range_max, vel_max, num_samples, num_chirps):
    # Plot FFT Results
    fft_shifted = np.fft.fftshift(fft_result_2d)
    magnitude_db = 20 * np.log10(np.abs(fft_shifted) + 1e-10)
    normalized_db = magnitude_db - np.max(magnitude_db)

    plt.figure(figsize=(10, 7))
    
    # --- NEU IN STEP 2: Task 2.2 Achsen erstellen und Plot anpassen ---
    
    # Erstelle die Achsenvektoren
    # Y-Achse (Range) geht von 0 bis range_max
    range_axis = np.linspace(0, range_max, num_samples)
    
    # X-Achse (Velocity) geht von -vel_max bis +vel_max
    velocity_axis = np.linspace(-vel_max, vel_max, num_chirps)
    
    # Verwende 'extent' um die Achsen im Plot zu skalieren
    # [x_min, x_max, y_min, y_max]
    plot_extent = [velocity_axis[0], velocity_axis[-1], range_axis[0], range_axis[-1]]
    
    # 'origin='lower'' setzt den (0,0)-Punkt nach unten links
    plt.imshow(normalized_db, aspect='auto', cmap='seismic', vmin=-50, vmax=0,
                extent=plot_extent, origin='lower')
    
    plt.colorbar(label='Relative Amplitude (dB)')
    
    # Titel und Labels für Step 2 aktualisieren
    plot_title = f'Task 2: Range-Doppler Plot (Scaled) - Channel {channel} \nData File: {radar_file.name}'
    plt.title(plot_title)
    
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Distance (m)')
    # -----------------------------------------------------------------
    
    plt.show()


def Task_Step_2(radar_file, fft_result_2d, channel, num_samples, num_chirps, num_channels):


    # TASK 2.1 Calculation of Key Parameters
    chirp_slope, range_res, range_max, vel_res, vel_max = calculation_of_key_parameter(num_samples, num_chirps)

    # TASK 2.2 Plot Range-Doppler Map with Scaled Axes
    plot_fft_results(radar_file, fft_result_2d, channel, range_max, vel_max, num_samples, num_chirps)

# Main processing function
def main():
    # Paths to radar cube data
    radar_files = Path("RadarCube/USRR_Dynamic10m/3D/RadarCube7.npy") #Path("RadarCube/MRR_CornField/3D/RadarCube1.npy"),
        
    # Window types to test
    window_types = ['blackman', 'hanning']
    #window_types = ['hanning']  # For simplicity, only use Hanning in this example
    #window_types = ['hamming']  # For simplicity, only use Hamming in this example
    #window_types = ['blackman']  # For simplicity, only use Blackman in this example
    #window_types = ['bartlett']  # For simplicity, only use Bartlett in this example
    #window_types = ['kaiser']  # For simplicity, only use Kaiser in this example    
    #window_types = ['boxcar']  # For simplicity, only use Boxcar in this example
    #window_types = ['flattop']  # For simplicity, only use Flattop in this example

    fft_result_2d, channel, num_samples, num_chirps, num_channels = Task_Step_1(radar_files, window_types)



    Task_Step_2(radar_files, fft_result_2d, channel, num_samples, num_chirps, num_channels)
        

# Main entry point
if __name__ == "__main__":
    main()
