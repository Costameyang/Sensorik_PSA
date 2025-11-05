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


# Main processing function
def main():
    # Paths to radar cube data
    radar_files = [
        #Path("RadarCube/MRR_CornField/3D/RadarCube1.npy"),
        Path("RadarCube/USRR_Dynamic10m/3D/RadarCube7.npy")
    ]
    
    # Window types to test
    window_types = ['blackman', 'hanning']
    #window_types = ['hanning']  # For simplicity, only use Hanning in this example
    #window_types = ['hamming']  # For simplicity, only use Hamming in this example
    #window_types = ['blackman']  # For simplicity, only use Blackman in this example
    #window_types = ['bartlett']  # For simplicity, only use Bartlett in this example
    #window_types = ['kaiser']  # For simplicity, only use Kaiser in this example    
    #window_types = ['boxcar']  # For simplicity, only use Boxcar in this example
    #window_types = ['flattop']  # For simplicity, only use Flattop in this example
    
    # Combine all window types into a single string for the title
    all_windows = ""
    for window in window_types:
        all_windows += f"{window}, \n"

    # Process each radar cube
    for radar_file in radar_files:
        print(f"\nProcessing {radar_file.name}")
        



        # Task 1.1 Load radar cube
        AntennaArray = load_radar_cube(radar_file)

        # Plotting Shape of Data
        num_samples, num_chirps, num_channels = AntennaArray.shape
        print(f"Loaded {radar_file.name}\n - Number of Samples: {num_samples}\n - Number of Chirps: {num_chirps}\n - Number of Channels: {num_channels}\n\n")

        # Choose frame 0 by default (modify if you want to loop frames)
        if AntennaArray.ndim == 3:
            # assume shape (range, antenna, frames) or (range, angle, frames)
            channel = 0
            single_channel = AntennaArray[:, :, channel]
        elif AntennaArray.ndim == 2:
            single_channel = AntennaArray
        else:
            raise ValueError("Unsupported radar_data dimensions: expected 2D or 3D array")




        # Task 1.2 Apply different windowing functions and process
        if isinstance(window_types, (list, tuple)):
            if len(window_types) == 0:
                print("Warnung: Keine Fenstertypen angegeben -> verwende 'boxcar'.")
                wr_name = wd_name = "boxcar"
            else:
                wr_name = _normalize_window_name(window_types[0])
                wd_name = _normalize_window_name(window_types[1]) if len(window_types) > 1 else wr_name
        else:
            wr_name = wd_name = _normalize_window_name(window_types)

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



        # Task 1.3 Führe die 2D-FFT durch
        fft_result_2d = np.fft.fft2(windowed_data)




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
        plot_title = f'Task 1: Range/Doppler Heat Map over channel {channel} \nData File: {radar_file.name} - Used Window: {all_windows}'
        plt.title(plot_title)
        plt.xlabel('velocity (m/s)')
        plt.ylabel('distance (m)')
        
        # Speichern oder anzeigen
        # plt.savefig(f"step1_{radar_file.stem}_{window_name}_norm.png")
        plt.show()
        

# Main entry point
if __name__ == "__main__":
    main()
