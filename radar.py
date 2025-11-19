# Import necessary libraries
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import get_window


class Radar:
    def __init__(self, radar_file, f_center, B, f_sampling, num_samples, tc, num_chirps, Rx_gain, Tx_Channels, Rx_Channels, num=0):
        # Path to radar cube data
        self.radar_file = Path(radar_file + f"{num}.npy")

        print(f"Initializing Radar with file: {self.radar_file}")

        # Konstants
        self.c = 299792458  # Lichtgeschwindigkeit in m/s

        # Parameter of Radar
        self.f_center = f_center
        self.B = B
        self.f_sampling = f_sampling
        self.num_samples = num_samples
        self.tc = tc
        self.num_chirps = num_chirps
        self.Rx_gain = Rx_gain
        
        self.Tx_Channels = Tx_Channels
        self.Rx_Channels = Rx_Channels

        self.window_name = ['hanning', 'hanning']   #['blackman', 'hanning']  



    def Task_Step_1(self):
        # Print wht File is being processed
        print(f"\nProcessing {self.radar_file.name}")
        

        # TASK 1.1 Load radar cube
        self.load_radar_cube()

        # Plot Shape of Data
        self._plot_shape_of_data()

        # Select single channel for processing (e.g., channel 0)
        self.channel = 0
        self._select_single_channel()


        # TASK 1.2 Apply different windowing functions and process
        self.apply_window(self.single_channel)


        # TASK 1.3 Perform 2D-FFT    
        self.perform_2d_fft()


        # TASK 1.4 Plot FFT Result
        self.plot_fft_result()



    def Task_Step_2(self):
        # TASK 2.1 Calculation of Key Parameters
        self.calculation_of_key_parameter()

        # TASK 2.2 Plot Range-Doppler Map with Scaled Axes
        self.plot_fft_results(name=f"Channel {self.channel}")


        # With all Channels
        # sum all Channels
        sum_Channels = np.sum(self.AntennaArray, axis=2)
        self.apply_window(sum_Channels)

        # TASK 1.3 Perform 2D-FFT    
        self.perform_2d_fft()


        # TASK 2.2 Plot Range-Doppler Map with Scaled Axes
        #self.plot_fft_results(name="All Channels Summed")

        self.plot_fft_results_3d(name="All Channels Summed 3D")



    def task3(self):
        # Platz für Task 3 Implementierung
        print("Task 3 wird ausgeführt")
        # Hier können Sie die Berechnungen für Task 3 implementieren
        pass



    ################
    # FOR TASK 1   #
    ################

    # Task 1.1 Function to load radar cube data
    def load_radar_cube(self):
        """Load radar cube data from .npy file"""
        self.AntennaArray = np.load(self.radar_file)

    # Helper function to normalize window names
    def _normalize_window_name(self, name):
        """Map common aliases to scipy/get_window accepted names."""
        if not isinstance(name, str):
            return name
        n = name.lower()
        if n == "hanning":
            return "hann"
        if n == "rect":
            return "boxcar"
        return n


    def _plot_shape_of_data(self):
        # Plotting Shape of Data
        num_samples, num_chirps, self.num_channels = self.AntennaArray.shape
        
        if num_samples != self.num_samples or num_chirps != self.num_chirps:
            print(f"Warnung: Expected Dimension ({self.num_samples}, {self.num_chirps}) are not equal to Format ({num_samples}, {num_chirps}).")
        else:
            print(f"Loaded {self.radar_file.name}\n - Number of Samples: {num_samples}\n - Number of Chirps: {num_chirps}\n - Number of Channels: {self.num_channels}\n\n")
        

    def _select_single_channel(self):
        self.single_channel = self.AntennaArray[:, :, self.channel]


    def apply_window(self, channel):
        # Task 1.2 Apply different windowing functions and process
        if isinstance(self.window_name, (list, tuple)):
            if len(self.window_name) == 0:
                print("Warnung: Keine Fenstertypen angegeben -> verwende 'boxcar'.")
                wr_name = wd_name = "boxcar"
            else:
                self.wr_name = self._normalize_window_name(self.window_name[0])
                self.wd_name = self._normalize_window_name(self.window_name[1]) if len(self.window_name) > 1 else self.wr_name
        else:
            self.wr_name = self.wd_name = self._normalize_window_name(self.window_name)

        try:
            # get_window ist flexibler als die einzelnen np.funktionen
            self.window_range = get_window(self.wr_name, self.num_samples)
            self.window_doppler = get_window(self.wd_name, self.num_chirps)
            
        except (ValueError, TypeError) as e:
            print(f"Warnung: Fenstertypen '{self.wr_name}', '{self.wd_name}' nicht erkannt oder Parameter falsch: {e}. Verwende 'boxcar' als Fallback.")
            self.window_range = np.ones(self.num_samples)
            self.window_doppler = np.ones(self.num_chirps)
            # continue  # don't skip entire file; proceed with rectangular window
        
        # 2. 2D-Fenster erstellen
        self.window_2d = np.outer(self.window_range, self.window_doppler)

        # 3. Fenster anwenden
        self.windowed_data = channel #* self.window_2d


    def perform_2d_fft(self):   
        # Führe die 2D-FFT durch
        print(f"FFT: {self.single_channel.shape}\n")
        self.fft_result_2d = np.fft.fft2(self.windowed_data)
        # 1. Verschiebe den Nullfrequenzpunkt (wie bisher)
        self.fft_shifted = np.fft.fftshift(self.fft_result_2d, axes=(1,))


    def plot_fft_result(self):  
        # Task 1.4 Plot FFT Ergebnis
        #fft_shifted = self.fft_result_2d # np.fft.fftshift(self.fft_result_2d)
        # 2. Berechne die absolute Amplitude in dB (wie bisher)
        magnitude_db = 20 * np.log10(np.abs(self.fft_shifted) + 1e-10)

        # 3. NEU: Normalisieren auf den Peak (0 dB)
        # Finde den maximalen Wert und ziehe ihn von allen Werten ab.
        self.normalized_db = magnitude_db - np.max(magnitude_db)

        # 4. Plotte die normalisierten Daten
        plt.figure(figsize=(10, 7))
        
        # Verwende 'normalized_db' für den Plot
        # Setze vmax=0 (der Peak) und vmin=-50 (um das Rauschen abzuschneiden)
        plt.imshow(self.normalized_db, aspect='auto', cmap='seismic', vmin=-50, vmax=0) # cmap='seismic'
        plt.colorbar()
        
        # Dynamischer Titel
        plot_title = f'Task 1: Range/Doppler Heat Map over channel {self.channel} \nData File: {self.radar_file.name}'
        plt.title(plot_title)
        plt.xlabel('Doppler Index (Bins)')
        plt.ylabel('Range Index (Bins)')
        
        # Speichern oder anzeigen
        # plt.savefig(f"step1_{radar_file.stem}_{window_name}_norm.png")
        plt.show()

    ################
    # FOR TASK 2   #
    ################

    def _print_key_parameters(self):
        print("--- Calculated Key Parameters (Step 2) ---")
        print(f"  Chirp Slope: {self.chirp_slope:.2e} Hz/s")
        print(f"  Range Resolution: {self.range_res:.2f} m")
        print(f"  Max Range: {self.range_max:.2f} m")
        print(f"  Velocity Resolution: {self.vel_res:.4f} m/s")
        print(f"  Max Velocity: {self.vel_max:.3f} m/s (Range: +/- {self.vel_max:.3f} m/s)")
        print("-------------------------------------------\n")


    def calculation_of_key_parameter(self):
        # 1. Chirp Slope
        self.chirp_slope = self.B / ((1/self.f_sampling)*self.num_samples) 

        # 2. Range-Resolution
        self.range_res = self.c / (2 * self.B)

        # 3. Max Range
        self.range_max = (self.f_sampling * self.c) / (2 * self.chirp_slope)

        # 4. Max Velocity
        self.vel_max = self.c / (2 * self.f_center * self.tc * self.Tx_Channels)

        # 5. Velocity Resolution
        self.vel_res = self.c / (2 * self.f_center * self.tc * self.num_chirps * self.Tx_Channels)  
            
        self._print_key_parameters()


    def plot_fft_results(self, name=""):
        #fft_shifted = self.fft_result_2d # np.fft.fftshift(self.fft_result_2d)
        # 2. Berechne die absolute Amplitude in dB (wie bisher)
        magnitude_db = 20 * np.log10(np.abs(self.fft_shifted) + 1e-10)

        # 3. NEU: Normalisieren auf den Peak (0 dB)
        # Finde den maximalen Wert und ziehe ihn von allen Werten ab.
        self.normalized_db = magnitude_db - np.max(magnitude_db)

        # Plot FFT Results
        plt.figure(figsize=(10, 7))

        # --- NEU IN STEP 2: Task 2.2 Achsen erstellen und Plot anpassen ---

        # Erstelle die Achsenvektoren
        # Y-Achse (Range) geht von 0 bis range_max
        range_axis = np.linspace(0, self.range_max, self.num_samples)

        # X-Achse (Velocity) geht von -vel_max bis +vel_max
        velocity_axis = np.linspace(-self.vel_max, self.vel_max, self.num_chirps)

        # Verwende 'extent' um die Achsen im Plot zu skalieren
        # [x_min, x_max, y_min, y_max]
        plot_extent = [velocity_axis[0], velocity_axis[-1], range_axis[0], range_axis[-1]]

        # 'origin='lower'' setzt den (0,0)-Punkt nach unten links
        plt.imshow(self.normalized_db, aspect='auto', cmap='seismic', vmin=-50, vmax=0,
                    extent=plot_extent, origin='lower')

        plt.colorbar(label='Relative Amplitude (dB)')

        # Titel und Labels für Step 2 aktualisieren
        plot_title = f'Task 2: Range-Doppler Plot (Scaled) - {name} \nData File: {self.radar_file.name}'
        plt.title(plot_title)

        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Distance (m)')
        # -----------------------------------------------------------------

        plt.show()

        print("DONE: Plot FFT Results with Scaled Axes")



    def plot_fft_results_3d(self, name="", decimate=(1,1), elev=30, azim=-60):
        """
        3D-Plot der bereits berechneten Range-Doppler-Daten (self.normalized_db).
        - name: Titelzusatz
        - decimate: tuple (r_step, v_step) um ggf. Daten zu reduzieren (z.B. (2,2))
        - elev, azim: Ansichtswinkel
        """
        # Falls noch nicht berechnet, berechne normalized_db
        if not hasattr(self, "normalized_db") or self.normalized_db is None:
            magnitude_db = 20 * np.log10(np.abs(self.fft_shifted) + 1e-10)
            self.normalized_db = magnitude_db - np.max(magnitude_db)

        # Achsenvektoren wie in plot_fft_results
        range_axis = np.linspace(0, self.range_max, self.num_samples)
        velocity_axis = np.linspace(-self.vel_max, self.vel_max, self.num_chirps)

        # Erzeuge Gitter (Shapes: (num_samples, num_chirps))
        V, R = np.meshgrid(velocity_axis, range_axis)  # V: velocity, R: range
        Z = self.normalized_db

        # Optional: Decimation für Performance
        r_step, v_step = decimate
        if r_step > 1 or v_step > 1:
            R = R[::r_step, ::v_step]
            V = V[::r_step, ::v_step]
            Z = Z[::r_step, ::v_step]

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Surface-Plot
        surf = ax.plot_surface(V, R, Z, cmap='seismic', linewidth=0, antialiased=True, vmin=-50, vmax=0)

        # Farbenleisten und Labels
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Relative Amplitude (dB)')

        ax.set_title(f'Task 2 (3D): Range-Doppler Surface - {name} \nData File: {self.radar_file.name}')
        ax.set_xlabel('Velocity (m/s)')
        ax.set_ylabel('Distance (m)')
        ax.set_zlabel('Relative Amplitude (dB)')

        ax.view_init(elev=elev, azim=azim)
        plt.show()