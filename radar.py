# Import necessary libraries
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import get_window
from tkinter_fenster import TkPrinter


###########################################################
# Radar Class Definition
###########################################################
class Radar:
 
    ### INITIALIZATION of Logging and Parameters ###
    def _log(self, *args, end="\n"):
        """Internal logging helper: sends output to Tk window if enabled, otherwise to stdout."""
        msg = ' '.join(str(a) for a in args) + end
        if getattr(self, "use_tk", False) and getattr(self, "printer", None) is not None:
            try:
                self.printer.write(msg)
            except Exception:
                # fallback to stdout on error
                print(msg, end='')
        else:
            print(msg, end='')


    def __init__(self, radar_file, f_center, B, f_sampling, num_samples, tc, num_chirps, Rx_gain, Tx_Channels, Rx_Channels, num=0, use_tk=False):
        # Path to radar cube data
        self.radar_file = Path(radar_file + f"{num}.npy")

        self.use_tk = bool(use_tk)
        self.printer = TkPrinter(title=f"Radar Logs: {self.radar_file.name}")

        self._log(f"\nInitializing Radar with file: {self.radar_file}")


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


        # Windowing parameters
        self.Range_window_type = 'hanning' # 'hanning' # 'boxcar'
        self.Doppler_window_type = 'hanning' # 'hanning'
        self.window_name = [self.Range_window_type, self.Doppler_window_type]   #['blackman', 'hanning']  , ['blackman', 'blackman']



    def Task_Step_1(self):
        # Print wht File is being processed
        self._log(f"\n------------------------------------------------------------------------------------------------")
        self._log(f"------------------------------------------------------------------------------------------------")
        self._log(f"\nTask 1: (Processing file {self.radar_file.name})")
        

        # TASK 1.1 Load radar cube
        self.load_radar_cube()

        # Plot Shape of Data
        self._plot_shape_of_data()

        # Select single channel for processing (e.g., channel 0)
        self.channel = 0
        self._select_single_channel()


        # TASK 1.2 Apply different windowing functions and process
        self.apply_window(self.single_channel, True)


        # TASK 1.3 Perform 2D-FFT    
        self.perform_2d_fft(first_time=True)


        # TASK 1.4 Plot FFT Result
        self.plot_fft_result()



    def Task_Step_2(self):
        # Task 2: Print Key Parameters and Plot Range-Doppler Map
        self._log(f"\n------------------------------------------------------------------------------------------------")
        self._log(f"------------------------------------------------------------------------------------------------")
        self._log(f"\nTask 2: (Processing file {self.radar_file.name})")

        # TASK 2.1 Calculation of Key Parameters
        self.calculation_of_key_parameter()

        # TASK 2.3 Plot Range-Doppler Map with Scaled Axes
        self.plot_fft_results(name=f"Channel {self.channel}", Task="2.3")


        # With all Channels
        # sum all Channels
        sum_Channels = np.sum(self.AntennaArray, axis=2)
        self.apply_window(sum_Channels, False)

        # TASK 1.3 Perform 2D-FFT    
        self.perform_2d_fft(first_time=False)


        # TASK 2.4 Plot Range-Doppler Map with Scaled Axes
        self.plot_fft_results(name="All Channels Summed", Task="2.4")

        self.plot_fft_results_3d(name="All Channels Summed 3D")



    def task3(self):
        # Platz für Task 3 Implementierung
        self._log(f"\n------------------------------------------------------------------------------------------------")
        self._log(f"------------------------------------------------------------------------------------------------")  
        self._log(f"\nTask 3: (Processing file {self.radar_file.name})")

        # Hier können Sie die Berechnungen für Task 3 implementieren
        pass



    ################
    # FOR TASK 1   #
    ################

    # Task 1.1 Function to load radar cube data
    def load_radar_cube(self):
        """Load radar cube data from .npy file"""
        self.AntennaArray = np.load(self.radar_file)
        self._log(f"\n- Task 1.1: \n\t- Load radar cube data from {self.radar_file}")

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
        #else:
            #print(f"Loaded {self.radar_file.name}\n - Number of Samples: {num_samples}\n - Number of Chirps: {num_chirps}\n - Number of Channels: {self.num_channels}\n\n")



    def _select_single_channel(self):
        self.single_channel = self.AntennaArray[:, :, self.channel]



    def apply_window(self, channel, plot_window=False):
        if plot_window:
            self._log(f"\n - Task 1.2: \n\t- Apply windowing function(s) with '{self.Range_window_type}' for Range and '{self.Doppler_window_type}' for Doppler")

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

        if plot_window:
            # im 3d plot die window funktion anzeigen
            self._log(f"\t- Plot 2D Window Function (Range: {self.Range_window_type} x Doppler: {self.Doppler_window_type})")
            try:
                # Achsen: Doppler (Spalten) und Range (Zeilen)
                doppler_axis = np.arange(self.num_chirps)
                range_axis = np.arange(self.num_samples)
                D, R = np.meshgrid(doppler_axis, range_axis)

                title = f'2D Window Function (Range: {self.Range_window_type} x Doppler: {self.Doppler_window_type})'
                # Verwende generische 3D-Funktion
                self._generic_plot_3d(D, R, self.window_2d,
                                      xlabel=f"Doppler Bin ('{self.Doppler_window_type}')", ylabel=f"Range Bin ('{self.Range_window_type}')", zlabel='Window Amplitude',
                                      title=title, cmap='viridis', vmin=None, vmax=None, elev=25, azim=-60)
            except Exception as e:
                print(f"Warnung: 3D-Plot der Fensterfunktion fehlgeschlagen: {e}")

        # 3. Fenster anwenden
        self.windowed_data = channel * self.window_2d



    def perform_2d_fft(self, first_time=True):   
        # Führe die 2D-FFT durch
        if first_time:
            self._log(f"\n - Task 1.3: \n\t- Perform 2D-FFT on windowed data with shape {self.windowed_data.shape} for single channel {self.channel}")

        self.fft_result_2d = np.fft.fft2(self.windowed_data)
        # 1. Verschiebe den Nullfrequenzpunkt (wie bisher)
        self.fft_shifted = np.fft.fftshift(self.fft_result_2d, axes=(1,))



    def plot_fft_result(self):  
        # Task 1.4 Plot FFT Ergebnis
        self._log(f"\n - Task 1.4: \n\t- Plot FFT Result for single channel {self.channel} without axes scaling")

        # 1. Berechne die absolute Amplitude in dB (wie bisher)
        magnitude_db = 20 * np.log10(np.abs(self.fft_shifted) + 1e-10)

        # 2. Normalisieren auf den Peak (0 dB)
        # Finde den maximalen Wert und ziehe ihn von allen Werten ab.
        self.normalized_db = magnitude_db - np.max(magnitude_db)

        # 4. Plotte die normalisierten Daten (ohne skalierte Achsen -> einfache Darstellung)
        title = f'Task 1: Range/Doppler Heat Map over channel {self.channel} \nData File: {self.radar_file.name}'
        self._generic_plot_2d(self.normalized_db, extent=None,
                              xlabel='Doppler Index (Bins)', ylabel='Range Index (Bins)',
                              title=title, cmap='seismic', vmin=-50, vmax=0, colorbar_label='Relative Amplitude (dB)')



    ################
    # FOR TASK 2   #
    ################

    def _print_key_parameters(self):
        self._log(f"\t   - Chirp Slope: {self.chirp_slope:.2e} Hz/s")
        self._log(f"\t   - Max Range: {self.range_max:.2f} m")
        self._log(f"\t   - Range Resolution: {self.range_res:.2f} m")
        self._log(f"\t   - Max Velocity: {self.vel_max:.3f} m/s")
        self._log(f"\t   - Velocity Resolution: {self.vel_res:.4f} m/s")
        



    def calculation_of_key_parameter(self):
        self._log(f"\n - Task 2.1: \n\t- Calculation of Key Parameters")

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



    def _prepare_range_velocity_axes(self):
        # Erstelle Achsenvektoren für Range- / Velocity-Skalierung
        range_axis = np.linspace(0, self.range_max, self.num_samples)
        velocity_axis = np.linspace(-self.vel_max, self.vel_max, self.num_chirps)
        plot_extent = [velocity_axis[0], velocity_axis[-1], range_axis[0], range_axis[-1]]
        return range_axis, velocity_axis, plot_extent



    def plot_fft_results(self, name="", Task="2.2"):
        self._log(f"\n - Task {Task}: \n\t- Plot Range-Doppler Map with Scaled Axes for {name}")

        #fft_shifted = self.fft_result_2d # np.fft.fftshift(self.fft_result_2d)
        # 2. Berechne die absolute Amplitude in dB (wie bisher)
        magnitude_db = 20 * np.log10(np.abs(self.fft_shifted) + 1e-10)

        # 3. NEU: Normalisieren auf den Peak (0 dB)
        # Finde den maximalen Wert und ziehe ihn von allen Werten ab.
        self.normalized_db = magnitude_db - np.max(magnitude_db)

        # Plot FFT Results
        range_axis, velocity_axis, plot_extent = self._prepare_range_velocity_axes()

        title = f'Task 2: Range-Doppler Plot (Scaled) - {name} \nData File: {self.radar_file.name}'
        self._generic_plot_2d(self.normalized_db, extent=plot_extent,
                              xlabel='Velocity (m/s)', ylabel='Distance (m)',
                              title=title, cmap='seismic', vmin=-50, vmax=0, colorbar_label='Relative Amplitude (dB)')



    def plot_fft_results_3d(self, name="", decimate=(1,1), elev=30, azim=-60):
        self._log(f"\t- 3D Plot of Range-Doppler Map for {name}")    

        """
        3D-Plot der bereits berechneten Range-Doppler-Daten (self.normalized_db).
        - name: Titelzusatz
        - decimate: tuple (r_step, v_step) um ggf. Daten zu reduzieren (z.B. (2,2))
        - elev, azim: Ansichtswinkel
        """

        # Achsenvektoren wie in plot_fft_results
        range_axis, velocity_axis, _ = self._prepare_range_velocity_axes()

        # Erzeuge Gitter (Shapes: (num_samples, num_chirps))
        V, R = np.meshgrid(velocity_axis, range_axis)  # V: velocity, R: range
        Z = self.normalized_db

        title = f'Task 2 (3D): Range-Doppler Surface - {name} \nData File: {self.radar_file.name}'
        self._generic_plot_3d(V, R, Z,
                              xlabel='Velocity (m/s)', ylabel='Distance (m)', zlabel='Relative Amplitude (dB)',
                              title=title, cmap='seismic', vmin=-150, vmax=0, elev=elev, azim=azim, decimate=decimate)



    ########################
    # GENERIC PLOT HELPERS
    ########################

    def _generic_plot_2d(self, data, extent=None, xlabel='X', ylabel='Y', title='', cmap='seismic', vmin=None, vmax=None, colorbar_label=None, origin='lower', figsize=(10,7)):
        plt.figure(figsize=figsize)
        if extent is None:
            img = plt.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            img = plt.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, origin=origin)
        cbar = plt.colorbar(img)
        if colorbar_label:
            cbar.set_label(colorbar_label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()


    def _generic_plot_3d(self, X, Y, Z, xlabel='X', ylabel='Y', zlabel='Z', title='', cmap='seismic', vmin=None, vmax=None, elev=30, azim=-60, decimate=(1,1), figsize=(12,8)):
        # Optional decimation
        r_step, v_step = decimate
        if r_step > 1 or v_step > 1:
            X = X[::r_step, ::v_step]
            Y = Y[::r_step, ::v_step]
            Z = Z[::r_step, ::v_step]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Verwende explizite Norm/ScalarMappable, damit die Farben (z.B. Rot bei 0 dB) konsistent zu 2D sind
        from matplotlib import cm
        from matplotlib.colors import Normalize

        z_min = vmin if vmin is not None else np.nanmin(Z)
        z_max = vmax if vmax is not None else np.nanmax(Z)
        norm = Normalize(vmin=z_min, vmax=z_max)
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(Z)

        facecolors = mappable.to_rgba(Z)
        surf = ax.plot_surface(X, Y, Z, facecolors=facecolors, linewidth=0, antialiased=True, shade=False)
        # Clip Z to the colorbar range so Z-axis matches the color scale
        Z_plot = np.clip(Z, z_min, z_max)
        facecolors = mappable.to_rgba(Z_plot)
        surf = ax.plot_surface(X, Y, Z_plot, facecolors=facecolors, linewidth=0, antialiased=True, shade=False)

        cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label(zlabel)

        # Set z-axis limits to match color scale
        ax.set_zlim(z_min, z_max)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        plt.show()