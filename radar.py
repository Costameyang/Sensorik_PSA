# Import necessary libraries
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import get_window
from scipy.ndimage import label, center_of_mass, maximum_filter
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


    def __init__(self, radar_file_3D, radar_file_4D, threshold_factor, factor_formular_max_velocity, f_center, B, f_sampling, num_samples, tc, chirp_slope, num_chirps, Rx_gain, Tx_Channels, Rx_Channels, num=0, use_tk=False, output_print=True):
        # Path to radar cube data
        self.radar_file_3D = Path(radar_file_3D + f"{num}.npy")
        self.radar_file_4D = Path(radar_file_4D + f"{num}.npy")

        self.output_print = output_print
        if self.output_print:
            self.use_tk = bool(use_tk)
            self.printer = TkPrinter(title=f"Radar Logs: {self.radar_file_3D.name}")

            self._log(f"\nInitializing Radar with file: {self.radar_file_3D}")


        # Konstants
        self.c = 299792458  # Lichtgeschwindigkeit in m/s

        # Parameter of Radar
        self.f_center = f_center
        self.B = B
        self.f_sampling = f_sampling
        self.num_samples = num_samples
        self.tc = tc
        self.num_chirps = num_chirps
        self.chirp_slope = chirp_slope
        self.Rx_gain = Rx_gain
        self.f_0 = self.f_center - self.B / 2  # Startfrequenz
        
        self.Tx_Channels = Tx_Channels
        self.Rx_Channels = Rx_Channels

        self.factor_formular_max_velocity = factor_formular_max_velocity  # Faktor im Nenner der Formel zur Berechnung der maximalen Geschwindigkeit (2 für TDM-MIMO)


        # Windowing parameters
        self.Range_window_type = 'hanning' # 'hanning' # 'boxcar'
        self.Doppler_window_type = 'hanning' # 'hanning'
        self.window_name = [self.Range_window_type, self.Doppler_window_type]   #['blackman', 'hanning']  , ['blackman', 'blackman']

        # CFAR-Kernel Paraeter
        self.kernel_matrix = None       # Angabe einer 2D-Matrix als Kernel möglich (0=außerhalb, 1=train, 2=guard, 3=CUT)
        self.train_range = 10 # 15
        self.train_doppler = 10 # 15
        self.guard_range = 8 # 10
        self.guard_doppler = 8 #  10
        self.threshold_factor = threshold_factor     # Parameter zur Schwellwertberechnung

        self.window_cifar_max_size = 5  # Fenstergröße für Non-Maximum Suppression bei CFAR (ungerade Zahl)

        # Load Antenna Array
        antenna_array_path = Path("RadarCube/AntennaArray.npy")
        antenna_array_raw = np.load(antenna_array_path).astype(np.float64)
        self.AntennaPositions = antenna_array_raw

        ##############
        # f_center oder f_0 ?
        self.wavelength = float(self.c / self.f_center)

        # Skaliere: Wenn Positionen in "halben Wellenlängen" sind
        #self.AntennaPositions = antenna_array_raw * (self.wavelength / 2)
        
        if self.output_print:
            print(f"Antenna Positions loaded from {antenna_array_path}, shape: {self.AntennaPositions.shape}")
            print(f"Antenna Positions dtype: {self.AntennaPositions.dtype}")
            print(f"X-Range: {np.min(self.AntennaPositions[:,0])*1000:.2f} to {np.max(self.AntennaPositions[:,0])*1000:.2f} mm")
            print(f"Y-Range: {np.min(self.AntennaPositions[:,1])*1000:.2f} to {np.max(self.AntennaPositions[:,1])*1000:.2f} mm\n")
        

        print(f"Antenna Positions loaded from {antenna_array_path}, shape: {self.AntennaPositions.shape}\n\n")
        #print(self.AntennaPositions)

    def Task_Step_1(self):
        if self.output_print:
            # Print wht File is being processed
            self._log(f"\n------------------------------------------------------------------------------------------------")
            self._log(f"------------------------------------------------------------------------------------------------")
            self._log(f"\nTask 1: (Processing file {self.radar_file_3D.name})")
    

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
        self.plot_fft_results(name=f"Channel {self.channel}", Task="1.4")



    def Task_Step_2(self):
        if self.output_print:
            # Task 2: Print Key Parameters and Plot Range-Doppler Map
            self._log(f"\n------------------------------------------------------------------------------------------------")
            self._log(f"------------------------------------------------------------------------------------------------")
            self._log(f"\nTask 2: (Processing file {self.radar_file_3D.name})")
       
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



    def Task_Step_3(self):
        if self.output_print:
            # Task 3 Implementierung
            self._log(f"\n------------------------------------------------------------------------------------------------")
            self._log(f"------------------------------------------------------------------------------------------------")  
            self._log(f"\nTask 3: (Processing file {self.radar_file_3D.name})")


        # Task 3.1 Kernel generieren und in Instanz abspeichern
        self.CFAR_kernel = self.create_cfar_kernel(
            kernel_matrix=self.kernel_matrix,
            train_range=self.train_range,
            train_doppler=self.train_doppler,
            guard_range=self.guard_range,
            guard_doppler=self.guard_doppler
        )

        # Task 3.2 Ausgabe der CFAR-Kernel-Kenngrößen im log
        self.print_output_cifar()
        
        # Task 3.3 Kernel als figure plotten
        self.plot_cfar_kernel(self.CFAR_kernel)

        # Task 3.3 CA-CFAR-Kernel auf Range-Doppler-Map anwenden und Schwellenwerte berechnen
        self.thresholds = self.compute_cfar_thresholds(data=None, kernel=self.CFAR_kernel, threshold_factor=self.threshold_factor, pad_fill=0)

        # Task 3.4 CFAR-3D-Plot mit Overlay der Schwellenwerte aufrufen
        self.plot_CFAR_results_3d(name="Task 3: All Channels Summed 3D with CFAR", decimate=(1,1), elev=30, azim=-60)

        # Task 3.5 Objekte durch Schwellwertvergleich identifizieren und ausgeben
        detections = self.apply_cfar_detection(data=self.fft_shifted, thresholds=self.thresholds)
        
        # Task 3.6 Visualisierung der detektierten Objekte als 2D-Plot
        self.plot_cfar_detections(name=f"CFAR Detections (factor={self.threshold_factor})", Task="3.6", mode=1) # mode=1 für bins / mode = 0 für meter


        # Task 3.7 Visualisierung der detektierten Objekte als Velocity Profile
        self.plot_range_profile_at_detection(detection_index=0)


    def Task_Step_4(self):
        if self.output_print:
            self._log(f"\n------------------------------------------------------------------------------------------------")
            self._log(f"------------------------------------------------------------------------------------------------")
            self._log(f"\nTask 4: Angle Estimation & 3D Localization (Processing file {self.radar_file_3D.name})")
        
        # 4.1 Berechne Wellenlänge und Antennenabstand
        self._calculate_antenna_parameters()
        
        # 4.2 Erzeuge virtuelle Antennen-Array (MIMO)
        self._create_virtual_array()
        
        # 4.3 Winkelschätzung für alle detektierten Objekte
        self.detections_3d = self._estimate_angles_for_detections()
        
        # 4.4 3D-Visualisierung der Objekte
        self._plot_3d_detections()
       

    ################
    # FOR TASK 1   #
    ################

    # Task 1.1 Function to load radar cube data
    def load_radar_cube(self):
        """Load radar cube data from .npy file"""
        self.AntennaArray = np.load(self.radar_file_3D)

        if self.output_print:
            self._log(f"\n- Task 1.1: \n\t- Load radar cube data from {self.radar_file_3D}")

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
            self._log(f"Loaded {self.radar_file_3D.name}\n - Number of Samples: {num_samples}\n - Number of Chirps: {num_chirps}\n - Number of Channels: {self.num_channels}\n\n")



    def _select_single_channel(self):
        self.single_channel = self.AntennaArray[:, :, self.channel]



    def apply_window(self, channel, plot_window=False):
        if plot_window:
            if self.output_print:
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
            if self.output_print:
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
            if self.output_print:
                self._log(f"\n - Task 1.3: \n\t- Perform 2D-FFT on windowed data with shape {self.windowed_data.shape} for single channel {self.channel}")

        self.fft_result_2d = np.fft.fft2(self.windowed_data)
        # 1. Verschiebe den Nullfrequenzpunkt (wie bisher)
        self.fft_shifted = np.fft.fftshift(self.fft_result_2d, axes=(1,))


    ################
    # FOR TASK 2   #
    ################

    def _print_key_parameters(self):
        if self.output_print:
            self._log(f"\t   - Chirp Slope: {self.chirp_slope:.2e} Hz/s")
            self._log(f"\t   - Max Range: {self.range_max:.2f} m")
            self._log(f"\t   - Range Resolution: {self.range_res:.2f} m")
            self._log(f"\t   - Max Velocity: {self.vel_max:.3f} m/s")
            self._log(f"\t   - Velocity Resolution: {self.vel_res:.4f} m/s")
        



    def calculation_of_key_parameter(self):
        if self.output_print:
            self._log(f"\n - Task 2.1: \n\t- Calculation of Key Parameters")

        # 1. Chirp Slope
        if self.chirp_slope == 0:
            self.chirp_slope = self.B / ((1/self.f_sampling)*self.num_samples) 
        #print(self.chirp_slope/1e12, " Hz/s")

        # 2. Range-Resolution
        self.range_res = self.c / (2 * self.B)

        # 3. Max Range
        self.range_max = (self.f_sampling * self.c) / (2 * self.chirp_slope)

        # 4. Max Velocity
        # MARKER 4 bei MRR richtiges Ergebnis und USRR falsch?
        self.vel_max = self.c / (self.factor_formular_max_velocity * self.f_center * self.tc * self.Tx_Channels)

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

        #fft_shifted = self.fft_result_2d # np.fft.fftshift(self.fft_result_2d)
        # 2. Berechne die absolute Amplitude in dB (wie bisher)
        magnitude_db = 20 * np.log10(np.abs(self.fft_shifted) + 1e-10)

        # 3. NEU: Normalisieren auf den Peak (0 dB)
        # Finde den maximalen Wert und ziehe ihn von allen Werten ab.
        self.normalized_db = magnitude_db - np.max(magnitude_db)

        # Plot FFT Results
        if Task[0] == '1':
            if self.output_print:
                self._log(f"\n - Task 1.4: \n\t- Plot FFT Result for single channel {self.channel} without axes scaling")

            title = f'Task 1: Range/Doppler Heat Map over channel {self.channel} \nData File: {self.radar_file_3D.name}'
            self._generic_plot_2d(self.normalized_db, extent=None,
                                xlabel='Doppler Index (Bins)', ylabel='Range Index (Bins)',
                                title=title, cmap='seismic', vmin=-50, vmax=0, colorbar_label='Relative Amplitude (dB)')
        if Task[0] == '2':
            if self.output_print:
                self._log(f"\n - Task {Task}: \n\t- Plot Range-Doppler Map with Scaled Axes for {name}")

            range_axis, velocity_axis, plot_extent = self._prepare_range_velocity_axes()

            title = f'Task 2: Range-Doppler Plot (Scaled) - {name} \nData File: {self.radar_file_3D.name}'
            self._generic_plot_2d(self.normalized_db, extent=plot_extent,
                                xlabel='Velocity (m/s)', ylabel='Distance (m)',
                                title=title, cmap='seismic', vmin=-50, vmax=0, colorbar_label='Relative Amplitude (dB)')



    def plot_fft_results_3d(self, name="", decimate=(1,1), elev=30, azim=-60):
        if self.output_print:
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

        title = f'Task 2 (3D): Range-Doppler Surface - {name} \nData File: {self.radar_file_3D.name}'
        self._generic_plot_3d(V, R, Z,
                              xlabel='Velocity (m/s)', ylabel='Distance (m)', zlabel='Relative Amplitude (dB)',
                              title=title, cmap='seismic', vmin=-150, vmax=0, elev=elev, azim=azim, decimate=decimate)



    ################
    # FOR TASK 3   #
    ################

    def create_cfar_kernel(self, kernel_matrix=None, train_range=8, train_doppler=4, guard_range=4, guard_doppler=2):
        """
        Erzeuge CA-CFAR-Kernel aus Parametern (train_range und guard_range fuer Distanz, train_doppler und guard_doppler fuer Doppler)
        > Rückgabe: kernel (np.int8)
        """

        r_total = train_range + guard_range
        d_total = train_doppler + guard_doppler

        # symmetrischer Kernel mit CUT in der Mitte
        rows = 2 * r_total + 1      
        cols = 2 * d_total + 1

        kernel = np.zeros((rows, cols), dtype=np.int8)

        # Indizes des Zentrums 
        r_c = rows // 2
        d_c = cols // 2

        # Markiere die äußeren Ecken des Kernels (Trainings + Guard)
        r_min, r_max = r_c - r_total, r_c + r_total
        d_min, d_max = d_c - d_total, d_c + d_total
        kernel[r_min:r_max+1, d_min:d_max+1] = 1  # vorläufig Trainings/Guard

        # Markiere Guard-Zellen (inneres Rechteck)
        g_r_min, g_r_max = r_c - guard_range, r_c + guard_range
        g_d_min, g_d_max = d_c - guard_doppler, d_c + guard_doppler
        kernel[g_r_min:g_r_max+1, g_d_min:g_d_max+1] = 2  # Guard

        # Setze CUT in der Mitte
        kernel[r_c, d_c] = 3

        return kernel

    def print_output_cifar(self):
        n_train = int(np.sum(self.CFAR_kernel == 1))
        n_guard = int(np.sum(self.CFAR_kernel == 2))
        n_cut = int(np.sum(self.CFAR_kernel == 3))

        if self.output_print:
            self._log(f"\n - Task 3.1: \n\t- Kernel erzeugt: Trainingszellen={n_train}, Guard-Zellen={n_guard}, CUT={n_cut}")
            self._log(f"\n\t- Kernelgrößen in Doppler-Dimension: Anzahl Trainingszellen = {self.train_doppler}, Anzahl Guardzellen = {self.guard_doppler}")
            self._log(f"\n\t- Kernelgrößen in Distanz-Dimension: Anzahl Trainingszellen = {self.train_range}, Anzahl Guardzellen = {self.guard_range}")


    def plot_cfar_kernel(self, kernel, figsize=(6,6), origin='lower'):
        """
        Visualisiere den CFAR-Kernel:
          - Trainingszellen: türkis
          - Guard-Zellen: grün
          - CUT (Mitte): orange
          - außerhalb: hellgrau
        """

        from matplotlib.colors import ListedColormap
        import matplotlib.patches as mpatches
        import matplotlib.patches as patches

        title = f"\n -Task 3: CA-CFAR Kernel \n(Train (Distanz,Doppler)={self.train_range},{self.train_doppler} ; Guard (Distanz,Doppler)={self.guard_range},{self.guard_doppler})"
        

        kernel = np.asarray(kernel, dtype=np.int8)
        rows, cols = kernel.shape
        #print(f"\n - Task 3.2: \n\t- Plot CFAR Kernel with shape {kernel.shape}")

        # Farbliste für Codes 0..3
        colors = ['lightgray', '#40E0D0', 'green', 'orange']  # 0=outside,1=train,2=guard,3=CUT
        cmap = ListedColormap(colors)

        # _generic_plot_2d zeigt standardmäßig sofort via plt.show().
        # Wir unterdrücken temporär plt.show(), rufen die Funktion auf, ergänzen dann Gitter/Annot.
        old_show = plt.show
        plt.show = lambda *a, **k: None
        try:
            # vmin/vmax so wählen, dass die Farbskala den integer-Codes 0..3 entspricht
            self._generic_plot_2d(kernel, extent=None,
                                  xlabel="Doppler-Bins (relative)", ylabel="Range-Bins (relative)",
                                  title=title, cmap=cmap, vmin=0, vmax=3, colorbar_label='Kernel Code',
                                  origin=origin, figsize=figsize)
        finally:
            # show-Funktion wiederherstellen
            plt.show = old_show

        # Jetzt das gerade erzeugte Figure/Axes ergänzen
        fig = plt.gcf()
        ax = plt.gca()

        # Gitterlinien an Zellgrenzen (Minor ticks)
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=0.6, alpha=0.6)

        # Major ticks/labels entfernen (nur Gitter sichtbar)
        ax.set_xticks([])
        ax.set_yticks([])

        # CUT-Zelle dicker umranden
        r_c = rows // 2
        d_c = cols // 2
        rect = patches.Rectangle((d_c - 0.5, r_c - 0.5), 1, 1,
                                 linewidth=2.0, edgecolor='orange', facecolor='none')
        ax.add_patch(rect)

        # Colorbar: falls vorhanden, beschriften (letzte Axes ist üblicherweise Colorbar-Achse)
        if len(fig.axes) > 1:
            cb_ax = fig.axes[-1]
            try:
                cb_ax.set_yticks([0,1,2,3])
                cb_ax.set_yticklabels(['outside', 'train', 'guard', 'CUT'])
            except Exception:
                # Fallback: nichts tun, falls die Achse keine Colorbar-Achse ist
                pass

        # Legende
        legend_patches = [
            mpatches.Patch(color=colors[1], label='Trainingszellen (turquoise)'),
            mpatches.Patch(color=colors[2], label='Guard-Zellen (grün)'),
            mpatches.Patch(color=colors[3], label='CUT (orange)'),
            mpatches.Patch(color=colors[0], label='Außerhalb (nicht genutzt)')
        ]
        ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.35, 1.0))

        plt.tight_layout()
        plt.show()


    def compute_cfar_thresholds(self, data=None, kernel=None, threshold_factor=1.5, pad_fill=0):
        """
        Berechne CA-CFAR-Schwellenwerte über die 2D-Range-Doppler-Map.
        Vorgehen:
         - Für jedes CUT wird die Summe der Quadrate (Leistung) der Trainingszellen berechnet.
         - Mittelung durch die Anzahl tatsächlich überlappender Trainingszellen ergibt die mittlere Leistung.
         - Die mittlere Leistung wird mit `threshold_factor` multipliziert -> Schwellenwert.
        Rückgabe:
         - thresholds: 2D-Array gleicher Form wie `data` mit den Schwellenwerten (np.nan, wenn keine Trainingszellen überlappen)
        Parameter:
         - data: 2D-Array (Complex/Real). Default: abs(self.fft_shifted) (lineare Amplitude)
         - kernel: 2D-Kernel wie von `create_cfar_kernel` erzeugt (0=outside,1=train,2=guard,3=CUT).
                   Default: self.CFAR_kernel (falls vorhanden), sonst erzeugt aus self.train_*/self.guard_*.
         - threshold_factor: Skalar zum Multiplizieren der mittleren Leistung (z.B. 1.5, 2.0 ...)
         - pad_fill: Wert zum Füllen außerhalb (wird bei der Korrelation benutzt); Standard 0.
        Speichert zusätzlich in der Instanz:
         - self.cfar_thresholds
         - self.cfar_train_counts
         - self.cfar_threshold_factor
        """

        if self.output_print:
            self._log(f"\n - Task 3.3: \n\t- Berechne CA-CFAR Schwellenwerte (threshold_factor={self.threshold_factor})")


        # Validierung / Default-Handling
        if data is None:
            if hasattr(self, "fft_shifted"):
                data = np.abs(self.fft_shifted)
            else:
                raise ValueError("Kein `data` übergeben und `self.fft_shifted` nicht vorhanden.")
        else:
            data = np.asarray(data)

        kernel = self.CFAR_kernel       # definierter CFAR-Kernel 
          
        kernel = np.asarray(kernel, dtype=np.int8)
        if kernel.ndim != 2:
            raise ValueError("kernel muss 2D sein.")

        # Trainingsmask extrahieren (1 = Training)
        train_mask = (kernel == 1).astype(np.int8)
        n_train_total = int(np.sum(train_mask))

        # Verwende 2D-Korrelation (keine Kernel-Flip) für effiziente Berechnung
        from scipy.signal import correlate2d

        # Leistung je Zelle (linear): |x|^2
        power_map = np.abs(data) ** 2

        # Anzahl tatsächlich überlappender Trainingszellen an jeder Position
        ones_map = np.ones_like(power_map, dtype=np.int32)
        train_counts = correlate2d(ones_map, train_mask, mode='same', boundary='fill', fillvalue=0)

        # Summe der Leistungen über Trainingszellen an jeder Position
        train_power_sum = correlate2d(power_map, train_mask, mode='same', boundary='fill', fillvalue=pad_fill)

        # Vermeide Division durch 0: nur dort berechnen, wo train_counts > 0
        thresholds = np.full_like(train_power_sum, np.nan, dtype=float)
        valid = train_counts > 0
        thresholds[valid] = (train_power_sum[valid] / train_counts[valid]) * float(threshold_factor)

        # Speichere Ergebnisse in der Instanz
        self.cfar_thresholds = thresholds
        self.cfar_train_counts = train_counts
        self.cfar_threshold_factor = threshold_factor

        if self.output_print:
            self._log(f"\t- CA-CFAR Schwellenwerte berechnet (threshold_factor={threshold_factor}).")
            self._log(f"\t  - Kernel-Trainingszellen (global): {n_train_total}")
            self._log(f"\t  - Shape thresholds: {thresholds.shape}")

        return thresholds


    def plot_cfar_threshold_map(self, title=None, cmap='viridis', figsize=(10,7)):
        """
        Zeigt die in self.cfar_thresholds gespeicherten CFAR-Schwellen als 2D-Plot.
        """
        # Sicherstellen, dass Schwellenwerte vorhanden sind
        finite = np.isfinite(self.cfar_thresholds)
        if np.any(finite):
            vmin = float(np.nanmin(self.cfar_thresholds))
            vmax = float(np.nanmax(self.cfar_thresholds))
        else:
            vmin, vmax = None, None

        # Verwende die vorhandene generische Plot-Funktion
        self._generic_plot_2d(self.cfar_thresholds, extent=None,
                                xlabel='Doppler Index (Bins)', ylabel='Range Index (Bins)',
                                title=title, cmap=cmap, vmin=vmin, vmax=vmax, colorbar_label='Threshold (power)', figsize=figsize)
        

    def plot_CFAR_results_3d(self, name="", decimate=(1,1), elev=30, azim=-60):
        """
        Ruft die bestehende 3D-Plot-Funktion für die Range-Doppler-Map auf und überlagert
        optional die zuvor berechneten CFAR-Schwellen (self.cfar_thresholds).
        Vorgehen:
         - Temporär plt.show() unterdrücken, damit die bestehende Darstellung erstellt, aber
           nicht sofort angezeigt wird.
         - Aktuelle Figure/Axes holen (plt.gcf()/plt.gca()) und CFAR-Threshold-Oberfläche
           decimieren/umrechnen und als halbtransparente Surface drüber zeichnen.
        Hinweis:
         - compute_cfar_thresholds() muss vorher ausgeführt worden sein (z.B. in Task_Step_3);
           falls nicht vorhanden, wird compute_cfar_thresholds() automatisch aufgerufen.
        """

        if self.output_print:
            self._log(f"\t- 3D CFAR-Overlay Plot for {name}")

        # Sicherstellen, dass Range-Doppler-Daten vorhanden sind
        if not hasattr(self, "normalized_db") or not hasattr(self, "fft_shifted"):
            raise RuntimeError("FFT-Daten fehlen. Führe zuvor Task_Step_1/Task_Step_2 aus.")


        # Temporär plt.show unterdrücken beim Aufruf der vorhandenen 3D-Plot-Funktion
        old_show = plt.show
        plt.show = lambda *a, **k: None
        try:
            # Aufruf der vorhandenen Darstellung (erzeugt Figure/Axes im Hintergrund)
            self.plot_fft_results_3d(name=name, decimate=decimate, elev=elev, azim=azim)
        finally:
            plt.show = old_show

        # Hole Figure / Axes, die plot_fft_results_3d erzeugt hat
        fig = plt.gcf()
        ax = plt.gca()

        # Bereite Achsen-Gitter in originaler (nicht-decimierter) Auflösung vor
        range_axis, velocity_axis, _ = self._prepare_range_velocity_axes()
        V, R = np.meshgrid(velocity_axis, range_axis)  # V: velocity (cols), R: range (rows)

        # Hole und verarbeite CFAR-Schwellen (Power)
        thr = getattr(self, "cfar_thresholds", None)
        if thr is None:
            if self.output_print:
                self._log("\t- Keine CFAR-Schwellen vorhanden, nichts zu überlagern.")
            plt.show()
            return

        # Umrechnung: Power -> dB (10*log10), dann auf denselben Peak wie magnitude_db normieren
        magnitude_db = 20.0 * np.log10(np.abs(self.fft_shifted) + 1e-10)
        peak_db = np.max(magnitude_db)  # Referenz wie bei normalized_db = magnitude_db - peak_db

        thr_db = np.full_like(thr, np.nan, dtype=float)
        valid_thr = np.isfinite(thr) & (thr > 0)
        thr_db[valid_thr] = 10.0 * np.log10(thr[valid_thr])

        # Normiere wie normalized_db
        thr_db_norm = thr_db - peak_db  # nun vergleichbar mit self.normalized_db

        # Decimation anwenden (wie bei plot_fft_results_3d)
        r_step, v_step = decimate
        if r_step > 1 or v_step > 1:
            Vp = V[::r_step, ::v_step]
            Rp = R[::r_step, ::v_step]
            thr_p = thr_db_norm[::r_step, ::v_step]
            Zp = self.normalized_db[::r_step, ::v_step]
        else:
            Vp, Rp, thr_p, Zp = V, R, thr_db_norm, self.normalized_db

        # Plot overlay nur dort, wo thr_p gültig ist
        mask = np.isfinite(thr_p)
        if np.any(mask):
            from matplotlib import cm
            from matplotlib.colors import Normalize

            # Norm und Colormap für Threshold-Overlay
            vmin_thr = float(np.nanmin(thr_p[mask]))
            vmax_thr = float(np.nanmax(thr_p[mask]))
            norm_thr = Normalize(vmin=vmin_thr, vmax=vmax_thr)
            cmap_thr = cm.get_cmap('viridis')

            # Erzeuge RGBA-Farben; setze Alpha
            rgba = cmap_thr(norm_thr(np.nan_to_num(thr_p, nan=vmin_thr)))
            rgba[..., -1] = 0.6  # Transparenz

            # Damit NaNs nicht stören, ersetze NaNs in Z-Position durch einen Wert unterhalb des Minimums
            zmin_plot = float(np.nanmin(Zp)) - 1.0
            Z_plot_thr = np.where(mask, thr_p, zmin_plot)

            # Zeichne Threshold-Oberfläche
            try:
                surf_thr = ax.plot_surface(Vp, Rp, Z_plot_thr, facecolors=rgba, linewidth=0, antialiased=True, shade=False)
            except Exception as e:
                print(f"\t- Warnung: Overlay der Threshold-Oberfläche fehlgeschlagen: {e}")

            # Colorbar für Threshold-Overlay hinzufügen
            try:
                mappable_thr = cm.ScalarMappable(norm=norm_thr, cmap=cmap_thr)
                mappable_thr.set_array(thr_p[mask])
                cbar = fig.colorbar(mappable_thr, ax=ax, shrink=0.6, pad=0.04)
                cbar.set_label('CFAR Threshold (dB, norm.)')
            except Exception:
                pass
        else:
            print("\t- Keine gültigen Threshold-Werte zum Plotten.")

        # Finale Anpassungen und Anzeige
        ax.set_title(f"{ax.get_title()}  (mit CFAR-Thresholds)")
        plt.tight_layout()
        plt.show()


    def _apply_nms(self, power_map, detections, window_size=5):
        """
        Unterdrücke Detektionen, die keine lokalen Maxima sind.
        
        Args:
            power_map: 2D-Array mit Power-Werten
            detections: 2D-Array mit binären Detektionen (0/1)
            window_size: Größe des Fensters für lokale Maxima-Suche
        
        Returns:
            Gefilterte Detektionen (nur lokale Maxima)
        """
        local_max = maximum_filter(power_map, size=window_size) == power_map
        detections_nms = detections & local_max
        
        return detections_nms.astype(np.int8)


    def _cluster_detections(self, detections_binary, min_distance=3):
        """
        Gruppiere benachbarte Detektionen zu Clustern und finde Zentroid.
        
        Args:
            detections_binary: 2D-Array mit 0/1-Detektionen
            min_distance: Minimaler Abstand zwischen Peaks (in Bins) - aktuell nicht verwendet
        
        Returns:
            List of (r_bin, d_bin) tuples für Peak-Zentren
        """
        # Finde zusammenhängende Regionen
        labeled_array, num_features = label(detections_binary)
        
        peak_positions = []
        for i in range(1, num_features + 1):
            # Finde Schwerpunkt jedes Clusters
            coords = center_of_mass(detections_binary, labeled_array, i)
            r_bin = int(round(coords[0]))
            d_bin = int(round(coords[1]))
            peak_positions.append((r_bin, d_bin))
        
        if self.output_print:
            total_detections = int(np.sum(detections_binary))
            self._log(f"\t- Clustering: {total_detections} Detektionen → {len(peak_positions)} Objekte")
        
        return peak_positions


    def apply_cfar_detection(self, data=None, thresholds=None):
        """
        CFAR-Detektion mit zusätzlicher Validierung, Non-Maximum Suppression und Clustering.
        """
        if self.output_print:
            self._log(f"\n - Task 3.6: \n\t- CFAR-Detektion mit Validierung")
        
        if data is None:
            data = self.fft_shifted
        if thresholds is None:
            thresholds = self.cfar_thresholds
        
        # Formprüfung
        if data.shape != thresholds.shape:
            raise ValueError(f"Shape mismatch: data={data.shape}, thresholds={thresholds.shape}")
        
        # Leistungsmap
        power_map = np.abs(data) ** 2
        detections = np.zeros_like(power_map, dtype=np.int8)
        
        # NUR gültige Bereiche (isfinite) UND über Schwelle
        valid = np.isfinite(thresholds)
        exceeds_threshold = power_map > thresholds
        
        detections[valid & exceeds_threshold] = 1
        
        # Zähle rohe Detektionen vor NMS
        raw_detections = int(np.sum(detections))
        
        # **NEU: Non-Maximum Suppression**
        detections = self._apply_nms(power_map, detections, window_size=self.window_cifar_max_size)
        nms_detections = int(np.sum(detections))
        
        # **NEU: Validierung - Rand-Detektionen ausschließen**
        kernel_h, kernel_w = self.CFAR_kernel.shape
        border_r = min(5, kernel_h // 4)
        border_d = min(5, kernel_w // 4)
        
        # Setze Rand auf 0
        detections[:border_r, :] = 0
        detections[-border_r:, :] = 0
        detections[:, :border_d] = 0
        detections[:, -border_d:] = 0
        
        # Speichern
        self.cfar_detections = detections
        
        # **NEU: Clustering**
        self.detected_objects = self._cluster_detections(detections)
        
        n_det = int(np.sum(detections))
        
        if self.output_print:
            self._log(f"\t- Rohe Detektionen: {raw_detections}")
            self._log(f"\t- Nach NMS: {nms_detections}")
            self._log(f"\t- Nach Randausschluss: {n_det}")
            self._log(f"\t- Geclusterte Objekte: {len(self.detected_objects)}")
            
            # **Statistik über Range/Doppler-Verteilung**
            det_indices = np.argwhere(detections == 1)
            if len(det_indices) > 0:
                range_bins = det_indices[:, 0]
                doppler_bins = det_indices[:, 1]
                
                self._log(f"\t- Range-Bins: min={np.min(range_bins)}, max={np.max(range_bins)}")
                self._log(f"\t- Doppler-Bins: min={np.min(doppler_bins)}, max={np.max(doppler_bins)}")
                
                # Physikalische Werte
                range_axis = np.linspace(0, self.range_max, self.num_samples)
                velocity_axis = np.linspace(-self.vel_max, self.vel_max, self.num_chirps)
                
                ranges = range_axis[range_bins]
                velocities = velocity_axis[doppler_bins]
                
                self._log(f"\t- Distanzen: {np.min(ranges):.2f}m bis {np.max(ranges):.2f}m")
                self._log(f"\t- Geschwindigkeiten: {np.min(velocities):.3f}m/s bis {np.max(velocities):.3f}m/s")
        
        return detections


    def plot_cfar_detections(self, name="", Task="3.6", mode=1): # mode 1=vel and m / 2 = bins
        """
        Zeigt die Range-Doppler-Map, auf die die CFAR-Detektionen (0/1) angewendet wurden.
        Vorgehen:
         - Berechne magnitude_db und normalisiere wie in `plot_fft_results`.
         - Wende `self.cfar_detections` als Maske an (0 -> sehr kleines Signal, 1 -> original).
         - Benutze dieselben Achsen-/Farb-Parameter wie `plot_fft_results` für direkte Vergleichbarkeit.
        Rückgabe: die gezeigte, maskierte, normalisierte dB-Map.
        """
        if self.output_print:
            self._log(f"\n - Task {Task}: \n\t- Plot Range-Doppler Map mit CFAR-Maskierung (zeigt Detektionsergebnisse)")

        # Erzeuge magnitude und Normalisierung wie in plot_fft_results
        magnitude_db = 20.0 * np.log10(np.abs(self.fft_shifted) + 1e-10)
        peak_db = np.max(magnitude_db)
        normalized_db = magnitude_db - peak_db  # Referenz wie plot_fft_results

        # Hole Detection-Maske (0/1) und prüfe Form
        mask = np.asarray(self.cfar_detections, dtype=np.int8)

        # Wende Maske an: in linearer Amplitude-Menge maskieren, dann zurück in dB normalisieren
        amp = np.abs(self.fft_shifted)
        amp_masked = amp * mask
        # Vermeide 0 -> -inf dB, setze sehr kleine Zahl dort, wo mask==0
        eps = 1e-12
        amp_masked[mask == 0] = eps

        # zurück in dB und auf selben Peak normieren
        amplitude_db_masked = 20.0 * np.log10(amp_masked + 1e-20)
        normalized_masked_db = amplitude_db_masked - peak_db

        if mode == 1:
            # Achsenskalierung
            range_axis, velocity_axis, plot_extent = self._prepare_range_velocity_axes()

            title = f'Task {Task}: CFAR-masked Range-Doppler - {name} \nData File: {self.radar_file_3D.name}'
            # gleiche Farben / Skala wie plot_fft_results
            self._generic_plot_2d(normalized_masked_db, extent=plot_extent,
                                xlabel='Velocity (m/s)', ylabel='Distance (m)',
                                title=title, cmap='seismic', vmin=-50, vmax=0, colorbar_label='Relative Amplitude (dB)')
        if mode == 2:
            title = f'Task {Task}: CFAR-masked Range-Doppler - {name} \nData File: {self.radar_file_3D.name}'
            # gleiche Farben / Skala wie plot_fft_results
            self._generic_plot_2d(normalized_masked_db, extent=None,
                                xlabel='Doppler Index (Bins)', ylabel='Range Index (Bins)',
                                title=title, cmap='seismic', vmin=-50, vmax=0, colorbar_label='Relative Amplitude (dB)')


        # speichere für externe Nutzung und return
        self.normalized_cfar_masked_db = normalized_masked_db
        return normalized_masked_db
    

    def plot_range_profile_at_detection(self, detection_index=0):
        """
        Zeigt das Entfernungsprofil (Range-Dimension) für eine bestimmte Detektion.
        Verwendet absolute Power in dB (nicht normalisiert), um positive Werte zu zeigen.
        """

        # Fallback auf cfar_detections (nur Task 3)
        if hasattr(self, 'cfar_detections'):
            det_indices = np.argwhere(self.cfar_detections == 1)
            
            if len(det_indices) == 0:
                self._log("Keine Detektionen vorhanden.")
                return
            
            if detection_index >= len(det_indices):
                self._log(f"Detection Index {detection_index} außerhalb des Bereichs (max: {len(det_indices)-1})")
                return
            
            r_bin, d_bin = det_indices[detection_index]
            azimuth_deg = None
            has_angle_info = False
            
        else:
            self._log("Fehler: Keine CFAR-Detektionen vorhanden. Führe zuerst Task_Step_3 aus.")
            return
        
        # Extrahiere Range-Profil bei dieser Doppler-Position
        range_profile = self.fft_shifted[:, d_bin]
        
        # NEU: Konvertiere zu absoluter Power in dB (NICHT normalisiert)
        # Power = |signal|^2, dann 10*log10 für Power in dB
        power_linear = np.abs(range_profile) ** 2
        power_db = 10 * np.log10(power_linear + 1e-20)  # Vermeide log(0)
        
        # NEU: Extrahiere CFAR-Threshold für diesen Doppler-Bin
        if hasattr(self, 'cfar_thresholds'):
            cfar_threshold_linear = self.cfar_thresholds[:, d_bin]
            # Konvertiere zu dB (Power-Skala)
            cfar_threshold_db = 10 * np.log10(cfar_threshold_linear + 1e-20)
        else:
            cfar_threshold_db = None
        
        # Finde den Peak im Power-Spektrum
        peak_bin = np.argmax(power_db)
        
        # Erstelle Achsen
        velocity_axis = np.linspace(-self.vel_max, self.vel_max, self.num_chirps)
        range_axis = np.linspace(0, self.range_max, self.num_samples)
        
        # Physikalische Werte
        range_detection = range_axis[r_bin]      # CFAR-Detektion
        range_peak = range_axis[peak_bin]        # 1D-Peak
        velocity_m_s = velocity_axis[d_bin]
        
        # Plot erstellen
        plt.figure(figsize=(12, 6))
        plt.plot(range_axis, power_db, 'b-', linewidth=1.5, label='Signal')
        
        # NEU: Plotte CFAR-Threshold
        if cfar_threshold_db is not None:
            plt.plot(range_axis, cfar_threshold_db, color='orange', linewidth=2, 
                    label='Threshold', alpha=0.8)
        
        # NEU: Markiere ALLE Detektionen bei diesem Doppler-Bin mit roten Punkten
        detections_at_doppler = det_indices[det_indices[:, 1] == d_bin]
        if len(detections_at_doppler) > 0:
            detection_ranges = range_axis[detections_at_doppler[:, 0]]
            detection_powers = power_db[detections_at_doppler[:, 0]]
            plt.plot(detection_ranges, detection_powers, 'o', color='yellow', 
                    markersize=10, markeredgecolor='red', markeredgewidth=2,
                    label=f'Detections ({len(detections_at_doppler)})', zorder=5)
        
        # Markiere die aktuell ausgewählte CFAR-Detektion mit vertikaler Linie
        plt.axvline(x=range_detection, color='r', linestyle='--', linewidth=2, 
                    label=f'Selected Detection at {range_detection:.2f} m', alpha=0.7)
    
        
        # Beschriftung
        plt.xlabel('Range (m)', fontsize=12)
        plt.ylabel('Receive Power (dB)', fontsize=12)  # NEU: Absolute Power
        
        title_str = f'Range Profile at Velocity = {velocity_m_s:.3f} m/s ({velocity_m_s*3.6:.2f} km/h)\n'
        title_str += f'Detection #{detection_index+1} (Range Bin: {r_bin}, Doppler Bin: {d_bin}'
        if has_angle_info:
            title_str += f', Azimuth: {azimuth_deg:.1f}°'
        title_str += f')\nData File: {self.radar_file_3D.name}'
        
        plt.title(title_str, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # NEU: Automatische Y-Achsen-Limits basierend auf Daten
        y_min = np.percentile(power_db[np.isfinite(power_db)], 1)  # 1. Perzentil
        y_max = np.max(power_db[np.isfinite(power_db)]) + 5
        plt.ylim([y_min, y_max])
        
        plt.tight_layout()
        plt.show()
        
        # Log-Ausgabe
        if self.output_print:
            self._log(f"\n - Range Profile Plot:")
            self._log(f"\t- Detection #{detection_index+1}")
            self._log(f"\t- CFAR Detection: {range_detection:.2f} m (Bin {r_bin}, Power: {power_db[r_bin]:.1f} dB)")
            self._log(f"\t- 1D Peak: {range_peak:.2f} m (Bin {peak_bin}, Power: {power_db[peak_bin]:.1f} dB)")
            self._log(f"\t- Velocity: {velocity_m_s:.3f} m/s = {velocity_m_s*3.6:.2f} km/h (Bin {d_bin})")
            if has_angle_info:
                self._log(f"\t- Azimuth Angle: {azimuth_deg:.1f}°")


    ################
    # FOR TASK 4   #
    ################

    def _calculate_antenna_parameters(self):
        """Berechne Wellenlänge und Antennenabstände"""
        #self.wavelength = float(self.c / self.f_center)  # Explizit float
        
        if self.output_print:
            self._log(f"\n - Task 4.1:")
            self._log(f"\t- Wellenlänge λ = {self.wavelength*1000:.2f} mm")
        
        # *** FIX: Konvertiere Positionen zu float ***
        x_positions = self.AntennaPositions[:, 0].astype(np.float64)
        y_positions = self.AntennaPositions[:, 1].astype(np.float64)
        
        # Berechne tatsächlichen Elementabstand (meist λ/2)
        x_unique = np.unique(x_positions)
        y_unique = np.unique(y_positions)
        
        # Finde kleinsten nicht-null Abstand
        x_diffs = np.diff(np.sort(x_unique))
        y_diffs = np.diff(np.sort(y_unique)) if len(y_unique) > 1 else np.array([0.0])
        
        self.d_azimuth = float(np.min(x_diffs[x_diffs > 0])) if len(x_diffs[x_diffs > 0]) > 0 else float(self.wavelength/2)
        self.d_elevation = float(np.min(y_diffs[y_diffs > 0])) if len(y_diffs[y_diffs > 0]) > 0 else 0.0
        
        # **NEU: Berechne FOV-Grenzen für konsistente Darstellung**
        d_antenna = self.wavelength / 2
        max_azimuth_rad = np.arcsin(self.wavelength / (2 * d_antenna))  # ±90° für d=λ/2
        max_azimuth_deg = np.degrees(max_azimuth_rad)
        self.FOV_min = -max_azimuth_deg
        self.FOV_max = max_azimuth_deg
        
        # DEBUG: Prüfe ob d_azimuth sinnvoll ist
        if self.output_print:
            self._log(f"\t- Anzahl virtuelle Antennen: {len(x_positions)}")
            self._log(f"\t- Field of View: {self.FOV_min:.1f}° bis {self.FOV_max:.1f}°")


    def _create_virtual_array(self):
        """Erstelle virtuelles MIMO-Array aus Tx x Rx Kanälen"""
        if self.output_print:
            self._log(f"\n - Task 4.2:")
            self._log(f"\t- Erzeuge virtuelles Array: {self.Tx_Channels} Tx × {self.Rx_Channels} Rx = {self.num_channels} virtuelle Antennen")
        
        # Virtuelles Array ist bereits in AntennaArray enthalten
        # Shape: (num_samples, num_chirps, num_channels)
        self.num_virtual_antennas = self.num_channels


    def _estimate_angles_for_detections(self):
        """Winkelschätzung für CFAR-Detektionen"""
        
        if self.output_print:
            self._log(f"\n - Task 4.3:")
            self._log(f"\t- Winkelschätzung für detektierte Objekte")
        
        # Filtere Azimuth-Antennen (y=0)
        ind = np.where(self.AntennaPositions[:, 1] == 0)
        ind = np.asarray(ind)
        ind = np.swapaxes(ind, 0, 1)
        AntennaSelected = self.AntennaPositions[ind, 0]
        val, indices = np.unique(AntennaSelected, return_index=True) 
        AzimuthAntennaOnly = indices
        
        # Antennenabstand berechnen
        d_antenna = self.wavelength / 2
        
        if self.output_print:
            self._log(f"\t- Anzahl Range-Doppler-Bins mit Detektionen: {np.sum(self.cfar_detections)}")
            self._log(f"\t- Azimuth-Antennen: {len(AzimuthAntennaOnly)} von {self.num_channels}")
            self._log(f"\t- Antennenabstand: {d_antenna*1000:.3f} mm")
            self._log(f"\t- Wellenlänge: {self.wavelength*1000:.3f} mm")

        # Für jede Detektion Winkel schätzen
        detections = []
        det_indices = np.argwhere(self.cfar_detections == 1)
        
        range_axis = np.linspace(0, self.range_max, self.num_samples)
        velocity_axis = np.linspace(-self.vel_max, self.vel_max, self.num_chirps)

        if self.output_print:
            self._log(f"\t- Verarbeite {len(det_indices)} Detektionen...")
        
        for det_num, idx in enumerate(det_indices):
            r_bin, d_bin = idx
            
            # Extrahiere Antennensignale für diese Detektion
            antenna_signal = self.AntennaArray[r_bin, d_bin, :]
            x_azimuth_only = antenna_signal[AzimuthAntennaOnly]

            # Windowing + FFT mit Zero-Padding
            window = get_window('hann', len(x_azimuth_only))
            nfft = 512  # Höhere Auflösung
            angle_fft = np.fft.fft(x_azimuth_only * window, n=nfft)
            angle_fft_shifted = np.fft.fftshift(angle_fft)
            angle_spectrum = np.abs(angle_fft_shifted)

            # Berechnung der Winkelachse
            N_FFT = len(angle_spectrum)
            k_bins = np.arange(N_FFT) - N_FFT / 2 
            sin_theta = (k_bins / (N_FFT/2)) * (self.wavelength / (2 * d_antenna))
            
            # Begrenze sin_theta auf [-1, 1] für arcsin
            sin_theta = np.clip(sin_theta, -1, 1)
            theta_rad = np.arcsin(sin_theta)
            theta_deg = np.degrees(theta_rad)

            # Power-Spektrum in dB
            power_db = 10 * np.log10(angle_spectrum**2 + 1e-10)

            # Finde Peak (maximale Leistung)
            max_index = np.argmax(power_db)
            estimated_angle = theta_deg[max_index]
            max_power = power_db[max_index]

            # Berechne physikalische Werte
            range_m = range_axis[r_bin]
            velocity_m_s = velocity_axis[d_bin]
            magnitude = np.abs(self.fft_shifted[r_bin, d_bin])

            # Speichere Detektion
            detection_dict = {
                'range_m': range_m,
                'velocity_m_s': velocity_m_s,
                'azimuth_deg': estimated_angle,
                'elevation_deg': 0.0,  # Keine Elevation-Schätzung
                'magnitude': magnitude,
                'r_bin': r_bin,
                'd_bin': d_bin,
                'power_db': max_power
            }
            detections.append(detection_dict)

            # --- Plotting für jede Detektion ---
            plt.figure(figsize=(10, 6))
            plt.plot(theta_deg, power_db, 'b-', linewidth=1.5, label="Azimut-Spektrum")
            
            # Markiere Peak
            plt.plot(estimated_angle, max_power, 'ro', markersize=10,
                    label=f"Peak bei {estimated_angle:.2f}°")
            plt.axvline(estimated_angle, color='r', linestyle='--', linewidth=1.5, alpha=0.7)

            # Titel und Beschriftungen
            title = f"Azimut-Spektrum - Detektion #{det_num+1}\n"
            title += f"Range: {range_m:.2f}m (Bin {r_bin}), "
            title += f"Velocity: {velocity_m_s:.3f}m/s ({velocity_m_s*3.6:.2f} km/h) (Bin {d_bin})\n"
            title += f"Geschätzter Winkel: {estimated_angle:.2f}°, Leistung: {max_power:.1f} dB"
            plt.title(title, fontsize=11)
            
            plt.xlabel("Azimut-Winkel [Grad]", fontsize=12)
            plt.ylabel("Leistung [dB]", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)

            # Setze X-Achsen-Grenzen auf konsistente FOV-Werte
            if np.all(np.isreal(theta_deg)):
                plt.xlim(self.FOV_min, self.FOV_max)

            plt.tight_layout()
            plt.show()

            # Log-Ausgabe für diese Detektion
            if self.output_print:
                self._log(f"\t  Detektion #{det_num+1}: Range={range_m:.2f}m, Vel={velocity_m_s:.3f}m/s, "
                         f"Azimut={estimated_angle:.1f}°, Power={max_power:.1f}dB")

        if self.output_print:
            self._log(f"\n\t- Winkelschätzung abgeschlossen")
            self._log(f"\t- Anzahl gefundener Objekte: {len(detections)}")
        
        return detections




    def _plot_3d_detections(self):
        """
        3D-Plot: Zeige detektierte Objekte im kartesischen Raum (x, y, z)
        """
        if self.output_print:
            self._log(f"\n - Task 4.4:")
            self._log(f"\t- 3D-Visualisierung der Objekte")
        
        if len(self.detections_3d) == 0:
            self._log(f"\t- Keine Objekte zu plotten")
            return
        
        # Konvertiere zu kartesischen Koordinaten
        x_coords = []
        y_coords = []
        z_coords = []
        magnitudes = []
        
        for det in self.detections_3d:
            r = det['range_m']
            az = np.radians(det['azimuth_deg'])
            el = np.radians(det['elevation_deg'])
            
            # Kartesische Koordinaten
            x = r * np.cos(el) * np.sin(az)
            y = r * np.cos(el) * np.cos(az)
            z = r * np.sin(el)
            
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
            magnitudes.append(det['magnitude'])
        
        # 3D Scatter Plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Normalisiere Magnitudes für Farbe/Größe
        magnitudes = np.array(magnitudes)
        sizes = 100 + 400 * (magnitudes / magnitudes.max())  # Punktgröße
        
        scatter = ax.scatter(x_coords, y_coords, z_coords, 
                            c=magnitudes, cmap='hot', s=sizes, 
                            alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Signal Magnitude')
        
        # Achsenbeschriftung
        ax.set_xlabel('X (m) - Lateral')
        ax.set_ylabel('Y (m) - Longitudinal')
        ax.set_zlabel('Z (m) - Vertical')
        ax.set_title(f'Task 4: 3D Object Localization\nData File: {self.radar_file_3D.name}')
        
        # **OPTIMIERT: Achsengrenzen basierend auf FOV und range_max**
        R_max = self.range_max
        
        # Berechne maximale laterale Ausdehnung basierend auf FOV
        max_azimuth_rad = np.radians(self.FOV_max)
        x_max = R_max * np.sin(max_azimuth_rad)
        
        # Setze Achsengrenzen mit ausreichend Spielraum
        ax.set_xlim([-x_max * 1.1, x_max * 1.1])  # Lateral: ±FOV-Bereich
        ax.set_ylim([0, R_max * 1.1])              # Longitudinal: 0 bis max_range
        ax.set_zlim([-x_max * 0.3, x_max * 0.3])   # Vertical: kleiner Bereich (meist 0)
        
        # **NEU: Gleiche Skalierung für X und Y Achsen**
        ax.set_box_aspect([1, 1, 0.3])  # X:Y:Z = 1:1:0.3 für gleichmäßige Schritte
        
        # **NEU: Radar Field-of-View (FOV) Visualisierung**
        # Erzeuge Winkelkeule (Kegel) in 3D
        azimuth_angles = np.linspace(-max_azimuth_rad, max_azimuth_rad, 30)
        ranges = np.linspace(0, R_max, 20)
        
        # Meshgrid für Azimut und Range
        Az, R = np.meshgrid(azimuth_angles, ranges)
        
        # Kartesische Koordinaten für die Keule (z=0 Ebene)
        X_fov = R * np.sin(Az)
        Y_fov = R * np.cos(Az)
        Z_fov = np.zeros_like(X_fov)
        
        # Plotte die Winkelkeule als transparente Fläche
        ax.plot_surface(X_fov, Y_fov, Z_fov, alpha=0.15, color='cyan', 
                       edgecolor='darkblue', linewidth=2, label='Radar FOV')
        
        # **NEU: Gestrichelte Mittellinie bei 0° Azimut**
        center_line_y = np.linspace(0, R_max, 50)
        center_line_x = np.zeros_like(center_line_y)
        center_line_z = np.zeros_like(center_line_y)
        ax.plot(center_line_x, center_line_y, center_line_z, 
               'k--', linewidth=2.5, label='0° Azimut', alpha=0.7)
        
        # **NEU: Reichweitenkreise (optional)**
        # Zeichne Kreise bei verschiedenen Reichweiten
        for r in [R_max * 0.25, R_max * 0.5, R_max * 0.75, R_max]:
            theta = np.linspace(-max_azimuth_rad, max_azimuth_rad, 100)
            x_circle = r * np.sin(theta)
            y_circle = r * np.cos(theta)
            z_circle = np.zeros_like(theta)
            ax.plot(x_circle, y_circle, z_circle, 'b-', linewidth=1, alpha=0.3)
        
        # Legende
        ax.legend(loc='upper left', fontsize=10)
        
        # Gitter
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()





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

        # Verwende die originalen Z-Werte für die Farbzuordnung
        facecolors = mappable.to_rgba(Z)
        surf = ax.plot_surface(X, Y, Z, facecolors=facecolors, linewidth=0, antialiased=True, shade=False)

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