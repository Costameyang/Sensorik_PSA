# Import necessary libraries
from pathlib import Path

# Import the Radar class
from radar import Radar

def USRR_Dynamic_Config():
    # Paths to radar cube data
    radar_file = Path("RadarCube/USRR_Dynamic10m/3D/RadarCube7.npy")
    
    # Parameter of Radar
    f_center = 78.26375e9  # Center Frequency (Hz)
    B = 2.5275e9           # Radar Bandwidth (Hz)
    f_sampling = 8e6       # Sampling Rate (Hz)
    num_samples = 256      # N samples per chirp (256)
    tc = 55e-6             # Chirp time / interval (s)
    num_chirps = 128       # N Chirps per CPI (128)
    chirp_slope = B / tc   # Chirp slope (Hz/s)
    Rx_gain = 45           # RX Gain (dB)

    Tx_Channels = 12       # Aus "TDM-MIMO (12 Tx..."
    Rx_Channels = 16       # Aus "TDM-MIMO (...16 Rx)"
  
    return radar_file, f_center, B, f_sampling, num_samples, tc, num_chirps, Rx_gain, Tx_Channels, Rx_Channels


def MRR_CornField_Config():
    # Paths to radar cube data
    radar_file = Path("RadarCube/MRR_CornField/3D/RadarCube1.npy")
    
    # Parameter of Radar
    f_center = 77.27e9     # Center Frequency (Hz)
    B = 0.26e9             # Radar Bandwidth (Hz)
    f_sampling = 15e6      # Sampling Rate (Hz)
    num_samples = 256      # N samples per chirp (256)
    tc = 45e-6             # Chirp time / interval (s)
    chirp_slope = B / tc   # Chirp slope (Hz/s)
    num_chirps = 64        # N Chirps per CPI (128)
    Rx_gain = 48           # RX Gain (dB)

    Tx_Channels = 12       # Aus "TDM-MIMO (12 Tx..."
    Rx_Channels = 16       # Aus "TDM-MIMO (...16 Rx)"
  
    return radar_file, f_center, B, f_sampling, num_samples, tc, num_chirps, Rx_gain, Tx_Channels, Rx_Channels


def main():
    
    # Create Radar object
    radar = Radar(*USRR_Dynamic_Config())
    
    # Tasks ausführen
    radar.Task_Step_1()
    radar.Task_Step_2()
    #radar.task3()

if __name__ == "__main__":
    main()