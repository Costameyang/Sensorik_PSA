# Import necessary libraries
from pathlib import Path

# Import the Radar class
from radar import Radar

def USRR_Dynamic_Config():
    # Paths to radar cube data
    radar_file = "RadarCube/USRR_Dynamic10m/3D/RadarCube"
    factor_formular_max_velocity = 2

    # Parameter of Radar
    f_center = 78.26375e9  # Center Frequency (Hz)
    B = 2.5275e9           # Radar Bandwidth (Hz)
    f_sampling = 8e6       # Sampling Rate (Hz)
    num_samples = 256      # N samples per chirp (256)
    tc = 55e-6             # Chirp time / interval (s)
    num_chirps = 128       # N Chirps per CPI (128)
    chirp_slope = 0        # Chirp slope (Hz/s)
    Rx_gain = 48           # RX Gain (dB)

    Tx_Channels = 12       # Aus "TDM-MIMO (12 Tx..."
    Rx_Channels = 16       # Aus "TDM-MIMO (...16 Rx)"
  
    return radar_file, factor_formular_max_velocity, f_center, B, f_sampling, num_samples, tc, chirp_slope, num_chirps, Rx_gain, Tx_Channels, Rx_Channels


def MRR_CornField_Config():
    # Paths to radar cube data
    radar_file = "RadarCube/MRR_CornField/3D/RadarCube"
    factor_formular_max_velocity = 4

    # Parameter of Radar
    f_center = 77.27e9     # Center Frequency (Hz)
    B = 0.26e9             # Radar Bandwidth (Hz)
    f_sampling = 15e6      # Sampling Rate (Hz)
    num_samples = 256      # N samples per chirp (256)
    tc = 45e-6             # Chirp time / interval (s)
    chirp_slope = 15e12   # Chirp slope (Hz/s)
    num_chirps = 64        # N Chirps per CPI (128)
    Rx_gain = 48           # RX Gain (dB)

    Tx_Channels = 12       # Aus "TDM-MIMO (12 Tx..."
    Rx_Channels = 16       # Aus "TDM-MIMO (...16 Rx)"
  
    return radar_file, factor_formular_max_velocity, f_center, B, f_sampling, num_samples, tc, chirp_slope, num_chirps, Rx_gain, Tx_Channels, Rx_Channels


def main():
    
    # Create Radar object
    USSR_radar = Radar(*USRR_Dynamic_Config(), num=1, use_tk=True, output_print=True)
    
    # Tasks ausführen
    USSR_radar.Task_Step_1()
    USSR_radar.Task_Step_2()
    USSR_radar.Task_Step_3()
    USSR_radar.Task_Step_4()

    # Create Radar object
    #MRR_radar = Radar(*MRR_CornField_Config(), num=0, use_tk=True, output_print=True)
    # Tasks ausführen
    #MRR_radar.Task_Step_1()
    #MRR_radar.Task_Step_2()
    #MRR_radar.Task_Step_3()
    #MRR_radar.Task_Step_4()

if __name__ == "__main__":
    main()