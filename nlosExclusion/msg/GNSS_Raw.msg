float64 GNSS_week
float64 GNSS_time
float64 total_sv
float64 prn_satellites_index
float64 pseudorange
float64 raw_pseudorange
float64 carrier_phase
float64 doppler
float64 lamda
float64 snr
int64 LLI
int64 slip #0 no cycle slip or no carrier, 1 yes
float64 elevation
float64 azimuth
float64 err_tropo
float64 err_iono
float64 sat_clk_err
float64 sat_pos_x
float64 sat_pos_y
float64 sat_pos_z
float64 ttx # satellite obstime - tof - satellite clock bias
float64 vel_x # satellite velocity 
float64 vel_y # satellite velocity 
float64 vel_z # satellite velocity 
float64 dt # satellite clock drift
float64 ddt # satellite clock drift rate
float64 tgd # /* group delay parameters (Prange function in RTKLIB) */
#float64 GNSSCovariance
int64 visable #0-Not sure 1-visable 2-invisable
string  sat_system
int64 visable3DMA # ground truth visibility from 3DMA
float64 prE3dMA # ground truth pseudorange error from 3DMA

