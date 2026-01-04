''' 
This code processes data from a series of .pkl files, extracting the DC and RF photocurrent and calculating the beam waist for two segments of the QPD (quadA and quadD). 
This data was aquired on 20250114 and 20250113 with the fiber laser setup.

Compatible data: 
"final_data/20250114/BeamProfile (i.e. first data from Timesh)/20250114",
"final_data/20250114/BeamProfile (i.e. first data from Timesh)/20250113"
'''

import re
import sys
import h5py
import pickle
import scipy
import json
import pathlib
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import glob
from scipy.optimize import curve_fit
from scipy.special import erf

font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
cmu_serif = fm.FontProperties(fname=font_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

# Error function
def erf_model(x, A, B, C, D):
    return A * erf(B * (x - C)) + D

# ====================================== DATA ==================================================== #
data_dirs = [
    "/Users/asca/Documents/University/Master Thesis/code/Data/BeamProfile (i.e. first data from Timesh)/20250114",
    "/Users/asca/Documents/University/Master Thesis/code/Data/BeamProfile (i.e. first data from Timesh)/20250113"
]
fig_dir = os.path.join(data_dirs[0], "fig")
os.makedirs(fig_dir, exist_ok=True)

# Collect all .pkl files
file_list = []
for data_dir in data_dirs:
    print(f"Checking directory: {data_dir}")
    print("All files in directory:", os.listdir(data_dir))
    files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    print(f"Found .pkl files in {data_dir}:", files)
    file_list.extend(files)

print("All .pkl files:", file_list)

# Function to load data
def load_data(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

# -------------------- Construct figure environments -------------------- #
save_dir = fig_dir
print(save_dir)

# =================================== PARAMETERS ================================================= #
procdata = {'quadA': {'z_positions': [], 'beamwaist_rf': [], 'beamwaist_dc': []},
            'quadB': {'z_positions': [], 'beamwaist_rf': [], 'beamwaist_dc': []},
            'quadC': {'z_positions': [], 'beamwaist_rf': [], 'beamwaist_dc': []},
            'quadD': {'z_positions': [], 'beamwaist_rf': [], 'beamwaist_dc': []}}

# Power conversion parameters
RBW_Hz = 10e3  # 10 kHz
num_freq_points = 1183  # Number of measurable points
bin_width_Hz = RBW_Hz / num_freq_points  # ~8.45 Hz per bin

# Collect all data first, sorted by Z position
all_data = []
for pkl_file in file_list:
    # Load the data
    data = load_data(pkl_file)
    print(f"Loaded: {pkl_file}")

    # Check if Z positions are available in the data
    if 'stage_laser_zposition' not in data['rawdata']:
        print(f"Warning: 'stage_laser_zposition' not found in {pkl_file}. Skipping.")
        continue

    z_positions = data['rawdata']['stage_laser_zposition']  # Assume this is in mm
    if z_positions.size == 0:  # Check if the array is empty
        print(f"Warning: 'stage_laser_zposition' is empty in {pkl_file}. Skipping.")
        continue

    # Use the last Z position (mimicking your supervisor's approach)
    current_z = z_positions[-1]  # Assume mm; adjust if in µm (e.g., divide by 1000)
    # If in µm, convert to mm
    # current_z = z_positions[-1] / 1000

    all_data.append((current_z, data, pkl_file))

# Sort by Z position
all_data.sort(key=lambda x: x[0])
print("Sorted Z positions:", [item[0] for item in all_data])

# ==================================== PLOTTING ================================================== #

# Plot configurations (unchanged)
fig00, ax00 = plt.subplots(figsize=(10, 6))
fig00_figname = 'QPDGapScan_LaserProfile_RF'
ax00.set_xlabel(r'Z Position [mm]', fontsize=14)
ax00.set_ylabel(r'Beam Waist $\rm{w}_0$ [\textmu{}m]', fontsize=14)
ax00.grid(True, linestyle='--', alpha=0.6)
ax00.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5, direction='in')
title00 = r'\textbf{Beam Profile (RF)}'
ax00.set_title(title00, fontsize=14, fontweight='bold', pad=10)

fig10, ax10 = plt.subplots(figsize=(10, 6), layout='constrained')
fig10_figname = 'QPDGapScan_LaserProfile_DC'
ax10.set_xlabel(r'Z Position [mm]', fontsize=14)
ax10.set_ylabel(r'Beam Waist $\rm{w}_0$ [\textmu{}m]', fontsize=14)
ax10.grid(True, linestyle='--', alpha=0.6)
ax10.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5, direction='in')
title10 = r'\textbf{Beam Profile (DC)}'
ax10.set_title(title10, fontsize=14, fontweight='bold', pad=10)

fig20, ax20 = plt.subplots(figsize=(10, 6))
fig20_figname = 'QPDGapScan_RFData'
ax20.set_xlabel(r'Translation Stage Position [mm]', fontsize=14)
ax20.set_ylabel(r'RF Peak Magnitude [dBm]', fontsize=14)
ax20.grid(True, linestyle='--', alpha=0.6)
ax20.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5, direction='in')
title20 = r'\textbf{RF Peak Magnitude}'
ax20.set_title(title20, fontsize=14, fontweight='bold', pad=10)

fig30, ax30 = plt.subplots(figsize=(10, 6), layout='constrained')
fig30_figname = 'QPDGapScan_DCData'
ax30.set_xlabel(r'Translation Stage Position [mm]', fontsize=14)
ax30.set_ylabel(r'DC Photocurrent [A]', fontsize=14)
ax30.grid(True, linestyle='--', alpha=0.6)
ax30.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5, direction='in')
title30 = r'\textbf{DC Photocurrent}'
ax30.set_title(title30, fontsize=14, fontweight='bold', pad=10)

# Process sorted data
for current_z, data, pkl_file in all_data:
    print(f"Processing: {pkl_file} with Z = {current_z}")

    # Determine if X or Y line scan and calculate the position array (in µm)
    if data['global_params']['xstart_um'] == data['global_params']['xstop_um']:
        step = int((data['global_params']['ystop_um'] - data['global_params']['ystart_um']) / data['global_params']['xystep_um']) + 1
        pos_array = np.linspace(data['global_params']['ystart_um'], data['global_params']['ystop_um'], step)
    elif data['global_params']['ystart_um'] == data['global_params']['ystop_um']:
        step = int((data['global_params']['xstop_um'] - data['global_params']['xstart_um']) / data['global_params']['xystep_um']) + 1
        pos_array = np.linspace(data['global_params']['xstart_um'], data['global_params']['xstop_um'], step)
    else:
        # Handle case where neither X nor Y is constant (unlikely for a beam profile scan)
        print(f"Warning: Scan direction ambiguous in {pkl_file}. Skipping.")
        continue


    # Loop over segments
    for idq, (segn, stat) in enumerate(data['rfsw00']['cmd']['quad_select'].items()):
        if stat:
            # Power conversion and beam waist calculation
            power_dbm_RBW = data['rawdata'][segn]['spa_peak1_dbm']
            power_mw = 10 ** (power_dbm_RBW / 10)  # mW
            power_watts_raw = power_mw * 1e-3  # W
            power_dbm_hz = power_dbm_RBW - 10 * np.log10(RBW_Hz)
            power_watts_hz = 10 ** ((power_dbm_hz - 30) / 10)  # W/Hz
            power_watts_avg = np.mean(power_watts_hz, axis=1)  # Average over measurements

            if len(pos_array) != len(power_watts_avg):
                print(f"Error: pos_array and power_watts_avg length mismatch for {pkl_file}, segment {segn}.")
                continue

            # Fit the error function model for RF
            initial_guess = [np.max(power_watts_avg) / 2, 0.01, np.mean(pos_array), np.min(power_watts_avg)]
            try:
                params, _ = curve_fit(erf_model, pos_array, power_watts_avg, p0=initial_guess)
                A_fit, B_fit, x0_fit, C_fit = params
                spot_size = 1 / (np.sqrt(2) * B_fit)  # Beam waist in µm
            except RuntimeError:
                print(f"Fit failed for {pkl_file}, segment {segn}. Skipping.")
                continue

            # Store RF data
            procdata[segn]['z_positions'].append(current_z)
            procdata[segn]['beamwaist_rf'].append(spot_size)

            # --- DC Beam Waist Calculation ---
            if 'dmm00_curr_amp' in data['rawdata'][segn]:
                dc_curr_avg = np.mean(data['rawdata'][segn]['dmm00_curr_amp'], axis=1)  # Average DC photocurrent

                if len(pos_array) != len(dc_curr_avg):
                    print(f"Error: pos_array and dc_curr_avg length mismatch for {pkl_file}, segment {segn}.")
                    continue

                # Fit the error function model for DC
                initial_guess_dc = [np.max(dc_curr_avg) / 2, 0.01, np.mean(pos_array), np.min(dc_curr_avg)]
                try:
                    params_dc, _ = curve_fit(erf_model, pos_array, dc_curr_avg, p0=initial_guess_dc)
                    A_fit_dc, B_fit_dc, x0_fit_dc, C_fit_dc = params_dc
                    spot_size_dc = 1 / (np.sqrt(2) * B_fit_dc)  # Beam waist in µm
                    procdata[segn]['beamwaist_dc'].append(spot_size_dc)
                except RuntimeError:
                    print(f"DC Fit failed for {pkl_file}, segment {segn}. Skipping.")
                    continue

            # Plot the raw RF and DC data
            ax20.plot(pos_array / 1000, data['rawdata'][segn]['spa_peak1_dbm'].mean(axis=1), label=f"{segn} at {current_z:.2f} mm")  # Convert µm to mm
            if 'dmm00_curr_amp' in data['rawdata'][segn]:
                ax30.plot(pos_array / 1000, data['rawdata'][segn]['dmm00_curr_amp'].mean(axis=1), label=f"{segn} at {current_z:.2f} mm")

# Plot the processed data
for segn in procdata.keys():
    if procdata[segn]['z_positions']:  # Only plot if data exists
        # Sort the data for this segment by Z position
        z_and_waist_rf = sorted(zip(procdata[segn]['z_positions'], procdata[segn]['beamwaist_rf']), key=lambda x: x[0])
        z_sorted, waist_rf_sorted = zip(*z_and_waist_rf) if z_and_waist_rf else ([], [])
        ax00.plot(z_sorted, waist_rf_sorted, label=segn)

        if procdata[segn]['beamwaist_dc']:  # Check if DC data exists
            z_and_waist_dc = sorted(zip(procdata[segn]['z_positions'], procdata[segn]['beamwaist_dc']), key=lambda x: x[0])
            z_sorted, waist_dc_sorted = zip(*z_and_waist_dc) if z_and_waist_dc else ([], [])
            ax10.plot(z_sorted, waist_dc_sorted, label=f"{segn} DC")

# Add legends
ax00.legend()
ax10.legend()

# Save figures
fig00.savefig(os.path.join(fig_dir, f"{fig00_figname}_RF.png"), dpi=300)
fig10.savefig(os.path.join(fig_dir, f"{fig10_figname}_DC.png"), dpi=300)
fig20.savefig(os.path.join(fig_dir, f"{fig20_figname}.png"), dpi=300)
fig30.savefig(os.path.join(fig_dir, f"{fig30_figname}.png"), dpi=300)

plt.show()

# ================================================================================================ #
#                                 MINIMUM DC WAIST CALCULATION                                     #
# ================================================================================================ #

# Collect all DC beam waist values into a single list
all_dc_waists = []
for segn in procdata:
    all_dc_waists.extend(procdata[segn]['beamwaist_dc'])

# Find the minimum DC beam waist
if all_dc_waists:
    min_dc_waist = np.min(all_dc_waists)
    
    # Locate the segment and Z position corresponding to this minimum value
    min_info = None
    for segn in procdata:
        for z, waist in zip(procdata[segn]['z_positions'], procdata[segn]['beamwaist_dc']):
            if np.isclose(waist, min_dc_waist): # Use isclose for floating point comparison
                min_info = (segn, z, waist)
                break
        if min_info:
            break
            
    print("\n" + "="*80)
    print("RESULTS: MINIMUM DC BEAM WAIST ACHIEVED")
    print("="*80)
    print(f"The minimum DC beam waist ($\rm{{w}}_0$) achieved across all segments is: \n")
    print(f"  {min_dc_waist:.2f} um")
    
    if min_info:
        segment, z_pos, waist = min_info
        print(f"  (Achieved by segment {segment} at Z = {z_pos:.3f} mm)")
    
    print("="*80)
else:
    print("\n[WARNING] No valid DC beam waist values were calculated to determine the minimum.")