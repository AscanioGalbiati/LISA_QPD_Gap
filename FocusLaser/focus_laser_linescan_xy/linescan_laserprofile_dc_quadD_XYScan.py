''' 
This code processes data for linescans executed along the x direction at a fixed Y height.
This initial code version processes data from a series orom a series of .pkl files, extracting the DC photocurrent and calculating the beam waist for the quadD segment.

Compatible data: 
final_data/20250506/VIGO17_NS089008_QPD_0750_20_AS_015_CC_250506_LB1761C_quadD_XYScan/Y...
final_data/20250507/VIGO17_NS089008_QPD_0750_20_AS_015_CC_250507_LB1761C_quadD/Y5900
final_data/20250508/VIGO17_NS089008_QPD_0750_20_AS_015_CC_250508_LB1761C_quadD/Y...
final_data/20250512/VIGO17_NS089008_QPD_0750_20_AS_015_CC_250508_LB1761C_quadA/Y...

Folder structure: 
final_data/QPDspecs_date_lens_quad/Y...
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
data_dir = "/Users/asca/Documents/University/Master Thesis/code/Data/VIGO17_NS089008_QPD_0750_20_AS_015_CC_asca_14_A/Y6600um"
fig_dir = os.path.join(data_dir, "fig")  # Directory for saving figures
os.makedirs(fig_dir, exist_ok=True)

# Check directory contents and collect all .pkl files
print(f"Checking directory: {data_dir}")
print("All files in directory:", os.listdir(data_dir))
# Load .pkl files from this directory
file_list = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
#print(f"Found .pkl files in {data_dir}:", file_list)

# Extract Z positions from filenames for sorting
z_positions_with_files = []
for file_path in file_list:
    # Extract Z position from filename (e.g., "Z15000um")
    match = re.search(r'Z(\d+)um', file_path)
    if match:
        z_um = int(match.group(1))  # Z position in µm
        z_mm = z_um / 1000  # Convert to mm
        z_positions_with_files.append((z_mm, file_path))
    else:
        print(f"Warning: Could not extract Z position from {file_path}")

# Sort files by Z position
z_positions_with_files.sort(key=lambda x: x[0])  # Sort by Z position
z_values = [item[0] for item in z_positions_with_files]  # Extract sorted Z positions
file_list = [item[1] for item in z_positions_with_files]  # Extract sorted file paths

print("Sorted .pkl files:", file_list)
print(f"Extracted Z positions: {z_values}")
print(f"Number of Z positions: {len(z_values)}")

# Function to load data
def load_data(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

# -------------------- Construct figure environments -------------------- #
save_dir = fig_dir
print(save_dir)

# =================================== PARAMETERS ================================================= #
# Only include quadD since that's the only data available
procdata = {'quadD': {'z_positions': [], 'beamwaist_dc': []}}

# ==================================== PLOTTING ================================================== #

# ----- Plot configuration for DC beam waist -----
fig10, ax10 = plt.subplots(figsize=(10, 6), layout='constrained')
fig10_figname = 'QPDGapScan_LaserProfile_DC'
ax10.set_xlabel(r'Z Position [mm]', fontsize=14)
ax10.set_ylabel(r'Beam Waist $\rm{w}_0$ [\textmu{}m]', fontsize=14)
ax10.grid(True, linestyle='--', alpha=0.6)
ax10.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5, direction='in')
title10 = r'\textbf{Beam Profile (DC)}'
ax10.set_title(title10, fontsize=14, fontweight='bold', pad=10)

# ----- Plot configuration for DC quadrant data -----
fig30, ax30 = plt.subplots(figsize=(10, 6), layout='constrained')
fig30_figname = 'QPDGapScan_DCData'
ax30.set_xlabel(r'Translation Stage Position [mm]', fontsize=14)
ax30.set_ylabel(r'DC Photocurrent [A]', fontsize=14)
ax30.grid(True, linestyle='--', alpha=0.6)
ax30.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5, direction='in')
title30 = r'\textbf{DC Photocurrent}'
ax30.set_title(title30, fontsize=14, fontweight='bold', pad=10)

ystep = 100
# Loop over .pkl files
for i, pkl_file in enumerate(file_list):
    # Load the data
    data = load_data(pkl_file)
    #print(f"Loaded: {pkl_file}")

    # Use the Z position extracted from the filename
    current_z = z_values[i]  # Z in mm

    # Use X scan parameters (since experiment scans X for each Y, Z)
    step = int((data['global_params']['xstop_um'] - data['global_params']['xstart_um']) / data['global_params']['xstep_um']) + 1
    pos_array = np.linspace(data['global_params']['xstart_um'], data['global_params']['xstop_um'], step)

    '''
    # Determine if X or Y line scan and calculate the position array (in µm)
    if data['global_params']['xstart_um'] == data['global_params']['xstop_um']:
        step = int((data['global_params']['ystop_um'] - data['global_params']['ystart_um']) / data['global_params']['xystep_um']) + 1
        pos_array = np.linspace(data['global_params']['ystart_um'], data['global_params']['ystop_um'], step)
    elif data['global_params']['ystart_um'] == data['global_params']['ystop_um']:
        step = int((data['global_params']['xstop_um'] - data['global_params']['xstart_um']) / ystep) + 1
        pos_array = np.linspace(data['global_params']['xstart_um'], data['global_params']['xstop_um'], step)
    '''
    # Process only quadD data
    segn = 'quadD'
    if segn in data['rawdata'] and 'dmm00_curr_amp' in data['rawdata'][segn]:
        # --- DC Beam Waist Calculation ---
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
            procdata[segn]['z_positions'].append(current_z)
            procdata[segn]['beamwaist_dc'].append(spot_size_dc)
        except RuntimeError:
            print(f"DC Fit failed for {pkl_file}, segment {segn}. Skipping.")
            continue

        # Plot the raw DC data
        ax30.plot(pos_array / 1000, data['rawdata'][segn]['dmm00_curr_amp'].mean(axis=1), label=f'Z={current_z:.1f} mm')  # Convert µm to mm
        '''
        # Plot filtered data
        if current_z < 6:
            ax30.plot(pos_array / 1000, data['rawdata'][segn]['dmm00_curr_amp'].mean(axis=1), label=f'Z={current_z:.1f} mm')  # Convert µm to mm
        '''
        print(f"B fit DC: {B_fit_dc:.4f}")
        print(f"Z position: {current_z:.1f} mm, Beam waist (DC): {spot_size_dc:.2f} µm")

# Plot the processed DC data for quadD
if procdata['quadD']['z_positions'] and procdata['quadD']['beamwaist_dc']:
    ax10.plot(procdata['quadD']['z_positions'], procdata['quadD']['beamwaist_dc'], label='quadD DC')

'''
# Plot the filtered data for quadD
if procdata['quadD']['z_positions']:  # Check if data exists
    # Filter z_positions and beamwaist_dc for z < 10
    valid_indices = [i for i, z in enumerate(procdata['quadD']['z_positions']) if z < 10]
    filtered_z_positions = [procdata['quadD']['z_positions'][i] for i in valid_indices]
    filtered_beamwaist_dc = [procdata['quadD']['beamwaist_dc'][i] for i in valid_indices]
    
    if filtered_z_positions and filtered_beamwaist_dc:  # Check if filtered data exists
        ax10.plot(filtered_z_positions, filtered_beamwaist_dc, label='quadD DC')
'''
# Add legend
ax10.legend()
ax30.legend()

# Save figures
fig10.savefig(os.path.join(fig_dir, f"{fig10_figname}_DC.png"), dpi=300)
fig30.savefig(os.path.join(fig_dir, f"{fig30_figname}.png"), dpi=300)

plt.show()