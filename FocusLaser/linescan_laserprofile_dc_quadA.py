''' 
This code processes data from a series of .pkl files, extracting the DC photocurrent and calculating the beam waist for the quadA segment.

Compatible data: 
/Users/asca/Documents/University/Master Thesis/code/Data/VIGO17_NS089008_QPD_0750_20_AS_015_CC_asca_250605_quadA/Y7900um
/Users/asca/Documents/University/Master Thesis/code/Data/VIGO17_NS089008_QPD_0750_20_AS_015_CC_asca_250607_Y8000_quadA/Y7900um
/Users/asca/Documents/University/Master Thesis/code/Data/VIGO17_NS089008_QPD_0750_20_AS_015_CC_asca_250611_YScan_quadA/Y7900um"
/Users/asca/Documents/University/Master Thesis/code/Data/VIGO17_NS089008_QPD_0750_20_AS_015_CC_asca_250613_YScan_LB1471C_shortrange_quadA/Y5800um
/Users/asca/Documents/University/Master Thesis/code/Data/VIGO17_NS089008_QPD_0750_20_AS_015_CC_asca_250614_YScan_LB1471C_quadA/Y5800um
/Users/asca/Documents/University/Master Thesis/code/Data/VIGO17_NS089008_QPD_0750_20_AS_015_CC_asca_250611_YScan_planocvx_quadA/Y7800um
/Users/asca/Documents/University/Master Thesis/code/Data/VIGO17_NS089008_QPD_0750_20_AS_015_CC_asca_250614_YScan_LB1471C_quadA/Y5900um
/Users/asca/Documents/University/Master Thesis/code/Data/VIGO17_NS089008_QPD_0750_20_AS_015_CC_asca_250616_YScan_LB1761C_quadA_final/Y5100um
'''

import re
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import curve_fit
from scipy.special import erf

font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
cmu_serif = fm.FontProperties(fname=font_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

# Error function for curve fitting
def erf_model(x, A, B, C, D):
    return A * erf(B * (x - C)) + D

# ====================================== DATA ==================================================== #
data_dirs = {
    'Y_A': "/Users/asca/Documents/University/Master Thesis/code/Data/VIGO17_NS089008_QPD_0750_20_AS_015_CC_asca_250617_YScan_LB1471C_quadA_1/Y6100um"
}

# Function to load data 
def load_data(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

# Function to extract Z position from filenames
def extract_z_position(file_path):
    match = re.search(r'Z(\d+)um', file_path)
    if match:
        z_um = int(match.group(1))  # Z position in µm
        z_mm = z_um / 1000  # Convert to mm
        return z_mm
    return None

# Collect data from the directory
data_collection = {'Y_A': []}
procdata = {
    'Y_A': {'quadA': {'z_positions': [], 'beamwaist_dc': []}}
}
all_z_positions = set()

# Determine the X-range from Y_A
x_range = {'xstart_um': None, 'xstop_um': None}
for y_pos, data_dir in data_dirs.items():
    # Create figure directory
    fig_dir = os.path.join(data_dir, "fig")
    os.makedirs(fig_dir, exist_ok=True)
    
    # Load .pkl files
    file_list = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    
    # Extract Z positions and sort files
    z_positions_with_files = []
    for file_path in file_list:
        z_mm = extract_z_position(file_path)
        if z_mm is not None:
            z_positions_with_files.append((z_mm, file_path))
            all_z_positions.add(z_mm)
    
    # Sort by Z position
    z_positions_with_files.sort(key=lambda x: x[0])
    z_values = [item[0] for item in z_positions_with_files]
    file_list = [item[1] for item in z_positions_with_files]
    
    # Process each file
    for z_mm, pkl_file in z_positions_with_files:
        data = load_data(pkl_file)
        
        # Extract X position array
        step = int((data['global_params']['xstop_um'] - data['global_params']['xstart_um']) / data['global_params']['xstep_um']) + 1
        pos_array = np.linspace(data['global_params']['xstart_um'], data['global_params']['xstop_um'], step)
        
        # Update X-range for Y_A
        if y_pos == 'Y_A':
            if x_range['xstart_um'] is None or data['global_params']['xstart_um'] > x_range['xstart_um']:
                x_range['xstart_um'] = data['global_params']['xstart_um']
            if x_range['xstop_um'] is None or data['global_params']['xstop_um'] < x_range['xstop_um']:
                x_range['xstop_um'] = data['global_params']['xstop_um']
        
        # Extract DC photocurrent and calculate beam waist for quadA only
        data_point = {'z_mm': z_mm, 'x_pos_um': pos_array}
        segn = 'quadA'
        if segn in data['rawdata'] and 'dmm00_curr_amp' in data['rawdata'][segn]:
            dc_curr_avg = np.mean(data['rawdata'][segn]['dmm00_curr_amp'], axis=1)
            data_point[f'dc_curr_{segn}'] = dc_curr_avg
            
            # Calculate beam waist
            if len(pos_array) != len(dc_curr_avg):
                print(f"Error: pos_array and dc_curr_avg length mismatch for {pkl_file}, segment {segn}.")
                continue
            initial_guess_dc = [np.max(dc_curr_avg) / 2, 0.01, np.mean(pos_array), np.min(dc_curr_avg)]
            try:
                params_dc, _ = curve_fit(erf_model, pos_array, dc_curr_avg, p0=initial_guess_dc)
                A_fit_dc, B_fit_dc, x0_fit_dc, C_fit_dc = params_dc
                spot_size_dc = 1 / (np.sqrt(2) * B_fit_dc)  # Beam waist in µm
                procdata[y_pos][segn]['z_positions'].append(z_mm)
                procdata[y_pos][segn]['beamwaist_dc'].append(spot_size_dc)
                print(f"{y_pos}, {segn} - Z position: {z_mm:.1f} mm, Beam waist (DC): {spot_size_dc:.2f} µm")
            except RuntimeError:
                print(f"DC Fit failed for {pkl_file}, segment {segn}. Skipping.")
                continue
        else:
            data_point[f'dc_curr_{segn}'] = None  # Handle missing data
            print(f"Warning: No data for {segn} in {pkl_file}")
        data_collection[y_pos].append(data_point)

# ==================================== PLOTTING ================================================== #

# Define color map
cmap = plt.cm.RdYlBu
all_z_positions = sorted(list(all_z_positions))  # Sort for consistent color assignment
norm = plt.Normalize(min(all_z_positions), max(all_z_positions))
color_dict = {}
for i, z in enumerate(all_z_positions):
    if i < len(all_z_positions) // 2:  # First half for reds/yellows
        color_dict[z] = cmap(0.1 + 0.4 * (i / (len(all_z_positions) // 2)))  # Range 0.1 to 0.5
    else:  # Second half for blues
        color_dict[z] = cmap(0.6 + 0.4 * ((i - len(all_z_positions) // 2) / (len(all_z_positions) - len(all_z_positions) // 2)))  # Range 0.6 to 1.0

# ----- Plot configuration for DC beam waist -----
fig10, ax10 = plt.subplots(figsize=(10, 6), layout='constrained')
fig10_figname = 'QPDGapScan_LaserProfile_DC'
ax10.set_xlabel(r'Z Position [mm]', fontsize=14)
ax10.set_ylabel(r'Beam Waist $\rm{w}_0$ [\textmu{}m]', fontsize=14)
ax10.grid(True, linestyle='--', alpha=0.6)
ax10.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5, direction='in')
title10 = r'\textbf{Beam Profile (DC)}'
ax10.set_title(title10, fontsize=14, fontweight='bold', pad=10)

# Plot beam waist data for quadA
for y_pos in procdata:
    segn = 'quadA'
    if procdata[y_pos][segn]['z_positions'] and procdata[y_pos][segn]['beamwaist_dc']:
        line_color = 'red'
        linestyle = '-'
        # Plot points with Z-dependent colors
        for z, w0 in zip(procdata[y_pos][segn]['z_positions'], procdata[y_pos][segn]['beamwaist_dc']):
            ax10.plot(z, w0, 'o', color=color_dict[z])
        # Connect points with a colored line
        ax10.plot(procdata[y_pos][segn]['z_positions'], procdata[y_pos][segn]['beamwaist_dc'],
                  linestyle=linestyle, color=line_color, alpha=0.5, label=segn)
        ax10.legend(fontsize=12, loc='upper left', frameon=True, bbox_to_anchor=(0.02, 0.98))

# ----- Plot configuration for DC photocurrent -----
fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
fig_figname = 'QPDGapScan_DCData_Y7900um_A' # ADJUST FOR Y_A VALUE

# Plot DC photocurrent data
for y_pos in data_collection:
    linestyle = '-'
    for data_point in data_collection[y_pos]:
        # Filter data to the restricted X-range
        mask = (data_point['x_pos_um'] >= x_range['xstart_um']) & (data_point['x_pos_um'] <= x_range['xstop_um'])
        x_pos_filtered = data_point['x_pos_um'][mask]
        line_color = color_dict[data_point['z_mm']]
        
        # Plot for quadA
        if data_point['dc_curr_quadA'] is not None:
            dc_curr_filtered = data_point['dc_curr_quadA'][mask]
            ax.plot(x_pos_filtered / 1000, dc_curr_filtered, 
                    linestyle=linestyle, color=line_color, alpha=1.0,
                    label=f'Z={data_point["z_mm"]:.2f} mm')

# Configure DC photocurrent plot
ax.set_xlabel(r'Translation Stage Position [mm]', fontsize=14)
ax.set_ylabel(r'DC Photocurrent [A]', fontsize=14)
title = r'\textbf{DC Photocurrent (Y=5100\textmu{}m: quadA)}' # ADJUST FOR Y_A VALUE
ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
ax.grid(True, linestyle='--', alpha=0.6)
ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5, direction='in')
ax.legend()

# Save figures
for y_pos, data_dir in data_dirs.items():
    fig_dir = os.path.join(data_dir, "fig")
    fig10.savefig(os.path.join(fig_dir, f"{fig10_figname}_good.png"), dpi=300)
    fig.savefig(os.path.join(fig_dir, f"{fig_figname}_good.png"), dpi=300)

print(f"Figures saved in {fig_dir}")
# Print minimum beam waist for quadA
for y_pos in procdata:
    min_waist_A = min(procdata[y_pos]['quadA']['beamwaist_dc']) if procdata[y_pos]['quadA']['beamwaist_dc'] else float('inf')
    print(f"Dataset {y_pos}: Min waist A: {min_waist_A:.4f} µm")
    
plt.show()