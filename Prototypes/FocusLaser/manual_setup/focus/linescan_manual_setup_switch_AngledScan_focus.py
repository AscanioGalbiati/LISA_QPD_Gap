import re
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import curve_fit
from scipy.special import erf

# Set up font and LaTeX rendering
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
cmu_serif = fm.FontProperties(fname=font_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

# Error function for curve fitting
def erf_model(u, A, B, C, D):
    return A * erf(B * (u - C)) + D

# Base directory containing .pkl files
base_dir = "/Users/asca/Documents/University/Master Thesis/code/Data/manual setup/20250926/VIGO17_333-1_QPD_0750_20_AW_011_CC_250926_LB1471C_quadA&D_manual_setup_AngledScan"

# Segments to process
segns = ['quadA', 'quadD']

# Function to load data
def load_data(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

# Function to extract Along position from filename
def extract_along_position(file_path):
    match = re.search(r'(?:Along|Y)(\d+)um', file_path)
    if match:
        return int(match.group(1))  # Return in µm
    print(f"Warning: Could not extract Along/Y position from {file_path}")
    return None

# Function to extract Z position from filename
def extract_z_position(file_path):
    match = re.search(r'Z(\d+)um', file_path)
    if match:
        z_um = int(match.group(1))  # Z position in µm
        z_mm = z_um / 1000  # Convert to mm
        return z_mm
    print(f"Warning: Could not extract Z position from {file_path}")
    return None

# Collect data from the directory
data_collection = {}
procdata = {segn: {'z_positions': [], 'beamwaist_dc': []} for segn in segns}
all_z_positions = set()
all_along_positions = set()

# Load .pkl files
file_list = sorted(glob.glob(os.path.join(base_dir, "*.pkl")))
if not file_list:
    print(f"Error: No .pkl files found in {base_dir}")
    exit(1)

# Extract Along and Z positions and sort files
files_with_positions = []
for file_path in file_list:
    along_um = extract_along_position(file_path)
    z_mm = extract_z_position(file_path)
    if along_um is not None and z_mm is not None:
        files_with_positions.append((along_um, z_mm, file_path))
        all_along_positions.add(along_um)  # Store as µm
        all_z_positions.add(z_mm)
    else:
        print(f"Skipping {file_path}: Missing Along or Z position")

if not files_with_positions:
    print(f"Error: No valid files with both Along and Z positions found in {base_dir}")
    exit(1)

# Sort by along_um, then z_mm
files_with_positions.sort(key=lambda x: (x[0], x[1]))

# Process each file
for along_um, z_mm, pkl_file in files_with_positions:
    if along_um not in data_collection:
        data_collection[along_um] = []
    
    # Create figure directory
    fig_dir = os.path.join(base_dir, "fig")
    os.makedirs(fig_dir, exist_ok=True)
    
    # Load data
    data = load_data(pkl_file)
    
    # Extract u position array (in mm)
    u_array = data['rawdata']['u_position']
    
    # Extract DC photocurrent and calculate beam waist for each segment
    data_point = {'z_mm': z_mm, 'u_pos_mm': u_array}
    for segn in segns:
        if segn in data['rawdata'] and 'dmm00_curr_amp' in data['rawdata'][segn]:
            dc_curr_avg = np.mean(data['rawdata'][segn]['dmm00_curr_amp'], axis=1)
            data_point[f'dc_curr_{segn}'] = dc_curr_avg
            
            # Calculate beam waist
            if len(u_array) != len(dc_curr_avg):
                print(f"Error: u_array and dc_curr_avg length mismatch for {pkl_file}, segment {segn}.")
                continue
            initial_guess_dc = [np.max(dc_curr_avg) - np.min(dc_curr_avg), 1.0, np.mean(u_array), np.min(dc_curr_avg)]
            try:
                params_dc, pcov = curve_fit(erf_model, u_array, dc_curr_avg, p0=initial_guess_dc, bounds=([-np.inf, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]))
                A_fit_dc, B_fit_dc, u0_fit_dc, C_fit_dc = params_dc
                spot_size_dc = 1 / (np.sqrt(2) * B_fit_dc) * 1000  # Beam waist in µm
                # Check for large beam waist values (potential bad fits)
                if spot_size_dc > 300:  # Arbitrary threshold (µm)
                    print(f"Warning: Large beam waist ({spot_size_dc:.2f} µm) for {pkl_file}, segment {segn}. Possible bad fit.")
                else:
                    procdata[segn]['z_positions'].append(z_mm)
                    procdata[segn]['beamwaist_dc'].append(spot_size_dc)
                    print(f"Along={along_um}um, {segn} - Z position: {z_mm:.1f} mm, Beam waist (DC): {spot_size_dc:.2f} µm")
            except RuntimeError:
                print(f"DC Fit failed for {pkl_file}, segment {segn}. Skipping.")
            except ValueError as e:
                print(f"Fit error for {pkl_file}, segment {segn}: {e}. Skipping.")
        else:
            data_point[f'dc_curr_{segn}'] = None  # Handle missing data
            print(f"Warning: No data for {segn} in {pkl_file}")
    data_collection[along_um].append(data_point)

# Check if any data was collected
if not all_z_positions:
    print(f"Error: No valid Z positions found in any .pkl files. Check file naming or directory structure.")
    exit(1)

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
fig10_figname = 'QPDGapScan_LaserProfile_DC_Angled'
ax10.set_xlabel(r'Z Position [mm]', fontsize=14)
ax10.set_ylabel(r'Beam Waist $\rm{w}_0$ [\textmu{}m]', fontsize=14)
ax10.grid(True, linestyle='--', alpha=0.6)
ax10.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5, direction='in')
title10 = r'\textbf{Beam Profile (DC) - Angled Scans}'
ax10.set_title(title10, fontsize=14, fontweight='bold', pad=10)

# Plot beam waist data for each segment
for segn in segns:
    if procdata[segn]['z_positions'] and procdata[segn]['beamwaist_dc']:
        line_color = 'blue' if segn == 'quadA' else 'green'
        linestyle = '-'
        # Plot points with Z-dependent colors
        for z, w0 in zip(procdata[segn]['z_positions'], procdata[segn]['beamwaist_dc']):
            ax10.plot(z, w0, 'o', color=color_dict[z])
        # Connect points with a colored line
        ax10.plot(procdata[segn]['z_positions'], procdata[segn]['beamwaist_dc'],
                  linestyle=linestyle, color=line_color, alpha=0.5, label=segn)
ax10.legend(fontsize=12, loc='upper left', frameon=True, bbox_to_anchor=(0.02, 0.98))

# ----- Plot configuration for DC photocurrent (one plot per Along position) -----
all_along_positions = sorted(list(all_along_positions))
for along_um in all_along_positions:  # Now in µm
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    fig_figname = f'QPDGapScan_DCData_Along{int(along_um)}um_A_and_D'
    
    # Plot DC photocurrent data
    for data_point in data_collection[along_um]:
        line_color = color_dict[data_point['z_mm']]
        alpha_val = 1.0 if 'dc_curr_quadA' in data_point and data_point['dc_curr_quadA'] is not None else 0.7
        for segn in segns:
            if f'dc_curr_{segn}' in data_point and data_point[f'dc_curr_{segn}'] is not None:
                ax.plot(data_point['u_pos_mm'], data_point[f'dc_curr_{segn}'],
                        linestyle='-', color=line_color, alpha=alpha_val,
                        label=f'{segn} Z={data_point["z_mm"]:.2f} mm')
    
    # Configure DC photocurrent plot
    ax.set_xlabel(r'u Position (Perpendicular to Gap) [mm]', fontsize=14)
    ax.set_ylabel(r'DC Photocurrent [A]', fontsize=14)
    title = rf'\textbf{{DC Photocurrent (Along={int(along_um)}\textmu{{}}m: quadA \& quadD)}}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5, direction='in')
    ax.legend()

    # Save photocurrent figure
    fig_dir = os.path.join(base_dir, "fig")
    fig.savefig(os.path.join(fig_dir, f"{fig_figname}_good.png"), dpi=300)
    print(f"Photocurrent figure saved: {os.path.join(fig_dir, f'{fig_figname}_good.png')}")

    plt.close(fig)  # Close figure to free memory

# Save beam waist figure
fig10_dir = os.path.join(base_dir, "fig")
os.makedirs(fig10_dir, exist_ok=True)
fig10.savefig(os.path.join(fig10_dir, f"{fig10_figname}_good.png"), dpi=300)
print(f"Beam waist figure saved: {os.path.join(fig10_dir, f'{fig10_figname}_good.png')}")

# Print minimum beam waist for each segment
for segn in segns:
    min_waist = min(procdata[segn]['beamwaist_dc']) if procdata[segn]['beamwaist_dc'] else float('inf')
    print(f"Segment {segn}: Min waist: {min_waist:.4f} µm")

print("All plots generated.")
plt.show()