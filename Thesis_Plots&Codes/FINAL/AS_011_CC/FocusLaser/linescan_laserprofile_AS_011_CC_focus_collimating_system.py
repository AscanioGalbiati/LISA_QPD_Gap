'''
Focusing attempt with a collimating system of lenses to couple into L4
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
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable

# Set up font and LaTeX rendering
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
cmu_serif = fm.FontProperties(fname=font_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']


# Error function for curve fitting
def erf_model(x, A, B, C, D):
    return A * erf(B * (x - C)) + D

# Custom tick formatter: 0 -> "0", everything else -> 3 decimal places
def y_formatter(val, pos):
    return '0' if val == 0 else f'{val:.5f}'
y_fmt = FuncFormatter(y_formatter)

# Base directory containing .pkl files
base_dir = "/Users/asca/Documents/University/Master Thesis/code/Data/20251215_final_data/20251120/Y19000um"

# Directory to save figures
fig_dir = "/Users/asca/Documents/University/Master Thesis/code/Data/20251215_final_data/20251120/Y19000um/fig"
os.makedirs(fig_dir, exist_ok=True)

# Segments to process (plotting only A and D)
segns = ['quadA', 'quadB', 'quadC', 'quadD'] # Process all to calculate w0 for all
plot_segns = ['quadA', 'quadD'] # Only plot these two

# Line styles for quadrants (consistent across plots)
line_styles = {'quadA': '-', 'quadB': '--', 'quadC': '-.', 'quadD': ':'}

# Colors for quadrants in beam waist plot (more distinct for A and D)
colors = {'quadA': 'blue', 'quadB': 'gray', 'quadC': 'gray', 'quadD': 'red'}

# Function to load data
def load_data(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

# Function to extract Y position from filename
def extract_y_position(file_name):
    match = re.search(r'Y(\d+)um', file_name)
    if match:
        return int(match.group(1))  # Return in µm
    print(f"Warning: Could not extract Y position from {file_name}")
    return None

# Function to extract Z position from filename
def extract_z_position(file_name):
    match = re.search(r'Z(\d+)um', file_name)
    if match:
        z_um = int(match.group(1))  # Z position in µm
        z_mm = z_um / 1000  # Convert to mm
        return z_mm
    print(f"Warning: Could not extract Z position from {file_name}")
    return None

# Collect data from .pkl files
data_collection = {}
procdata = {segn: {'z_positions': [], 'beamwaist_dc': []} for segn in segns}
all_z_positions = set()
all_y_positions = set()

# Find all .pkl files in base_dir
file_list = sorted(glob.glob(os.path.join(base_dir, "*.pkl")))
if not file_list:
    print(f"Error: No .pkl files found in {base_dir}")
    exit(1)

# Extract Y and Z positions and sort files
files_with_positions = []
for file_path in file_list:
    file_name = os.path.basename(file_path)
    y_um = extract_y_position(file_name)
    z_mm = extract_z_position(file_name)
    if y_um is not None and z_mm is not None:
        files_with_positions.append((y_um, z_mm, file_path))
        all_y_positions.add(y_um)
        all_z_positions.add(z_mm)
    else:
        print(f"Skipping {file_path}: Missing Y or Z position")

if not files_with_positions:
    print(f"Error: No valid files with both Y and Z positions found in {base_dir}")
    exit(1)

# Sort by y_um, then z_mm
files_with_positions.sort(key=lambda x: (x[0], x[1]))

# X-range restriction for photocurrent plotting (in mm) - TARGET RANGE
x_min = 9.8
x_max = 10.65

# Process each file
for y_um, z_mm, pkl_file in files_with_positions:
    if y_um not in data_collection:
        data_collection[y_um] = []
    
    # Load data
    data = load_data(pkl_file)
    
    # Generate X position array with adaptive steps
    xstart = data['global_params']['xstart_mm']
    xstop = data['global_params']['xstop_mm']
    xstep_big = data['global_params']['xstep_big_mm']
    xstep_fine = data['global_params']['xstep_fine_mm']
    x_threshold_start = data['global_params']['x_threshold_start_mm']
    x_threshold_stop = data['global_params']['x_threshold_stop_mm']
    
    x_array = []
    x_current = xstart
    while x_current <= xstop:
        if x_threshold_start <= x_current <= x_threshold_stop:
            x_array.append(x_current)
            x_current += xstep_fine
        else:
            x_array.append(x_current)
            x_current += xstep_big
    x_array = np.array(x_array)
    
    # Apply X-range restriction for photocurrent plotting
    # Add a small tolerance to ensure x_max is included if needed
    mask = (x_array >= x_min) & (x_array <= x_max + 1e-6) 
    x_array_filtered = x_array[mask]
    
    # Extract DC photocurrent and calculate beam waist for each segment
    data_point = {'z_mm': z_mm, 'x_pos_mm': x_array_filtered}
    for segn in segns:
        if segn in data['rawdata'] and 'dmm00_curr_amp' in data['rawdata'][segn]:
            dc_curr_avg = np.mean(data['rawdata'][segn]['dmm00_curr_amp'], axis=1)
            if len(x_array) != len(dc_curr_avg):
                print(f"Error: x_array and dc_curr_avg length mismatch for {pkl_file}, segment {segn}.")
                continue
            dc_curr_filtered = dc_curr_avg[mask]
            data_point[f'dc_curr_{segn}'] = dc_curr_filtered
            
            # Calculate beam waist using full X-range
            if len(x_array) < 4:  # Minimum points for fitting
                #print(f"Warning: Insufficient data points for {pkl_file}, segment {segn}. Skipping beam waist.")
                continue
            initial_guess_dc = [np.max(dc_curr_avg) - np.min(dc_curr_avg), 1.0, np.mean(x_array), np.min(dc_curr_avg)]
            try:
                params_dc, _ = curve_fit(erf_model, x_array, dc_curr_avg, p0=initial_guess_dc, bounds=([-np.inf, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]))
                A_fit_dc, B_fit_dc, x0_fit_dc, C_fit_dc = params_dc
                spot_size_dc = 1 / (np.sqrt(2) * B_fit_dc) * 1000  # Beam waist in µm
                if spot_size_dc > 300:  # Arbitrary threshold (µm)
                    #print(f"Warning: Large beam waist ({spot_size_dc:.2f} µm) for {pkl_file}, segment {segn}. Possible bad fit.")
                    pass
                else:
                    procdata[segn]['z_positions'].append(z_mm)
                    procdata[segn]['beamwaist_dc'].append(spot_size_dc)
                    #print(f"Y={y_um}um, {segn} - Z position: {z_mm:.2f} mm, Beam waist (DC): {spot_size_dc:.2f} µm")
            except RuntimeError:
                #print(f"DC Fit failed for {pkl_file}, segment {segn}. Skipping.")
                pass
            except ValueError as e:
                #print(f"Fit error for {pkl_file}, segment {segn}: {e}. Skipping.")
                pass
        else:
            data_point[f'dc_curr_{segn}'] = None
            #print(f"Warning: No data for {segn} in {pkl_file}")
    data_collection[y_um].append(data_point)

# Check if any data was collected
if not all_z_positions or not all_y_positions:
    print(f"Error: No valid Y or Z positions found. Check folder structure or file naming.")
    exit(1)

# Sort all Z positions for colormap normalization
all_z_positions = sorted(list(all_z_positions))

# Define color map for Z-dependent points (using RdYlBu)
cmap = plt.cm.RdYlBu
norm = plt.Normalize(min(all_z_positions), max(all_z_positions))
color_dict = {}
num_z = len(all_z_positions)
# Distribute colors: first half for reds/yellows, second for blues
for i, z in enumerate(all_z_positions):
    if i < num_z // 2:
        color_dict[z] = cmap(0.1 + 0.4 * (i / max(1, (num_z // 2))))
    else:
        color_dict[z] = cmap(0.6 + 0.4 * ((i - num_z // 2) / max(1, (num_z - num_z // 2))))

# =============================================================
# 1. BEAM WAIST PLOT (quadA and quadD only)
# =============================================================

# Font sizes are already consistent here: 
# Title/Labels=14, Ticks=10
fig10, ax10 = plt.subplots(figsize=(10, 6), layout='constrained')
fig10_figname = 'QPDGapScan_LaserProfile_DC_AD_Yscan'
# MODIFIED: Labels/Title from 14 to 16. Ticks from 10 to 13.
ax10.set_xlabel(r'Translation Stage Z Position [mm]', fontsize=16)
ax10.set_ylabel(r'Beam Waist $\rm{w}_0$ [\textmu{}m]', fontsize=16)
ax10.grid(True, linestyle='--', alpha=0.6)
ax10.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in')
title10 = r'\textbf{Beam Profile (DC)}'
ax10.set_title(title10, fontsize=16, fontweight='bold', pad=10) # MODIFIED: Title from 14 to 16

# Plot beam waist data for quadA and quadD only
for segn in plot_segns:
    if procdata[segn]['z_positions'] and procdata[segn]['beamwaist_dc']:
        linestyle = line_styles[segn]
        line_color = colors[segn]
        marker = 'o' if segn == 'quadA' else 'D' # Use distinct markers for A and D
        for z, w0 in zip(procdata[segn]['z_positions'], procdata[segn]['beamwaist_dc']):
            ax10.plot(z, w0, marker=marker, color=color_dict[z], markersize=9) # Points colored by Z
        # Plot line connecting points
        # MODIFIED: Line width increased from default to 2.1
        ax10.plot(procdata[segn]['z_positions'], procdata[segn]['beamwaist_dc'],
                  linestyle=linestyle, color=line_color, alpha=0.5, label=segn, linewidth=2.2)

# --- Find minimum waist positions for quadA and quadD and highlight ---
min_info = []
for segn in plot_segns:
    if procdata[segn]['z_positions'] and procdata[segn]['beamwaist_dc']:
        z_vals = np.array(procdata[segn]['z_positions'])
        w_vals = np.array(procdata[segn]['beamwaist_dc'])
        # Sort by Z to ensure correct order before finding min
        sort_indices = np.argsort(z_vals)
        z_vals_sorted = z_vals[sort_indices]
        w_vals_sorted = w_vals[sort_indices]

        if len(w_vals_sorted) > 0:
            i_min = np.argmin(w_vals_sorted)
            z_min_val, w_min_val = z_vals_sorted[i_min], w_vals_sorted[i_min]
            marker = 'o' if segn == 'quadA' else 'D'
            # Highlight the minimum point
            ax10.plot(z_min_val, w_min_val, marker, mec='red', mfc='none', mew=2.5, ms=13.5)
            min_info.append((segn, w_min_val))

# ---- Inset result box (for w_0,min values) -------------------------------------------------
if min_info:
    # Prepare text lines for the inset box
    tex_lines = []
    for seg, wmin in min_info:
        tex_lines.append(r'$w_{{0,\min}}$(KE, quad{}) $= {:.2f}\,\mu\mathrm{{m}}$'.format(seg[-1], wmin))
    
    # Prepare handles for the inset legend
    handles = []
    for seg, _ in min_info:
        marker = 'o' if seg == 'quadA' else 'D'
        line = Line2D([0], [0], marker=marker, color='w',
                      markeredgecolor='red', markerfacecolor='none',
                      markersize=10.5, markeredgewidth=2.5)
        handles.append(line)
        
    # Create and configure the inset axes
    inset = ax10.inset_axes([0.28, 0.6, 0.35, 0.25])
    inset.axis('off')
    
    # Add the legend to the inset box
    # Using the raw tex lines as labels in the legend, which removes the need for separate text rendering
    # MODIFIED: Inset legend text size from 14 to 22.
    inset.legend(handles=handles, labels=tex_lines, loc='upper left', fontsize=18.7, frameon=False, handlelength=1.2)


# Main plot legend for line styles (quadA and quadD)
legend_handles_main = []
legend_labels_main = []
for segn in plot_segns:
    # MODIFIED: Line width increased from default to 2.1 for legend handle
    line = Line2D([0], [0], linestyle=line_styles[segn], color=colors[segn], alpha=0.5, label=segn, linewidth=2.2)
    legend_handles_main.append(line)
    legend_labels_main.append(segn)
# MODIFIED: Main legend text size from 13 to 14.
ax10.legend(legend_handles_main, legend_labels_main, fontsize=15, loc='upper left', frameon=True, bbox_to_anchor=(0.05, 0.975))

# Save beam waist figure
fig10.savefig(os.path.join(fig_dir, f"{fig10_figname}_good.png"), dpi=300)
print(f"Beam waist figure saved: {os.path.join(fig_dir, f'{fig10_figname}_good.png')}")

# =============================================================
# 2. DC PHOTOCURRENT PLOT (quadA and quadD only with colorbar)
# =============================================================

# Define Z position to exclude from this plot
Z_EXCLUDE = 7.47 # mm

fig_combined, ax_combined = plt.subplots(figsize=(12, 8), layout='constrained')
fig_combined_figname = 'QPDGapScan_DCData_AD_Y_All_Quadrants_thres'

# First pass: plot all data for quadA and quadD
for y_um in all_y_positions:
    for data_point in data_collection[y_um]:
        z_pos = data_point['z_mm']
        
        # --- MODIFICATION 3: Exclude Z=7.47 from plotting ---
        if abs(z_pos - Z_EXCLUDE) < 1e-6:
            continue
            
        line_color = color_dict[z_pos]
        for segn in plot_segns:
            if f'dc_curr_{segn}' in data_point and data_point[f'dc_curr_{segn}'] is not None:
                # MODIFIED: Line width increased from default to 2.1
                ax_combined.plot(data_point['x_pos_mm'], data_point[f'dc_curr_{segn}'],
                                 linestyle=line_styles[segn], color=line_color, alpha=0.8, linewidth=2.2)

# Configure DC photocurrent plot
# --- MODIFICATION 4: Ensured font sizes match beam waist plot (Labels/Title=16, Ticks=13) ---
# MODIFIED: Labels/Title from 14 to 16. Ticks from 10 to 13.
ax_combined.set_xlabel(r'Translation Stage X Position [mm]', fontsize=16)
ax_combined.set_ylabel(r'DC Photocurrent [A]', fontsize=16)
#y_um_str = f"Y={int(list(all_y_positions)[0])}\textmu{{}}m" if len(all_y_positions) == 1 else "All Y Positions"
#y_um_str = f"Y={int(list(all_y_positions)[0])}\textmu{{}}m" if len(all_y_positions) == 1 else "All Y Positions"
# CORRECT FIX is applied here:
y_um_str = rf"Y={int(list(all_y_positions)[0])}\textmu{{}}m" if len(all_y_positions) == 1 else "All Y Positions"
# Title already in bold and size 14
title_combined = rf'\textbf{{DC Photocurrent ({y_um_str}: quadA \& quadD)}}'
ax_combined.set_title(title_combined, fontsize=16, fontweight='bold', pad=10) # MODIFIED: Title from 14 to 16
ax_combined.grid(True, linestyle='--', alpha=0.6)
ax_combined.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in') # MODIFIED: Ticks from 10 to 13
ax_combined.set_xlim(x_min+0.15, x_max-0.15)  # Explicitly set restricted X-axis limits
ax_combined.set_ylim(bottom = 0, top=0.00024)
ax_combined.yaxis.set_major_formatter(y_fmt) # Apply custom Y-axis formatter

# --- Colorbar (Legend for Z colormap) ---
# --- MODIFICATION 1: Bring the colorbar down (bottom from 0.75 to 0.65) ---
cax = fig_combined.add_axes([0.12, 0.55, 0.02, 0.18]) 
cbar = fig_combined.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical')
# MODIFIED: Colorbar label size from 12 to 14.
cbar.set_label(r'Z Position [mm]', fontsize=14.2, labelpad=10)
# MODIFIED: Colorbar tick size from 10 to 13.
cbar.ax.tick_params(labelsize=14.2)

# Set min / mid / max Z ticks on the colorbar
if all_z_positions:
    z_min_val, z_max_val = min(all_z_positions), max(all_z_positions)
    z_mid = (z_min_val + z_max_val) / 2
    cbar.set_ticks([z_min_val, z_mid, z_max_val])
    cbar.set_ticklabels([f'{z_min_val:.2f}', f'{z_mid:.2f}', f'{z_max_val:.2f}'])

# --- MODIFICATION 2: Removed the Legend for Line Styles (Quadrants) ---

# Save photocurrent figure
fig_combined.savefig(os.path.join(fig_dir, f"{fig_combined_figname}_good.png"), dpi=300, bbox_inches='tight')
print(f"Photocurrent figure saved: {os.path.join(fig_dir, f'{fig_combined_figname}_good.png')}")

# Print minimum beam waist for each segment (data analysis is unaffected)
for segn in plot_segns:
    min_waist = min(procdata[segn]['beamwaist_dc']) if procdata[segn]['beamwaist_dc'] else float('inf')
    print(f"Segment {segn}: Min waist: {min_waist:.4f} µm")

print("All plots generated.")
plt.show()