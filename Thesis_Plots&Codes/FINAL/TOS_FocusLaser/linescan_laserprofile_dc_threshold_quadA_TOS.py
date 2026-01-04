'''
@author: A_Galbiati
This code processes data from a series of .pkl files, extracting the DC photocurrent and calculating the beam waist for the quadA segment.
The Test Optical Setup (TOS) was used.
Compatible data:
/Users/asca/Documents/University/Master Thesis/code/Data/VIGO17_NS089008_QPD_0750_20_AS_015_CC_asca_250617_focus_laser_thres_LB1471C_quadA/Y...
Folder structure:
final_data/QPDspecs_date_lens_quad/Y...
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
    'Y_A': "/Users/asca/Documents/University/Master Thesis/code/Data/20251215_final_data/20250617/VIGO17_NS089008_QPD_0750_20_AS_015_CC_asca_250617_focus_laser_thres_LB1471C_quadA/Y6100um"
}

# Function to load data
def load_data(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

# Function to extract Z position from filenames
def extract_z_position(file_path):
    match = re.search(r'Z(\d+)um', file_path)
    if match:
        z_um = int(match.group(1)) # Z position in µm
        z_mm = z_um / 1000 # Convert to mm
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
        if z_mm is not None and z_mm >= 32: # Filter Z positions >= 32 mm
            z_positions_with_files.append((z_mm, file_path))
            all_z_positions.add(z_mm)

    # Sort by Z position
    z_positions_with_files.sort(key=lambda x: x[0])
    z_values = [item[0] for item in z_positions_with_files]
    file_list = [item[1] for item in z_positions_with_files]

    # Process each file
    for z_mm, pkl_file in z_positions_with_files:
        data = load_data(pkl_file)

        # Generate X position array with adaptive steps
        xstart = data['global_params']['xstart_um']
        xstop = data['global_params']['xstop_um']
        xstep_big = data['global_params']['xstep_big_um']
        xstep_fine = data['global_params']['xstep_fine_um']
        x_threshold_start = data['global_params']['x_threshold_start_um']
        x_threshold_stop = data['global_params']['x_threshold_stop_um']
        x_array = []
        x_current = xstart
        while x_current <= xstop:
            if x_threshold_start <= x_current <= x_threshold_stop:
                x_array.append(x_current)
                x_current += xstep_fine
            else:
                x_array.append(x_current)
                x_current += xstep_big
        pos_array = np.array(x_array)

        # Update X-range for Y_A
        if y_pos == 'Y_A':
            if x_range['xstart_um'] is None or xstart > x_range['xstart_um']:
                x_range['xstart_um'] = xstart
            if x_range['xstop_um'] is None or xstop < x_range['xstop_um']:
                x_range['xstop_um'] = xstop

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
                spot_size_dc = 1 / (np.sqrt(2) * B_fit_dc) # Beam waist in µm
                procdata[y_pos][segn]['z_positions'].append(z_mm)
                procdata[y_pos][segn]['beamwaist_dc'].append(spot_size_dc)
                print(f"{y_pos}, {segn} - Z position: {z_mm:.1f} mm, Beam waist (DC): {spot_size_dc:.2f} µm")
            except RuntimeError:
                print(f"DC Fit failed for {pkl_file}, segment {segn}. Skipping.")
                continue
        else:
            data_point[f'dc_curr_{segn}'] = None # Handle missing data
            print(f"Warning: No data for {segn} in {pkl_file}")
        data_collection[y_pos].append(data_point)

# ==================================== PLOTTING ================================================== #
# Define color map
cmap = plt.cm.RdYlBu
all_z_positions = sorted(list(all_z_positions)) # Sort for consistent color assignment
norm = plt.Normalize(min(all_z_positions), max(all_z_positions))
color_dict = {}
for i, z in enumerate(all_z_positions):
    if i < len(all_z_positions) // 2: # First half for reds/yellows
        color_dict[z] = cmap(0.1 + 0.4 * (i / (len(all_z_positions) // 2))) # Range 0.1 to 0.5
    else: # Second half for blues
        color_dict[z] = cmap(0.6 + 0.4 * ((i - len(all_z_positions) // 2) / (len(all_z_positions) - len(all_z_positions) // 2))) # Range 0.6 to 1.0

# ----- Plot configuration for DC beam waist -----
fig10, ax10 = plt.subplots(figsize=(10, 6), layout='constrained')
fig10_figname = 'QPDGapScan_LaserProfile_DC_thres_restriced_range'
#x10.set_xlim(43.5, 56.5) # 42 , 56
#ax10.set_ylim(top=1.3)
# **UPDATES for fig10 (Beam Waist Plot):**
# * Labels/Title fontsize: 14 -> 16
# * Tick labels fontsize: 10 -> 13
# * Line width: Default -> 2.2
# * Marker size: Default -> 7 (for colored points)
# * Minimum highlight size: 12 (kept same)
# * Inset legend fontsize: 14 -> 18.7
# * Main legend fontsize: 13 -> 15

ax10.set_xlabel(r'Translatin Stage Z Position [mm]', fontsize=16) # FONTSIZE 16
ax10.set_ylabel(r'Beam Waist $w_0$ [\textmu{}m]', fontsize=16) # FONTSIZE 16
ax10.grid(True, linestyle='--', alpha=0.6)
ax10.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in') # FONTSIZE 13
title10 = r'\textbf{Beam Profile (DC)}'
ax10.set_title(title10, fontsize=16, fontweight='bold', pad=10) # FONTSIZE 16

# Plot beam waist data for quadA
for y_pos in procdata:
    segn = 'quadA'
    if procdata[y_pos][segn]['z_positions'] and procdata[y_pos][segn]['beamwaist_dc']:
        line_color = 'blue'
        linestyle = '-'
        # Plot points with Z-dependent colors
        for z, w0 in zip(procdata[y_pos][segn]['z_positions'], procdata[y_pos][segn]['beamwaist_dc']):
            if z >= 32: # Ensure only Z >= 32 mm is plotted
                ax10.plot(z, w0, 'o', color=color_dict[z], markersize=9) # MARKERSIZE 7
        # Connect points with a colored line
        filtered_z = [z for z in procdata[y_pos][segn]['z_positions'] if z >= 32]
        filtered_w0 = [w0 for z, w0 in zip(procdata[y_pos][segn]['z_positions'], procdata[y_pos][segn]['beamwaist_dc']) if z >= 32]
        ax10.plot(filtered_z, filtered_w0, linestyle=linestyle, color=line_color, alpha=0.5, label=segn, linewidth=2.2) # LINEWIDTH 2.2

# --- Find minimum waist and highlight with red circle + inset box ---
z_vals = np.array(procdata['Y_A']['quadA']['z_positions'])
w_vals = np.array(procdata['Y_A']['quadA']['beamwaist_dc'])
mask = z_vals >= 32
z_vals, w_vals = z_vals[mask], w_vals[mask]
i_min = np.argmin(w_vals)
z_min, w_min = z_vals[i_min], w_vals[i_min]

# Red open circle around the minimum
ax10.plot(z_min, w_min, '', mec='red', mfc='none', mew=2.5, ms=13.5) # MARKERSIZE 12

# ---- FIXED LaTeX inset label -------------------------------------------------
tex_label = r'$w_{{0,\min}}$(KE) $= {:.2f}\,\mu\mathrm{{m}}$'.format(w_min)
result_line = Line2D([0], [0], marker='o', color='w',
                     markeredgecolor='red', markerfacecolor='none',
                     markersize=10.5, markeredgewidth=2.5,
                     label=tex_label)
# Inset position adjusted slightly toward center for better fit
inset = ax10.inset_axes([0.31, 0.6, 0.35, 0.25])
inset.axis('off')
inset.legend(handles=[result_line], loc='upper left', fontsize=18.7, frameon=False, handlelength=1.2) # FONTSIZE 18.7

# Legend: larger font, shifted right
ax10.legend(fontsize=15, loc='upper left', frameon=True, bbox_to_anchor=(0.12, 0.975)) # FONTSIZE 15

# ----- Custom tick formatter: 0 → "0", others → 1 decimal -----
def tick_formatter(val, pos):
    # Reference uses 5 decimal places for Y-axis of photocurrent, so we update the formatter for that plot only.
    # Keep this one for beam waist plot (w0) which typically has 2 or 3 decimals.
    return '0' if val == 0 else f'{val:.2f}'
y_fmt10 = FuncFormatter(tick_formatter)
ax10.yaxis.set_major_formatter(y_fmt10) # Apply formatter to beam waist plot

# ----- Custom tick formatter: 0 → "0", others → 5 decimal (for photocurrent) -----
def tick_formatter_photocurrent(val, pos):
    return '0' if val == 0 else f'{val:.3f}'
y_fmt_photo = FuncFormatter(tick_formatter_photocurrent)

# ----- Plot configuration for DC photocurrent (X restricted to 3.8–4.9 mm) -----
fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
fig_figname = 'QPDGapScan_DCData_Y6100um_A_thres_restriced_range'

# **UPDATES for fig (Photocurrent Plot):**
# * Labels/Title fontsize: 14 -> 16
# * Tick labels fontsize: 10 -> 13
# * Line width: Default -> 2.2
# * Colorbar label/tick fontsize: 12/10 -> 14.2 (approx 14)
# * Colorbar position: [0.12, 0.695, 0.02, 0.18] -> [0.12, 0.55, 0.02, 0.18] (moved down)
# * Y-axis formatter: f'{val:.3f}' -> f'{val:.5f}' (for high precision)

# Plot DC photocurrent data
line_handles = []
line_labels = []
x_plot_min = 3.94 # 3.9
x_plot_max = 4.85 # 4.8

for y_pos in data_collection:
    linestyle = '-'
    for data_point in data_collection[y_pos]:
        if data_point['z_mm'] < 32: # Skip data points with Z < 32 mm
            continue
        x_pos_mm = data_point['x_pos_um'] / 1000
        mask = (x_pos_mm >= x_plot_min) & (x_pos_mm <= x_plot_max)
        x_pos_filtered = x_pos_mm[mask]
        line_color = color_dict[data_point['z_mm']]

        if data_point['dc_curr_quadA'] is not None:
            dc_curr_filtered = data_point['dc_curr_quadA'][mask]
            line, = ax.plot(x_pos_filtered, dc_curr_filtered,
                            linestyle=linestyle, color=line_color, alpha=1.0, linewidth=2.2) # LINEWIDTH 2.2

            # Keep only first and last Z for line legend
            z_pos = data_point['z_mm']
            if z_pos in (all_z_positions[0], all_z_positions[-1]):
                line_handles.append(line)
                line_labels.append(f'Z={z_pos:.2f} mm')

# Configure DC photocurrent plot
ax.set_xlabel(r'Translation Stage X Position [mm]', fontsize=16) # FONTSIZE 16
ax.set_ylabel(r'DC Photocurrent [A]', fontsize=16) # FONTSIZE 16
title = r'\textbf{DC Photocurrent (Y=6100\textmu{}m, quadA)}'
ax.set_xlim(3.97, 4.82) # 42 , 56
#ax10.set_ylim(top=1.3)
ax.set_title(title, fontsize=16, fontweight='bold', pad=10) # FONTSIZE 16
ax.grid(True, linestyle='--', alpha=0.6)
ax.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in') # FONTSIZE 13
ax.set_xlim(x_plot_min+0.02, x_plot_max-0.02)
ax.set_ylim(bottom=0)
ax.yaxis.set_major_formatter(y_fmt_photo) # USE f'{val:.5f}' formatter

# --- Colorbar replaces the old legend (same position) ---
cax = fig.add_axes([0.14, 0.68, 0.02, 0.18])  # [left, bottom, width, height] ADJUSTED BOTTOM FROM 0.695 TO 0.55
# Colormap bar
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical')
cbar.set_label(r'Z Position [mm]', fontsize=14.2, labelpad=10) # FONTSIZE 14.2
cbar.ax.tick_params(labelsize=14.2) # FONTSIZE 14.2

# Min / mid / max Z ticks
z_min, z_max = min(all_z_positions), max(all_z_positions)
z_mid = (z_min + z_max) / 2
cbar.set_ticks([z_min, z_mid, z_max])
cbar.set_ticklabels([f'{z_min:.0f}', f'{z_mid:.0f}', f'{z_max:.0f}'])

# Small line legend for first/last Z (Reference code removed this, keeping it commented)
#ax.legend(line_handles, line_labels, fontsize=11, loc='upper left', frameon=True, bbox_to_anchor=(0.05, 0.98))

# Save figures
for y_pos, data_dir in data_dirs.items():
    fig_dir = os.path.join(data_dir, "fig")
    fig10.savefig(os.path.join(fig_dir, f"{fig10_figname}.png"), dpi=300)
    fig.savefig(os.path.join(fig_dir, f"{fig_figname}.png"), dpi=300)
print(f"Figures saved in {fig_dir}")

# Print minimum beam waist for quadA
for y_pos in procdata:
    min_waist_A = min(procdata[y_pos]['quadA']['beamwaist_dc']) if procdata[y_pos]['quadA']['beamwaist_dc'] else float('inf')
    print(f"Dataset {y_pos}: Min waist A: {min_waist_A:.4f} µm")

plt.show()