# -*- coding: utf-8 -*-
"""
DIAGONAL KNIFE-EDGE SCAN - ZOOM COMPARISON (LOAD vs GND)
Gap AD (quadrants A & D)

Compares LOAD and GND terminations in the gap region, calculating
overshoot based on the wide plateau regions defined in the full scan script.
"""
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import re
import pandas as pd

# ==================== FONT & STYLE ====================
# NOTE: Ensure this font path is correct on your system!
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
cm_bold_path = "/Users/asca/Library/Fonts/cmunbx.ttf"
fm.FontProperties(fname=font_path)
fm.FontProperties(fname=cm_bold_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

# ==================== CONFIG (Based on your first script's definitions) ====================

# Base directory (MUST point to the parent or a single Along*um folder)
base_dir = "/Users/asca/Documents/University/Master Thesis/code/Data/all quadrants/20251124/VIGO_FPW01_QPD_1500_20_B17R11_251122_LB1471C_quadABCD_manual_setup_Z13.56mm_LINESCANGap_AD/Along6700.0um"

# Plot/Region Parameters (Y_INDEX = 0, as defined in your full scan script)
u_min_plot_default = 0.28
u_max_plot_default = 0.72
left_region_default  = (0.28, 0.35)
right_region_default = (0.55, 0.72)
gap_region_default   = (0.35, 0.55)

# === ZOOMED PLOT RANGE (Focuses on the gap and immediate plateaus) ===
# We use the *plateau regions* (left/right_region_default) for calculation
# and the *gap region* (gap_region_default) for finding the max/min.
# The plot range itself will span the total region, or a slightly tighter version.
zoom_left = left_region_default[0] - 0.02 # Extend slightly to the left plateau
zoom_right = right_region_default[1] + 0.02 # Extend slightly to the right plateau
plot_u_min = 0.30
plot_u_max = 0.70

# === LASER POWER (from your first script) ===
P1 = 0.0005389  # W
sigma1 = 9.3463e-7  # W
P2 = 0.00053803  # W
sigma2 = 8.5199e-7  # W
w1 = 1 / (sigma1**2)
w2 = 1 / (sigma2**2)
P_combined = (P1 * w1 + P2 * w2) / (w1 + w2)


# ==================== FUNCTIONS ====================

def u_pos(f):
    """Extract Y position from 'Along[Y]um' folder name."""
    m = re.search(r'Along([\d.]+)um', f)
    return int(float(m.group(1))) if m else None

def extract_sum(data, u_key='u_position'):
    """Extracts quadA + quadD sum array from data dictionary."""
    raw = data["rawdata"]
    
    def get(key):
        arr = raw.get(key, {}).get("dmm00_curr_amp", None)
        if arr is None: return np.zeros(len(raw[u_key]))
        if arr.ndim == 1: arr = arr.reshape(-1,1)
        return arr.mean(axis=1)
        
    current_A = get("quadA")
    current_D = get("quadD")
    return current_A + current_D 

def load_data(p):
    with open(p, "rb") as f: return pickle.load(f)
    
def curr_fmt(val, pos):
    """Custom formatter for Photocurrent y-axis."""
    return '0' if abs(val) < 1e-12 else f'{val:.4f}'
curr_formatter = FuncFormatter(curr_fmt)

# ==================== FIND & SELECT DATA FOLDER ====================
# Find all Along*um folders and sort by Y position
along_folders = glob.glob(os.path.join(base_dir, "Along*um"))
y_folders = [(u_pos(os.path.basename(f)), f) for f in along_folders if u_pos(os.path.basename(f)) is not None]
y_folders.sort()

if not y_folders:
    raise FileNotFoundError(f"No 'Along*um' folders found in {base_dir}")

# We only process the first Y position found
y_um_to_plot, folder_to_plot = y_folders[0]
print(f"Selecting Diagonal Scan at Y = {y_um_to_plot} µm for comparison plot.")

fig_dir = os.path.join(folder_to_plot, "fig"); os.makedirs(fig_dir, exist_ok=True)

load_files = glob.glob(os.path.join(folder_to_plot, "*_load2.pkl")) or glob.glob(os.path.join(folder_to_plot, "*_load.pkl"))
gnd_files = glob.glob(os.path.join(folder_to_plot, "*_gnd.pkl"))

if not load_files or not gnd_files:
     raise FileNotFoundError(f"Missing LOAD or GND .pkl files in {folder_to_plot}")

load_file = load_files[0]
gnd_file = gnd_files[0]

# ==================== PROCESSING & CALCULATION ====================

# 1. Load the full data arrays
data_load_full = load_data(load_file)
data_gnd_full = load_data(gnd_file)
# Key for the diagonal position is 'u_position'
u_full = np.array(data_load_full['rawdata']['u_position'])

# Mask for the wide plateau region (The full calc range)
mask_wide = (u_full >= left_region_default[0]) & (u_full <= right_region_default[1])
u_wide = u_full[mask_wide]

# Extract sum currents over the wide range
I_load_wide = extract_sum(data_load_full)[mask_wide]
I_gnd_wide = extract_sum(data_gnd_full)[mask_wide]

# 2. Define the explicit left/right plateau masks on the WIDE data
left_mask  = (u_wide >= left_region_default[0])  & (u_wide <= left_region_default[1])
right_mask = (u_wide >= right_region_default[0]) & (u_wide <= right_region_default[1])

# 3. Calculate the robust plateau average (Mean of Left Plateau Mean + Right Plateau Mean)
plat_l = (np.mean(I_load_wide[left_mask]) + np.mean(I_load_wide[right_mask])) / 2.0
plat_g = (np.mean(I_gnd_wide[left_mask]) + np.mean(I_gnd_wide[right_mask])) / 2.0

# 4. Apply the zoom mask for plotting and finding the gap max/min
mask_zoom = (u_full >= plot_u_min) & (u_full <= plot_u_max)
uf = u_full[mask_zoom]
I_load = extract_sum(data_load_full)[mask_zoom]
I_gnd = extract_sum(data_gnd_full)[mask_zoom]

# Calculate gap maximums using the zoomed data (only within the defined gap region)
gap_mask = (uf >= gap_region_default[0]) & (uf <= gap_region_default[1])

# LOAD: find max
idx_max_load = np.argmax(I_load[gap_mask])
max_load = np.max(I_load[gap_mask])
u_load = uf[gap_mask][idx_max_load]

# GND: find max
idx_max_gnd = np.argmax(I_gnd[gap_mask])
max_gnd = np.max(I_gnd[gap_mask])
u_gnd = uf[gap_mask][idx_max_gnd]


# 5. Calculate overshoot percentages using the corrected plateaus
ov_l = (max_load/plat_l - 1)*100 if plat_l > 0 else 0
ov_g = (max_gnd/plat_g - 1)*100 if plat_g > 0 else 0


# ==================== PLOT GENERATION ====================
fig, ax = plt.subplots(figsize=(9.5, 5.8), layout='constrained')

# --- Data Plotting ---
line_gnd, = ax.plot(uf, I_gnd, color='#992f7f', lw=3.2, ls='--', label='quadA+quadD (GND)')
line_load,  = ax.plot(uf, I_load,  color='#992f7f', lw=3.2, ls='--', alpha=0.6, label='quadA+quadD (LOAD)')

# --- Region Highlight ---
ax.axvspan(gap_region_default[0], gap_region_default[1], color='gray', alpha=0.22, label='Gap region')

# --- Extremum Markers ---
ax.plot(u_gnd, max_gnd, 'o', color='#d62728', mec='darkred', mew=2.2, ms=12, zorder=10)
ax.plot(u_load,  max_load,  'o', color='#ff7f0e', mec='#cc5e00', mew=2.2, ms=12, zorder=10)

# --- RESULT BOX (Inset) ---
inset = ax.inset_axes([0.28, 0.655, 0.42, 0.20], transform=ax.transAxes)
inset.axis('off')

# GND Handle (Thick purple line, red marker)
gnd_handle = Line2D([0], [0], color='#992f7f', lw=3.2, ls='--',
                     marker='o', markerfacecolor='#d62728',
                     markeredgecolor='darkred', markeredgewidth=2.2, markersize=12)
# LOAD Handle (Thick purple line with alpha, orange marker)
load_handle = Line2D([0], [0],
                color=(*plt.cm.colors.to_rgb('#992f7f'), 0.6),
                lw=3.2, ls='--',
                marker='o',
                markerfacecolor='#ff7f0e',
                markeredgecolor='#cc5e00',
                markeredgewidth=2.2,
                markersize=12)

inset.legend(handles=[load_handle, gnd_handle],
             labels=[rf'Gap overshoot (LOAD): \textbf{{{ov_l:+.2f}\%}}',
                     rf'Gap overshoot (GND):  \textbf{{{ov_g:+.2f}\%}}'],
             loc='center', fontsize=18, frameon=False,
             handletextpad=1.2, labelspacing=1.4)

# --- Final Styling ---
title_prop = fm.FontProperties(fname=cm_bold_path, weight='bold', size=15)
ax.set_title(rf'\textbf{{DC Photocurrent Gap Region Zoom (Y={y_um_to_plot}\,\textmu{{m}}) – Diagonal AD}}',
             fontproperties=title_prop, pad=10)
ax.set_xlabel(r'Perpendicular offset $u$ [mm]', fontsize=14)
ax.set_ylabel(r'Photocurrent [A]', fontsize=14)
ax.grid(True, ls='--', alpha=0.6)
ax.tick_params(axis='both', which='major', labelsize=11, length=6, width=1.5, direction='in')
ax.yaxis.set_major_formatter(curr_formatter)
ax.set_ylim(bottom=0)
# Dynamic top limit based on max of both curves
ax.set_ylim(top = 1.05 * np.max(np.concatenate([I_load, I_gnd]))) 
ax.set_xlim(plot_u_min, plot_u_max)

# --- Main Legend (only for the two sum curves and the gap region) ---
handles, labels = ax.get_legend_handles_labels()
# Filter out the gap region from the main legend
handles_clean = [h for h, l in zip(handles, labels) if l != 'Gap region']
labels_clean = [l for l in labels if l != 'Gap region']

ax.legend(handles_clean, labels_clean, loc='upper right', fontsize=13, frameon=True, fancybox=False, edgecolor='black')


# --- Save Figure ---
save_name = f"ZOOM_Diagonal_Gap_Comparison_AD_Y{y_um_to_plot:04d}um.png"
save_path = os.path.join(fig_dir, save_name)
fig.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()
plt.close(fig)

print(f"\n--- Zoom Comparison Summary ---")
print(f"LOAD Overshoot: {ov_l:+.2f}% (Max at u={u_load:.3f} mm)")
print(f"GND Overshoot:  {ov_g:+.2f}% (Max at u={u_gnd:.3f} mm)")
print(f"Plot saved to: {save_path}")