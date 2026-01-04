'''
FINAL!!!! - Quadrants B+C
@author: A_Galbiati
Horizontal linescans data analysis for QUADRANTS B+C
Combined power from two 2025-11-03 measurements
+ Quantum Efficiency (η = I_sum / P) on right Y-axis
*** IMPLEMENTED BIGGER FONTS AND THICKER LINES ***
'''
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
# import csv # CSV removed as requested

# Set up font and LaTeX rendering
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
cmu_serif = fm.FontProperties(fname=font_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

# === CORRECTED BASE DIRECTORY ===
base_dir = '/Users/asca/Documents/University/Master Thesis/code/Data/all quadrants/20251022/VIGO_NS089008_QPD_750_20_AS_0011_CC_251022_objLens_quadABCD_manual_setup_Z7.91mm_Yscan_thresholdX_YScan_FineSteps'

# === PLOT RANGE (narrow) ===
x_min_plot = 7.15
x_max_plot = 7.45

# === CALCULATION RANGES (Wide and explicit for Gap Overshoot) ===
calc_min = 7.10 
calc_max = 7.45
left_region  = (calc_min, 7.25)   # Left plateau
right_region = (7.35, calc_max)   # Right plateau
gap_region   = (7.25, 7.35)       # Gap 

# === TARGET Y POSITIONS (AS REQUESTED) ===
target_y_ums = [2300, 2500]

# === COMBINED POWER FROM TWO MEASUREMENTS (1σ) ===
P1 = 0.019158 # W
sigma1 = 1.2336e-5 # W
P2 = 0.019151 # W
sigma2 = 1.1417e-5 # W
w1 = 1 / (sigma1**2)
w2 = 1 / (sigma2**2)
P_combined = (P1 * w1 + P2 * w2) / (w1 + w2)
sigma_combined_1sigma = np.sqrt(1 / (w1 + w2))

# === 3σ FOR PLOTTING ===
sigma_combined_3sigma = 3 * sigma_combined_1sigma 
# rel_uncertainty_percent_3sigma calculation not needed for plotting, kept for reference
# rel_uncertainty_percent_3sigma = (sigma_combined_3sigma / P_combined) * 100

# QE references
eta_100 = 1.0
eta_real = 0.8
I_100 = eta_100 * P_combined
I_real_mean = eta_real * P_combined
# I_real_err_3sigma = eta_real * sigma_combined_3sigma

# <<< CSV SETUP REMOVED >>>

# === ALIGN ZEROS FUNCTION ===
def align_yaxis_zeros(ax1, ax2):
    """Force y=0 of ax1 and ax2 to be at the same physical height."""
    y1_min, y1_max = ax1.get_ylim()
    y2_min, y2_max = ax2.get_ylim()
    if y1_max == y1_min:
        return
    zero_frac = (0 - y1_min) / (y1_max - y1_min)
    y2_range = y2_max - y2_min
    ax2.set_ylim(0 - zero_frac * y2_range, 0 - zero_frac * y2_range + y2_range)

# Load data
def load_data(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

def extract_y_position(folder_name):
    match = re.search(r'Y(\d+)um', folder_name)
    return int(match.group(1)) if match else None

# Find Y folders
y_folders = glob.glob(os.path.join(base_dir, "Y*um"))
y_positions = [(extract_y_position(os.path.basename(f)), f) for f in y_folders if extract_y_position(os.path.basename(f)) is not None]
y_positions.sort()

# Formatters
def curr_fmt(val, pos): return '0' if val == 0 else f'{val:.4f}'
def qe_fmt(val, pos): return '0' if val == 0 else f'{val:.1f}'
y_fmt = FuncFormatter(curr_fmt)
y_fmt_qe = FuncFormatter(qe_fmt)

# Process each Y folder
for y_um, y_folder in y_positions:
    # === FILTER Y POSITIONS ===
    if y_um not in target_y_ums:
        continue
    
    fig_dir = os.path.join(y_folder, "fig_AS_011_CC_BC_BiggerFonts")
    os.makedirs(fig_dir, exist_ok=True)

    # === FIXED: FIND SINGLE RAW DATA FILE and load it for both LOAD and GND ===
    raw_pkl_files = glob.glob(os.path.join(y_folder, "*.pkl"))
    raw_pkl_files = [f for f in raw_pkl_files if not f.endswith('_load2.pkl') and not f.endswith('_gnd.pkl')]

    if not raw_pkl_files:
        print(f"Skipping Y={y_um} µm: Missing the single raw .pkl file in {y_folder}")
        continue

    raw_data_file = raw_pkl_files[0]
    
    try:
        data_raw = load_data(raw_data_file)
        data_load = data_raw # Use raw data for 'LOAD' calculation
        data_gnd = data_raw # Use raw data for 'GND' calculation
    except Exception as e:
        print(f"Error loading pickle data for {y_folder}: {e}. Skipping.")
        continue
    # === END FIX ===

    x_array = data_load['rawdata']['stage_laser_xposition']
    
    # Filter for plot/calculation
    mask_plot = (x_array >= x_min_plot) & (x_array <= x_max_plot)
    x_array_filtered = x_array[mask_plot]
    
    if len(x_array_filtered) == 0:
        print(f"No data in X range for {y_folder}")
        continue

    # Create calculation masks on the filtered array (which aligns with x_min_plot/x_max_plot)
    left_mask  = (x_array_filtered >= left_region[0])  & (x_array_filtered <= left_region[1])
    right_mask = (x_array_filtered >= right_region[0]) & (x_array_filtered <= right_region[1])
    gap_mask   = (x_array_filtered >= gap_region[0])   & (x_array_filtered <= gap_region[1])

    # =============================================================
    # LOAD PLOT – QUADRANTS B & C 
    # =============================================================
    fig_load, ax_load = plt.subplots(figsize=(10, 6), layout='constrained')

    # Data extraction
    quadB_mean_load = data_load['rawdata'].get('quadB', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)[mask_plot]
    quadC_mean_load = data_load['rawdata'].get('quadC', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)[mask_plot]
    I_sum_load = quadB_mean_load + quadC_mean_load
    
    # Plotting with larger linewidths (2.1)
    ax_load.plot(x_array_filtered, quadB_mean_load, label='quadB', color='tab:orange', linewidth=2.1)
    ax_load.plot(x_array_filtered, quadC_mean_load, label='quadC', color='tab:green', linewidth=2.1)
    ax_load.plot(x_array_filtered, I_sum_load, '--', color="#992f7f", linewidth=2.1, label='quadB+quadC', alpha=0.6)

    # Plotting Colored Regions
    ax_load.axvspan(left_region[0], left_region[1], color='tab:orange', alpha=0.12, label='Left region')
    ax_load.axvspan(right_region[0], right_region[1], color='tab:green', alpha=0.12, label='Right region')
    ax_load.axvspan(gap_region[0], gap_region[1], color='gray', alpha=0.22, label='Gap region')
    
    # Reference lines with linewidth=1.8
    ax_load.axhline(I_100, color="#c39d7a", linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 1\,$A/W')
    ax_load.axhline(I_real_mean, color='#8b5a2b', linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 0.8\,$A/W')

    # === QUANTUM EFFICIENCY ===
    QE_load = I_sum_load / P_combined
    ax_qe_load = ax_load.twinx()
    ax_qe_load.plot(x_array_filtered, QE_load, color="#00ff88", linewidth=2.1, alpha=0.2, label=r'$\eta_{\rm{mes}}$ [A/W]') 
    
    # Larger font size for Y-label (16)
    ax_qe_load.set_ylabel(r'Quantum Efficiency $\eta$ [A/W]', color='#8b5a2b', fontsize=16)
    
    # Larger font size for Y-tick labels (13)
    ax_qe_load.tick_params(axis='y', which='major', color='#8b5a2b', length=8, width=1.5, direction='in', labelsize=13)
    for tick in ax_qe_load.get_yticklabels():
        tick.set_color('#8b5a2b')

    # Force zero alignment
    ax_load.set_ylim(bottom=0)
    ax_qe_load.set_ylim(bottom=0, top=1.05)
    align_yaxis_zeros(ax_load, ax_qe_load)
    ax_qe_load.yaxis.set_major_formatter(y_fmt_qe)

    # ==================== GAP OVERSHOOT ANALYSIS (LOAD) ====================
    mean_left_load = np.mean(I_sum_load[left_mask]) if np.any(left_mask) else 0
    mean_right_load = np.mean(I_sum_load[right_mask]) if np.any(right_mask) else 0
    mean_plateaus_load = (mean_left_load + mean_right_load) / 2.0
    max_in_gap_load = np.max(I_sum_load[gap_mask]) if np.any(gap_mask) else 0
    idx_max = np.argmax(I_sum_load[gap_mask]) if np.any(gap_mask) else 0
    x_at_max = x_array_filtered[gap_mask][idx_max] if np.any(gap_mask) else np.mean(x_array_filtered)
    overshoot_percent_load = (max_in_gap_load / mean_plateaus_load - 1.0) * 100.0 if mean_plateaus_load > 0 else 0

    # Marker at maximum in gap (ms=10, mew=1.8)
    ax_load.plot(x_at_max, max_in_gap_load, 'o', color='#ff7f0e', mec='#cc5e00', mew=1.8, ms=10, zorder=10)

    # Result box (Raised position: [0.32, 0.78, 0.38, 0.16])
    inset_load = ax_load.inset_axes([0.32, 0.78, 0.38, 0.16], transform=ax_load.transAxes)
    inset_load.axis('off')
    # Marker style adjusted (lw=2.2, mew=1.8, ms=10)
    legend_handle = Line2D([0], [0], color=(*plt.cm.colors.to_rgb('#992f7f'), 0.6), lw=2.2, ls='--',
                           marker='o', markerfacecolor='#ff7f0e',
                           markeredgecolor='#cc5e00', markeredgewidth=1.8, markersize=10)
    # Larger font size for inset legend (22)
    inset_load.legend(handles=[legend_handle],
                      labels=[rf'GS: \textbf{{{overshoot_percent_load:+.2f}\%}}'],
                      loc='center', fontsize=22, frameon=False,
                      handletextpad=0.8, handlelength=1.8)

    # ==================== LEGEND & FINALIZE ====================
    lines1, labels1 = ax_load.get_legend_handles_labels()
    lines2, labels2 = ax_qe_load.get_legend_handles_labels()
    # Larger font size for main legend (14)
    ax_load.legend(lines1 + lines2, labels1 + labels2,
                   fontsize=14, loc='upper right', bbox_to_anchor=(0.98, 0.98),
                   frameon=True, fancybox=False, edgecolor='black')
                   
    # Larger font size for Title (16)
    ax_load.set_title(rf'\textbf{{DC Photocurrent (HL: Y={y_um}\textmu{{m}})}}',
                      fontsize=16, fontweight='bold', pad=10)
    # Larger font size for X-label (16)
    ax_load.set_xlabel(r'X Position [mm]', fontsize=16)
    # Larger font size for Y-label (16)
    ax_load.set_ylabel(r'Photocurrent [A]', fontsize=16)
    
    ax_load.grid(True, linestyle='--', alpha=0.6)
    # Larger font size for axis ticks (13)
    ax_load.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in')
    ax_load.yaxis.set_major_formatter(y_fmt)
    ax_load.set_xlim(x_min_plot + 0.015, x_max_plot - 0.015)

    fig_name_load = f"DC_Photocurrent_Y{y_um:04d}um_Quadrants_BC_GapOvershoot_LOAD_results_inv.png"
    fig_load.savefig(os.path.join(fig_dir, fig_name_load), dpi=300, bbox_inches='tight')
    print(f"LOAD (B+C + overshoot) saved: {os.path.join(fig_dir, fig_name_load)}")
    plt.close(fig_load)

    # =============================================================
    # GND PLOT – QUADRANTS B & C 
    # =============================================================
    fig_gnd, ax_gnd = plt.subplots(figsize=(10, 6), layout='constrained')

    quadB_mean_gnd = data_gnd['rawdata'].get('quadB', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)[mask_plot]
    quadC_mean_gnd = data_gnd['rawdata'].get('quadC', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)[mask_plot]
    I_sum_gnd = quadB_mean_gnd + quadC_mean_gnd
    
    # Plotting with larger linewidths (2.1)
    ax_gnd.plot(x_array_filtered, quadB_mean_gnd, label='quadB', color='tab:orange', linewidth=2.1)
    ax_gnd.plot(x_array_filtered, quadC_mean_gnd, label='quadC', color='tab:green', linewidth=2.1)
    ax_gnd.plot(x_array_filtered, I_sum_gnd, '--', color="#992f7f", linewidth=2.1, alpha=1.0, label='quadB+quadC')

    # Plotting Colored Regions
    ax_gnd.axvspan(left_region[0], left_region[1], color='tab:orange', alpha=0.12, label='Left region')
    ax_gnd.axvspan(right_region[0], right_region[1], color='tab:green', alpha=0.12, label='Right region')
    ax_gnd.axvspan(gap_region[0], gap_region[1], color='gray', alpha=0.22, label='Gap region')

    # Reference lines with linewidth=1.8
    ax_gnd.axhline(I_100, color='#d2b48c', linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 1\,$A/W')
    ax_gnd.axhline(I_real_mean, color='#8b5a2b', linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 0.8\,$A/W')

    # QE GND
    QE_gnd = I_sum_gnd / P_combined
    ax_qe_gnd = ax_gnd.twinx()
    ax_qe_gnd.plot(x_array_filtered, QE_gnd, color="#44fd00", linewidth=2.1, label=r'$\eta_{\rm{mes}}$ [A/W]', alpha=0.2) 
    
    # Larger font size for Y-label (16)
    ax_qe_gnd.set_ylabel(r'Quantum Efficiency $\eta$ [A/W]', color='#8b5a2b', fontsize=16)
    
    # Larger font size for Y-tick labels (13)
    ax_qe_gnd.tick_params(axis='y', which='major', color='#8b5a2b', length=8, width=1.5, direction='in', labelsize=13)
    for tick in ax_qe_gnd.get_yticklabels():
        tick.set_color('#8b5a2b')

    ax_gnd.set_ylim(bottom=0)
    ax_qe_gnd.set_ylim(bottom=0, top=1.05)
    align_yaxis_zeros(ax_gnd, ax_qe_gnd)
    ax_qe_gnd.yaxis.set_major_formatter(y_fmt_qe)

    # ==================== GAP OVERSHOOT ANALYSIS (GND) ====================
    mean_left_gnd = np.mean(I_sum_gnd[left_mask]) if np.any(left_mask) else 0
    mean_right_gnd = np.mean(I_sum_gnd[right_mask]) if np.any(right_mask) else 0
    mean_plateaus_gnd = (mean_left_gnd + mean_right_gnd) / 2.0
    max_in_gap_gnd = np.max(I_sum_gnd[gap_mask]) if np.any(gap_mask) else 0
    idx_max_g = np.argmax(I_sum_gnd[gap_mask]) if np.any(gap_mask) else 0
    x_at_max_g = x_array_filtered[gap_mask][idx_max_g] if np.any(gap_mask) else np.mean(x_array_filtered)
    overshoot_percent_gnd = (max_in_gap_gnd / mean_plateaus_gnd - 1.0) * 100.0 if mean_plateaus_gnd > 0 else 0

    # Marker (ms=10, mew=1.8)
    ax_gnd.plot(x_at_max_g, max_in_gap_gnd, 'o', color='#d62728', mec='darkred', mew=1.8, ms=10, zorder=10)

    # Result box (Raised position: [0.32, 0.78, 0.38, 0.16])
    inset_gnd = ax_gnd.inset_axes([0.32, 0.78, 0.38, 0.16], transform=ax_gnd.transAxes)
    inset_gnd.axis('off')
    # Marker style adjusted (lw=2.2, mew=1.8, ms=10)
    legend_handle_g = Line2D([0], [0], color='#992f7f', lw=2.2, ls='--',
                             marker='o', markerfacecolor='#d62728',
                             markeredgecolor='darkred', markeredgewidth=1.8, markersize=10)
    # Larger font size for inset legend (22)
    inset_gnd.legend(handles=[legend_handle_g],
                     labels=[rf'GS: \textbf{{{overshoot_percent_gnd:+.2f}\%}}'],
                     loc='center', fontsize=22, frameon=False,
                     handletextpad=0.8, handlelength=1.8)

    print(f"\n=== GAP OVERSHOOT SUMMARY Y = {y_um} µm (B+C) ===")
    print(f"LOAD → GS = {overshoot_percent_load:+.2f} %")
    print(f"GROUND → GS = {overshoot_percent_gnd:+.2f} %")

    # Legend & finalize
    lines1g, labels1g = ax_gnd.get_legend_handles_labels()
    lines2g, labels2g = ax_qe_gnd.get_legend_handles_labels()
    # Larger font size for main legend (14)
    ax_gnd.legend(lines1g + lines2g, labels1g + labels2g,
                  fontsize=14, loc='upper right', bbox_to_anchor=(0.98, 0.98),
                  frameon=True, fancybox=False, edgecolor='black')
                  
    # Larger font size for Title (16)
    ax_gnd.set_title(rf'\textbf{{DC Photocurrent (HL: Y={y_um}\textmu{{m}})}}',
                     fontsize=16, fontweight='bold', pad=10)
    # Larger font size for X-label (16)
    ax_gnd.set_xlabel(r'X Position [mm]', fontsize=16)
    # Larger font size for Y-label (16)
    ax_gnd.set_ylabel(r'Photocurrent [A]', fontsize=16)
    
    ax_gnd.grid(True, linestyle='--', alpha=0.6)
    # Larger font size for axis ticks (13)
    ax_gnd.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in')
    ax_gnd.yaxis.set_major_formatter(y_fmt)
    ax_gnd.set_xlim(x_min_plot + 0.015, x_max_plot - 0.015)

    fig_name_gnd = f"DC_Photocurrent_Y{y_um:04d}um_Quadrants_BC_GapOvershoot_GND_results_inv.png"
    fig_gnd.savefig(os.path.join(fig_dir, fig_name_gnd), dpi=300, bbox_inches='tight')
    print(f"GND (B+C + overshoot) saved: {os.path.join(fig_dir, fig_name_gnd)}")
    plt.close(fig_gnd)

print(f"\n=== ALL DONE (HORIZONTAL SCAN – QUADRANTS B+C) ===\n"
      f"Plots generated and saved for Y={target_y_ums} with increased font/line size. No CSV output was performed.")