'''
FINAL!!!!
Created on Nov 22 2025
@author: A_Galbiati
Horizontal linescans data analysis
Combined power from two 2025-11-03 measurements
+ Quantum Efficiency (η = I_sum / P) on right Y-axis
*** IMPLEMENTED BIGGER FONTS AND THICKER LINES (MATCHING REFERENCE) ***
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
# <<< REMOVED CSV IMPORTS >>>

# Set up font and LaTeX rendering
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
cmu_serif = fm.FontProperties(fname=font_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

# Base directory
# === CORRECTED BASE DIR ===
base_dir = '/Users/asca/Documents/University/Master Thesis/code/Data/all quadrants/20251022/VIGO_NS089008_QPD_750_20_AS_0011_CC_251022_objLens_quadABCD_manual_setup_Z7.91mm_Yscan_thresholdX_YScan_FineSteps'

# === PLOT RANGE (narrow) ===
x_min_plot = 7.15 
x_max_plot = 7.45

# === CALCULATION RANGE (wide – for accurate GS) ===
calc_min = 7.10 #FROM CONFIG FILE
calc_max = 7.45
left_region = (7.10, 7.25)
right_region = (7.35, 7.45)
gap_region = (7.25, 7.35)

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
rel_uncertainty_percent_3sigma = (sigma_combined_3sigma / P_combined)

# QE references
eta_100 = 1.0
eta_real = 0.8
I_100 = eta_100 * P_combined
I_real_mean = eta_real * P_combined
I_real_err_3sigma = eta_real * sigma_combined_3sigma
sigma_eta_3sigma = eta_real * sigma_combined_3sigma / P_combined

# === ALIGN ZEROS FUNCTION ===
def align_yaxis_zeros(ax1, ax2):
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

# Process each Y folder (ONLY UP TO 2200 µm)
for y_um, y_folder in y_positions:
    if y_um > 2200:
        continue

    fig_dir = os.path.join(y_folder, "fig_AS_011_CC_BiggerFonts") # NEW fig_dir name
    os.makedirs(fig_dir, exist_ok=True)

    # === FIX: FIND SINGLE RAW DATA FILE ===
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

    # Plot mask (narrow)
    mask_plot = (x_array >= x_min_plot) & (x_array <= x_max_plot)
    x_array_filtered = x_array[mask_plot]

    # Calc mask (wide)
    mask_calc = (x_array >= calc_min) & (x_array <= calc_max)
    x_calc = x_array[mask_calc]

    if len(x_array_filtered) == 0:
        print(f"No data in X range for {y_folder}")
        continue

    # ------------------------------------------------------------------
    # Custom tick formatter: 0 → "0" , everything else → 4 decimal
    # ------------------------------------------------------------------
    def y_formatter(val, pos):
        return '0' if val == 0 else f'{val:.4f}'
    y_fmt = FuncFormatter(y_formatter)

    def y_formatter_qe(val, pos):
        return '0' if val == 0 else f'{val:.1f}'
    y_fmt_qe = FuncFormatter(y_formatter_qe)

    # =============================================================
    # LOAD PLOT & CALCULATIONS
    # =============================================================
    
    quadA_mean_load = data_load['rawdata'].get('quadA', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)
    quadD_mean_load = data_load['rawdata'].get('quadD', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)

    I_sum_load_plot = (quadA_mean_load + quadD_mean_load)[mask_plot]
    I_sum_load_calc = (quadA_mean_load + quadD_mean_load)[mask_calc]

    # <<< GAP SENSITIVITY (wide range) >>>
    left_mask = (x_calc >= left_region[0]) & (x_calc <= left_region[1])
    right_mask = (x_calc >= right_region[0]) & (x_calc <= right_region[1])
    gap_mask = (x_calc >= gap_region[0]) & (x_calc <= gap_region[1])
    
    mean_left_sum_load = np.mean(I_sum_load_calc[left_mask]) if np.any(left_mask) else 0
    mean_right_sum_load = np.mean(I_sum_load_calc[right_mask]) if np.any(right_mask) else 0
    mean_plateaus_load = (mean_left_sum_load + mean_right_sum_load) / 2.0
    max_in_gap_load = np.max(I_sum_load_calc[gap_mask]) if np.any(gap_mask) else 0
    
    relative_percent_increase_load = (max_in_gap_load / mean_plateaus_load - 1.0) * 100.0 if mean_plateaus_load > 0 else 0
    
    idx_max_gap_load = np.argmax(I_sum_load_calc[gap_mask]) if np.any(gap_mask) else -1
    x_at_max_load = x_calc[gap_mask][idx_max_gap_load] if np.any(gap_mask) and idx_max_gap_load != -1 else x_array_filtered[len(x_array_filtered)//2]


    # PLOTTING LOAD
    fig_load, ax_load = plt.subplots(figsize=(10, 6), layout='constrained')
    
    # Line width adjustment: 2.1
    ax_load.plot(x_array_filtered, quadA_mean_load[mask_plot], label='quadA', color='tab:blue', linewidth=2.1)
    ax_load.plot(x_array_filtered, quadD_mean_load[mask_plot], label='quadD', color='tab:red', linewidth=2.1)
    ax_load.plot(x_array_filtered, I_sum_load_plot, '--', color="#992f7f", linewidth=2.1, label='quadA+quadD', alpha=0.6)
    
    # Colored regions
    ax_load.axvspan(7.10, 7.25, color='tab:blue', alpha=0.12, label='Left region')
    ax_load.axvspan(7.35, 7.45, color='tab:red', alpha=0.12, label='Right region')
    ax_load.axvspan(7.25, 7.35, color='gray', alpha=0.22, label='Gap region')
    
    # Horizontal lines: linewidth=1.8
    ax_load.axhline(I_100, color="#c39d7a", linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 1\,$A/W')
    ax_load.axhline(I_real_mean, color='#8b5a2b', linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 0.8\,$A/W')
    
    # QUANTUM EFFICIENCY LOAD
    QE_load = I_sum_load_plot / P_combined
    ax_qe_load = ax_load.twinx()
    # Line width adjustment: 2.1
    ax_qe_load.plot(x_array_filtered, QE_load, color="#00ff88", lw=2.1, ls='-', alpha=0.15, label=r'$\eta_{\rm{mes}}$ [A/W]')
    
    # Font size adjustment: 16
    ax_qe_load.set_ylabel(r'Quantum Efficiency $\eta$ [A/W]', color='#8b5a2b', fontsize=16)
    
    # Font size adjustment: 13
    ax_qe_load.tick_params(axis='y', which='major', color='#8b5a2b', length=8, width=1.5, direction='in', labelsize=13)
    for tick in ax_qe_load.get_yticklabels():
        tick.set_color('#8b5a2b')
    ax_load.set_ylim(bottom=0)
    ax_qe_load.set_ylim(bottom=0)
    ax_qe_load.set_ylim(top=1.05)
    ax_qe_load.yaxis.set_major_formatter(y_fmt_qe)
    align_yaxis_zeros(ax_load, ax_qe_load)
    
    # Marker: ms=10, mew=1.8
    ax_load.plot(x_at_max_load, max_in_gap_load, 'o', color='#ff7f0e', mec='#cc5e00', mew=1.8, ms=10, zorder=10)
    
    # RESULT BOX – GS (LOAD)
    # Inset position adjustment: [0.32, 0.78, 0.38, 0.16]
    inset = ax_load.inset_axes([0.32, 0.78, 0.38, 0.16], transform=ax_load.transAxes)
    inset.axis('off')
    # Marker and line adjustments: lw=2.2, mew=1.8, ms=10, handlelength=1.8
    red_dot_load = Line2D([0], [0],
                          color=(*plt.cm.colors.to_rgb('#992f7f'), 0.6),
                          lw=2.2, ls='--',
                          marker='o',
                          markerfacecolor='#ff7f0e',
                          markeredgecolor='#cc5e00',
                          markeredgewidth=1.8,
                          markersize=10)
    # Font size adjustment: 22
    inset.legend(handles=[red_dot_load],
                 labels=[rf'GS: \textbf{{{relative_percent_increase_load:+.2f}\%}}'],
                 loc='center', fontsize=22, frameon=False,
                 handletextpad=0.8, handlelength=1.8)

    # COMBINE LEGENDS
    lines1, labels1 = ax_load.get_legend_handles_labels()
    lines2, labels2 = ax_qe_load.get_legend_handles_labels()
    # Font size adjustment: 14
    legend = ax_load.legend(lines1 + lines2, labels1 + labels2,
                           fontsize=14, loc='upper right',
                           bbox_to_anchor=(0.98, 0.98),
                           frameon=True, fancybox=False, edgecolor='black')

    # Title/Label font size adjustment: 16
    ax_load.set_title(rf'\textbf{{DC Photocurrent (HL: Y={y_um}\textmu{{}}m)}}', fontsize=16, fontweight='bold', pad=10)
    ax_load.set_xlabel(r'X Position [mm]', fontsize=16)
    ax_load.set_ylabel(r'Photocurrent [A]', fontsize=16)
    ax_load.grid(True, linestyle='--', alpha=0.6)
    
    # Axis tick font size adjustment: 13
    ax_load.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in')
    ax_load.yaxis.set_major_formatter(y_fmt)
    ax_load.set_xlim(x_min_plot+0.015, x_max_plot-0.015)

    fig_name_load = f"DC_Photocurrent_Y{y_um:04d}um_Quadrants_AD_thresholdX_RestricedRange_LOAD_results_inv"
    fig_load.savefig(os.path.join(fig_dir, f"{fig_name_load}.png"), dpi=300, bbox_inches='tight')
    print(f"LOAD saved: {os.path.join(fig_dir, f'{fig_name_load}.png')}")
    plt.close(fig_load)

    # =============================================================
    # GND PLOT & CALCULATIONS
    # =============================================================
    quadA_mean_gnd = data_gnd['rawdata'].get('quadA', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)
    quadD_mean_gnd = data_gnd['rawdata'].get('quadD', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)

    I_sum_gnd_plot = (quadA_mean_gnd + quadD_mean_gnd)[mask_plot]
    I_sum_gnd_calc = (quadA_mean_gnd + quadD_mean_gnd)[mask_calc]

    # GAP SENSITIVITY GND (wide range)
    mean_left_sum_gnd = np.mean(I_sum_gnd_calc[left_mask]) if np.any(left_mask) else 0
    mean_right_sum_gnd = np.mean(I_sum_gnd_calc[right_mask]) if np.any(right_mask) else 0
    mean_plateaus_gnd = (mean_left_sum_gnd + mean_right_sum_gnd) / 2.0
    max_in_gap_gnd = np.max(I_sum_gnd_calc[gap_mask]) if np.any(gap_mask) else 0
    
    relative_percent_increase_gnd = (max_in_gap_gnd / mean_plateaus_gnd - 1.0) * 100.0 if mean_plateaus_gnd > 0 else 0
    
    idx_max_gap_gnd = np.argmax(I_sum_gnd_calc[gap_mask]) if np.any(gap_mask) else -1
    x_at_max_gnd = x_calc[gap_mask][idx_max_gap_gnd] if np.any(gap_mask) and idx_max_gap_gnd != -1 else x_array_filtered[len(x_array_filtered)//2]


    # PLOTTING GND
    fig_gnd, ax_gnd = plt.subplots(figsize=(10, 6), layout='constrained')
    # Line width adjustment: 2.1
    ax_gnd.plot(x_array_filtered, quadA_mean_gnd[mask_plot], label='quadA', color='tab:blue', linewidth=2.1)
    ax_gnd.plot(x_array_filtered, quadD_mean_gnd[mask_plot], label='quadD', color='tab:red', linewidth=2.1)
    ax_gnd.plot(x_array_filtered, I_sum_gnd_plot, '--', color="#992f7f", linewidth=2.1, alpha=1.0, label='quadA+quadD')

    # Colored regions
    ax_gnd.axvspan(7.10, 7.25, color='tab:blue', alpha=0.12, label='Left region')
    ax_gnd.axvspan(7.35, 7.45, color='tab:red', alpha=0.12, label='Right region')
    ax_gnd.axvspan(7.25, 7.35, color='gray', alpha=0.22, label='Gap region')
    
    # Horizontal lines: linewidth=1.8
    ax_gnd.axhline(I_100, color='#d2b48c', linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 1\,$A/W')
    ax_gnd.axhline(I_real_mean, color='#8b5a2b', linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 0.8\,$A/W')
    
    # QUANTUM EFFICIENCY GND
    QE_gnd = I_sum_gnd_plot / P_combined
    ax_qe_gnd = ax_gnd.twinx()
    # Line width adjustment: 2.1
    ax_qe_gnd.plot(x_array_filtered, QE_gnd, color="#44fd00", lw=2.1, alpha=0.15, label=r'$\eta_{\rm{mes}}$ [A/W]')
    
    # Font size adjustment: 16
    ax_qe_gnd.set_ylabel(r'Quantum Efficiency $\eta$ [A/W]', color='#8b5a2b', fontsize=16)
    
    # Font size adjustment: 13
    ax_qe_gnd.tick_params(axis='y', which='major', color='#8b5a2b', length=8, width=1.5, direction='in', labelsize=13)
    for tick in ax_qe_gnd.get_yticklabels():
        tick.set_color('#8b5a2b')
    ax_gnd.set_ylim(bottom=0)
    ax_qe_gnd.set_ylim(bottom=0)
    ax_qe_gnd.set_ylim(top=1.05)
    ax_qe_gnd.yaxis.set_major_formatter(y_fmt_qe)
    align_yaxis_zeros(ax_gnd, ax_qe_gnd)

    # Marker: ms=10, mew=1.8
    ax_gnd.plot(x_at_max_gnd, max_in_gap_gnd, 'o', color='#d62728', mec='darkred', mew=1.8, ms=10, zorder=10)

    # RESULT BOX – GS (GND)
    # Inset position adjustment: [0.32, 0.78, 0.38, 0.16]
    inset = ax_gnd.inset_axes([0.32, 0.78, 0.38, 0.16], transform=ax_gnd.transAxes)
    inset.axis('off')
    # Marker and line adjustments: lw=2.2, mew=1.8, ms=10, handlelength=1.8
    red_dot_gnd = Line2D([0], [0],
                         color='#992f7f', lw=2.2, ls='--',
                         marker='o', markerfacecolor='#d62728',
                         markeredgecolor='darkred', markeredgewidth=1.8,
                         markersize=10)
    # Font size adjustment: 22
    inset.legend(handles=[red_dot_gnd],
                 labels=[rf'GS: \textbf{{{relative_percent_increase_gnd:+.2f}\%}}'],
                 loc='center', fontsize=22, frameon=False,
                 handletextpad=0.8, handlelength=1.8)
    
    print(f"\n=== GAP SENSITIVITY FOR Y = {y_um} µm ===")
    print(f"→ Overshoot LOAD : {relative_percent_increase_load:+.2f} %")
    print(f"→ Overshoot GND : {relative_percent_increase_gnd:+.2f} %")
    
    # === LEGEND GND ===
    lines1g, labels1g = ax_gnd.get_legend_handles_labels()
    lines2g, labels2g = ax_qe_gnd.get_legend_handles_labels()
    # Font size adjustment: 14
    legend_gnd = ax_gnd.legend(
        lines1g + lines2g, labels1g + labels2g,
        fontsize=14, loc='upper right',
        bbox_to_anchor=(0.98, 0.98),
        frameon=True, fancybox=False, edgecolor='black'
    )

    # Title/Label font size adjustment: 16
    ax_gnd.set_title(rf'\textbf{{DC Photocurrent (HL: Y={y_um}\textmu{{}}m)}}', fontsize=16, fontweight='bold', pad=10)
    ax_gnd.set_xlabel(r'X Position [mm]', fontsize=16)
    ax_gnd.set_ylabel(r'Photocurrent [A]', fontsize=16)
    ax_gnd.grid(True, linestyle='--', alpha=0.6)
    
    # Axis tick font size adjustment: 13
    ax_gnd.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in')
    ax_gnd.yaxis.set_major_formatter(y_fmt)
    ax_gnd.set_ylim(bottom=0)
    ax_gnd.set_xlim(x_min_plot+0.015, x_max_plot-0.015)

    fig_name_gnd = f"DC_Photocurrent_Y{y_um:04d}um_Quadrants_AD_thresholdX_RestricedRange_GND_results_inv"
    fig_gnd.savefig(os.path.join(fig_dir, f"{fig_name_gnd}.png"), dpi=300, bbox_inches='tight')
    print(f"GND saved: {os.path.join(fig_dir, f'{fig_name_gnd}.png')}")
    plt.close(fig_gnd)

print("\n=== ALL DONE ===\nPlots successfully generated and saved. No CSV output was performed.")