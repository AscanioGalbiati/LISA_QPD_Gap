'''
@author: A_Galbiati
Horizontal linescans data analysis
Combined power from two 2025-11-03 measurements
+ Responsivity(η = I_sum / P) on right Y-axis
Author: Ascanio Galbiati
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
# <<< ADDED >>> CSV writing
import csv
# <<< END ADDED >>>

# Set up font and LaTeX rendering
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
cmu_serif = fm.FontProperties(fname=font_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

# Base directory
base_dir = "/Users/asca/Documents/University/Master Thesis/code/Data/20251215_final_data/20251027/VIGO_NS089008_QPD_750_20_AS_0011_CC_objLens_quadABCD_manual_setup_GND_LOAD_XY/HorizontalScan"

# === PLOT RANGE (narrow) ===
x_min_plot = 7.145
x_max_plot = 7.405

# === CALCULATION RANGE (wide – for accurate GS) ===
calc_min = 7.10 #FROM CONFIG FILE
calc_max = 7.45
left_region = (7.10, 7.20)
right_region = (7.35, 7.45)
gap_region = (7.20, 7.35)

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

# <<< FINAL CSV SETUP >>>
# Initialize a list to hold all rows before writing to ensure atomicity
csv_dir = "/Users/asca/Documents/University/Master Thesis/code/LISA_QPD_Gap/Thesis_Plots&Codes/FINAL/AS_011_CC/statistical study"
os.makedirs(csv_dir, exist_ok=True)
csv_path = os.path.join(csv_dir, "Overshoot_Results_AD_Y_up_to_2100um.csv")

# 1. DEFINE NEW FINAL OUTPUT DIRECTORY
FINAL_FIG_DIR = '/Users/asca/Documents/University/Master Thesis/code/LISA_QPD_Gap/Thesis_Plots&Codes/FINAL/AS_011_CC/20251027/fig'
os.makedirs(FINAL_FIG_DIR, exist_ok=True) # Ensure it exists


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

# --- CSV Data Collection List ---
csv_data_rows = []
# --------------------------------

# Process each Y folder (ONLY UP TO 2100 µm)
for y_um, y_folder in y_positions:
    if y_um > 2200:
        continue

    # 2. DELETED fig_dir CREATION
    # fig_dir = os.path.join(y_folder, "fig_AS_011_CC")
    # os.makedirs(fig_dir, exist_ok=True)

    load_pkl_files = glob.glob(os.path.join(y_folder, "*_load2.pkl"))
    gnd_pkl_files = glob.glob(os.path.join(y_folder, "*_gnd.pkl"))

    if not load_pkl_files or not gnd_pkl_files:
        print(f"Missing .pkl files in {y_folder}")
        continue

    load_pkl_file = load_pkl_files[0]
    gnd_pkl_file = gnd_pkl_files[0]
    
    # Try-except block to safely load files and handle potential errors
    try:
        data_load = load_data(load_pkl_file)
        data_gnd = load_data(gnd_pkl_file)
    except Exception as e:
        print(f"Error loading pickle data for {y_folder}: {e}")
        continue
    
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

    # Check for missing data
    if quadA_mean_load.size == 0 or np.all(quadA_mean_load == 0):
        print(f"Warning: quadA missing/zero in LOAD {load_pkl_file}")
    if quadD_mean_load.size == 0 or np.all(quadD_mean_load == 0):
        print(f"Warning: quadD missing/zero in LOAD {load_pkl_file}")

    I_sum_load_plot = (quadA_mean_load + quadD_mean_load)[mask_plot]
    I_sum_load_calc = (quadA_mean_load + quadD_mean_load)[mask_calc]

    # <<< GAP SENSITIVITY (wide range) >>>
    left_mask = (x_calc >= left_region[0]) & (x_calc <= left_region[1])
    right_mask = (x_calc >= right_region[0]) & (x_calc <= right_region[1])
    gap_mask = (x_calc >= gap_region[0]) & (x_calc <= gap_region[1])
    
    # Calculate means and max
    mean_left_sum_load = np.mean(I_sum_load_calc[left_mask]) if np.any(left_mask) else 0
    mean_right_sum_load = np.mean(I_sum_load_calc[right_mask]) if np.any(right_mask) else 0
    mean_plateaus_load = (mean_left_sum_load + mean_right_sum_load) / 2.0
    max_in_gap_load = np.max(I_sum_load_calc[gap_mask]) if np.any(gap_mask) else 0
    
    # Avoid division by zero
    relative_percent_increase_load = (max_in_gap_load / mean_plateaus_load - 1.0) * 100.0 if mean_plateaus_load > 0 else 0
    
    # Find x_at_max for plotting
    idx_max_gap_load = np.argmax(I_sum_load_calc[gap_mask]) if np.any(gap_mask) else -1
    x_at_max_load = x_calc[gap_mask][idx_max_gap_load] if np.any(gap_mask) and idx_max_gap_load != -1 else x_array_filtered[len(x_array_filtered)//2]


    # PLOTTING LOAD
    fig_load, ax_load = plt.subplots(figsize=(10, 6), layout='constrained')
    # Increased line width: 2.1
    ax_load.plot(x_array_filtered, quadA_mean_load[mask_plot], label='quadA', color='tab:blue', linewidth=2.1)
    ax_load.plot(x_array_filtered, quadD_mean_load[mask_plot], label='quadD', color='tab:red', linewidth=2.1)
    ax_load.plot(x_array_filtered, I_sum_load_plot, '--', color="#992f7f", linewidth=2.1, label='quadA+quadD', alpha=0.6)
    
    # Colored regions
    ax_load.axvspan(7.10, 7.20, color='tab:blue', alpha=0.12, label='Left region')
    ax_load.axvspan(7.35, 7.45, color='tab:red', alpha=0.12, label='Right region')
    ax_load.axvspan(7.20, 7.35, color='gray', alpha=0.22, label='Gap region')
    
    # Horizontal lines: linewidth=1.8 (kept original)
    ax_load.axhline(I_100, color="#c39d7a", linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 1\,$A/W')
    ax_load.axhline(I_real_mean, color='#8b5a2b', linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 0.8\,$A/W')
    
    # QUANTUM EFFICIENCY LOAD
    QE_load = I_sum_load_plot / P_combined
    ax_qe_load = ax_load.twinx()
    # Increased line width: 2.1
    ax_qe_load.plot(x_array_filtered, QE_load, color="#00ff88", lw=2.1, ls='-', alpha=0.15, label=r'$\eta_{\rm{LOAD}}$ [A/W]')
    # Increased font size: 16
    ax_qe_load.set_ylabel(r'Responsivity $\eta$ [A/W]', color='#8b5a2b', fontsize=16) # ax_qe_load.set_ylabel(r'Quantum Efficiency $\eta$ [A/W]', color='#8b5a2b', fontsize=16)
    # Increased tick label size: 13
    ax_qe_load.tick_params(axis='y', which='major', color='#8b5a2b', length=8, width=1.5, direction='in', labelsize=13)
    ax_qe_load.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
    for tick in ax_qe_load.get_yticklabels():
        tick.set_color('#8b5a2b')
    ax_load.set_ylim(bottom=0)
    ax_qe_load.set_ylim(bottom=0)
    ax_qe_load.set_ylim(top=1.05)
    ax_qe_load.yaxis.set_major_formatter(y_fmt_qe)
    align_yaxis_zeros(ax_load, ax_qe_load)
    
    # Marker size: ms=10, mew=1.8 (kept original)
    ax_load.plot(x_at_max_load, max_in_gap_load, 'o', color='#ff7f0e', mec='#cc5e00', mew=1.8, ms=10, zorder=10)
    
    # RESULT BOX – GS (LOAD)
    # Adjusted inset position: [0.32, 0.78, 0.38, 0.16]
    inset = ax_load.inset_axes([0.32, 0.54, 0.38, 0.16], transform=ax_load.transAxes)
    inset.axis('off')
    # Increased line width: 2.2
    red_dot_load = Line2D([0], [0],
                          color=(*plt.cm.colors.to_rgb('#992f7f'), 0.6),
                          lw=2.2, ls='--',
                          marker='o',
                          markerfacecolor='#ff7f0e',
                          markeredgecolor='#cc5e00',
                          markeredgewidth=1.8,
                          markersize=10)
    # Increased font size: 22
    inset.legend(handles=[red_dot_load],
                 labels=[rf'GS (LOAD): \textbf{{{relative_percent_increase_load:+.2f}\%}}'],
                 loc='center', fontsize=22, frameon=False,
                 handletextpad=0.8, handlelength=1.8)

    # COMBINE LEGENDS
    lines1, labels1 = ax_load.get_legend_handles_labels()
    lines2, labels2 = ax_qe_load.get_legend_handles_labels()
    # Increased font size: 14
    legend = ax_load.legend(lines1 + lines2, labels1 + labels2,
                           fontsize=14, loc='upper right',
                           bbox_to_anchor=(0.98, 0.98),
                           frameon=True, fancybox=False, edgecolor='black')

    # Increased font size: 16
    ax_load.set_title(rf'\textbf{{DC Photocurrent LOAD (HL: Y={y_um}\textmu{{}}m)}}', fontsize=16, fontweight='bold', pad=10)
    ax_load.set_xlabel(r'X Position [mm]', fontsize=16)
    ax_load.set_ylabel(r'Photocurrent [A]', fontsize=16)
    ax_load.grid(True, linestyle='--', alpha=0.6)
    # Increased tick label size: 13
    ax_load.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in')
    ax_load.yaxis.set_major_formatter(y_fmt)
    ax_load.set_xlim(x_min_plot+0.015, x_max_plot-0.015)

    # 3. UPDATED SAVE PATH FOR LOAD FIGURE
    fig_name_load = f"DC_Photocurrent_Y{y_um:04d}um_Quadrants_AD_thresholdX_RestricedRange_LOAD_results_inv"
    fig_load.savefig(os.path.join(FINAL_FIG_DIR, f"{fig_name_load}.png"), dpi=300, bbox_inches='tight')
    print(f"LOAD saved: {os.path.join(FINAL_FIG_DIR, f'{fig_name_load}.png')}")
    plt.show()
    plt.close(fig_load)

    # =============================================================
    # GND PLOT & CALCULATIONS
    # =============================================================
    quadA_mean_gnd = data_gnd['rawdata'].get('quadA', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)
    quadD_mean_gnd = data_gnd['rawdata'].get('quadD', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)

    # Check for missing data
    if quadA_mean_gnd.size == 0 or np.all(quadA_mean_gnd == 0):
        print(f"Warning: quadA missing/zero in GND {gnd_pkl_file}")
    if quadD_mean_gnd.size == 0 or np.all(quadD_mean_gnd == 0):
        print(f"Warning: quadD missing/zero in GND {gnd_pkl_file}")

    I_sum_gnd_plot = (quadA_mean_gnd + quadD_mean_gnd)[mask_plot]
    I_sum_gnd_calc = (quadA_mean_gnd + quadD_mean_gnd)[mask_calc]

    # GAP SENSITIVITY GND (wide range)
    mean_left_sum_gnd = np.mean(I_sum_gnd_calc[left_mask]) if np.any(left_mask) else 0
    mean_right_sum_gnd = np.mean(I_sum_gnd_calc[right_mask]) if np.any(right_mask) else 0
    mean_plateaus_gnd = (mean_left_sum_gnd + mean_right_sum_gnd) / 2.0
    max_in_gap_gnd = np.max(I_sum_gnd_calc[gap_mask]) if np.any(gap_mask) else 0
    
    # Avoid division by zero
    relative_percent_increase_gnd = (max_in_gap_gnd / mean_plateaus_gnd - 1.0) * 100.0 if mean_plateaus_gnd > 0 else 0
    
    # Find x_at_max for plotting
    idx_max_gap_gnd = np.argmax(I_sum_gnd_calc[gap_mask]) if np.any(gap_mask) else -1
    x_at_max_gnd = x_calc[gap_mask][idx_max_gap_gnd] if np.any(gap_mask) and idx_max_gap_gnd != -1 else x_array_filtered[len(x_array_filtered)//2]


    # PLOTTING GND
    fig_gnd, ax_gnd = plt.subplots(figsize=(10, 6), layout='constrained')
    # Increased line width: 2.1
    ax_gnd.plot(x_array_filtered, quadA_mean_gnd[mask_plot], label='quadA', color='tab:blue', linewidth=2.1)
    ax_gnd.plot(x_array_filtered, quadD_mean_gnd[mask_plot], label='quadD', color='tab:red', linewidth=2.1)
    ax_gnd.plot(x_array_filtered, I_sum_gnd_plot, '--', color="#992f7f", linewidth=2.1, alpha=1.0, label='quadA+quadD')

    # Colored regions (now in legend)
    ax_gnd.axvspan(7.10, 7.20, color='tab:blue', alpha=0.12, label='Left region')
    ax_gnd.axvspan(7.35, 7.45, color='tab:red', alpha=0.12, label='Right region')
    ax_gnd.axvspan(7.20, 7.35, color='gray', alpha=0.22, label='Gap region')
    
    # Horizontal lines: linewidth=1.8 (kept original)
    ax_gnd.axhline(I_100, color='#d2b48c', linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 1\,$A/W')
    ax_gnd.axhline(I_real_mean, color='#8b5a2b', linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 0.8\,$A/W')
    
    # QUANTUM EFFICIENCY GND
    QE_gnd = I_sum_gnd_plot / P_combined
    ax_qe_gnd = ax_gnd.twinx()
    # Increased line width: 2.1
    ax_qe_gnd.plot(x_array_filtered, QE_gnd, color="#44fd00", lw=2.1, alpha=0.15, label=r'$\eta_{\rm{GND}}$ [A/W]')
    # Increased font size: 16
    ax_qe_gnd.set_ylabel(r'Responsivity $\eta$ [A/W]', color='#8b5a2b', fontsize=16)
    # Increased tick label size: 13
    ax_qe_gnd.tick_params(axis='y', which='major', color='#8b5a2b', length=8, width=1.5, direction='in', labelsize=13)
    ax_qe_gnd.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
    for tick in ax_qe_gnd.get_yticklabels():
        tick.set_color('#8b5a2b')
    ax_gnd.set_ylim(bottom=0)
    ax_qe_gnd.set_ylim(bottom=0)
    ax_qe_gnd.set_ylim(top=1.05)
    ax_qe_gnd.yaxis.set_major_formatter(y_fmt_qe)
    align_yaxis_zeros(ax_gnd, ax_qe_gnd)

    # Marker size: ms=10, mew=1.8 (kept original)
    ax_gnd.plot(x_at_max_gnd, max_in_gap_gnd, 'o', color='#d62728', mec='darkred', mew=1.8, ms=10, zorder=10)

    # RESULT BOX – GS (GND)
    # Adjusted inset position: [0.32, 0.78, 0.38, 0.16]
    inset = ax_gnd.inset_axes([0.32, 0.54, 0.38, 0.16], transform=ax_gnd.transAxes)
    inset.axis('off')
    # Increased line width: 2.2
    red_dot_gnd = Line2D([0], [0],
                         color='#992f7f', lw=2.2, ls='--',
                         marker='o', markerfacecolor='#d62728',
                         markeredgecolor='darkred', markeredgewidth=1.8,
                         markersize=10)
    # Increased font size: 22
    inset.legend(handles=[red_dot_gnd],
                 labels=[rf'GS (GND): \textbf{{{relative_percent_increase_gnd:+.2f}\%}}'],
                 loc='center', fontsize=22, frameon=False,
                 handletextpad=0.8, handlelength=1.8)

    # <<< CSV ROW PREPARATION (Before plotting ends, data is fresh) >>>
    csv_row = [
        y_um,
        f"{relative_percent_increase_load:+.3f}",
        f"{relative_percent_increase_gnd:+.3f}",
        f"{max_in_gap_load*1e6:.3f}",
        f"{max_in_gap_gnd*1e6:.3f}",
        f"{mean_plateaus_load*1e6:.3f}",
        f"{mean_plateaus_gnd*1e6:.3f}"
    ]
    csv_data_rows.append(csv_row)
    
    print(f"\n=== GAP SENSITIVITY FOR Y = {y_um} µm ===")
    print(f"→ Overshoot LOAD : {relative_percent_increase_load:+.2f} %")
    print(f"→ Overshoot GND : {relative_percent_increase_gnd:+.2f} %")
    
    # === LEGEND GND ===
    lines1g, labels1g = ax_gnd.get_legend_handles_labels()
    lines2g, labels2g = ax_qe_gnd.get_legend_handles_labels()
    # Increased font size: 14
    legend_gnd = ax_gnd.legend(
        lines1g + lines2g, labels1g + labels2g,
        fontsize=14, loc='upper right',
        bbox_to_anchor=(0.98, 0.98),
        frameon=True, fancybox=False, edgecolor='black'
    )

    # Increased font size: 16
    ax_gnd.set_title(rf'\textbf{{DC Photocurrent GND (HL: Y={y_um}\textmu{{}}m)}}', fontsize=16, fontweight='bold', pad=10)
    ax_gnd.set_xlabel(r'X Position [mm]', fontsize=16)
    ax_gnd.set_ylabel(r'Photocurrent [A]', fontsize=16)
    ax_gnd.grid(True, linestyle='--', alpha=0.6)
    # Increased tick label size: 13
    ax_gnd.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in')
    ax_gnd.yaxis.set_major_formatter(y_fmt)
    ax_gnd.set_ylim(bottom=0)
    ax_gnd.set_xlim(x_min_plot+0.015, x_max_plot-0.015)

    # 3. UPDATED SAVE PATH FOR GND FIGURE
    fig_name_gnd = f"DC_Photocurrent_Y{y_um:04d}um_Quadrants_AD_thresholdX_RestricedRange_GND_results_inv"
    fig_gnd.savefig(os.path.join(FINAL_FIG_DIR, f"{fig_name_gnd}.png"), dpi=300, bbox_inches='tight')
    print(f"GND saved: {os.path.join(FINAL_FIG_DIR, f'{fig_name_gnd}.png')}")
    plt.show()
    plt.close(fig_gnd)

# <<< FINAL CSV WRITE (Guarantees all data is written once at the end) >>>
# This replaces your original file open/close logic and should fix the "broken" file issue.
header = ['Y_position_um', 'Overshoot_LOAD_%', 'Overshoot_GND_%',
          'Max_in_gap_LOAD_µA', 'Max_in_gap_GND_µA',
          'Mean_plateau_LOAD_µA', 'Mean_plateau_GND_µA']

# Use 'with open' to ensure the file is closed correctly even if an error occurs
with open(csv_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)
    csv_writer.writerows(csv_data_rows)
    
print(f"\n=== \nOvershoot results successfully saved to:\n{csv_path}")