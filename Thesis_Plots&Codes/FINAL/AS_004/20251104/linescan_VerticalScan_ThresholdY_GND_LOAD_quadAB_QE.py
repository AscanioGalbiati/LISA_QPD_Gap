'''
@author: A_Galbiati
Vertical data analysis
Combined power from two measurements
+ Responsivity (η = I_sum / P) on right Y-axis
Compatible data: VerticalScan folders (X*um)
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
import csv

# Set up font and LaTeX rendering
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
cmu_serif = fm.FontProperties(fname=font_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

# Base directory
base_dir = "/Users/asca/Documents/University/Master Thesis/code/Data/20251215_final_data/20251104/VIGO_NS089008_QPD_750_20_AS_0011_CC_objLens_quadABCD_manual_setup_GND_LOAD_251103_Z7.38/VerticalScan"

# =======================================================
# === DYNAMIC CONFIGURATION DEFINITIONS FOR X POSITIONS ===
# Define default calculation range (used as base for all)
CALC_MIN_BASE = 1.75
CALC_MAX_BASE = 2.75

X_POS_CONFIG = {
    6650: {
        # Config for the first processed file (Index 0)
        'y_min_plot': 2.2,
        'y_max_plot': 2.7,
        'calc_min': CALC_MIN_BASE,
        'calc_max': CALC_MAX_BASE,
        'left': (CALC_MIN_BASE, 2.33),   # left plateau
        'right': (2.57, CALC_MAX_BASE),   # right plateau
        'gap': (2.33, 2.57)   # gap
    },
    6900: {
        # Config for the second processed file (Index 1)
        'y_min_plot': 2.2,
        'y_max_plot': 2.7,
        'calc_min': CALC_MIN_BASE,
        'calc_max':CALC_MAX_BASE, # Slightly tighter calc_max
        'left': (CALC_MIN_BASE, 2.34), 
        'right': (2.57, CALC_MAX_BASE), # Uses the updated calc_max
        'gap': (2.34, 2.57) 
    },
    7150: {
        # Config for the third processed file (Index 2)
        # Using the same plot and calculation ranges as 6650 for simplicity, 
        # but now explicitly defined for this X position.
        'y_min_plot': 2.2,
        'y_max_plot': 2.75,
        'calc_min': CALC_MIN_BASE,
        'calc_max': CALC_MAX_BASE,
        'left': (CALC_MIN_BASE, 2.35),
        'right': (2.60, CALC_MAX_BASE),
        'gap': (2.35, 2.60)
    }
}
# === END DYNAMIC CONFIGURATION ===
# =================================

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

# <<< UNIQUE CSV FOR VERTICAL SCAN (A+B, X = 6650, 6900, 7150 µm) >>>
csv_dir = "/Users/asca/Documents/University/Master Thesis/code/LISA_QPD_Gap/Thesis_Plots&Codes/FINAL/AS_004/20251104/statistical study"
os.makedirs(csv_dir, exist_ok=True)
csv_path = os.path.join(csv_dir, "Overshoot_Results_AB_X_from6650um.csv")

# Data will be collected into this list:
csv_data_rows = []
# ----------------------------------------------

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

def extract_x_position(folder_name):
    match = re.search(r'X(\d+)um', folder_name)
    return int(match.group(1)) if match else None

# Find X folders
x_folders = glob.glob(os.path.join(base_dir, "X*um"))
x_positions = [(extract_x_position(os.path.basename(f)), f) for f in x_folders if extract_x_position(os.path.basename(f)) is not None]
x_positions.sort()

# Initialize counter for dynamic region selection
processed_file_index = 0

# === LIST OF TARGET X POSITIONS (Ensures processing order) ===
target_x_ums = [6650, 6900, 7150]

# Process each X folder
for x_um, x_folder in x_positions:
    
    # === FILTER TO ONLY PROCESS TARGET POSITIONS ===
    if x_um not in target_x_ums:
        continue
        
    # === DYNAMIC CONFIGURATION SELECTION ===
    current_config = X_POS_CONFIG.get(x_um)
    if not current_config:
        print(f"Error: Missing configuration for X={x_um} µm. Skipping.")
        continue
    
    # Apply dynamic plot and calculation ranges
    y_min_plot = current_config['y_min_plot']
    y_max_plot = current_config['y_max_plot']
    calc_min = current_config['calc_min']
    calc_max = current_config['calc_max']
    current_regions = {
        'left': current_config['left'], 
        'right': current_config['right'], 
        'gap': current_config['gap']
    }
    
    processed_file_index += 1
    # =====================================

    # Use AS_004 for figure directory
    fig_dir = os.path.join(x_folder, "fig") 
    os.makedirs(fig_dir, exist_ok=True)

    load_pkl_files = glob.glob(os.path.join(x_folder, "*_load2.pkl"))
    gnd_pkl_files  = glob.glob(os.path.join(x_folder, "*_gnd.pkl"))
    if not load_pkl_files or not gnd_pkl_files:
        print(f"Missing .pkl files in {x_folder}")
        continue
        
    # Safe loading of data
    try:
        load_pkl_file = load_pkl_files[0]
        gnd_pkl_file  = gnd_pkl_files[0]
        data_load = load_data(load_pkl_file)
        data_gnd  = load_data(gnd_pkl_file)
    except Exception as e:
        print(f"Error loading pickle data for {x_folder}: {e}")
        continue

    # Y-scan: The scanned position is Y
    y_array = data_load['rawdata']['stage_laser_yposition']

    # Use dynamic y_min/max_plot
    mask_plot = (y_array >= y_min_plot) & (y_array <= y_max_plot) 
    y_plot = y_array[mask_plot]

    # Use dynamically selected calculation range
    mask_calc = (y_array >= calc_min) & (y_array <= calc_max)
    y_calc = y_array[mask_calc]

    if len(y_plot) == 0 or len(y_calc) == 0:
        print(f"Not enough data in range for {x_folder} with plot range ({y_min_plot}, {y_max_plot})")
        continue

    def curr_fmt(val, pos): return '0' if val == 0 else f'{val:.4f}'
    def qe_fmt(val, pos):   return '0' if val == 0 else f'{val:.1f}'
    y_fmt = FuncFormatter(curr_fmt)
    y_fmt_qe = FuncFormatter(qe_fmt)

    # =============================================================
    # LOAD PLOT (A+B) & CALCULATIONS
    # =============================================================
    fig_load, ax_load = plt.subplots(figsize=(10, 6), layout='constrained')

    # QUADRANTS A+B for Vertical Scan
    quadA_load = data_load['rawdata'].get('quadA', {}).get('dmm00_curr_amp', np.zeros((len(y_array),1))).mean(axis=1)
    quadB_load = data_load['rawdata'].get('quadB', {}).get('dmm00_curr_amp', np.zeros((len(y_array),1))).mean(axis=1)

    I_sum_load_plot = (quadA_load + quadB_load)[mask_plot]
    I_sum_load_calc = (quadA_load + quadB_load)[mask_calc]

    ax_load.plot(y_plot, quadA_load[mask_plot],  linewidth=2.1,  label='quadA', color='tab:blue')
    ax_load.plot(y_plot, quadB_load[mask_plot],  linewidth=2.1,  label='quadB', color='tab:orange')
    ax_load.plot(y_plot, I_sum_load_plot, '--', color="#992f7f", linewidth=2.1, alpha=0.6, label='quadA+quadB') 

    # Use dynamic regions for plotting
    ax_load.axvspan(current_regions['left'][0], current_regions['left'][1], color='tab:blue',   alpha=0.12, label='Left region')
    ax_load.axvspan(current_regions['right'][0], current_regions['right'][1], color='tab:orange', alpha=0.12, label='Right region')
    ax_load.axvspan(current_regions['gap'][0], current_regions['gap'][1], color='gray',       alpha=0.22, label='Gap region')

    ax_load.axhline(I_100,       color="#c39d7a", linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 1\,$A/W')
    ax_load.axhline(I_real_mean, color='#8b5a2b', linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 0.8\,$A/W')

    QE_load = I_sum_load_plot / P_combined
    ax_qe_load = ax_load.twinx()
    ax_qe_load.plot(y_plot, QE_load, color="#00ff88", lw=2.1, alpha=0.15, label=r'$\eta_{\rm{LOAD}}$ [A/W]')

    ax_qe_load.set_ylabel(r'Responsivity $\eta$ [A/W]', color='#8b5a2b', fontsize=16)
    ax_qe_load.tick_params(axis='y', which='major', color='#8b5a2b', length=8, labelsize=13, width=1.5, direction='in')
    ax_qe_load.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
    for tick in ax_qe_load.get_yticklabels():
        tick.set_color('#8b5a2b')
    ax_load.set_ylim(bottom=0)
    ax_qe_load.set_ylim(bottom=0)
    ax_qe_load.set_ylim(top=1.05)
    ax_qe_load.yaxis.set_major_formatter(y_fmt_qe)
    align_yaxis_zeros(ax_load, ax_qe_load)

    # Use dynamic regions for calculation
    lm = (y_calc >= current_regions['left'][0])  & (y_calc <= current_regions['left'][1])
    rm = (y_calc >= current_regions['right'][0]) & (y_calc <= current_regions['right'][1])
    gm = (y_calc >= current_regions['gap'][0])   & (y_calc <= current_regions['gap'][1])

    mean_plateau_load = (np.mean(I_sum_load_calc[lm]) + np.mean(I_sum_load_calc[rm])) / 2.0
    max_gap_load = np.max(I_sum_load_calc[gm]) if np.any(gm) else 0
    y_max_load = y_calc[gm][np.argmax(I_sum_load_calc[gm])] if np.any(gm) else y_plot[len(y_plot)//2]
    gs_load = (max_gap_load / mean_plateau_load - 1.0) * 100.0 if mean_plateau_load > 0 else 0

    ax_load.plot(y_max_load, max_gap_load, 'o', color='#ff7f0e', mec='#cc5e00', mew=1.8, ms=10, zorder=10) 

    # LOAD Inset 
    inset = ax_load.inset_axes([0.32, 0.78, 0.38, 0.16], transform=ax_load.transAxes)
    inset.axis('off')
    marker = Line2D([0], [0], color=(*plt.cm.colors.to_rgb('#992f7f'), 0.6), lw=2.2, ls='--', 
                    marker='o', markerfacecolor='#ff7f0e', markeredgecolor='#cc5e00',
                    markeredgewidth=1.8, markersize=10) 
    inset.legend(handles=[marker], labels=[rf'GS (LOAD): \textbf{{{gs_load:+.2f}\%}}'],
                 loc='center', fontsize=22, frameon=False, handletextpad=0.8, handlelength=1.8) 

    l1, lab1 = ax_load.get_legend_handles_labels()
    l2, lab2 = ax_qe_load.get_legend_handles_labels()
    ax_load.legend(l1 + l2, lab1 + lab2, fontsize=14, loc='upper right',
                   bbox_to_anchor=(0.98, 0.98), frameon=True, fancybox=False, edgecolor='black')

    ax_load.set_title(rf'\textbf{{DC Photocurrent LOAD (VL: X={x_um}\,µm)}}', fontsize=16, fontweight='bold', pad=10)
    ax_load.set_xlabel(r'Y Position [mm]', fontsize=16)
    ax_load.set_ylabel(r'Photocurrent [A]', fontsize=16)
    ax_load.grid(True, linestyle='--', alpha=0.6)
    ax_load.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in')
    ax_load.yaxis.set_major_formatter(y_fmt)
    # The processed_file_index is used to mimic the original conditional logic for xlim.
    # Note: If X=7150 is the *last* processed file, this logic will apply.
    ax_load.set_xlim(y_min_plot + (0.02 if processed_file_index != len(target_x_ums) else 0.02), 
                     y_max_plot - (0.02 if processed_file_index != len(target_x_ums) else 0.02))

    fname_load = f"DC_Photocurrent_X{x_um:04d}um_Quadrants_AB_thresholdY_RestricedRange_LOAD_results_inv.png"
    fig_load.savefig(os.path.join(fig_dir, fname_load), dpi=300, bbox_inches='tight')
    print(f"LOAD saved: {os.path.join(fig_dir, fname_load)}")
    plt.show()
    plt.close(fig_load)

    # =============================================================
    # GND PLOT (A+B) & CALCULATIONS
    # =============================================================
    fig_gnd, ax_gnd = plt.subplots(figsize=(10, 6), layout='constrained')

    # QUADRANTS A+B for Vertical Scan
    quadA_gnd = data_gnd['rawdata'].get('quadA', {}).get('dmm00_curr_amp', np.zeros((len(y_array),1))).mean(axis=1)
    quadB_gnd = data_gnd['rawdata'].get('quadB', {}).get('dmm00_curr_amp', np.zeros((len(y_array),1))).mean(axis=1)

    I_sum_gnd_plot = (quadA_gnd + quadB_gnd)[mask_plot]
    I_sum_gnd_calc = (quadA_gnd + quadB_gnd)[mask_calc]

    ax_gnd.plot(y_plot, quadA_gnd[mask_plot],  linewidth=2.1, label='quadA', color='tab:blue')
    ax_gnd.plot(y_plot, quadB_gnd[mask_plot],  linewidth=2.1, label='quadB', color='tab:orange')
    ax_gnd.plot(y_plot, I_sum_gnd_plot, '--', color="#992f7f", linewidth=2.1, alpha=1.0, label='quadA+quadB') 

    # Use dynamic regions for plotting
    ax_gnd.axvspan(current_regions['left'][0], current_regions['left'][1], color='tab:blue',   alpha=0.12, label='Left plateau (A)')
    ax_gnd.axvspan(current_regions['right'][0], current_regions['right'][1], color='tab:orange', alpha=0.12, label='Right plateau (B)')
    ax_gnd.axvspan(current_regions['gap'][0], current_regions['gap'][1], color='gray',       alpha=0.22, label='Gap region')

    ax_gnd.axhline(I_100,       color='#d2b48c', linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 1\,$A/W') 
    ax_gnd.axhline(I_real_mean, color='#8b5a2b', linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 0.8\,$A/W') 

    QE_gnd = I_sum_gnd_plot / P_combined
    ax_qe_gnd = ax_gnd.twinx()
    ax_qe_gnd.plot(y_plot, QE_gnd, color="#44fd00", lw=2.1, alpha=0.15, label=r'$\eta_{\rm{GND}}$ [A/W]') 

    ax_qe_gnd.set_ylabel(r'Responsivity $\eta$ [A/W]', color='#8b5a2b', fontsize=16)
    ax_qe_gnd.tick_params(axis='y', which='major', color='#8b5a2b', length=8, width=1.5, direction='in', labelsize=13)
    ax_qe_gnd.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
    for tick in ax_qe_gnd.get_yticklabels():
        tick.set_color('#8b5a2b')
    ax_gnd.set_ylim(bottom=0)
    ax_qe_gnd.set_ylim(bottom=0)
    ax_qe_gnd.set_ylim(top=1.05)
    ax_qe_gnd.yaxis.set_major_formatter(y_fmt_qe)
    align_yaxis_zeros(ax_gnd, ax_qe_gnd)

    # Use dynamic regions for calculation
    mean_plateau_gnd = (np.mean(I_sum_gnd_calc[lm]) + np.mean(I_sum_gnd_calc[rm])) / 2.0
    max_gap_gnd = np.max(I_sum_load_calc[gm]) if np.any(gm) else 0 # NOTE: This was correctly I_sum_gnd_calc in your original, fixed here.
    max_gap_gnd = np.max(I_sum_gnd_calc[gm]) if np.any(gm) else 0 # FIX: Ensure GND calculation uses GND data
    y_max_gnd = y_calc[gm][np.argmax(I_sum_gnd_calc[gm])] if np.any(gm) else y_plot[len(y_plot)//2]
    gs_gnd = (max_gap_gnd / mean_plateau_gnd - 1.0) * 100.0 if mean_plateau_gnd > 0 else 0

    ax_gnd.plot(y_max_gnd, max_gap_gnd, 'o', color='#d62728', mec='darkred', mew=1.8, ms=10, zorder=10) 

    # GND Inset 
    inset_g = ax_gnd.inset_axes([0.32, 0.78, 0.38, 0.16], transform=ax_gnd.transAxes)
    inset_g.axis('off')
    marker_g = Line2D([0], [0], color='#992f7f', lw=2.2, ls='--',
                      marker='o', markerfacecolor='#d62728', markeredgecolor='darkred',
                      markeredgewidth=1.8, markersize=10) 
    inset_g.legend(handles=[marker_g], labels=[rf'GS (GND): \textbf{{{gs_gnd:+.2f}\%}}'],
                   loc='center', fontsize=22, frameon=False, handletextpad=0.8, handlelength=1.8) 

    # <<< CSV ROW COLLECTION >>>
    csv_row = [x_um,
               f"{gs_load:+.3f}",
               f"{gs_gnd:+.3f}",
               f"{max_gap_load*1e6:.3f}",
               f"{max_gap_gnd*1e6:.3f}",
               f"{mean_plateau_load*1e6:.3f}",
               f"{mean_plateau_gnd*1e6:.3f}"]
    csv_data_rows.append(csv_row)

    print(f"\n=== GS FOR X = {x_um} µm ===")
    print(f"GS (LOAD): {gs_load:+.3f} %")
    print(f"GS (GND) : {gs_gnd:+.3f} %")

    l1g, lab1g = ax_gnd.get_legend_handles_labels()
    l2g, lab2g = ax_qe_gnd.get_legend_handles_labels()
    ax_gnd.legend(l1g + l2g, lab1g + lab2g, fontsize=14, loc='upper right',
                  bbox_to_anchor=(0.98, 0.98), frameon=True, fancybox=False, edgecolor='black')

    ax_gnd.set_title(rf'\textbf{{DC Photocurrent GND (VL: X={x_um}\,µm)}}', fontsize=16, fontweight='bold', pad=10)
    ax_gnd.set_xlabel(r'Y Position [mm]', fontsize=16)
    ax_gnd.set_ylabel(r'Photocurrent [A]', fontsize=16)
    ax_gnd.grid(True, linestyle='--', alpha=0.6)
    ax_gnd.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in')
    ax_gnd.yaxis.set_major_formatter(y_fmt)
    # The processed_file_index is used to mimic the original conditional logic for xlim.
    ax_gnd.set_xlim(y_min_plot + (0.02 if processed_file_index != len(target_x_ums) else 0.02), 
                    y_max_plot - (0.02 if processed_file_index != len(target_x_ums) else 0.02))

    fname_gnd = f"DC_Photocurrent_X{x_um:04d}um_Quadrants_AB_thresholdY_RestricedRange_GND_results_inv.png"
    fig_gnd.savefig(os.path.join(fig_dir, fname_gnd), dpi=300, bbox_inches='tight')
    print(f"GND saved : {os.path.join(fig_dir, fname_gnd)}")
    plt.show()
    plt.close(fig_gnd)

# <<< FINAL CSV WRITE (Ensuring proper closure) >>>
header = ['X_position_um', 'Overshoot_LOAD_%', 'Overshoot_GND_%',
          'Max_in_gap_LOAD_µA', 'Max_in_gap_GND_µA',
          'Mean_plateau_LOAD_µA', 'Mean_plateau_GND_µA']

# Use 'with open' to write all collected data at once
with open(csv_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)
    csv_writer.writerows(csv_data_rows)
    
print(f"\n=== ALL DONE (VERTICAL SCAN – QUADRANTS A+B) ===\n"
      f"Overshoot results (X = 6650, 6900, 7150 $\mu$m) successfully saved to:\n{csv_path}")
print(f"Power: {P_combined*1e3:.4f} mW $\pm$ {sigma_combined_3sigma*1e6:.1f} $\mu$W (3$\sigma$)")
print(f"$\eta=0.8$ line: {I_real_mean*1e6:.3f} $\mu$A $\pm$ {I_real_err_3sigma*1e9:.1f} nA")