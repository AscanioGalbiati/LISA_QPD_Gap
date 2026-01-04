'''
@author: A_Galbiati
Vertical data analysis
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

# LaTeX font setup
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
fm.FontProperties(fname=font_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

# Base directory
base_dir = "/Users/asca/Documents/University/Master Thesis/code/Data/20251215_final_data/20251113/VIGO_NS089008_QPD_1000_20_AS_004_objLens_quadABCD_manual_setup_GND_LOAD_251111_Z7.0_NDfilter_POWER/VerticalScan"

# === PLOT RANGE (narrow – beautiful figures) ===
y_min_plot = 2.29
y_max_plot = 2.45

# === DYNAMIC REGION DEFINITIONS ===
# Default Regions (SYMMETRIC)
calc_min_default = 2.31
calc_max_default = 2.45
left_region_default  = (calc_min_default, 2.31)   # Adjusting L/R to be slightly wider for safety
right_region_default = (2.38, calc_max_default)   
gap_region_default   = (2.31, 2.38)   # gap

# Regions for specific faulty X positions (ASYMMETRIC - R-ONLY)
# These regions will be used when asymmetric_calc = True
calc_min_asymm = 2.32
calc_max_asymm = 2.45
left_region_asymm  = (calc_min_asymm, calc_min_asymm) # <<< MODIFIED: Disabled Left Plateau >>>
right_region_asymm = (2.36, calc_max_asymm)           # Right plateau (for normalization)
gap_region_asymm   = (2.31, 2.36)                     # Gap
y_min_plot_asymm = 2.29                               # Tweak plot start for missing left data
y_max_plot_asymm = y_max_plot

# === COMBINED POWER FROM TWO MEASUREMENTS (1σ) ===
P1 = 0.0019846 # W
sigma1 = 1.0257e-06# W
P2 = 0.0019885 # W
sigma2 = 2.489e-06 # W
w1 = 1 / (sigma1**2)
w2 = 1 / (sigma2**2)
P_combined = (P1 * w1 + P2 * w2) / (w1 + w2)
sigma_combined_1sigma = np.sqrt(1 / (w1 + w2))

# === 3σ FOR PLOTTING ===
sigma_combined_3sigma = 3 * sigma_combined_1sigma
rel_uncertainty_percent_3sigma = (sigma_combined_3sigma / P_combined)

# QE reference lines
eta_100 = 1.0
eta_real = 0.8
I_100 = eta_100 * P_combined
I_real_mean = eta_real * P_combined

# <<< CSV SETUP >>>
csv_dir = "/Users/asca/Documents/University/Master Thesis/code/LISA_QPD_Gap/Thesis_Plots&Codes/FINAL/AK_004/statistical study"
os.makedirs(csv_dir, exist_ok=True)
csv_path = os.path.join(csv_dir, "Overshoot_Results_CD_X_from_7300um_AsymmRight_V2.csv") # Updated CSV name

# Initialize a list to hold all rows before writing
csv_data_rows = []
# <<< END CSV SETUP >>>


# === ZERO ALIGNMENT HELPER ===
def align_yaxis_zeros(ax1, ax2):
    y1l, y1h = ax1.get_ylim()
    y2l, y2h = ax2.get_ylim()
    if y1h == y1l:
        return
    zero_frac = (0 - y1l) / (y1h - y1l)
    y2_range = y2h - y2l
    ax2.set_ylim(0 - zero_frac * y2_range, 0 - zero_frac * y2_range + y2_range)

# === DATA LOADING ===
def load_data(fp):
    with open(fp, "rb") as f:
        return pickle.load(f)

def get_x_um(folder_name):
    m = re.search(r'X(\d+)um', folder_name)
    return int(m.group(1)) if m else None

# Find all X*um folders
x_folders = glob.glob(os.path.join(base_dir, "X*um"))
x_positions = sorted([(get_x_um(os.path.basename(f)), f) for f in x_folders if get_x_um(os.path.basename(f))])

# Formatters
def curr_fmt(val, pos): return '0' if val == 0 else f'{val:.4f}'
def qe_fmt(val, pos): return '0' if val == 0 else f'{val:.1f}'
y_fmt = FuncFormatter(curr_fmt)
y_fmt_qe = FuncFormatter(qe_fmt)

# ================================================================
# MAIN LOOP – Quadrants C & D – STARTING FROM X ≥ 7400 µm
# ================================================================
for x_um, x_folder in x_positions:
    if x_um < 7400: # Skip X < 7400 µm
        continue

    # === DYNAMIC REGION AND PLOT RANGE ASSIGNMENT ===
    # Check if the current X position is one of the problematic files
    if x_um in [7500, 7700]: # <<< MODIFIED: Target specific X values for R-ONLY calc >>>
        calc_min = calc_min_asymm
        calc_max = calc_max_asymm
        left_region = left_region_asymm   # Disabled left
        right_region = right_region_asymm # Active right
        gap_region = gap_region_asymm
        y_min_plot_current = y_min_plot_asymm
        y_max_plot_current = y_max_plot_asymm
        asymmetric_calc = True # Flag for asymmetric calculation (R-only)
    else:
        # Use default (symmetric) settings for all other X >= 7400
        calc_min = calc_min_default
        calc_max = calc_max_default
        left_region = left_region_default
        right_region = right_region_default
        gap_region = gap_region_default
        y_min_plot_current = y_min_plot
        y_max_plot_current = y_max_plot
        asymmetric_calc = False 
    # =================================

    fig_dir = os.path.join(x_folder, "fig")
    os.makedirs(fig_dir, exist_ok=True)

    load_pkl = glob.glob(os.path.join(x_folder, "*_load2.pkl"))
    gnd_pkl = glob.glob(os.path.join(x_folder, "*_gnd.pkl"))
    if not load_pkl or not gnd_pkl:
        print(f"Missing pkl in {x_folder}")
        continue
    
    # Safe data loading
    try:
        data_load = load_data(load_pkl[0])
        data_gnd = load_data(gnd_pkl[0])
    except Exception as e:
        print(f"Error loading pickle data for {x_folder}: {e}")
        continue

    y_array = data_load['rawdata']['stage_laser_yposition']
    
    # Masks: narrow mask for plotting, wide mask for calculation
    mask_plot = (y_array >= y_min_plot_current) & (y_array <= y_max_plot_current) # Uses dynamic plot range
    y_arr = y_array[mask_plot]
    
    mask_calc = (y_array >= calc_min) & (y_array <= calc_max)
    y_calc = y_array[mask_calc]
    
    if len(y_arr) == 0:
        print(f"No data in Y range {x_folder}")
        continue

    # Create masks for the calculation range
    lm = (y_calc >= left_region[0]) & (y_calc <= left_region[1])
    rm = (y_calc >= right_region[0]) & (y_calc <= right_region[1])
    gm = (y_calc >= gap_region[0]) & (y_calc <= gap_region[1])
    
    # Handle the case where the calculation range is empty
    if not np.any(mask_calc):
        print(f"No data in calculation range for {x_folder}")
        continue

    # ====================== LOAD PLOT (C+D) ======================
    fig_load, ax_load = plt.subplots(figsize=(10,6), layout='constrained')

    # Data extraction for plotting (narrow range)
    quadC_load_plot = data_load['rawdata'].get('quadC', {}).get('dmm00_curr_amp', np.zeros_like(y_array)).mean(axis=1)[mask_plot]
    quadD_load_plot = data_load['rawdata'].get('quadD', {}).get('dmm00_curr_amp', np.zeros_like(y_array)).mean(axis=1)[mask_plot]
    I_sum_load_plot = quadC_load_plot + quadD_load_plot

    # Data extraction for calculation (wide range)
    quadC_load_calc = data_load['rawdata'].get('quadC', {}).get('dmm00_curr_amp', np.zeros_like(y_array)).mean(axis=1)[mask_calc]
    quadD_load_calc = data_load['rawdata'].get('quadD', {}).get('dmm00_curr_amp', np.zeros_like(y_array)).mean(axis=1)[mask_calc]
    I_sum_load_calc = quadC_load_calc + quadD_load_calc

    # Plotting:
    ax_load.plot(y_arr, quadC_load_plot, linewidth=2.1, label='quadC', color='tab:green')
    ax_load.plot(y_arr, quadD_load_plot, linewidth=2.1, label='quadD', color='tab:red') 
    ax_load.plot(y_arr, I_sum_load_plot, '--', color="#992f7f", lw=2.1, alpha=0.6, label='quadC+quadD') 

    # Plot colored regions based on calculation ranges
    # Only plot the left region if it's NOT disabled (i.e., if symmetric)
    if not asymmetric_calc:
        ax_load.axvspan(left_region[0], left_region[1], color='tab:red', alpha=0.12, label='Left region')
    # Always plot the right region and gap region
    ax_load.axvspan(right_region[0], right_region[1], color='tab:green', alpha=0.12, label='Right region')
    ax_load.axvspan(gap_region[0], gap_region[1], color='gray', alpha=0.22, label='Gap region')

    # ETA REFERENCE LINE WIDTHS
    ax_load.axhline(I_100, color="#c39d7a", ls='--', lw=2.1, alpha=0.95, label=r'$\eta = 1\,$A/W')
    ax_load.axhline(I_real_mean, color="#8b5a2b", ls='--', lw=2.1, alpha=0.95, label=r'$\eta = 0.8\,$A/W')

    # Quantum Efficiency
    QE_load = I_sum_load_plot / P_combined
    ax_qe = ax_load.twinx()
    ax_qe.plot(y_arr, QE_load, color="#00ff88", lw=2.1, alpha=0.2, label=r'$\eta_{\rm{LOAD}}$ [A/W]') 
    
    ax_qe.set_ylabel(r'Responsivity $\eta$ [A/W]', color='#8b5a2b', fontsize=16)
    ax_qe.tick_params(axis='y', colors='#8b5a2b', length=8, width=1.5, direction='in', labelsize=13)
    for t in ax_qe.get_yticklabels(): t.set_color('#8b5a2b')
    ax_qe.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f'{y:.2f}'))

    ax_load.set_ylim(bottom=0)
    ax_qe.set_ylim(bottom=0, top=1.05)
    align_yaxis_zeros(ax_load, ax_qe)
    ax_qe.yaxis.set_major_formatter(y_fmt_qe)

    # ==================== GAP OVERSHOOT ANALYSIS (LOAD) ====================
    mean_left_load = np.mean(I_sum_load_calc[lm]) if np.any(lm) else 0
    mean_right_load = np.mean(I_sum_load_calc[rm]) if np.any(rm) else 0

    if asymmetric_calc:
        # Use only the right plateau mean for normalization
        mean_plateau_load = mean_right_load
        label_suffix = r' (\textbf{R-only})' 
    else:
        # Use the average of both plateaus (default)
        mean_plateau_load = (mean_left_load + mean_right_load) / 2
        label_suffix = ''

    max_gap_load = np.max(I_sum_load_calc[gm]) if np.any(gm) else 0
    y_max_load = y_calc[gm][np.argmax(I_sum_load_calc[gm])] if np.any(gm) else y_arr[len(y_arr)//2]
    overshoot_pct_load = (max_gap_load / mean_plateau_load - 1) * 100 if mean_plateau_load > 0 else 0

    # PEAK MARKER
    ax_load.plot(y_max_load, max_gap_load, 'o', color='#ff7f0e', mec='#cc5e00', mew=2.1, ms=11, zorder=10) 

    # Result box - RAISED POSITION
    box = ax_load.inset_axes([0.32, 0.78, 0.38, 0.16], transform=ax_load.transAxes)
    box.axis('off')
    marker = Line2D([0], [0], color=(*plt.cm.colors.to_rgb('#992f7f'),0.6), lw=2.1, ls='--',
                    marker='o', mfc='#ff7f0e', mec='#cc5e00', mew=2.1, ms=11)
    
    # Apply dynamic label suffix
    box.legend(handles=[marker],
               labels=[rf'GS (LOAD): \textbf{{{overshoot_pct_load:+.2f}\%}}'], 
               loc='center', fontsize=22, frameon=False, handlelength=2.5) 

    # Legend & Finalize
    h1, l1 = ax_load.get_legend_handles_labels()
    h2, l2 = ax_qe.get_legend_handles_labels()
    ax_load.legend(h1+h2, l1+l2, fontsize=14, loc='upper right',
                   bbox_to_anchor=(0.98,0.98), frameon=True, fancybox=False, edgecolor='black')

    ax_load.set_title(rf'\textbf{{DC Photocurrent LOAD (VL: X={x_um}\,µm)}}', fontsize=16, pad=10) 
    ax_load.set_xlabel('Y Position [mm]', fontsize=16) 
    ax_load.set_ylabel('Photocurrent [A]', fontsize=16) 
    ax_load.grid(True, ls='--', alpha=0.6)
    ax_load.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in') 
    ax_load.yaxis.set_major_formatter(y_fmt)
    # Uses dynamic plot range
    ax_load.set_xlim(y_min_plot_current+0.02, y_max_plot_current-0.02) 

    fname = f"DC_Photocurrent_X{x_um:04d}um_Quadrants_CD_LOAD_results_inv"
    if asymmetric_calc:
        fname += "_AsymmRight"
    fname += ".png"
    fig_load.savefig(os.path.join(fig_dir, fname), dpi=300, bbox_inches='tight')
    print(f"LOAD (C+D) → {os.path.join(fig_dir, fname)}")
    plt.show()
    plt.close(fig_load)

    # ====================== GND PLOT (C+D) ======================
    fig_gnd, ax_gnd = plt.subplots(figsize=(10,6), layout='constrained')

    # Data extraction for plotting (narrow range)
    quadC_gnd_plot = data_gnd['rawdata'].get('quadC', {}).get('dmm00_curr_amp', np.zeros_like(y_array)).mean(axis=1)[mask_plot]
    quadD_gnd_plot = data_gnd['rawdata'].get('quadD', {}).get('dmm00_curr_amp', np.zeros_like(y_array)).mean(axis=1)[mask_plot]
    I_sum_gnd_plot = quadC_gnd_plot + quadD_gnd_plot
    
    # Data extraction for calculation (wide range)
    quadC_gnd_calc = data_gnd['rawdata'].get('quadC', {}).get('dmm00_curr_amp', np.zeros_like(y_array)).mean(axis=1)[mask_calc]
    quadD_gnd_calc = data_gnd['rawdata'].get('quadD', {}).get('dmm00_curr_amp', np.zeros_like(y_array)).mean(axis=1)[mask_calc]
    I_sum_gnd_calc = quadC_gnd_calc + quadD_gnd_calc

    # Plotting:
    ax_gnd.plot(y_arr, quadC_gnd_plot, linewidth=2.1, label='quadC', color='tab:green')
    ax_gnd.plot(y_arr, quadD_gnd_plot, linewidth=2.1, label='quadD', color='tab:red')
    ax_gnd.plot(y_arr, I_sum_gnd_plot, '--', color="#992f7f", lw=2.1, label='quadC+quadD') 

    # Plot colored regions based on calculation ranges
    # Only plot the left region if it's NOT disabled (i.e., if symmetric)
    if not asymmetric_calc:
        ax_gnd.axvspan(left_region[0], left_region[1], color='tab:red', alpha=0.12, label='Left region')
    # Always plot the right region and gap region
    ax_gnd.axvspan(right_region[0], right_region[1], color='tab:green', alpha=0.12, label='Right region')
    ax_gnd.axvspan(gap_region[0], gap_region[1], color='gray', alpha=0.22, label='Gap region')

    # ETA REFERENCE LINE WIDTHS
    ax_gnd.axhline(I_100, color='#d2b48c', ls='--', lw=2.1, alpha=0.95, label=r'$\eta = 1\,$A/W')
    ax_gnd.axhline(I_real_mean, color="#8b5a2b", ls='--', lw=2.1, alpha=0.95, label=r'$\eta = 0.8\,$A/W')

    # Quantum Efficiency GND
    QE_gnd = I_sum_gnd_plot / P_combined
    ax_qe_g = ax_gnd.twinx()
    ax_qe_g.plot(y_arr, QE_gnd, color="#44fd00", lw=2.1, alpha=0.22, label=r'$\eta_{\rm{GND}}$ [A/W]') 
    
    ax_qe_g.set_ylabel(r'Responsivity $\eta$ [A/W]', fontsize=16, color='#8b5a2b') 
    ax_qe_g.tick_params(axis='y', colors='#8b5a2b', length=8, width=1.5, direction='in', labelsize=13) 
    for t in ax_qe_g.get_yticklabels(): t.set_color('#8b5a2b')
    ax_qe_g.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f'{y:.2f}'))

    ax_gnd.set_ylim(bottom=0)
    ax_qe_g.set_ylim(bottom=0, top=1.05)
    align_yaxis_zeros(ax_gnd, ax_qe_g)
    ax_qe_g.yaxis.set_major_formatter(y_fmt_qe)

    # ==================== GAP OVERSHOOT ANALYSIS (GND) ====================
    mean_left_gnd = np.mean(I_sum_gnd_calc[lm]) if np.any(lm) else 0
    mean_right_gnd = np.mean(I_sum_gnd_calc[rm]) if np.any(rm) else 0

    if asymmetric_calc:
        # Use only the right plateau mean for normalization
        mean_plateau_gnd = mean_right_gnd
        label_suffix = r' (\textbf{R-only})'
    else:
        # Use the average of both plateaus (default)
        mean_plateau_gnd = (mean_left_gnd + mean_right_gnd) / 2
        label_suffix = ''

    max_gap_gnd = np.max(I_sum_gnd_calc[gm]) if np.any(gm) else 0
    y_max_gnd = y_calc[gm][np.argmax(I_sum_gnd_calc[gm])] if np.any(gm) else y_arr[len(y_arr)//2]
    overshoot_g_pct = (max_gap_gnd / mean_plateau_gnd - 1) * 100 if mean_plateau_gnd > 0 else 0

    # PEAK MARKER
    ax_gnd.plot(y_max_gnd, max_gap_gnd, 'o', color='#d62728', mec='darkred', mew=2.1, ms=11, zorder=10) 

    # Result box GND - RAISED POSITION
    box_g = ax_gnd.inset_axes([0.32, 0.78, 0.38, 0.16], transform=ax_gnd.transAxes)
    box_g.axis('off')
    marker_g = Line2D([0], [0], color='#992f7f', lw=2.3, ls='--',
                      marker='o', mfc='#d62728', mec='darkred', mew=2.1, ms=11)
    
    # Apply dynamic label suffix
    box_g.legend(handles=[marker_g],
                 labels=[rf'GS (GND): \textbf{{{overshoot_g_pct:+.2f}\%}}'], 
                 loc='center', fontsize=22, frameon=False, handlelength=2.5) 

    # <<< CSV ROW COLLECTION >>>
    csv_data_rows.append([x_um,
                         f"{overshoot_pct_load:+.3f}",
                         f"{overshoot_g_pct:+.3f}",
                         f"{max_gap_load*1e6:.3f}",
                         f"{max_gap_gnd*1e6:.3f}",
                         f"{mean_plateau_load*1e6:.3f}",
                         f"{mean_plateau_gnd*1e6:.3f}"])
    # <<< END CSV ROW COLLECTION >>>

    print(f"\nX = {x_um} µm → LOAD GS: {overshoot_pct_load:+.2f} % | GND GS: {overshoot_g_pct:+.2f} % (Asymmetric R-only: {asymmetric_calc})")

    # Legend & Finalize
    h1g, l1g = ax_gnd.get_legend_handles_labels()
    h2g, l2g = ax_qe_g.get_legend_handles_labels()
    ax_gnd.legend(h1g+h2g, l1g+l2g, fontsize=14, loc='upper right',
                  bbox_to_anchor=(0.98,0.98), frameon=True, fancybox=False, edgecolor='black')

    ax_gnd.set_title(rf'\textbf{{DC Photocurrent GND (VL: X={x_um}\,µm)}}', fontsize=16, pad=10) 
    ax_gnd.set_xlabel('Y Position [mm]', fontsize=16) 
    ax_gnd.set_ylabel('Photocurrent [A]', fontsize=16) 
    ax_gnd.grid(True, ls='--', alpha=0.6)
    ax_gnd.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in') 
    ax_gnd.yaxis.set_major_formatter(y_fmt)
    # Uses dynamic plot range
    ax_gnd.set_xlim(y_min_plot_current+0.02, y_max_plot_current-0.02)

    fname_gnd = f"DC_Photocurrent_X{x_um:04d}um_Quadrants_CD_GND_results_inv"
    if asymmetric_calc:
        fname_gnd += "_AsymmRight"
    fname_gnd += ".png"
    fig_gnd.savefig(os.path.join(fig_dir, fname_gnd), dpi=300, bbox_inches='tight')
    print(f"GND (C+D) → {os.path.join(fig_dir, fname_gnd)}")
    plt.show()
    plt.close(fig_gnd)

# <<< FINAL CSV WRITE (Guarantees proper close) >>>
header = ['X_position_um', 'Overshoot_LOAD_%', 'Overshoot_GND_%',
          'Max_in_gap_LOAD_µA', 'Max_in_gap_GND_µA',
          'Mean_plateau_LOAD_µA', 'Mean_plateau_GND_µA']

with open(csv_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)
    csv_writer.writerows(csv_data_rows)

print(f"\n=== ALL DONE (C+D VERTICAL) ===\nOvershoot results (X ≥ 7400 µm, R-only for 7500 and 7700) successfully saved to:\n    {csv_path}")
print(f"Power used: {P_combined*1e3:.4f} mW $\pm$ {sigma_combined_3sigma*1e6:.1f} $\mu$W (3$\sigma$)")
# <<< END FINAL CSV WRITE >>>

print("\nALL C&D VERTICAL PLOTS GENERATED SUCCESSFULLY – WITH CORRECTED ASYMMETRIC GAP OVERSHOOT BOXES")