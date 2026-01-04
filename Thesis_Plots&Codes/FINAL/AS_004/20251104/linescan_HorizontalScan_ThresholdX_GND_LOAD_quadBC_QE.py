'''
@author: A_Galbiati
Horizontal data analysis 
Combined power from two measurements
+ Responsivity (η = I_sum / P) on right Y-axis
Compatible data: HorizontalScan folders (Y*um)
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
base_dir = "/Users/asca/Documents/University/Master Thesis/code/Data/20251215_final_data/20251104/VIGO_NS089008_QPD_750_20_AS_0011_CC_objLens_quadABCD_manual_setup_GND_LOAD_251103_Z7.38/HorizontalScan"

# === PLOT RANGE (narrow – beautiful figures) ===
# Define default plot range (used for first processed file and all others except the second)
#x_min_plot_default = 6.9
#_max_plot_default = 7.55
# Initialize current plot range with default
#x_min_plot = x_min_plot_default 
#x_max_plot = x_max_plot_default

'''# === DYNAMIC REGION DEFINITIONS ===
# Default Regions (e.g., for first file processed)
calc_min_default = 6.65
calc_max_default = 7.65
left_region_default  = (calc_min_default, 7.15)   # Left plateau
right_region_default = (7.4, calc_max_default)   # Right plateau
gap_region_default   = (7.15, 7.4)       # Gap '''

# Regions for the SECOND PROCESSED FILE (y_index == 1)
# The request states calc_min and calc_max must STAY THE SAME.
calc_min_default = 6.65 # Keep the calculation range constant
calc_max_default = 7.65 # Keep the calculation range constant
left_region_default  = (calc_min_default, 7.14)   # Tweaked region 1
right_region_default = (7.34, calc_max_default)   # Tweaked region 2
gap_region_default  = (7.14, 7.34)       # Tweaked gap 

# Plot range for the SECOND PROCESSED FILE (y_index == 1), as requested
x_min_plot_default = 7.0
x_max_plot_default = 7.5
x_min_plot = x_min_plot_default 
x_max_plot = x_max_plot_default

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
rel_uncertainty_percent_3sigma = (sigma_combined_3sigma / P_combined) * 100

# QE references
eta_100 = 1.0
eta_real = 0.8
I_100 = eta_100 * P_combined
I_real_mean = eta_real * P_combined
I_real_err_3sigma = eta_real * sigma_combined_3sigma
sigma_eta_3sigma = eta_real * sigma_combined_3sigma / P_combined

# <<< CSV SETUP >>>
csv_dir = "/Users/asca/Documents/University/Master Thesis/code/LISA_QPD_Gap/Thesis_Plots&Codes/FINAL/AS_004/20251104/statistical study"
os.makedirs(csv_dir, exist_ok=True)
csv_path = os.path.join(csv_dir, "Overshoot_Results_BC_Y_from_2300um.csv")

# Initialize a list to hold all rows before writing
csv_data_rows = []
# <<< END CSV SETUP >>>

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

# Initialize a counter for the files that are *actually processed* (Y >= 2300)
processed_file_index = 0

# Process each Y folder (ONLY FROM Y = 2300 µm onwards)
for y_um, y_folder in y_positions:
    if y_um < 2300: # Skip Y < 2300 µm
        continue
        
    # === DYNAMIC REGION AND PLOT RANGE ASSIGNMENT ===
    if processed_file_index == 0:
        # First processed file (e.g., Y=2300)
        calc_min = calc_min_default
        calc_max = calc_max_default
        left_region = left_region_default
        right_region = right_region_default
        gap_region = gap_region_default
        # Plot range is the initial default
        x_min_plot = x_min_plot_default
        x_max_plot = x_max_plot_default
    elif processed_file_index == 1:
        # Second processed file (the one to be customized)
        calc_min = calc_min_y2
        calc_max = calc_max_y2
        left_region = left_region_y2
        right_region = right_region_y2
        gap_region = gap_region_y2
        # Apply the specific plot range, as requested
        x_min_plot = x_min_plot_y2
        x_max_plot = x_max_plot_y2
    else:
        # All subsequent files revert to default
        calc_min = calc_min_default
        calc_max = calc_max_default
        left_region = left_region_default
        right_region = right_region_default
        gap_region = gap_region_default
        # Revert to default plot range
        x_min_plot = x_min_plot_default
        x_max_plot = x_max_plot_default
        
    processed_file_index += 1
    # =================================

    fig_dir = os.path.join(y_folder, "fig")
    os.makedirs(fig_dir, exist_ok=True)

    load_pkl_files = glob.glob(os.path.join(y_folder, "*_load2.pkl"))
    gnd_pkl_files = glob.glob(os.path.join(y_folder, "*_gnd.pkl"))
    if not load_pkl_files or not gnd_pkl_files:
        print(f"Missing .pkl files in {y_folder}")
        continue
    
    # Safe data loading
    try:
        load_pkl_file = load_pkl_files[0]
        gnd_pkl_file = gnd_pkl_files[0]
        data_load = load_data(load_pkl_file)
        data_gnd = load_data(gnd_pkl_file)
    except Exception as e:
        print(f"Error loading pickle data for {y_folder}: {e}")
        continue

    x_array = data_load['rawdata']['stage_laser_xposition']
    
    # Use wide mask for calculation and narrow mask for plotting
    mask_plot = (x_array >= x_min_plot) & (x_array <= x_max_plot) # Uses dynamic x_min/max_plot
    x_array_filtered_plot = x_array[mask_plot]
    
    mask_calc = (x_array >= calc_min) & (x_array <= calc_max) # Uses constant calc_min/max
    x_array_filtered_calc = x_array[mask_calc]
    
    if len(x_array_filtered_plot) == 0:
        print(f"No data in X range for {y_folder} with plot range ({x_min_plot}, {x_max_plot})")
        continue

    # Create calculation masks on the wide calculation array
    left_mask  = (x_array_filtered_calc >= left_region[0])  & (x_array_filtered_calc <= left_region[1])
    right_mask = (x_array_filtered_calc >= right_region[0]) & (x_array_filtered_calc <= right_region[1])
    gap_mask   = (x_array_filtered_calc >= gap_region[0])   & (x_array_filtered_calc <= gap_region[1])

    # =============================================================
    # LOAD PLOT – QUADRANTS B & C (with GAP OVERSHOOT)
    # =============================================================
    fig_load, ax_load = plt.subplots(figsize=(10, 6), layout='constrained')

    # Data extraction for I_sum (narrow plot array)
    quadB_mean_load_plot = data_load['rawdata'].get('quadB', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)[mask_plot]
    quadC_mean_load_plot = data_load['rawdata'].get('quadC', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)[mask_plot]
    I_sum_load_plot = quadB_mean_load_plot + quadC_mean_load_plot
    
    # Data extraction for calculation (wide calc array)
    quadB_mean_load_calc = data_load['rawdata'].get('quadB', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)[mask_calc]
    quadC_mean_load_calc = data_load['rawdata'].get('quadC', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)[mask_calc]
    I_sum_load_calc = quadB_mean_load_calc + quadC_mean_load_calc
    
    # Plotting
    # <<< LINE WIDTH ADJUSTMENT: 2.0 -> 2.1 >>>
    ax_load.plot(x_array_filtered_plot, quadB_mean_load_plot, label='quadB', color='tab:orange', linewidth=2.1)
    ax_load.plot(x_array_filtered_plot, quadC_mean_load_plot, label='quadC', color='tab:green', linewidth=2.1)
    # <<< LINE WIDTH ADJUSTMENT: 2.0 -> 2.1 >>>
    ax_load.plot(x_array_filtered_plot, I_sum_load_plot, '--', color="#992f7f", linewidth=2.1, label='quadB+quadC', alpha=0.6)

    # Plotting Colored Regions (using calc ranges, which extend outside plot range)
    ax_load.axvspan(left_region[0], left_region[1], color='tab:orange', alpha=0.12, label='Left region')
    ax_load.axvspan(right_region[0], right_region[1], color='tab:green', alpha=0.12, label='Right region')
    ax_load.axvspan(gap_region[0], gap_region[1], color='gray', alpha=0.22, label='Gap region')
    
    # Reference lines
    # <<< LINE WIDTH ADJUSTMENT: 1.8 -> 1.8 (kept constant) >>>
    ax_load.axhline(I_100, color="#c39d7a", linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 1\,$A/W')
    ax_load.axhline(I_real_mean, color='#8b5a2b', linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 0.8\,$A/W')

    # === QUANTUM EFFICIENCY ===
    QE_load = I_sum_load_plot / P_combined
    ax_qe_load = ax_load.twinx()
    # <<< LINE WIDTH ADJUSTMENT: 2 -> 2.1 >>>
    ax_qe_load.plot(x_array_filtered_plot, QE_load, color="#00ff88", linewidth=2.1, alpha=0.2, label=r'$\eta_{\rm{LOAD}}$ [A/W]') 
    # FONTSIZE CHANGE (LOAD QE Y-label)
    ax_qe_load.set_ylabel(r'Responsivity $\eta$ [A/W]', color='#8b5a2b', fontsize=16)
    # FONTSIZE CHANGE (LOAD QE Y-tick)
    ax_qe_load.tick_params(axis='y', which='major', color='#8b5a2b', length=8, width=1.5, direction='in', labelsize=13)
    ax_qe_load.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
    for tick in ax_qe_load.get_yticklabels():
        tick.set_color('#8b5a2b')

    # Force zero alignment
    ax_load.set_ylim(bottom=0)
    ax_qe_load.set_ylim(bottom=0, top=1.05)
    align_yaxis_zeros(ax_load, ax_qe_load)
    ax_qe_load.yaxis.set_major_formatter(y_fmt_qe)

    # ==================== GAP OVERSHOOT ANALYSIS (LOAD) ====================
    # Calculations use the wide I_sum_load_calc array
    mean_left_load = np.mean(I_sum_load_calc[left_mask]) if np.any(left_mask) else 0
    mean_right_load = np.mean(I_sum_load_calc[right_mask]) if np.any(right_mask) else 0
    mean_plateaus_load = (mean_left_load + mean_right_load) / 2.0
    max_in_gap_load = np.max(I_sum_load_calc[gap_mask]) if np.any(gap_mask) else 0
    idx_max = np.argmax(I_sum_load_calc[gap_mask]) if np.any(gap_mask) else 0
    x_at_max = x_array_filtered_calc[gap_mask][idx_max] if np.any(gap_mask) else np.mean(x_array_filtered_plot)
    overshoot_percent_load = (max_in_gap_load / mean_plateaus_load - 1.0) * 100.0 if mean_plateaus_load > 0 else 0

    # Marker at maximum in gap
    # <<< MARKER SIZE ADJUSTMENT: ms=10 (kept constant), mew=1.8 (kept constant) >>>
    ax_load.plot(x_at_max, max_in_gap_load, 'o', color='#ff7f0e', mec='#cc5e00', mew=1.8, ms=10, zorder=10)

    # Result box - RAISED POSITION [0.32, 0.78, 0.38, 0.16]
    inset_load = ax_load.inset_axes([0.32, 0.78, 0.38, 0.16], transform=ax_load.transAxes)
    inset_load.axis('off')
    # <<< INSET MARKER ADJUSTMENT: lw=2 -> 2.2, mew=1.8 (kept constant), ms=10 (kept constant), handlelength=1.8 (kept constant) >>>
    legend_handle = Line2D([0], [0], color=(*plt.cm.colors.to_rgb('#992f7f'), 0.6), lw=2.2, ls='--',
                           marker='o', markerfacecolor='#ff7f0e',
                           markeredgecolor='#cc5e00', markeredgewidth=1.8, markersize=10)
    # FONTSIZE CHANGE (LOAD Inset Legend)
    inset_load.legend(handles=[legend_handle],
                      labels=[rf'GS (LOAD): \textbf{{{overshoot_percent_load:+.2f}\%}}'], 
                      loc='center', fontsize=22, frameon=False, # Increased to 22
                      handletextpad=0.8, handlelength=1.8)

    # ==================== LEGEND & FINALIZE ====================
    lines1, labels1 = ax_load.get_legend_handles_labels()
    lines2, labels2 = ax_qe_load.get_legend_handles_labels()
    # FONTSIZE CHANGE (LOAD Legend)
    ax_load.legend(lines1 + lines2, labels1 + labels2,
                   fontsize=14, loc='upper right', bbox_to_anchor=(0.98, 0.98), # Increased to 14
                   frameon=True, fancybox=False, edgecolor='black')
    # FONTSIZE CHANGE (LOAD Title)
    ax_load.set_title(rf'\textbf{{DC Photocurrent LOAD (BC: Y={y_um}\textmu{{m}})}}',
                      fontsize=16, fontweight='bold', pad=10) # Increased to 16
    # FONTSIZE CHANGE (LOAD X-label)
    ax_load.set_xlabel(r'X Position [mm]', fontsize=16) # Increased to 16
    # FONTSIZE CHANGE (LOAD Y-label)
    ax_load.set_ylabel(r'Photocurrent [A]', fontsize=16) # Increased to 16
    ax_load.grid(True, linestyle='--', alpha=0.6)
    # FONTSIZE CHANGE (LOAD Axis ticks)
    ax_load.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in') # Increased to 13
    ax_load.yaxis.set_major_formatter(y_fmt)
    ax_load.set_xlim(x_min_plot + 0.015, x_max_plot-0.015) # Uses dynamic x_min/max_plot

    fig_name_load = f"DC_Photocurrent_Y{y_um:04d}um_Quadrants_BC_GapOvershoot_LOAD_results_inv"
    fig_load.savefig(os.path.join(fig_dir, f"{fig_name_load}.png"), dpi=300, bbox_inches='tight')
    print(f"LOAD (B+C + overshoot) saved: {os.path.join(fig_dir, f'{fig_name_load}.png')}")
    plt.show()
    plt.close(fig_load)

    # =============================================================
    # GND PLOT – QUADRANTS B & C (with GAP OVERSHOOT)
    # =============================================================
    fig_gnd, ax_gnd = plt.subplots(figsize=(10, 6), layout='constrained')

    # Data extraction for I_sum (narrow plot array)
    quadB_mean_gnd_plot = data_gnd['rawdata'].get('quadB', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)[mask_plot]
    quadC_mean_gnd_plot = data_gnd['rawdata'].get('quadC', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)[mask_plot]
    I_sum_gnd_plot = quadB_mean_gnd_plot + quadC_mean_gnd_plot
    
    # Data extraction for calculation (wide calc array)
    quadB_mean_gnd_calc = data_gnd['rawdata'].get('quadB', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)[mask_calc]
    quadC_mean_gnd_calc = data_gnd['rawdata'].get('quadC', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)[mask_calc]
    I_sum_gnd_calc = quadB_mean_gnd_calc + quadC_mean_gnd_calc
    
    # <<< LINE WIDTH ADJUSTMENT: 2.0 -> 2.1 >>>
    ax_gnd.plot(x_array_filtered_plot, quadB_mean_gnd_plot, label='quadB', color='tab:orange', linewidth=2.1)
    ax_gnd.plot(x_array_filtered_plot, quadC_mean_gnd_plot, label='quadC', color='tab:green', linewidth=2.1)
    # <<< LINE WIDTH ADJUSTMENT: 2.0 -> 2.1 >>>
    ax_gnd.plot(x_array_filtered_plot, I_sum_gnd_plot, '--', color="#992f7f", linewidth=2.1, alpha=1.0, label='quadB+quadC')

    # Plotting Colored Regions
    ax_gnd.axvspan(left_region[0], left_region[1], color='tab:orange', alpha=0.12, label='Left region')
    ax_gnd.axvspan(right_region[0], right_region[1], color='tab:green', alpha=0.12, label='Right region')
    ax_gnd.axvspan(gap_region[0], gap_region[1], color='gray', alpha=0.22, label='Gap region')

    # <<< LINE WIDTH ADJUSTMENT: 1.8 -> 1.8 (kept constant) >>>
    ax_gnd.axhline(I_100, color='#d2b48c', linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 1\,$A/W')
    ax_gnd.axhline(I_real_mean, color='#8b5a2b', linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 0.8\,$A/W')

    # QE GND
    QE_gnd = I_sum_gnd_plot / P_combined
    ax_qe_gnd = ax_gnd.twinx()
    # <<< LINE WIDTH ADJUSTMENT: 2 -> 2.1 >>>
    ax_qe_gnd.plot(x_array_filtered_plot, QE_gnd, color="#44fd00", linewidth=2.1, label=r'$\eta_{\rm{GND}}$ [A/W]', alpha=0.2) 
    # FONTSIZE CHANGE (GND QE Y-label)
    ax_qe_gnd.set_ylabel(r'Responsivity $\eta$ [A/W]', color='#8b5a2b', fontsize=16) # Increased to 16
    # FONTSIZE CHANGE (GND QE Y-tick)
    ax_qe_gnd.tick_params(axis='y', which='major', color='#8b5a2b', length=8, width=1.5, direction='in', labelsize=13) # Increased to 13
    ax_qe_gnd.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
    for tick in ax_qe_gnd.get_yticklabels():
        tick.set_color('#8b5a2b')

    ax_gnd.set_ylim(bottom=0)
    ax_qe_gnd.set_ylim(bottom=0, top=1.05)
    align_yaxis_zeros(ax_gnd, ax_qe_gnd)
    ax_qe_gnd.yaxis.set_major_formatter(y_fmt_qe)

    # ==================== GAP OVERSHOOT ANALYSIS (GND) ====================
    # Calculations use the wide I_sum_gnd_calc array
    mean_left_gnd = np.mean(I_sum_gnd_calc[left_mask]) if np.any(left_mask) else 0
    mean_right_gnd = np.mean(I_sum_gnd_calc[right_mask]) if np.any(right_mask) else 0
    mean_plateaus_gnd = (mean_left_gnd + mean_right_gnd) / 2.0
    max_in_gap_gnd = np.max(I_sum_gnd_calc[gap_mask]) if np.any(gap_mask) else 0
    idx_max_g = np.argmax(I_sum_gnd_calc[gap_mask]) if np.any(gap_mask) else 0
    x_at_max_g = x_array_filtered_calc[gap_mask][idx_max_g] if np.any(gap_mask) else np.mean(x_array_filtered_plot)
    overshoot_percent_gnd = (max_in_gap_gnd / mean_plateaus_gnd - 1.0) * 100.0 if mean_plateaus_gnd > 0 else 0

    # Marker
    # <<< MARKER SIZE ADJUSTMENT: ms=10 (kept constant), mew=1.8 (kept constant) >>>
    ax_gnd.plot(x_at_max_g, max_in_gap_gnd, 'o', color='#d62728', mec='darkred', mew=1.8, ms=10, zorder=10)

    # Result box - RAISED POSITION [0.32, 0.78, 0.38, 0.16]
    inset_gnd = ax_gnd.inset_axes([0.32, 0.78, 0.38, 0.16], transform=ax_gnd.transAxes)
    inset_gnd.axis('off')
    # <<< INSET MARKER ADJUSTMENT: lw=2.2 (kept constant), mew=1.8 (kept constant), ms=10 (kept constant), handlelength=1.8 (kept constant) >>>
    legend_handle_g = Line2D([0], [0], color='#992f7f', lw=2.2, ls='--',
                             marker='o', markerfacecolor='#d62728',
                             markeredgecolor='darkred', markeredgewidth=1.8, markersize=10)
    # FONTSIZE CHANGE (GND Inset Legend)
    inset_gnd.legend(handles=[legend_handle_g],
                     labels=[rf'GS (GND): \textbf{{{overshoot_percent_gnd:+.2f}\%}}'], 
                     loc='center', fontsize=22, frameon=False, # Increased to 22
                     handletextpad=0.8, handlelength=1.8)

    # <<< CSV ROW COLLECTION >>>
    csv_data_rows.append([y_um,
                         f"{overshoot_percent_load:+.3f}",
                         f"{overshoot_percent_gnd:+.3f}",
                         f"{max_in_gap_load*1e6:.3f}",
                         f"{max_in_gap_gnd*1e6:.3f}",
                         f"{mean_plateaus_load*1e6:.3f}",
                         f"{mean_plateaus_gnd*1e6:.3f}"])
    # <<< END CSV ROW COLLECTION >>>

    # Legend & finalize
    lines1g, labels1g = ax_gnd.get_legend_handles_labels()
    lines2g, labels2g = ax_qe_gnd.get_legend_handles_labels()
    # FONTSIZE CHANGE (GND Legend)
    ax_gnd.legend(lines1g + lines2g, labels1g + labels2g,
                  fontsize=14, loc='upper right', bbox_to_anchor=(0.98, 0.98), # Increased to 14
                  frameon=True, fancybox=False, edgecolor='black')
    # FONTSIZE CHANGE (GND Title)
    ax_gnd.set_title(rf'\textbf{{DC Photocurrent GND (BC: Y={y_um}\textmu{{m}})}}',
                     fontsize=16, fontweight='bold', pad=10) # Increased to 16
    # FONTSIZE CHANGE (GND X-label)
    ax_gnd.set_xlabel(r'X Position [mm]', fontsize=16) # Increased to 16
    # FONTSIZE CHANGE (GND Y-label)
    ax_gnd.set_ylabel(r'Photocurrent [A]', fontsize=16) # Increased to 16
    ax_gnd.grid(True, linestyle='--', alpha=0.6)
    # FONTSIZE CHANGE (GND Axis ticks)
    ax_gnd.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in') # Increased to 13
    ax_gnd.yaxis.set_major_formatter(y_fmt)
    ax_gnd.set_xlim(x_min_plot + 0.01, x_max_plot - 0.01) # Uses dynamic x_min/max_plot

    fig_name_gnd = f"DC_Photocurrent_Y{y_um:04d}um_Quadrants_BC_GapOvershoot_GND_results_inv"
    fig_gnd.savefig(os.path.join(fig_dir, f"{fig_name_gnd}.png"), dpi=300, bbox_inches='tight')
    print(f"GND (B+C + overshoot) saved: {os.path.join(fig_dir, f'{fig_name_gnd}.png')}")
    plt.show()
    plt.close(fig_gnd)

    # Console summary
    print(f"\n=== GAP OVERSHOOT SUMMARY Y = {y_um} µm (B+C) ===")
    print(f"LOAD → GS = {overshoot_percent_load:+.2f} %")
    print(f"GROUND → GS = {overshoot_percent_gnd:+.2f} %")

# <<< FINAL CSV WRITE (Guarantees proper close) >>>
header = ['Y_position_um', 'Overshoot_LOAD_%', 'Overshoot_GND_%',
          'Max_in_gap_LOAD_µA', 'Max_in_gap_GND_µA',
          'Mean_plateau_LOAD_µA', 'Mean_plateau_GND_µA']

with open(csv_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)
    csv_writer.writerows(csv_data_rows)

print(f"\n=== ALL DONE (HORIZONTAL SCAN – QUADRANTS B+C) ===\n"
      f"Overshoot results (Y ≥ 2300 µm) successfully saved to:\n    {csv_path}")
print(f"Power: {P_combined*1e3:.4f} mW $\pm$ {sigma_combined_3sigma*1e6:.1f} $\mu$W (3$\sigma$)")
# <<< END FINAL CSV WRITE >>>