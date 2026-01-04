''' 
Horizontal data analysis – FINAL 3σ uncertainty | ZEROS ALIGNED | QE in BROWN | NO INSETS 
Combined power from two 2025-11-03 measurements
* Quantum Efficiency (η = I_sum / P) on right Y-axis 

Compatible data:
/Users/asca/Documents/University/Master Thesis/code/Data/all quadrants/20251104/VIGO_NS089008_QPD_750_20_AS_0011_CC_objLens_quadABCD_manual_setup_GND_LOAD_251103_Z7.38/HorizontalScan
# X-range
x_min_plot = 7.0
x_max_plot = 7.5

# === COMBINED POWER FROM TWO MEASUREMENTS (1σ) ===
P1 = 0.019158  # W
sigma1 = 1.2336e-5  # W
P2 = 0.019151  # W
sigma2 = 1.1417e-5  # W
w1 = 1 / (sigma1**2)
w2 = 1 / (sigma2**2)
P_combined = (P1 * w1 + P2 * w2) / (w1 + w2)
sigma_combined_1sigma = np.sqrt(1 / (w1 + w2))

/Users/asca/Documents/University/Master Thesis/code/Data/all quadrants/20251110/VIGO_NS089008_QPD_1000_20_AS_004_objLens_quadABCD_manual_setup_GND_LOAD_251107_Z7.0_NDfilter/HorizontalScan

# X-range
x_min_plot = 7.1
x_max_plot = 7.350

# FOR 20251106 DATA
# === COMBINED POWER FROM TWO MEASUREMENTS (1σ) ===
P1 = 0.0019846  # W
sigma1 = 1.0257e-06# W
P2 = 0.0019885  # W
sigma2 = 2.489e-06 # W
w1 = 1 / (sigma1**2)
w2 = 1 / (sigma2**2)
P_combined = (P1 * w1 + P2 * w2) / (w1 + w2)
sigma_combined_1sigma = np.sqrt(1 / (w1 + w2))

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

# Set up font and LaTeX rendering
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
cmu_serif = fm.FontProperties(fname=font_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

# Base directory
base_dir = "/Users/asca/Documents/University/Master Thesis/code/Data/all quadrants/20251027/VIGO_NS089008_QPD_750_20_AS_0011_CC_objLens_quadABCD_manual_setup_GND_LOAD_XY/HorizontalScan"

# X-range
x_min_plot = 7.198
x_max_plot = 7.362

# === COMBINED POWER FROM TWO MEASUREMENTS (1σ) ===
P1 = 0.019158  # W
sigma1 = 1.2336e-5  # W
P2 = 0.019151  # W
sigma2 = 1.1417e-5  # W
w1 = 1 / (sigma1**2)
w2 = 1 / (sigma2**2)
P_combined = (P1 * w1 + P2 * w2) / (w1 + w2)
sigma_combined_1sigma = np.sqrt(1 / (w1 + w2))

# === 3σ FOR PLOTTING ===
sigma_combined_3sigma = 6 * sigma_combined_1sigma
rel_uncertainty_percent_3sigma = (sigma_combined_3sigma / P_combined) * 100

# QE references
eta_100 = 1.0
eta_real = 0.8
I_100 = eta_100 * P_combined
I_real_mean = eta_real * P_combined
I_real_err_3sigma = eta_real * sigma_combined_3sigma
sigma_eta_3sigma = eta_real * sigma_combined_3sigma / P_combined

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

# Process each Y folder
for y_um, y_folder in y_positions:
    fig_dir = os.path.join(y_folder, "fig_final")
    os.makedirs(fig_dir, exist_ok=True)

    load_pkl_files = glob.glob(os.path.join(y_folder, "*_load2.pkl"))
    gnd_pkl_files = glob.glob(os.path.join(y_folder, "*_gnd.pkl"))

    if not load_pkl_files or not gnd_pkl_files:
        print(f"Missing .pkl files in {y_folder}")
        continue

    load_pkl_file = load_pkl_files[0]
    gnd_pkl_file = gnd_pkl_files[0]

    data_load = load_data(load_pkl_file)
    data_gnd = load_data(gnd_pkl_file)

    x_array = data_load['rawdata']['stage_laser_xposition']
    mask = (x_array >= x_min_plot) & (x_array <= x_max_plot)
    x_array_filtered = x_array[mask]

    if len(x_array_filtered) == 0:
        print(f"No data in X range for {y_folder}")
        continue

    # ------------------------------------------------------------------
    #  Custom tick formatter: 0 → "0" , everything else → 1 decimal
    # ------------------------------------------------------------------
    from matplotlib.ticker import FormatStrFormatter, FuncFormatter
    def y_formatter(val, pos):
        return '0' if val == 0 else f'{val:.4f}'
    y_fmt = FuncFormatter(y_formatter)

    # ------------------------------------------------------------------
    #  Custom tick formatter: 0 QE
    # ------------------------------------------------------------------
    from matplotlib.ticker import FormatStrFormatter, FuncFormatter
    def y_formatter(val, pos):
        return '0' if val == 0 else f'{val:.1f}'
    y_fmt_qe = FuncFormatter(y_formatter)
    # =============================================================
    # LOAD PLOT – QUADRANTS B & C  (with GAP OVERSHOOT)
    # =============================================================
    fig_load, ax_load = plt.subplots(figsize=(10, 6), layout='constrained')

    quadB_mean_load = data_load['rawdata'].get('quadB', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)[mask]
    quadC_mean_load = data_load['rawdata'].get('quadC', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)[mask]

    if quadB_mean_load.size == 0:
        quadB_mean_load = np.zeros_like(x_array_filtered)
        print(f"Warning: quadB missing in LOAD {load_pkl_file}")
    if quadC_mean_load.size == 0:
        quadC_mean_load = np.zeros_like(x_array_filtered)
        print(f"Warning: quadC missing in LOAD {load_pkl_file}")

    I_sum_load = quadB_mean_load + quadC_mean_load

    ax_load.plot(x_array_filtered, quadB_mean_load, label='quadB', color='tab:orange')
    ax_load.plot(x_array_filtered, quadC_mean_load, label='quadC', color='tab:green')
    ax_load.plot(x_array_filtered, I_sum_load, '--', color="#992f7f", linewidth=1.5, label='quadB+quadC', alpha=1.0)

    # Reference lines
    ax_load.axhline(I_100, color="#c39d7a", linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 1\,$A/W')
    ax_load.axhline(I_real_mean, color='#8b5a2b', linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 0.8\,$A/W')

    # === QUANTUM EFFICIENCY (BROWN) ===
    QE_load = I_sum_load / P_combined
    ax_qe_load = ax_load.twinx()
    ax_qe_load.plot(x_array_filtered, QE_load, color="#44fd00", linewidth=2, label=r'$\eta_{\rm{LOAD}}$ [A/W]', alpha=0.2)
    ax_qe_load.set_ylabel(r'Quantum Efficiency $\eta$ [A/W]', color='#8b5a2b', fontsize=14)
    ax_qe_load.tick_params(axis='y', which='major', color='#8b5a2b', length=8, width=1.5, direction='in')
    ax_qe_load.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
    for tick in ax_qe_load.get_yticklabels():
        tick.set_color('#8b5a2b')

    # Force zero alignment
    ax_load.set_ylim(bottom=0)
    ax_qe_load.set_ylim(bottom=0, top=1.05)
    align_yaxis_zeros(ax_load, ax_qe_load)
    ax_qe_load.yaxis.set_major_formatter(y_fmt_qe)

    # ==================== GAP OVERSHOOT ANALYSIS (LOAD) ====================
    left_region  = (x_min_plot, 7.22)   # adjust slightly if your gap is shifted
    right_region = (7.32, x_max_plot)
    gap_region   = (7.22, 7.32)

    left_mask  = (x_array_filtered >= left_region[0])  & (x_array_filtered <= left_region[1])
    right_mask = (x_array_filtered >= right_region[0]) & (x_array_filtered <= right_region[1])
    gap_mask   = (x_array_filtered >= gap_region[0])   & (x_array_filtered <= gap_region[1])

    mean_left  = np.mean(I_sum_load[left_mask])  if np.any(left_mask)  else 0
    mean_right = np.mean(I_sum_load[right_mask]) if np.any(right_mask) else 0
    mean_plateaus = (mean_left + mean_right) / 2.0

    max_in_gap = np.max(I_sum_load[gap_mask]) if np.any(gap_mask) else 0
    idx_max = np.argmax(I_sum_load[gap_mask])
    x_at_max = x_array_filtered[gap_mask][idx_max] if np.any(gap_mask) else np.mean(x_array_filtered)

    overshoot_percent = (max_in_gap / mean_plateaus - 1.0) * 100.0 if mean_plateaus > 0 else 0

    # Marker at maximum in gap
    ax_load.plot(x_at_max, max_in_gap, 'o', color='#d62728', mec='darkred', mew=1.8, ms=10, zorder=10)

    # Result box
    inset_load = ax_load.inset_axes([0.32, 0.54, 0.38, 0.16], transform=ax_load.transAxes)
    inset_load.axis('off')
    legend_handle = Line2D([0], [0], color='#992f7f', lw=2.2, ls='--',
                        marker='o', markerfacecolor='#d62728',
                        markeredgecolor='darkred', markeredgewidth=1.8, markersize=10)
    inset_load.legend(handles=[legend_handle],
                    labels=[rf'Gap overshoot: \textbf{{{overshoot_percent:+.2f}\%}}'],
                    loc='center', fontsize=18, frameon=False,
                    handletextpad=0.8, handlelength=1.8)

    # ==================== LEGEND & FINALIZE ====================
    lines1, labels1 = ax_load.get_legend_handles_labels()
    lines2, labels2 = ax_qe_load.get_legend_handles_labels()
    ax_load.legend(lines1 + lines2, labels1 + labels2,
                fontsize=12, loc='upper right', bbox_to_anchor=(0.98, 0.98),
                frameon=True, fancybox=False, edgecolor='black')

    ax_load.set_title(rf'\textbf{{DC Photocurrent LOAD (Y={y_um}\textmu{{m}})}}', 
                    fontsize=14, fontweight='bold', pad=10)
    ax_load.set_xlabel(r'X Position [mm]', fontsize=14)
    ax_load.set_ylabel(r'Photocurrent [A]', fontsize=14)
    ax_load.grid(True, linestyle='--', alpha=0.6)
    ax_load.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5, direction='in')
    ax_load.yaxis.set_major_formatter(y_fmt)
    ax_load.set_xlim(x_min_plot + 0.005, x_max_plot - 0.005)

    fig_name_load = f"DC_Photocurrent_Y{y_um:04d}um_Quadrants_BC_GapOvershoot_LOAD_results"
    fig_load.savefig(os.path.join(fig_dir, f"{fig_name_load}.png"), dpi=300, bbox_inches='tight')
    print(f"LOAD (B+C + overshoot) saved: {os.path.join(fig_dir, f'{fig_name_load}.png')}")
    plt.show()
    plt.close(fig_load)


    # =============================================================
    # GND PLOT – QUADRANTS B & C  (with GAP OVERSHOOT)
    # =============================================================
    fig_gnd, ax_gnd = plt.subplots(figsize=(10, 6), layout='constrained')

    quadB_mean_gnd = data_gnd['rawdata'].get('quadB', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)[mask]
    quadC_mean_gnd = data_gnd['rawdata'].get('quadC', {}).get('dmm00_curr_amp', np.zeros((len(x_array), 1))).mean(axis=1)[mask]

    if quadB_mean_gnd.size == 0:
        quadB_mean_gnd = np.zeros_like(x_array_filtered)
    if quadC_mean_gnd.size == 0:
        quadC_mean_gnd = np.zeros_like(x_array_filtered)

    I_sum_gnd = quadB_mean_gnd + quadC_mean_gnd

    ax_gnd.plot(x_array_filtered, quadB_mean_gnd, label='quadB', color='tab:orange')
    ax_gnd.plot(x_array_filtered, quadC_mean_gnd, label='quadC', color='tab:green')
    ax_gnd.plot(x_array_filtered, I_sum_gnd, '--', color="#992f7f", linewidth=1.7, alpha=0.6, label='quadB+quadC')

    ax_gnd.axhline(I_100, color='#d2b48c', linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 1\,$A/W')
    ax_gnd.axhline(I_real_mean, color='#8b5a2b', linestyle='--', linewidth=1.8, alpha=0.95, label=r'$\eta = 0.8\,$A/W')

    # QE GND
    QE_gnd = I_sum_gnd / P_combined
    ax_qe_gnd = ax_gnd.twinx()
    ax_qe_gnd.plot(x_array_filtered, QE_gnd, color="#00ff88", linewidth=2, alpha=0.2, label=r'$\eta_{\rm{GND}}$ [A/W]')
    ax_qe_gnd.set_ylabel(r'Quantum Efficiency $\eta$ [A/W]', color='#8b5a2b', fontsize=14)
    ax_qe_gnd.tick_params(axis='y', which='major', color='#8b5a2b', length=8, width=1.5, direction='in')
    ax_qe_gnd.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
    for tick in ax_qe_gnd.get_yticklabels():
        tick.set_color('#8b5a2b')

    ax_gnd.set_ylim(bottom=0)
    ax_qe_gnd.set_ylim(bottom=0, top=1.05)
    align_yaxis_zeros(ax_gnd, ax_qe_gnd)
    ax_qe_gnd.yaxis.set_major_formatter(y_fmt_qe)

    # ==================== GAP OVERSHOOT ANALYSIS (GND) ====================
    mean_left_g  = np.mean(I_sum_gnd[left_mask])  if np.any(left_mask)  else 0
    mean_right_g = np.mean(I_sum_gnd[right_mask]) if np.any(right_mask) else 0
    mean_plateaus_g = (mean_left_g + mean_right_g) / 2.0

    max_in_gap_g = np.max(I_sum_gnd[gap_mask]) if np.any(gap_mask) else 0
    idx_max_g = np.argmax(I_sum_gnd[gap_mask])
    x_at_max_g = x_array_filtered[gap_mask][idx_max_g] if np.any(gap_mask) else np.mean(x_array_filtered)

    overshoot_percent_g = (max_in_gap_g / mean_plateaus_g - 1.0) * 100.0 if mean_plateaus_g > 0 else 0

    # Marker
    ax_gnd.plot(x_at_max_g, max_in_gap_g, 'o', color='#ff7f0e', mec='#cc5e00', mew=1.8, ms=10, zorder=10)

    # Result box (orange version like in A+D GND)
    inset_gnd = ax_gnd.inset_axes([0.32, 0.54, 0.38, 0.16], transform=ax_gnd.transAxes)
    inset_gnd.axis('off')
    legend_handle_g = Line2D([0], [0], color=(*plt.cm.colors.to_rgb('#992f7f'), 0.6), lw=2, ls='--',
                            marker='o', markerfacecolor='#ff7f0e',
                            markeredgecolor='#cc5e00', markeredgewidth=1.8, markersize=10)
    inset_gnd.legend(handles=[legend_handle_g],
                    labels=[rf'Gap overshoot: \textbf{{{overshoot_percent_g:+.2f}\%}}'],
                    loc='center', fontsize=18, frameon=False,
                    handletextpad=0.8, handlelength=1.8)

    # Legend & finalize
    lines1g, labels1g = ax_gnd.get_legend_handles_labels()
    lines2g, labels2g = ax_qe_gnd.get_legend_handles_labels()
    ax_gnd.legend(lines1g + lines2g, labels1g + labels2g,
                fontsize=12, loc='upper right', bbox_to_anchor=(0.98, 0.98),
                frameon=True, fancybox=False, edgecolor='black')

    ax_gnd.set_title(rf'\textbf{{DC Photocurrent GND (Y={y_um}\textmu{{m}})}}',
                    fontsize=14, fontweight='bold', pad=10)
    ax_gnd.set_xlabel(r'X Position [mm]', fontsize=14)
    ax_gnd.set_ylabel(r'Photocurrent [A]', fontsize=14)
    ax_gnd.grid(True, linestyle='--', alpha=0.6)
    ax_gnd.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5, direction='in')
    ax_gnd.yaxis.set_major_formatter(y_fmt)
    ax_gnd.set_xlim(x_min_plot + 0.005, x_max_plot - 0.005)

    fig_name_gnd = f"DC_Photocurrent_Y{y_um:04d}um_Quadrants_BC_GapOvershoot_GND_results"
    fig_gnd.savefig(os.path.join(fig_dir, f"{fig_name_gnd}.png"), dpi=300, bbox_inches='tight')
    print(f"GND (B+C + overshoot) saved: {os.path.join(fig_dir, f'{fig_name_gnd}.png')}")
    plt.show()
    plt.close(fig_gnd)

    # Console summary
    print(f"\n=== GAP OVERSHOOT SUMMARY Y = {y_um} µm (B+C) ===")
    print(f"LOAD  → overshoot = {overshoot_percent:+.2f} %")
    print(f"GROUND → overshoot = {overshoot_percent_g:+.2f} %")