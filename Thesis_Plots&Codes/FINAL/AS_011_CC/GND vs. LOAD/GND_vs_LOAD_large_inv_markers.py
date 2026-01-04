'''
@author: A_Galbiati

Analyzes QPD horizontal scans for comparison of GND and LOAD terminations.
Plotting data points for quadA/quadD (LOAD/GND) with summed A+D traces (marker and thin line).
Highlights plateau/gap regions used to compute overshoot percentage.

Compatible data: 
20251027/VIGO_NS089008_QPD_750_20_AS_0011_CC_objLens_quadABCD_manual_setup_GND_LOAD_XY/HorizontalScan

Author: Ascanio Galbiati
'''
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches

# ==================== FONT & STYLE ====================
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
fm.FontProperties(fname=font_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

# ==================== CONFIG ====================
base_dir = "/Users/asca/Documents/University/Master Thesis/code/Data/20251215_final_data/20251027/VIGO_NS089008_QPD_750_20_AS_0011_CC_objLens_quadABCD_manual_setup_GND_LOAD_XY/HorizontalScan"
x_min_plot = 7.10
x_max_plot = 7.45
left_region = (x_min_plot, 7.20)
gap_region = (7.20, 7.35)
right_region = (7.35, x_max_plot)

# Power (Kept for calculating overshoot)
P1, sigma1 = 0.019158, 1.2336e-5
P2, sigma2 = 0.019151, 1.1417e-5
w1, w2 = 1/sigma1**2, 1/sigma2**2
P_combined = (P1*w1 + P2*w2)/(w1+w2)
I_eta08 = 0.8 * P_combined
I_eta10 = 1.0 * P_combined

# === FIGURE SAVE PATH ===
FINAL_FIG_DIR = '/Users/asca/Documents/University/Master Thesis/code/LISA_QPD_Gap/Thesis_Plots&Codes/FINAL/AS_011_CC/GND vs. LOAD/fig'
os.makedirs(FINAL_FIG_DIR, exist_ok=True)


# ==================== FUNCTIONS ====================
def load_data(p):
    with open(p, "rb") as f: return pickle.load(f)

def extract_y(folder):
    m = re.search(r'Y(\d+)um', folder)
    return int(m.group(1)) if m else None

def extract_quad_current(data, quad_key, mask):
    arr = data["rawdata"].get(quad_key, {}).get("dmm00_curr_amp", None)
    if arr is None: return np.zeros_like(data["rawdata"]["stage_laser_xposition"])[mask]
    if arr.ndim == 1: arr = arr.reshape(-1, 1)
    return arr.mean(axis=1)[mask]

# ==================== PROCESSING DATA ====================
y_folders = sorted([(extract_y(os.path.basename(f)), f)
                    for f in glob.glob(os.path.join(base_dir, "Y*um"))
                    if extract_y(os.path.basename(f))])

# === PLOT CONFIGURATION ===
QUAD_MARKER = 'o'
QUAD_MS_INDIVIDUAL = 3    # Smaller marker size for individual quads
QUAD_MS_SUM = 3.5           # Slightly larger marker size for sums
QUAD_LW_INDIVIDUAL = 0.5  # Very small line width
QUAD_LW_SUM = 0.5         # Small line width for sums
Y_MAX_PLOT = 0.0145 
# Adjusted result box position (was 0.52, now 0.60)
INSET_POSITION = [0.28, 0.72, 0.42, 0.20] 

for y_um, y_folder in y_folders:
    print(f"Y = {y_um} $\\mu$m")
    fig_dir = FINAL_FIG_DIR 
    os.makedirs(fig_dir, exist_ok=True)

    load_file = glob.glob(os.path.join(y_folder, "*_load2.pkl"))[0]
    gnd_file  = glob.glob(os.path.join(y_folder, "*_gnd.pkl"))[0]
    data_load = load_data(load_file)
    data_gnd  = load_data(gnd_file)

    x = data_load['rawdata']['stage_laser_xposition']
    mask = (x >= x_min_plot) & (x <= x_max_plot)
    x_f = x[mask]

    A_load = extract_quad_current(data_load, "quadA", mask)
    D_load = extract_quad_current(data_load, "quadD", mask)
    A_gnd  = extract_quad_current(data_gnd,  "quadA", mask)
    D_gnd  = extract_quad_current(data_gnd,  "quadD", mask)
    sum_load = A_load + D_load
    sum_gnd  = A_gnd  + D_gnd

    left_mask  = (x_f >= left_region[0])  & (x_f <= left_region[1])
    right_mask = (x_f >= right_region[0]) & (x_f <= right_region[1])
    gap_mask   = (x_f >= gap_region[0])   & (x_f <= gap_region[1])

    def get_overshoot(I_sum):
        plat = (np.mean(I_sum[left_mask]) + np.mean(I_sum[right_mask])) / 2
        mx   = np.max(I_sum[gap_mask])
        return plat, mx, (mx/plat - 1)*100, x_f[gap_mask][np.argmax(I_sum[gap_mask])]

    _, max_l, ov_l, x_l = get_overshoot(sum_load)
    _, max_g, ov_g, x_g = get_overshoot(sum_gnd)

    def curr_fmt(val, pos): return '0' if val == 0 else f'{val:.4f}'
    y_fmt = FuncFormatter(curr_fmt)

    # ==================== PLOT ====================
    fig, ax = plt.subplots(figsize=(10,6), layout='constrained')

    # Individual quads: circle marker, thin line
    ax.plot(x_f, A_load, label='quadA', color='tab:blue',  lw=QUAD_LW_INDIVIDUAL, marker=QUAD_MARKER, ms=QUAD_MS_INDIVIDUAL, alpha=0.8)
    ax.plot(x_f, D_load, label='quadD', color='tab:red',   lw=QUAD_LW_INDIVIDUAL, marker=QUAD_MARKER, ms=QUAD_MS_INDIVIDUAL, alpha=0.8)
    ax.plot(x_f, A_gnd,  label='quadA (GND)', color='tab:blue',  lw=QUAD_LW_INDIVIDUAL, marker=QUAD_MARKER, ms=QUAD_MS_INDIVIDUAL, alpha=0.5)
    ax.plot(x_f, D_gnd,  label='quadD (GND)', color='tab:red',   lw=QUAD_LW_INDIVIDUAL, marker=QUAD_MARKER, ms=QUAD_MS_INDIVIDUAL, alpha=0.5)

    # Purple sums: circle marker, thin line, slightly larger marker
    ax.plot(x_f, sum_gnd, color='#992f7f', lw=QUAD_LW_SUM, marker=QUAD_MARKER, ms=QUAD_MS_SUM, label='quadA+quadD (LOAD)')
    ax.plot(x_f, sum_load,  color='#992f7f', lw=QUAD_LW_SUM, marker=QUAD_MARKER, ms=QUAD_MS_SUM, alpha=0.6, label='quadA+quadD (GND)')

    # Regions
    ax.axvspan(*left_region,  color='tab:blue', alpha=0.12, label='Left region')
    ax.axvspan(*right_region, color='tab:red',   alpha=0.12, label='Right region')
    ax.axvspan(*gap_region,   color='gray',     alpha=0.25, label='Gap region')

    # Overshoot markers (Kept original ms=10, mew=1.8 for the distinct max marker)
    ax.plot(x_g, max_g, 'o', color='#d62728', mec='darkred', mew=1.8, ms=10, zorder=10)
    ax.plot(x_l, max_l, 'o', color='#ff7f0e', mec='#cc5e00', mew=1.8, ms=10, zorder=10)

    # Set Y-limit
    ax.set_ylim(bottom=0, top=Y_MAX_PLOT)

    # RESULT BOX (Overshoot) - Position raised
    inset = ax.inset_axes(INSET_POSITION, transform=ax.transAxes)
    inset.axis('off')

    # Load/GND Overshoot handles: Use the distinct circle marker
    load_handle_max = Line2D([0], [0], marker='o', markerfacecolor='#ff7f0e', markeredgecolor='#cc5e00', mew=1.8, markersize=10, ls='none')
    gnd_handle_max = Line2D([0], [0], marker='o', markerfacecolor='#d62728', markeredgecolor='darkred', mew=1.8, markersize=10, ls='none')

    # Increased font size: 22
    inset.legend(handles=[load_handle_max, gnd_handle_max],
                 labels=[rf'GS (LOAD): \textbf{{{ov_l:+.2f}\%}}',
                         rf'GS (GND):  \textbf{{{ov_g:+.2f}\%}}'],
                 loc='center',
                 fontsize=19,
                 frameon=False,
                 handletextpad=1.2,
                 labelspacing=1.4)

    # Increased font size: 16
    ax.set_title(rf'\textbf{{DC Photocurrent GND vs. LOAD (HL: Y={y_um}\textmu{{}}m)}}', fontsize=16, pad=10)
    ax.set_xlabel(r'X Position [mm]', fontsize=16)
    ax.set_ylabel(r'Photocurrent [A]', fontsize=16)
    ax.grid(True, ls='--', alpha=0.6)
    # Increased tick label size: 13
    ax.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in')
    ax.yaxis.set_major_formatter(y_fmt)
    ax.set_xlim(x_min_plot+0.01, x_max_plot-0.01)

    # Main Legend 
    
    # Custom handles to correctly show marker/line style
    quad_a_load_handle = Line2D([0], [0], marker=QUAD_MARKER, markerfacecolor='tab:blue', markeredgecolor='tab:blue', markersize=QUAD_MS_INDIVIDUAL, lw=QUAD_LW_INDIVIDUAL, color='tab:blue')
    quad_d_load_handle = Line2D([0], [0], marker=QUAD_MARKER, markerfacecolor='tab:red', markeredgecolor='tab:red', markersize=QUAD_MS_INDIVIDUAL, lw=QUAD_LW_INDIVIDUAL, color='tab:red')
    # Sum handles (slightly larger markers, different alpha)
    sum_load_handle = Line2D([0], [0], marker=QUAD_MARKER, markerfacecolor=(*plt.cm.colors.to_rgb('#992f7f'), 0.6), markeredgecolor=(*plt.cm.colors.to_rgb('#992f7f'), 0.6), markersize=QUAD_MS_SUM, lw=QUAD_LW_SUM, color=(*plt.cm.colors.to_rgb('#992f7f'), 0.6))
    sum_gnd_handle = Line2D([0], [0], marker=QUAD_MARKER, markerfacecolor='#992f7f', markeredgecolor='#992f7f', markersize=QUAD_MS_SUM, lw=QUAD_LW_SUM, color='#992f7f')
    
    # Region patches for legend
    patch_l = mpatches.Patch(color='tab:blue', alpha=0.12, label='Left region')
    patch_r = mpatches.Patch(color='tab:red', alpha=0.12, label='Right region')
    patch_g = mpatches.Patch(color='gray', alpha=0.25, label='Gap region')

    legend_handles = [
        quad_a_load_handle,
        quad_d_load_handle,
        sum_load_handle,
        sum_gnd_handle,
        patch_l, 
        patch_r, 
        patch_g
    ]
    legend_labels = [
        'quadA (LOAD)',
        'quadD (LOAD)',
        'quadA+quadD (LOAD)',
        'quadA+quadD (GND)',
        'Left region', 
        'Right region', 
        'Gap region'
    ]

    # Increased font size: 14
    ax.legend(legend_handles, legend_labels,
              loc='upper right', fontsize=12, bbox_to_anchor=(0.99,0.98),
              frameon=True, fancybox=False, edgecolor='black')

    save_name = f"GND_vs_LOAD_Y{y_um:04d}um_Final.png"
    fig.savefig(os.path.join(fig_dir, save_name), dpi=400, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f" Saved: {save_name} | LOAD {ov_l:+.2f}% | GND {ov_g:+.2f}%")