'''
@author: A_Galbiati

Analyzes QPD horizontal scans for comparison of GND and LOAD terminations.
Plotting photocurrent for quadA/quadD (LOAD/GND) with summed A+D traces, dual η curves (LOAD/GND).
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

# Power
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

def align_yaxis_zeros(ax1, ax2):
    y1l, y1h = ax1.get_ylim()
    if y1h == y1l: return
    zero_frac = -y1l / (y1h - y1l)
    y2l, y2h = ax2.get_ylim()
    # Adjusted y-axis alignment formula for the twinx axis
    y2_range = y2h - y2l
    ax2.set_ylim(0 - zero_frac * y2_range, 0 - zero_frac * y2_range + y2_range)

def extract_quad_current(data, quad_key, mask):
    arr = data["rawdata"].get(quad_key, {}).get("dmm00_curr_amp", None)
    if arr is None: return np.zeros_like(data["rawdata"]["stage_laser_xposition"])[mask]
    if arr.ndim == 1: arr = arr.reshape(-1, 1)
    return arr.mean(axis=1)[mask]

# ==================== PROCESSING DATA ====================
y_folders = sorted([(extract_y(os.path.basename(f)), f)
                    for f in glob.glob(os.path.join(base_dir, "Y*um"))
                    if extract_y(os.path.basename(f))])

for y_um, y_folder in y_folders:
    print(f"Y = {y_um} $\\mu$m")
    # fig_dir = os.path.join(y_folder, "fig_GND_LOAD") # Original line
    # os.makedirs(fig_dir, exist_ok=True) # Original line
    fig_dir = FINAL_FIG_DIR # Use the globally defined final directory

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

    # ------------------------------------------------------------------
    #  Custom tick formatter: 0 QE
    # ------------------------------------------------------------------
    from matplotlib.ticker import FormatStrFormatter, FuncFormatter
    def y_formatter(val, pos):
        return '0' if val == 0 else f'{val:.1f}'
    y_fmt_qe = FuncFormatter(y_formatter)

    # ==================== PLOT ====================
    fig, ax = plt.subplots(figsize=(10,6), layout='constrained')

    # Individual quads (no legend) (Increased line width: 1.5)
    ax.plot(x_f, A_load, label='quadA', color='tab:blue',  lw=1.5, alpha=0.8)
    ax.plot(x_f, D_load, label='quadD', color='tab:red',   lw=1.5, alpha=0.8)
    ax.plot(x_f, A_gnd,  color='tab:blue',  lw=1.5, alpha=0.5)
    ax.plot(x_f, D_gnd,  color='tab:red',   lw=1.5, alpha=0.5)

    # Purple sums (Increased line width: 2.2)
    line_gnd, = ax.plot(x_f, sum_gnd, '--', color='#992f7f', lw=2.2, label='quadA+quadD (LOAD)')
    line_load,  = ax.plot(x_f, sum_load,  '--', color='#992f7f', lw=2.2, alpha=0.6, label='quadA+quadD (GND)')

    # Regions
    ax.axvspan(*left_region,  color='tab:blue', alpha=0.12, label='Left region')
    ax.axvspan(*right_region, color='tab:red',   alpha=0.12, label='Right region')
    ax.axvspan(*gap_region,   color='gray',     alpha=0.25, label='Gap region')

    # Reference lines (Kept original lw=1.8)
    ax.axhline(I_eta08, color='#8b5a2b', ls='--', lw=1.8, alpha=0.95, label=r'$\eta = 0.8\,$A/W')
    ax.axhline(I_eta10, color='#c39d7a', ls='--', lw=1.8, alpha=0.95, label=r'$\eta = 1.0\,$A/W')

    # Overshoot markers (Kept original ms=10, mew=1.8)
    ax.plot(x_g, max_g, 'o', color='#d62728', mec='darkred', mew=1.8, ms=10, zorder=10)
    ax.plot(x_l, max_l, 'o', color='#ff7f0e', mec='#cc5e00', mew=1.8, ms=10, zorder=10)

    # TWO ETA CURVES
    ax_qe = ax.twinx()
    #eta_load_line, = ax_qe.plot(x_f, sum_load / P_combined, color="#44fd00", lw=2.0, alpha=0.3, label=r'$\eta_{\mathrm{LOAD}}$ (A/W)')   # \mathrm{} = safe
    #eta_gnd_line,  = ax_qe.plot(x_f, sum_gnd  / P_combined, color="#00ff88", lw=2.0, ls='--', alpha=0.3,label=r'$\eta_{\mathrm{GND}}$ (A/W)')

    # QE axis (brown) (Increased font size: 16)
    ax_qe.set_ylabel(r'Responsivity $\eta$ [A/W]', color='#8b5a2b', fontsize=16)
    # Increased tick label size: 13
    ax_qe.tick_params(axis='y', which='major', color='#8b5a2b', length=8, width=1.5, direction='in', labelsize=13)
    for tick in ax_qe.get_yticklabels():
        tick.set_color('#8b5a2b')
    ax_qe.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f'{y:.2f}'))

    ax.set_ylim(bottom=0)
    ax_qe.set_ylim(bottom=0, top=1.05)
    ax_qe.yaxis.set_major_formatter(y_fmt_qe)
    align_yaxis_zeros(ax, ax_qe)

    # RESULT BOX — LINE + DOT COMBINED IN ONE HANDLE 
    # Adjusted position: [0.28, 0.70, 0.42, 0.20]
    inset = ax.inset_axes([0.28, 0.52, 0.42, 0.20], transform=ax.transAxes)
    inset.axis('off')

    # LOAD: thick purple line + red filled dot
    # Increased line width: 2.2
    # Increased marker size: 12 (LOAD), 11 (GND)
    gnd_handle = Line2D([0], [0],
                         color='#992f7f', lw=2.2, ls='--',
                         marker='o', markerfacecolor='#d62728',
                         markeredgecolor='darkred', markeredgewidth=1.8,
                         markersize=12)

    # GND: same purple line (with alpha) + orange filled dot
    load_handle = Line2D([0], [0],
                    color=(*plt.cm.colors.to_rgb('#992f7f'), 0.6),  # line with alpha
                    lw=2.2, ls='--',
                    marker='o',
                    markerfacecolor='#ff7f0e',      # full opacity
                    markeredgecolor='#cc5e00',
                    markeredgewidth=1.8,
                    markersize=12)

    # Increased font size: 22
    inset.legend(handles=[load_handle, gnd_handle],
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

    # Legend 
    handles, labels = ax.get_legend_handles_labels()
    #handles += [eta_load_line, eta_gnd_line]
    labels  += [r'$\eta (LOAD$)(A/W)', r'$\eta (GND)$ (A/W)']
    # Increased font size: 14
    ax.legend(handles, labels,
              loc='upper right', fontsize=12, bbox_to_anchor=(0.99,0.98),
              frameon=True, fancybox=False, edgecolor='black')

    save_name = f"GND_vs_LOAD_Y{y_um:04d}um.png"
    fig.savefig(os.path.join(fig_dir, save_name), dpi=400, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f" Saved: {save_name} | LOAD {ov_l:+.2f}% | GND {ov_g:+.2f}%")