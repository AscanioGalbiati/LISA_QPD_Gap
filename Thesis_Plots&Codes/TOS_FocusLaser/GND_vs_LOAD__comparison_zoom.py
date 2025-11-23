'''
Created on Nov 22 2025

@author: A_Galbiati

Zoom in plot of the termination comparison (GND vs LOAD) in the gap region.

Compatible data: 
final_data/20251027/VIGO_NS089008_QPD_750_20_AS_0011_CC_objLens_quadABCD_manual_setup_GND_LOAD_XY/HorizontalScan

Author: Ascanio Galbiati
'''
import os, glob, pickle, numpy as np, matplotlib.pyplot as plt, re, matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
fm.FontProperties(fname="/Users/asca/Library/Fonts/cmunrm.ttf")
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

base_dir = "/Users/asca/Documents/University/Master Thesis/code/Data/all quadrants/20251027/VIGO_NS089008_QPD_750_20_AS_0011_CC_objLens_quadABCD_manual_setup_GND_LOAD_XY/HorizontalScan"
zoom_left, zoom_right = 7.17, 7.38
gap_left, gap_right = 7.20, 7.35

def y_pos(f):
    m = re.search(r'Y(\d+)um', f); return int(m.group(1)) if m else None

def extract_sum(data, mask):
    raw = data["rawdata"]
    def get(key):
        arr = raw.get(key, {}).get("dmm00_curr_amp", None)
        if arr is None: return np.zeros(len(raw["stage_laser_xposition"]))
        if arr.ndim == 1: arr = arr.reshape(-1,1)
        return arr.mean(axis=1)
    return get("quadA")[mask] + get("quadD")[mask]

for y_um, folder in sorted([(y_pos(os.path.basename(f)), f)
                           for f in glob.glob(os.path.join(base_dir,"Y*um"))
                           if y_pos(os.path.basename(f))]):
    fig_dir = os.path.join(folder,"fig"); os.makedirs(fig_dir, exist_ok=True)
    load_file = glob.glob(os.path.join(folder,"*_load2.pkl"))[0]
    gnd_file = glob.glob(os.path.join(folder,"*_gnd.pkl"))[0]
    x = pickle.load(open(load_file,"rb"))['rawdata']['stage_laser_xposition']
    mask = (x >= zoom_left) & (x <= zoom_right)
    xf = x[mask]
    I_load = extract_sum(pickle.load(open(load_file,"rb")), mask)
    I_gnd = extract_sum(pickle.load(open(gnd_file,"rb")), mask)
    gap_mask = (xf >= gap_left) & (xf <= gap_right)
    plat_mask = np.logical_or(xf < gap_left, xf > gap_right)
    max_load, x_load = np.max(I_load[gap_mask]), xf[gap_mask][np.argmax(I_load[gap_mask])]
    max_gnd, x_gnd = np.max(I_gnd[gap_mask]), xf[gap_mask][np.argmax(I_gnd[gap_mask])]
    plat_l = np.mean(I_load[plat_mask])
    plat_g = np.mean(I_gnd[plat_mask])
    ov_l = (max_load/plat_l - 1)*100
    ov_g = (max_gnd/plat_g - 1)*100

    fig, ax = plt.subplots(figsize=(9.5, 5.8), layout='constrained')

    line_load, = ax.plot(xf, I_load, color='#992f7f', lw=3.2, ls='--', label='quadA+quadD (LOAD)')
    line_gnd,  = ax.plot(xf, I_gnd,  color='#992f7f', lw=3.2, ls='--', alpha=0.6, label='quadA+quadD (GND)')

    ax.axvspan(gap_left, gap_right, color='gray', alpha=0.22) #label='Gap region'

    ax.plot(x_load, max_load, 'o', color='#d62728', mec='darkred', mew=2.2, ms=12, zorder=10)
    ax.plot(x_gnd,  max_gnd,  'o', color='#ff7f0e', mec='#cc5e00', mew=2.2, ms=12, zorder=10)

    # RESULT BOX — EXACTLY LIKE THE REFERENCE (line + dot combined)
    inset = ax.inset_axes([0.28, 0.655, 0.42, 0.20], transform=ax.transAxes)
    inset.axis('off')
    load_handle = Line2D([0], [0], color='#992f7f', lw=3.2, ls='--',
                         marker='o', markerfacecolor='#d62728',
                         markeredgecolor='darkred', markeredgewidth=2.2, markersize=12)
    gnd_handle = Line2D([0], [0],
                    color=(*plt.cm.colors.to_rgb('#992f7f'), 0.6),  # line with alpha
                    lw=2, ls='--',
                    marker='o',
                    markerfacecolor='#ff7f0e',      # full opacity
                    markeredgecolor='#cc5e00',
                    markeredgewidth=1.8,
                    markersize=12)
    inset.legend(handles=[load_handle, gnd_handle],
                 labels=[rf'Gap overshoot (LOAD): \textbf{{{ov_l:+.2f}\%}}',
                         rf'Gap overshoot (GND):  \textbf{{{ov_g:+.2f}\%}}'],
                 loc='center', fontsize=18, frameon=False,
                 handletextpad=1.2, labelspacing=1.4)

    ax.set_title(r'\textbf{Gap region zoom}', fontsize=15, pad=10)
    ax.set_xlabel('X Position [mm]', fontsize=14)
    ax.set_ylabel('Photocurrent [A]', fontsize=14)
    ax.grid(True, ls='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=11, length=6, width=1.5, direction='in')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p: '0' if v==0 else f'{v:.4f}'))
    ax.set_ylim(bottom=0.0082)
    ax.set_ylim(top = 1.105 * np.max(np.concatenate([I_load, I_gnd])))  # centers the traces beautifully
    ax.set_xlim(zoom_left+0.01, zoom_right-0.01)

    ax.legend(loc='upper right', fontsize=13, frameon=True, fancybox=False, edgecolor='black')

    save = os.path.join(fig_dir, f"ZOOM_Gap_Comparison_{y_um:04d}um.png")
    fig.savefig(save, dpi=400, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"Zoom saved to {save}")
