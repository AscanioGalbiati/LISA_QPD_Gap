'''
@author: A_Galbiati

Zoom in plot of the termination comparison (GND vs LOAD) in the gap region,
using the plateau definition from the wider scan for accurate overshoot calculation.

Compatible data: 
20251027/VIGO_NS089008_QPD_750_20_AS_0011_CC_objLens_quadABCD_manual_setup_GND_LOAD_XY/HorizontalScan

Author: Ascanio Galbiati
'''
import os, glob, pickle, numpy as np, matplotlib.pyplot as plt, re, matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# ==================== FONT & STYLE ====================
# NOTE: Ensure this font path is correct on your system!
fm.FontProperties(fname="/Users/asca/Library/Fonts/cmunrm.ttf")
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

# ==================== CONFIG ====================
base_dir = "/Users/asca/Documents/University/Master Thesis/code/Data/20251215_final_data/20251027/VIGO_NS089008_QPD_750_20_AS_0011_CC_objLens_quadABCD_manual_setup_GND_LOAD_XY/HorizontalScan"

# Zoomed plot range (used for plotting the gap region)
zoom_left, zoom_right = 7.17, 7.38
gap_left, gap_right = 7.20, 7.35

# Plateau regions (used for CORRECT overshoot calculation, taken from Code 1)
x_min_plot = 7.10
x_max_plot = 7.45
left_region = (x_min_plot, 7.20)
right_region = (7.35, x_max_plot)

# === FIGURE SAVE PATH ===
FINAL_FIG_DIR = '/Users/asca/Documents/University/Master Thesis/code/LISA_QPD_Gap/Thesis_Plots&Codes/FINAL/AS_011_CC/GND vs. LOAD/fig'
os.makedirs(FINAL_FIG_DIR, exist_ok=True)


# ==================== FUNCTIONS ====================
def y_pos(f):
    m = re.search(r'Y(\d+)um', f); return int(m.group(1)) if m else None

def extract_sum(data, mask):
    raw = data["rawdata"]
    def get(key):
        arr = raw.get(key, {}).get("dmm00_curr_amp", None)
        if arr is None: return np.zeros(len(raw["stage_laser_xposition"]))
        if arr.ndim == 1: arr = arr.reshape(-1,1)
        return arr.mean(axis=1)
    # The mask is now applied to the array of summed currents
    x_len = len(raw["stage_laser_xposition"])
    current_A = get("quadA")
    current_D = get("quadD")
    # Handle the case where the mask is for the full array size
    if len(mask) == x_len:
         return (current_A + current_D)[mask]
    # If the mask is already applied (e.g., in a recursive call), return the sum
    return current_A + current_D 

def load_data(p):
    with open(p, "rb") as f: return pickle.load(f)

# ==================== PROCESSING DATA ====================
for y_um, folder in sorted([(y_pos(os.path.basename(f)), f)
                           for f in glob.glob(os.path.join(base_dir,"Y*um"))
                           if y_pos(os.path.basename(f))]):

    # fig_dir = os.path.join(folder,"fig_GND_LOAD"); os.makedirs(fig_dir, exist_ok=True) # Original
    fig_dir = FINAL_FIG_DIR # Use the globally defined final directory
    
    load_file = glob.glob(os.path.join(folder,"*_load2.pkl"))[0]
    gnd_file = glob.glob(os.path.join(folder,"*_gnd.pkl"))[0]

    # 1. Load the full data arrays
    data_load_full = load_data(load_file)
    data_gnd_full = load_data(gnd_file)
    x_full = data_load_full['rawdata']['stage_laser_xposition']

    # Mask for the wide plateau region (Code 1's plot limits)
    mask_wide = (x_full >= x_min_plot) & (x_full <= x_max_plot)
    x_wide = x_full[mask_wide]
    I_load_wide = extract_sum(data_load_full, mask_wide)
    I_gnd_wide = extract_sum(data_gnd_full, mask_wide)

    # 2. Define the explicit left/right plateau masks on the WIDE data
    left_mask  = (x_wide >= left_region[0])  & (x_wide <= left_region[1])
    right_mask = (x_wide >= right_region[0]) & (x_wide <= right_region[1])

    # 3. Calculate the plateaus as the average of the two means (Code 1 logic)
    plat_l = (np.mean(I_load_wide[left_mask]) + np.mean(I_load_wide[right_mask])) / 2
    plat_g = (np.mean(I_gnd_wide[left_mask]) + np.mean(I_gnd_wide[right_mask])) / 2

    # 4. Apply the zoom mask for plotting and finding the gap max
    mask_zoom = (x_full >= zoom_left) & (x_full <= zoom_right)
    xf = x_full[mask_zoom]
    I_load = extract_sum(data_load_full, mask_zoom)
    I_gnd = extract_sum(data_gnd_full, mask_zoom)

    # Calculate gap maximums using the zoomed data
    gap_mask = (xf >= gap_left) & (xf <= gap_right)
    max_load, x_load = np.max(I_load[gap_mask]), xf[gap_mask][np.argmax(I_load[gap_mask])]
    max_gnd, x_gnd = np.max(I_gnd[gap_mask]), xf[gap_mask][np.argmax(I_gnd[gap_mask])]

    # 5. Calculate overshoot percentages using the corrected plateaus
    ov_l = (max_load/plat_l - 1)*100
    ov_g = (max_gnd/plat_g - 1)*100

    # ==================== PLOT ====================
    fig, ax = plt.subplots(figsize=(9.5, 5.8), layout='constrained')

    # Increased line width: 3.2
    line_gnd, = ax.plot(xf, I_gnd, color='#992f7f', lw=3.2, ls='--', label='quadA+quadD (GND)')
    line_load,  = ax.plot(xf, I_load,  color='#992f7f', lw=3.2, ls='--', alpha=0.6, label='quadA+quadD (LOAD)')

    ax.axvspan(gap_left, gap_right, color='gray', alpha=0.22, label='Gap region')

    # Increased marker edge width: 2.2; Increased marker size: 12
    ax.plot(x_gnd, max_gnd, 'o', color='#d62728', mec='darkred', mew=2.2, ms=12, zorder=10)
    ax.plot(x_load,  max_load,  'o', color='#ff7f0e', mec='#cc5e00', mew=2.2, ms=12, zorder=10)

    # RESULT BOX — EXACTLY LIKE THE REFERENCE (line + dot combined)
    # Adjusted position: [0.28, 0.70, 0.42, 0.20]
    inset = ax.inset_axes([0.28, 0.68, 0.42, 0.20], transform=ax.transAxes)
    inset.axis('off')

    # GND Handle (Thick purple line, red marker)
    # Increased line width: 3.2; Increased marker edge width: 2.2; Increased marker size: 12
    gnd_handle = Line2D([0], [0], color='#992f7f', lw=3.2, ls='--',
                         marker='o', markerfacecolor='#d62728',
                         markeredgecolor='darkred', markeredgewidth=2.2, markersize=12)
    # LOAD Handle (Thick purple line with alpha, orange marker)
    # Increased line width: 3.2; Increased marker edge width: 2.2; Increased marker size: 12
    load_handle = Line2D([0], [0],
                    color=(*plt.cm.colors.to_rgb('#992f7f'), 0.6),
                    lw=3.2, ls='--',
                    marker='o',
                    markerfacecolor='#ff7f0e',
                    markeredgecolor='#cc5e00',
                    markeredgewidth=2.2,
                    markersize=12)

    # Increased font size: 22
    inset.legend(handles=[load_handle, gnd_handle],
                 labels=[rf'GS (LOAD): \textbf{{{ov_l:+.2f}\%}}',
                         rf'GS (GND):  \textbf{{{ov_g:+.2f}\%}}'],
                 loc='center', fontsize=18.8, frameon=False,
                 handletextpad=1.2, labelspacing=1.4)

    # Increased font size: 16
    ax.set_title(rf'\textbf{{DC Photocurrent Gap Region Zoom (HL: Y={y_um}\textmu{{}}m)}}', fontsize=16, pad=10)
    # Increased font size: 16
    ax.set_xlabel(r'X Position [mm]', fontsize=16)
    ax.set_ylabel(r'Photocurrent [A]', fontsize=16)
    ax.grid(True, ls='--', alpha=0.6)
    # Increased tick label size: 13
    ax.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p: '0' if v==0 else f'{v:.4f}'))
    ax.set_ylim(bottom=0.0082) # Example bottom limit, can be made dynamic
    ax.set_ylim(top = 1.105 * np.max(np.concatenate([I_load, I_gnd])))
    ax.set_xlim(zoom_left+0.01, zoom_right-0.01)

    # Combine all handles/labels for the main legend, ensuring order is clear
    handles, labels = ax.get_legend_handles_labels()
    # Remove the axvspan label from the main legend
    handles_clean = [h for h, l in zip(handles, labels) if l != 'Gap region']
    labels_clean = [l for l in labels if l != 'Gap region']
    
    # Increased font size: 14
    ax.legend(handles_clean, labels_clean, loc='upper right', fontsize=13.2, frameon=True, fancybox=False, edgecolor='black')


    save = os.path.join(fig_dir, f"ZOOM_Gap_Comparison_Y{y_um:04d}um_Corrected.png")
    fig.savefig(save, dpi=400, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"Zoom saved to {save} | LOAD {ov_l:+.2f}% | GND {ov_g:+.2f}%")