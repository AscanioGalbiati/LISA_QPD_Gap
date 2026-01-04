'''
Focusing the laser with a telescoping system of lenses to couple into L4
A restriceted Z range is applied
'''

import os
import glob
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import curve_fit
from scipy.special import erf
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter # NECESSARY for photocurrent plot

# -------------------------
# USER SETTINGS (change if needed)
# -------------------------
# QPD directory (old pipeline)
QPD_DIR = "/Users/asca/Documents/University/Master Thesis/code/Data/20251215_final_data/20251022/Y2500um"

# Z shifts (structure supports non-zero later)
QPD_OFFSET_MM = 0    # apply to QPD z positions (was +14.62 etc.)

# QPD file Z filter (only consider files with Z >= ... mm)
# MODIFICATION: Restrict Z range to 7.88 to 8.0
QPD_Z_MIN_MM = 7.88
QPD_Z_MAX_MM = 8.0

# Warnings / thresholds
SPOT_SIZE_SANITY_UM = 20000.0   # skip absurdly large spots
# SPOT_SIZE_KE_WARN_UM = 1000.0  # (removed: not used)

# Plot output
# Using QPD_DIR as base for output now, since BP data is removed.
OUT_DIR = os.path.join(QPD_DIR, "fig_refactored_custom") # Use new output dir for clarity
os.makedirs(OUT_DIR, exist_ok=True)

# Try set serif font + LaTeX if available, otherwise continue
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
try:
    fm.FontProperties(fname=font_path)
    plt.rcParams.update({'text.usetex': True,
                         'font.family': 'serif',
                         'font.serif': ['Computer Modern']})
except Exception as e:
    print("Warning: couldn't set requested font/LaTeX; continuing with defaults.", e)

# -------------------------
# Models & helpers
# -------------------------
def erf_model(x, A, B, C, D):
    """A * erf(B*(x-C)) + D"""
    return A * erf(B * (x - C)) + D

def safe_float(s, default=None):
    try:
        return float(s)
    except Exception:
        return default

# Custom formatter for photocurrent y-axis
def tick_formatter_photocurrent(val, pos):
    return '0' if val == 0 else f'{val:.3f}'

y_fmt_photo = FuncFormatter(tick_formatter_photocurrent)

# -------------------------
# Load QPD .pkl data: quadB and quadC KE fits + Photocurrent traces
# -------------------------
print("\n--- Loading QPD .pkl data (quadB & quadC):", QPD_DIR)
def extract_z_from_name(fname):
    m = re.search(r'Z(\d+)um', fname)
    return int(m.group(1)) / 1000.0 if m else None

pkl_files = sorted(
    glob.glob(os.path.join(QPD_DIR, "*.pkl")),
    key=lambda f: extract_z_from_name(os.path.basename(f)) if extract_z_from_name(os.path.basename(f)) is not None else np.inf
)

qpd_z_shifted_B = []
qpd_w_um_B = []
qpd_z_shifted_C = []
qpd_w_um_C = []

# New structures to store raw data for photocurrent plot
qpd_data_collection_B = [] # List of tuples: (z_shift, x_arr, dc_B)
qpd_data_collection_C = [] # List of tuples: (z_shift, x_arr, dc_C)
all_unique_z_shifts = set() # For global colormap

def load_quad_dc(data, quad):
    """Return DC photocurrent averaged array for given quad label, or None"""
    try:
        return np.mean(data['rawdata'][quad]['dmm00_curr_amp'], axis=1)
    except Exception:
        return None

for pkl in pkl_files:
    base = os.path.basename(pkl)
    z_raw = extract_z_from_name(base)
    if z_raw is None:
        continue
    
    # Check Z filter against raw Z (before shift)
    if z_raw < QPD_Z_MIN_MM or z_raw > QPD_Z_MAX_MM:
        continue

    try:
        data = pickle.load(open(pkl, 'rb'))
    except Exception as e:
        print("Skipping pkl (can't load):", base, e)
        continue

    # Build x array robustly (same logic as V1)
    if 'global_params' not in data:
        print("Skipping pkl (missing global_params):", base)
        continue
    gp = data['global_params']
    if 'xstart_um' in gp:
        cur = gp['xstart_um']
        step_big = gp.get('xstep_big_um', gp.get('xstep_big', 50))
        step_fine = gp.get('xstep_fine_um', gp.get('xstep_fine', 1))
        th_s = gp.get('x_threshold_start_um', gp.get('x_threshold_start', -1e9))
        th_e = gp.get('x_threshold_stop_um', gp.get('x_threshold_stop', 1e9))
        arr = []
        while cur <= gp['xstop_um'] + 1e-9:
            arr.append(cur)
            cur += step_fine if (th_s <= cur <= th_e) else step_big
        x_arr = np.array(arr)
    elif 'xstart_mm' in gp:
        cur = gp['xstart_mm']
        step_big = gp.get('xstep_big_mm', gp.get('xstep_big', 0.1))
        step_fine = gp.get('xstep_fine_mm', gp.get('xstep_fine', 0.01))
        th_s = gp.get('x_threshold_start_mm', -1e9)
        th_e = gp.get('x_threshold_stop_mm', gp.get('x_threshold_stop_mm', 1e9)) if gp.get('x_threshold_stop_mm', None) is not None else gp.get('x_threshold_stop_mm', 1e9)
        arr = []
        while cur <= gp['xstop_mm'] + 1e-9:
            arr.append(cur)
            cur += step_fine if (th_s <= cur <= th_e) else step_big
        x_arr = np.array(arr) * 1000.0  # convert mm→µm
    else:
        print("Cannot determine x array:", base)
        continue

    # Extract quadB and quadC
    dc_B = load_quad_dc(data, 'quadB')
    dc_C = load_quad_dc(data, 'quadC')

    if dc_B is None and dc_C is None:
        print("Skipping (no quadB/quadC):", base)
        continue

    def fit_erf_spot(dc, label, base, z_raw):
        if dc is None:
            return None
        # Ensure lengths match before fit
        if len(dc) != len(x_arr):
            min_len = min(len(dc), len(x_arr))
            dc = dc[:min_len]
            x_local = x_arr[:min_len]
        else:
            x_local = x_arr
        
        if len(dc) < 4: # Minimum points needed for a fit
            print(f"Not enough points for {label} in {base}")
            return np.nan

        try:
            # Initial guess
            p0 = [np.ptp(dc), 1.0/(np.std(x_local)+1e-9), np.mean(x_local), np.min(dc)]
            params, _ = curve_fit(erf_model, x_local, dc, p0=p0,
                                bounds=([-np.inf, 1e-6, -np.inf, -np.inf],
                                        [np.inf, np.inf, np.inf, np.inf]), maxfev=5000)
            B_fit = params[1]
            spot = 1.0 / (np.sqrt(2.0) * B_fit)
            if spot > SPOT_SIZE_SANITY_UM:
                print(f"{label} in {base} (Z={z_raw:.1f}mm): huge waist {spot:.1f} µm — keeping anyway")
            return spot, x_local, dc
        except Exception as e:
            # print(f"Fit failed for {label} in {base} (Z={z_raw:.1f}mm): {e}") # Suppress error logging in loop
            return np.nan, x_local, dc

    z_shift = z_raw + QPD_OFFSET_MM
    all_unique_z_shifts.add(z_shift)

    resultB = fit_erf_spot(dc_B, 'quadB', base, z_raw)
    resultC = fit_erf_spot(dc_C, 'quadC', base, z_raw)

    # Handle quadB results
    if resultB is not None and not np.isnan(resultB[0]):
        spotB, x_local, dc_local = resultB
        qpd_z_shifted_B.append(z_shift)
        qpd_w_um_B.append(spotB)
        qpd_data_collection_B.append((z_shift, x_local, dc_local))
        print(f"quadB: Z={z_shift:.3f} mm, w={spotB:.2f} µm")
    elif resultB is not None:
         # Still collect raw data even if fit failed (using original x_arr length)
        qpd_data_collection_B.append((z_shift, resultB[1], resultB[2]))

    # Handle quadC results
    if resultC is not None and not np.isnan(resultC[0]):
        spotC, x_local, dc_local = resultC
        qpd_z_shifted_C.append(z_shift)
        qpd_w_um_C.append(spotC)
        qpd_data_collection_C.append((z_shift, x_local, dc_local))
        print(f"quadC: Z={z_shift:.3f} mm, w={spotC:.2f} µm")
    elif resultC is not None:
         # Still collect raw data even if fit failed
        qpd_data_collection_C.append((z_shift, resultC[1], resultC[2]))

# convert and sort by z
if len(qpd_z_shifted_B):
    order_B = np.argsort(qpd_z_shifted_B)
    qpd_z_shifted_B = np.array(qpd_z_shifted_B)[order_B]
    qpd_w_um_B = np.array(qpd_w_um_B)[order_B]
else:
    qpd_z_shifted_B = np.array([])
    qpd_w_um_B = np.array([])

if len(qpd_z_shifted_C):
    order_C = np.argsort(qpd_z_shifted_C)
    qpd_z_shifted_C = np.array(qpd_z_shifted_C)[order_C]
    qpd_w_um_C = np.array(qpd_w_um_C)[order_C]
else:
    qpd_z_shifted_C = np.array([])
    qpd_w_um_C = np.array([])

print(f"\nQPD quadB total: {len(qpd_w_um_B)}")
print(f"QPD quadC total: {len(qpd_w_um_C)}")

# -------------------------
# Prepare color mapping for QPD points (use combined z's for colormap)
# -------------------------
unique_z = sorted(all_unique_z_shifts)

cmap = plt.cm.RdYlBu
color_dict = {}
if unique_z:
    n_z = len(unique_z)
    for i, z in enumerate(unique_z):
        if n_z == 1:
            color_dict[z] = cmap(0.5)
        else:
            if i < n_z // 2:
                color_dict[z] = cmap(0.1 + 0.4 * (i / max(1, (n_z // 2 - 1))))
            else:
                color_dict[z] = cmap(0.6 + 0.4 * ((i - n_z // 2) / max(1, (n_z - n_z // 2 - 1))))

def color_for_z(z):
    if not unique_z:
        return 'C0'
    # Find closest Z for coloring (avoids floating point mismatch)
    z_r = min(unique_z, key=lambda u: abs(u - z))
    return color_dict.get(z_r, 'C0')

# -------------------------
# Plot 1: Beam Waist w(z) - STYLED (KE only)
# -------------------------
print("\n--- Creating Beam Waist w(z) figure ---")
fig_w, ax_w = plt.subplots(figsize=(12, 7), layout='constrained')

ax_w.set_xlabel(r'Translation Stage Z Position [mm]', fontsize=16)
ax_w.set_ylabel(r'Beam Waist $\rm{w_0}$ [\textmu{}m]', fontsize=16)
ax_w.set_title(r'\textbf{Beam Profile (DC)}', fontsize=18, fontweight='bold')
ax_w.grid(True, ls='--', alpha=0.6)
ax_w.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in')

# Plot QPD quadB & quadC
if qpd_z_shifted_B.size:
    for z_s, w in zip(qpd_z_shifted_B, qpd_w_um_B):
        ax_w.plot(z_s, w, 'o', color=color_for_z(z_s), markersize=7)
    ax_w.plot(qpd_z_shifted_B, qpd_w_um_B, '--', color='orange', lw=2.2, alpha=0.75, label=r'quadB')

if qpd_z_shifted_C.size:
    for z_s, w in zip(qpd_z_shifted_C, qpd_w_um_C):
        ax_w.plot(z_s, w, 'D', color=color_for_z(z_s), markersize=7)
    ax_w.plot(qpd_z_shifted_C, qpd_w_um_C, '-.', color='green', lw=2.2, alpha=0.75, label=r'quadC')

# Inset: minima from quadB, quadC
min_info = []
# quadB min
if qpd_w_um_B.size:
    idx_min_B = int(np.argmin(qpd_w_um_B))
    z_min_B = float(qpd_z_shifted_B[idx_min_B])
    w_min_B = float(qpd_w_um_B[idx_min_B])
    ax_w.plot(z_min_B, w_min_B, 'o', mec='red', mfc='none', mew=2.5, ms=14)
    min_info.append(("B", w_min_B, z_min_B))
# quadC min
if qpd_w_um_C.size:
    idx_min_C = int(np.argmin(qpd_w_um_C))
    z_min_C = float(qpd_z_shifted_C[idx_min_C])
    w_min_C = float(qpd_w_um_C[idx_min_C])
    ax_w.plot(z_min_C, w_min_C, 'D', mec='red', mfc='none', mew=2.5, ms=14)
    min_info.append(("C", w_min_C, z_min_C))

# Inset box with values
if min_info:
    inset = ax_w.inset_axes([0.38, 0.45, 0.45, 0.4])
    inset.axis('off')
    handles = []
    labels = []
    for tag, wv, zv in min_info:
        if tag == "B":
            marker = 'o'; mcol = 'red'; label = rf'$w_{{0,\min}}$ (KE, quadB) = ${wv:.2f}\,\mu\mathrm{{m}}$'
        elif tag == "C":
            marker = 'D'; mcol = 'red'; label = rf'$w_{{0,\min}}$ (KE, quadC) = ${wv:.2f}\,\mu\mathrm{{m}}$'
        handles.append(Line2D([0], [0], marker=marker, color='w', markeredgecolor=mcol,
                              markerfacecolor='none', markersize=10, markeredgewidth=2.0))
        labels.append(label)
    try:
        inset.legend(handles=handles, labels=labels, loc='upper left', fontsize=18, frameon=False)
    except Exception:
        txt = "\n".join(labels)
        inset.text(0, 0.9, txt, va='top', fontsize=10)

ax_w.legend(fontsize=14, loc='upper left', frameon=True, bbox_to_anchor=(0.02, 0.98))

# Axis limits
all_z_values = np.concatenate([qpd_z_shifted_B, qpd_z_shifted_C])
if all_z_values.size:
    ax_w.set_xlim(np.min(all_z_values) - 0.5, np.max(all_z_values) + 0.5)
    all_w_values = np.concatenate([qpd_w_um_B, qpd_w_um_C])
    wmax = np.max(all_w_values) if all_w_values.size else 600
    ax_w.set_ylim(bottom=0.0, top=wmax * 1.1)
else:
    # Use the restricted range for the plot limits if no data (though the plots will be empty)
    ax_w.set_xlim(QPD_Z_MIN_MM, QPD_Z_MAX_MM)
    ax_w.set_ylim(bottom=0, top=600)

out_path_w = os.path.join(OUT_DIR, "BeamWaist_KE_quadB_quadC_refactored_custom.png")
fig_w.savefig(out_path_w, dpi=300, bbox_inches='tight')
print(f"\nBeam Waist figure saved → {out_path_w}")
plt.show()
plt.close(fig_w)

# -------------------------
# Plot 2: Raw Photocurrent Traces (quadB & quadC) - STYLED
# -------------------------
print("\n--- Creating Photocurrent Traces figure ---")
fig_curr, ax_curr = plt.subplots(figsize=(12, 8), layout='constrained')

# Plot traces for quadB
for z_shift, x_arr, dc in qpd_data_collection_B:
    color = color_for_z(z_shift)
    # Using '--' for quadB traces
    ax_curr.plot(x_arr / 1000.0, dc, '--', color=color, linewidth=2.2, alpha=0.8)

# Plot traces for quadC
for z_shift, x_arr, dc in qpd_data_collection_C:
    color = color_for_z(z_shift)
    # Using '-.' for quadC traces
    ax_curr.plot(x_arr / 1000.0, dc, '-.', color=color, linewidth=2.2, alpha=0.8)

ax_curr.set_xlabel(r'Translation Stage X Position [mm]', fontsize=16)
ax_curr.set_ylabel(r'DC Photocurrent [A]', fontsize=16)
ax_curr.set_title(r'\textbf{DC Photocurrent (Y=2500\textmu{}m, quadB \& quadC)}', fontsize=18, fontweight='bold', pad=10)
ax_curr.grid(True, linestyle='--', alpha=0.6)
ax_curr.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in')

# Apply custom y-axis formatter
ax_curr.yaxis.set_major_formatter(y_fmt_photo)

# MODIFICATION: Restrict X range to 7.10 to 7.50
ax_curr.set_xlim(7.10-0.03, 7.50+0.03)
ax_curr.set_ylim(bottom=0.0)

# Create Z-position colorbar (uses normalized Z range)
if unique_z:
    norm = plt.Normalize(min(unique_z), max(unique_z))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([]) # Important for colormapping non-data plots

    # Colorbar position: adjusted to a better spot (e.g., top-left corner)
    cax = fig_curr.add_axes([0.15, 0.75, 0.02, 0.15])
    cbar = fig_curr.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label(r'Z Position [mm]', fontsize=14, labelpad=10)
    cbar.ax.tick_params(labelsize=13)
    
    # Set tick labels to min/max/mid for clarity
    z_min_val, z_max_val = min(unique_z), max(unique_z)
    z_mid = (z_min_val + z_max_val) / 2
    cbar.set_ticks([z_min_val, z_mid, z_max_val])
    cbar.set_ticklabels([f'{z_min_val:.2f}', f'{z_mid:.2f}', f'{z_max_val:.2f}'])

# MODIFICATION: Remove the legend from the photocurrent plot
# ax_curr.legend(handles=legend_handles, fontsize=14, loc='upper right', frameon=True)


out_path_curr = os.path.join(OUT_DIR, "Photocurrent_Traces_quadB_quadC_refactored_custom.png")
fig_curr.savefig(out_path_curr, dpi=300, bbox_inches='tight')
print(f"Photocurrent traces figure saved → {out_path_curr}")
plt.show()
plt.close(fig_curr)


# -------------------------
# FINAL SUMMARY
# -------------------------
print("\n=== FINAL RESULTS SUMMARY (KE only) ===")
# Report quadB min
if qpd_w_um_B.size:
    i_min_B = int(np.argmin(qpd_w_um_B))
    print(f"QPD (KE quadB): w0_min = {qpd_w_um_B[i_min_B]:.2f} µm at Z = {qpd_z_shifted_B[i_min_B]:.3f} mm")
else:
    print("QPD (KE quadB): no valid fits")
# Report quadC min
if qpd_w_um_C.size:
    i_min_C = int(np.argmin(qpd_w_um_C))
    print(f"QPD (KE quadC): w0_min = {qpd_w_um_C[i_min_C]:.2f} µm at Z = {qpd_z_shifted_C[i_min_C]:.3f} mm")
else:
    print("QPD (KE quadC): no valid fits")
print("============================\n")

# Show both plots
plt.show()