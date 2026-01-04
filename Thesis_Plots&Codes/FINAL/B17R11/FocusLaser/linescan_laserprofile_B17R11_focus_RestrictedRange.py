"""
DIAGONAL FOCUSING ANALYSIS — Averaging across duplicate quadrants
Only shows the averaged beam waist data points in the final combined plot.
"""
import re
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import curve_fit
from scipy.special import erf
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
# Add the missing import here
from matplotlib.ticker import FuncFormatter
# ====================== FONT & LATEX (kept identical) ======================
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf" # keep your font path
cmu_serif = fm.FontProperties(fname=font_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']
# ====================== ERF MODEL (unchanged) ======================
def erf_model(x, A, B, C, D):
    return A * erf(B * (x - C)) + D
# ====================== FOLDERS TO PROCESS ======================
# Edit here if you move directories
base_dirs = [
    "/Users/asca/Documents/University/Master Thesis/code/Data/20251215_final_data/20251125/Along6700um_AD",
    "/Users/asca/Documents/University/Master Thesis/code/Data/20251215_final_data/20251125/Along6500um_DC",
    "/Users/asca/Documents/University/Master Thesis/code/Data/20251215_final_data/20251125/Along7100um_AB",
    "/Users/asca/Documents/University/Master Thesis/code/Data/20251215_final_data/20251125/Along7200um_BC",
]
# Determine which quadrants are relevant per folder by suffix (AD -> A,D etc.)
folder_quadrants_map = {
    'AD': ['quadA', 'quadD'],
    'DC': ['quadD', 'quadC'],
    'AB': ['quadA', 'quadB'],
    'BC': ['quadB', 'quadC'],
}
# output directory for combined figures
combined_outdir = os.path.join(os.path.dirname(base_dirs[0]), "fig_all_folders")
os.makedirs(combined_outdir, exist_ok=True)
# ====================== QUADRANT STYLE (updated markers) ======================
segns = ['quadA', 'quadB', 'quadC', 'quadD']
line_styles = {'quadA': '-', 'quadB': '--', 'quadC': '-.', 'quadD': ':'}
colors = {'quadA': 'blue', 'quadB': 'orange', 'quadC': 'green', 'quadD': 'red'}

# NEW: marker style per quadrant (exactly like reference)
markers = {
    'quadA': '^',   # triangle
    'quadB': 'o',   # circle
    'quadC': 'D',   # diamond
    'quadD': 's'    # square
}
# zoom mask (same as yours) for plotting
u_mask_min, u_mask_max = -0.4, 0.4

# --- NEW: QUADRANT A CONDITIONAL FITTING RANGE PARAMETERS ---
Z_FIT_THRESHOLD = 13.58 # mm
u_fit_A_min, u_fit_A_max = -0.2, 0.4 # mm
# -----------------------------------------------------------

# helper regex extractors
def extract_along_um(fname):
    m = re.search(r'(?:Along|Y ?)(\d+)um', str(fname))
    return int(m.group(1)) if m else None
def extract_z_um(fname):
    m = re.search(r'Z(\d+)um', str(fname))
    return int(m.group(1)) if m else None
# ====================== MAIN DATA COLLECTION STRUCTURES ======================
# For each folder we'll store:
# folder_procdata[folder_path][seg] -> dict with lists 'along_um','z_mm','beamwaist_dc'
folder_procdata = {}
# For combined plotting of raw traces we collect all points like before
all_data_points = defaultdict(list) # along_um -> list of points (same structure as your data_collection earlier)
# We'll also collect every single (seg, z_mm) measurement into a list for averaging later:
# collected_per_seg[seg][z_mm] = list of w0 values
collected_per_seg = {s: defaultdict(list) for s in segns}
# ====================== PROCESS each base_dir ======================
pkl_pattern = "*_load2.pkl"
for base_dir in base_dirs:
    if not os.path.isdir(base_dir):
        print(f"Warning: folder not found (skipping): {base_dir}")
        continue
    folder_name = os.path.basename(base_dir)
    # determine relevant quadrants from folder name (look for AD, DC, AB, BC)
    relevant = None
    for key in folder_quadrants_map:
        if key in folder_name:
            relevant = folder_quadrants_map[key]
            break
    if relevant is None:
        # fallback: if not recognized, assume all quadrants may be present
        relevant = segns.copy()
    fig_dir = os.path.join(base_dir, "fig")
    os.makedirs(fig_dir, exist_ok=True)
    file_list = sorted(glob.glob(os.path.join(base_dir, "**", pkl_pattern), recursive=True))
    if not file_list:
        print(f"No {pkl_pattern} files found in {base_dir} - skipping")
        continue
    # initialize procdata for this folder
    procdata = {s: {'along_um': [], 'z_mm': [], 'beamwaist_dc': []} for s in segns}
    data_collection = {} # along_um -> list of points (mirrors your previous structure)
    all_z_mm = set()
    for pkl_file in file_list:
        along_um = extract_along_um(pkl_file)
        z_um = extract_z_um(pkl_file)
        if along_um is None or z_um is None:
            continue
        z_mm = z_um / 1000.0
        all_z_mm.add(z_mm)
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        u_array = data['rawdata']['u_position']
        mask = (u_array >= u_mask_min) & (u_array <= u_mask_max)
        u_plot = u_array[mask]
        if along_um not in data_collection:
            data_collection[along_um] = []
        point = {
            'along_um': along_um,
            'z_mm' : z_mm,
            'u_mm' : u_plot,
            }
        for seg in segns:
            if seg not in data['rawdata'] or 'dmm00_curr_amp' not in data['rawdata'][seg]:
                point[f'dc_curr_{seg}'] = None
                continue
            curr_full = np.mean(data['rawdata'][seg]['dmm00_curr_amp'], axis=1)
            point[f'dc_curr_{seg}'] = curr_full[mask] # zoomed

            # --- Differentiate fitting ranges based on quadrant and Z position ---
            u_fit = u_array
            curr_fit = curr_full
            
            if seg == 'quadA' and z_mm < Z_FIT_THRESHOLD:
                # Use restricted range for quadA fitting only if Z is below the threshold
                u_fit_mask = (u_array >= u_fit_A_min) & (u_array <= u_fit_A_max)
                u_fit = u_array[u_fit_mask]
                curr_fit = curr_full[u_fit_mask]
            # else: u_fit and curr_fit remain the full range (u_array, curr_full) for quadA at high Z and quads B, C, D at all Z.

            # ensure we have enough points for a stable fit
            if len(curr_fit) >= 8:
                try:
                    # Initial guess: based on the fitting range
                    p0 = [np.ptp(curr_fit), 2.0,
                          u_fit[np.argmax(np.abs(np.gradient(curr_fit)))],
                          np.min(curr_fit)]
                    popt, _ = curve_fit(erf_model, u_fit, curr_fit, p0=p0,
                                        bounds=([-np.inf, 0.1, -np.inf, -np.inf],
                                                [np.inf, 50, np.inf, np.inf]))
                    A, B, _, _ = popt
                    w0_um = 1 / (np.sqrt(2) * B) * 1000
                    if np.isfinite(w0_um) and w0_um < 500:
                        procdata[seg]['along_um'].append(along_um)
                        procdata[seg]['z_mm'].append(z_mm)
                        procdata[seg]['beamwaist_dc'].append(w0_um)
                        # store for global averaging only if this quadrant is relevant for this folder
                        if seg in relevant:
                            collected_per_seg[seg][z_mm].append(w0_um)
                except Exception:
                    # print(f"Fit failed for {seg} at Z={z_mm} mm in {folder_name}")
                    pass
            # ----------------------------------------------------

        data_collection[along_um].append(point)
    # save this folder's procdata for plotting of its own figure(s)
    folder_procdata[base_dir] = {
        'procdata': procdata,
        'data_collection': data_collection,
        'all_z_positions': sorted(list(all_z_mm)),
        'relevant_quads': relevant,
        'fig_dir': fig_dir,
        'folder_name': folder_name
        }
    # Also accumulate for global plotting of raw traces
    for along_um, pts in data_collection.items():
        all_data_points[along_um].extend(pts)
# ====================== COLOR MAP (global) ======================
global_z_positions = sorted({z for seg in segns for z in collected_per_seg[seg].keys()})
if not global_z_positions:
    print("No valid z positions found in any folder. Exiting.")
    raise SystemExit
cmap = plt.cm.RdYlBu
norm = plt.Normalize(min(global_z_positions), max(global_z_positions))
color_dict = {}
n = len(global_z_positions)
for i, z in enumerate(global_z_positions):
    if i < n // 2:
        color_dict[z] = cmap(0.1 + 0.4 * (i / max(1, (n // 2))))
    else:
        color_dict[z] = cmap(0.6 + 0.4 * ((i - n // 2) / max(1, (n - n // 2))))
# ====================== PER-FOLDER PLOTS (individual results) ======================
for base_dir, info in folder_procdata.items():
    procdata = info['procdata']
    fig_dir = info['fig_dir']
    folder_name = info['folder_name']
    relevant = info['relevant_quads']
    all_z_positions = info['all_z_positions']

    # --------------------------------------------------------------------------
    # Beam waist vs Z for this folder - STYLED
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    ax.set_xlabel(r'Z Position [mm]', fontsize=16) # FONTSIZE 16
    ax.set_ylabel(r'Beam Waist $\rm{w}_0$ [\textmu{}m]', fontsize=16) # FONTSIZE 16
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in') # FONTSIZE 13
    title = rf'\textbf{{Beam Profile ({folder_name}) — Diagonal Knife-Edge Scan}}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10) # FONTSIZE 16

    min_info = []

    for seg in relevant:
        if not procdata[seg]['beamwaist_dc']:
            continue
        z_vals = np.array(procdata[seg]['z_mm'])
        w_vals = np.array(procdata[seg]['beamwaist_dc'])
        order = np.argsort(z_vals)
        z_vals = z_vals[order]
        w_vals = w_vals[order]
        for z, w in zip(z_vals, w_vals):
            col = color_dict.get(z, list(color_dict.values())[0])
            ax.plot(z, w, marker=markers[seg], color=col, markersize=9) # MARKERSIZE 7
        ax.plot(z_vals, w_vals, line_styles[seg], color=colors[seg], alpha=0.6, linewidth=2.2, label=seg) # LINEWIDTH 2.2

        if len(w_vals) > 0:
            i_min = np.argmin(w_vals)
            ax.plot(z_vals[i_min], w_vals[i_min],
                    marker=markers[seg], mec='red', mfc='none', mew=2.5, ms=13.5) # MARKERSIZE 12
            min_info.append((seg, w_vals[i_min]))

    ax.legend(fontsize=15, loc='upper left', frameon=True, bbox_to_anchor=(0.12, 0.975)) # FONTSIZE 15

    if min_info:
        tex_lines = []
        legend_handles = []
        for seg, wmin in min_info:
            tex_lines.append(r'$w_{{0,\min}}$(KE, quad{}) $= {:.2f}\,\mu\mathrm{{m}}$'.format(seg[-1], wmin))
            line = Line2D([0], [0], marker=markers[seg], color='w',
                          markeredgecolor='red', markerfacecolor='none',
                          markersize=10.5, markeredgewidth=2.5)
            legend_handles.append(line)
        inset = ax.inset_axes([0.26, 0.60, 0.35, 0.25])
        inset.axis('off')
        inset.legend(handles=legend_handles, labels=tex_lines,
                     loc='upper left', fontsize=18.7, frameon=False, handlelength=1.2, bbox_to_anchor=(0.12, 0.975)) # FONTSIZE 18.7

    fname = os.path.join(fig_dir, f"BeamWaist_vsZ_{folder_name}.png")
    fig.savefig(fname, dpi=300)
    
    print(f"Saved per-folder beam waist figure → {fname}")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # PHOTOCURRENT PLOT WITH FULL COLORBAR - STYLED
    # --------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(12, 8), layout='constrained')
    for along_um in sorted(info['data_collection'].keys()):
        for pt in info['data_collection'][along_um]:
            z = pt['z_mm']
            col = color_dict.get(z, list(color_dict.values())[0])
            for seg in segns:
                curr = pt.get(f'dc_curr_{seg}')
                if curr is not None:
                    ax2.plot(pt['u_mm'], curr, linestyle=line_styles[seg], color=col, alpha=0.8, linewidth=2.2) # LINEWIDTH 2.2
    
    # Custom Y-axis formatter for high-precision current (like f'{val:.5f}')
    def tick_formatter_photocurrent(val, pos):
        return '0' if val == 0 else f'{val:.5f}'
    y_fmt_photo = FuncFormatter(tick_formatter_photocurrent)
    
    ax2.set_xlabel(r'Perpendicular offset $u$ [mm]', fontsize=16) # FONTSIZE 16
    ax2.set_ylabel(r'DC Photocurrent [A]', fontsize=16) # FONTSIZE 16
    ax2.set_title(r'\textbf{DC Photocurrent – Diagonal Knife-Edge Scans (Folder: %s)}' % folder_name,
                  fontsize=16, fontweight='bold', pad=10) # FONTSIZE 16
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in') # FONTSIZE 13
    ax2.yaxis.set_major_formatter(y_fmt_photo) # Use 5 decimal places

    # FULL COLORBAR — STYLED
    cax = fig2.add_axes([0.12, 0.55, 0.02, 0.18]) # ADJUSTED BOTTOM FROM 0.695 TO 0.55
    cbar = fig2.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical')
    cbar.set_label(r'Z Position [mm]', fontsize=14.2, labelpad=10) # FONTSIZE 14.2
    cbar.ax.tick_params(labelsize=14.2) # FONTSIZE 14.2
    z_min_val, z_max_val = min(all_z_positions), max(all_z_positions)
    z_mid = (z_min_val + z_max_val) / 2
    cbar.set_ticks([z_min_val, z_mid, z_max_val])
    cbar.set_ticklabels([f'{z_min_val:.2f}', f'{z_mid:.2f}', f'{z_max_val:.2f}'])

    fname2 = os.path.join(fig_dir, f"Photocurrent_Traces_{folder_name}.png")
    fig2.savefig(fname2, dpi=300, bbox_inches='tight')
    print(f"Saved per-folder photocurrent figure → {fname2}")
    plt.close(fig2)

# ====================== COMBINED AVERAGED PLOT (averaging duplicates) ======================
avg_procdata = {s: {'z_mm': [], 'beamwaist_mean': [], 'counts': []} for s in segns}
for seg in segns:
    z_keys = sorted(collected_per_seg[seg].keys())
    for z in z_keys:
        values = collected_per_seg[seg][z]
        if len(values) == 0:
            continue
        avg = float(np.nanmean(values))
        avg_procdata[seg]['z_mm'].append(z)
        avg_procdata[seg]['beamwaist_mean'].append(avg)
        avg_procdata[seg]['counts'].append(len(values))

# --------------------------------------------------------------------------
# Combined Averaged Plot - STYLED
# --------------------------------------------------------------------------
fig_avg, ax_avg = plt.subplots(figsize=(10, 6), layout='constrained')
ax_avg.set_xlabel(r'Tranlation Stage Z Position [mm]', fontsize=16) # FONTSIZE 16
ax_avg.set_ylabel(r'Beam Waist $w_0$ [\textmu{}m]', fontsize=16) # FONTSIZE 16
ax_avg.grid(True, linestyle='--', alpha=0.6)
ax_avg.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in') # FONTSIZE 13
ax_avg.set_title(r'\textbf{Beam Profile (DC)}',
                 fontsize=16, fontweight='bold', pad=10) # FONTSIZE 16
ax.set_xlim(13.55-0.03, 13.9+0.03)
ax_avg.set_ylim(bottom = 0, top=120)

min_info_avg = []

for seg in segns:
    # --- REMOVED RAW DATA PLOTTING ---
    # raw_z = []
    # raw_w = []
    # for z, vals in collected_per_seg[seg].items():
    #     for vv in vals:
    #         raw_z.append(z)
    #         raw_w.append(vv)
    # if raw_z:
    #     ax_avg.scatter(raw_z, raw_w, alpha=0.25, marker=markers[seg], color=colors[seg]) 
    # ---------------------------------

    if avg_procdata[seg]['z_mm']:
        z_vals = np.array(avg_procdata[seg]['z_mm'])
        w_vals = np.array(avg_procdata[seg]['beamwaist_mean'])
        order = np.argsort(z_vals)
        z_vals = z_vals[order]
        w_vals = w_vals[order]
        for z, w in zip(z_vals, w_vals):
            col = color_dict.get(z, list(color_dict.values())[0])
            # Plot the single, averaged point
            ax_avg.plot(z, w, marker=markers[seg], color=col, markersize=7) # MARKERSIZE 7
        # Plot the line connecting the averaged points
        ax_avg.plot(z_vals, w_vals, line_styles[seg], color=colors[seg], alpha=0.5,
                    label=f"{seg}", linewidth=2.2) # LINEWIDTH 2.2

        if len(w_vals) > 0:
            i_min = np.argmin(w_vals)
            ax_avg.plot(z_vals[i_min], w_vals[i_min],
                        marker=markers[seg], mec='red', mfc='none', mew=2.5, ms=12) # MARKERSIZE 12
            min_info_avg.append((seg, w_vals[i_min]))

ax_avg.legend(fontsize=15, loc='upper left', frameon=True, bbox_to_anchor=(0.05, 0.975)) # FONTSIZE 15

if min_info_avg:
    tex_lines = []
    legend_handles = []
    for seg, wmin in min_info_avg:
        tex_lines.append(r'$w_{{0,\min}}$(KE, quad{}) $= {:.2f}\,\mu\mathrm{{m}}$'.format(seg[-1], wmin))
        line = Line2D([0], [0], marker=markers[seg], color='w',
                      markeredgecolor='red', markerfacecolor='none',
                      markersize=10, markeredgewidth=2.5)
        legend_handles.append(line)
    inset = ax_avg.inset_axes([0.26, 0.60, 0.35, 0.25])
    inset.axis('off')
    inset.legend(handles=legend_handles, labels=tex_lines,
                 loc='upper left', fontsize=18.7, frameon=False, handlelength=1.2, bbox_to_anchor=(0.02, 0.975)) # FONTSIZE 18.7

avg_fname = os.path.join(combined_outdir, "BeamWaist_Averaged_AllQuadrants.png")
fig_avg.savefig(avg_fname, dpi=300)
print(f"Saved averaged beam waist figure → {avg_fname}")
plt.show()
plt.close(fig_avg) # Removed plt.show()

# ====================== GLOBAL PHOTOCURRENT WITH FULL COLORBAR ======================
fig_all, ax_all = plt.subplots(figsize=(12, 8), layout='constrained')
for along_um in sorted(all_data_points.keys()):
    for pt in all_data_points[along_um]:
        z = pt['z_mm']
        col = color_dict.get(z, list(color_dict.values())[0])
        for seg in segns:
            curr = pt.get(f'dc_curr_{seg}')
            if curr is not None:
                ax_all.plot(pt['u_mm'], curr, linestyle=line_styles[seg], color=col, alpha=0.8, linewidth=2.2) # LINEWIDTH 2.2

ax_all.set_xlabel(r'Perpendicular offset $u$ [mm]', fontsize=16) # FONTSIZE 16
ax_all.set_ylabel(r'DC Photocurrent [A]', fontsize=16) # FONTSIZE 16
ax_all.set_title(r'\textbf{DC Photocurrent – Diagonal Knife-Edge Scans (All Folders)}',
                 fontsize=16, fontweight='bold', pad=10) # FONTSIZE 16
ax_all.grid(True, linestyle='--', alpha=0.6)
ax_all.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in') # FONTSIZE 13
#ax.set_xlim(13.55-0.03, 13.9+0.03)
ax_all.yaxis.set_major_formatter(y_fmt_photo) # Use 5 decimal places formatter

# FULL COLORBAR — STYLED
cax = fig_all.add_axes([0.12, 0.55, 0.02, 0.18]) # ADJUSTED BOTTOM FROM 0.695 TO 0.55
cbar = fig_all.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical')
cbar.set_label(r'Z Position [mm]', fontsize=14.2, labelpad=10) # FONTSIZE 14.2
cbar.ax.tick_params(labelsize=14.2) # FONTSIZE 14.2
z_min_val, z_max_val = min(global_z_positions), max(global_z_positions)
z_mid = (z_min_val + z_max_val) / 2
cbar.set_ticks([z_min_val, z_mid, z_max_val])
cbar.set_ticklabels([f'{z_min_val:.2f}', f'{z_mid:.2f}', f'{z_max_val:.2f}'])

all_fname = os.path.join(combined_outdir, "Photocurrent_AllFolders.png")
fig_all.savefig(all_fname, dpi=300, bbox_inches='tight')
print(f"Saved combined photocurrent figure → {all_fname}")
plt.show()
plt.close(fig_all) # Removed plt.show()

# ====================== MIN WAIST SUMMARY ======================
print("\n=== Minimum beam waist per quadrant (from averaged values where available) ===")
for seg in segns:
    if avg_procdata[seg]['beamwaist_mean']:
        ws = np.array(avg_procdata[seg]['beamwaist_mean'])
        idx = np.argmin(ws)
        print(f"{seg}: Min waist = {ws[idx]:.2f} µm at Z = {avg_procdata[seg]['z_mm'][idx]*1000:.0f} µm (n={avg_procdata[seg]['counts'][idx]})")
    else:
        raw_ws = []
        raw_zs = []
        for z, vals in collected_per_seg[seg].items():
            for v in vals:
                raw_ws.append(v)
                raw_zs.append(z)
        if raw_ws:
            raw_ws = np.array(raw_ws)
            raw_zs = np.array(raw_zs)
            idx = np.argmin(raw_ws)
            print(f"{seg}: Min waist = {raw_ws[idx]:.2f} µm at Z = {raw_zs[idx]*1000:.0f} µm (raw)")
        else:
            print(f"{seg}: no valid data")