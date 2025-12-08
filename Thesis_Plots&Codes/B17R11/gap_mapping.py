# -*- coding: utf-8 -*-
'''
FINAL — THESIS-READY QPD GAP MAPPING SCRIPT (FINAL REVISION V3)
→ Figure size decreased (smaller canvas).
→ Blue triangle marker for A-D now faces UP.
→ Highlight box font size increased.
'''
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm 
import re
from scipy.stats import linregress
from matplotlib.lines import Line2D 

# ------------------------------ Setup ------------------------------
# Configure LaTeX and Computer Modern font
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf" # Assumes this path is correct
cmu_serif = fm.FontProperties(fname=font_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams['font.size'] = 12

base_dir = "/Users/asca/Documents/University/Master Thesis/code/Data/all quadrants/20251121/VIGO_FPW01_QPD_1500_20_B17R11_251121_objLens_quadABCD_manual_setup_Z12.3mm_YScan_thresholdX_NDfilter_LOAD_GapIdentification"
CURRENT_THRESHOLD = 0.000015  # 15 µA
HIGHLIGHT_Y_UM = 6500         # Target Y position to highlight

def load_data(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

def extract_y_position(folder_name):
    match = re.search(r'Y(\d+)um', folder_name)
    return int(match.group(1)) if match else None

def build_x_array(global_params):
    xstart = global_params['xstart_mm']
    xstop = global_params['xstop_mm']
    xstep_big = global_params['xstep_big_mm']
    xstep_fine = global_params['xstep_fine_mm']
    x_th_start = global_params['x_threshold_start_mm']
    x_th_stop = global_params['x_threshold_stop_mm']
    x_array = []
    x = xstart
    while x <= xstop + 1e-9:
        x_array.append(x)
        x += xstep_fine if x_th_start <= x <= x_th_stop else xstep_big
    return np.array(x_array)

# ------------------------------ Pairs & Ranges ------------------------------
pairs = [
    ('quadA', 'quadD', 'Bottom left diagonal A-D'),
    ('quadB', 'quadC', 'Top right diagonal B-C'),
    ('quadA', 'quadB', 'Top left diagonal A-B'),
    ('quadD', 'quadC', 'Bottom right D-C')
]

pair_ranges = {
    'Bottom left diagonal A-D': (6000, 6900),
    'Top right diagonal B-C': (7100, 8000),
    'Top left diagonal A-B': (7100, 8000),
    'Bottom right D-C': (6000, 6900),
}

colors = ['blue', 'green', 'red', 'orange']
# UPDATED: Changed default marker for A-D from 'v' (down) to '^' (up)
# Keep others the same for visual separation
markers = ['^', 's', 'D', 'o']
linestyles = ['-', '-.', ':', '--']
label_order = ['Bottom left diagonal A-D', 'Top right diagonal B-C', 'Top left diagonal A-B', 'Bottom right D-C']

result_box_titles = {
    'Bottom left diagonal A-D': r'Gap A-D:',
    'Top right diagonal B-C': r'Gap B-C:',
    'Top left diagonal A-B': r'Gap A-B:',
    'Bottom right D-C': r'Gap D-C:'
}
# Markers for the second highlight box (consistent with the new default A-D marker)
highlight_markers = {
    'Bottom left diagonal A-D': '^', # UPWARD TRIANGLE
    'Bottom right D-C': 'o'          # CIRCLE
}

# -------------------------- Data Collection --------------------------
y_folders = sorted(glob.glob(os.path.join(base_dir, "Y*um")))
y_positions = [(extract_y_position(os.path.basename(f)), f) for f in y_folders if extract_y_position(os.path.basename(f))]

gap_data = {label: {'y_mm': [], 'x_gap': [], 'current': []} for _, _, label in pairs}
fit_results = {}
highlight_points = {}

for y_um, y_folder in y_positions:
    pkl_files = glob.glob(os.path.join(y_folder, "*.pkl"))
    if not pkl_files: continue
    data = load_data(pkl_files[0])
    try: x_array = build_x_array(data['global_params'])
    except KeyError: continue

    means = {q: data['rawdata'][q]['dmm00_curr_amp'].mean(axis=1) if q in data['rawdata'] else None 
             for q in ['quadA','quadB','quadC','quadD']}

    for quad1, quad2, label in pairs:
        min_y, max_y = pair_ranges.get(label, (None, None))
        if min_y is not None and not (min_y <= y_um <= max_y): continue
        if means[quad1] is None or means[quad2] is None: continue
        diff = means[quad1] - means[quad2]
        crossings = np.where(np.diff(np.sign(diff)) != 0)[0]
        if len(crossings) == 0: continue
        valid_crossings = []
        for i in crossings:
            x1, x2 = x_array[i], x_array[i+1]
            d1, d2 = diff[i], diff[i+1]
            x_cross = x1 if abs(d2-d1) < 1e-12 else x1 - d1*(x2-x1)/(d2-d1)
            sl = slice(max(0,i-5), min(len(x_array),i+6))
            total_curr = means[quad1][sl] + means[quad2][sl]
            if np.max(total_curr) > CURRENT_THRESHOLD:
                valid_crossings.append((x_cross, np.max(total_curr)))
        if not valid_crossings: continue

        if label == 'Top left diagonal A-B': x_cross, curr = valid_crossings[1]
        elif label == 'Top right diagonal B-C': x_cross, curr = valid_crossings[0]
        else: x_cross, curr = valid_crossings[0]

        gap_data[label]['y_mm'].append(y_um / 1000.0)
        gap_data[label]['x_gap'].append(x_cross)
        gap_data[label]['current'].append(curr)
        
        if y_um == HIGHLIGHT_Y_UM and label in ['Bottom left diagonal A-D', 'Bottom right D-C']:
             highlight_points[label] = (y_um / 1000.0, x_cross)

# -------------------------- INVERTED AXES PLOTTING --------------------------
# DECREASED CANVAS SIZE: (13, 10) -> (11.05, 8.85)
fig, ax = plt.subplots(figsize=(11.05, 8.85), layout='constrained')

for idx, (_, _, label) in enumerate(pairs):
    if not gap_data[label]['y_mm']:
        continue

    y_mm = np.array(gap_data[label]['y_mm'])
    x_gap = np.array(gap_data[label]['x_gap'])

    # Scatter plot
    ax.scatter(y_mm, x_gap, label=label, color=colors[idx], marker=markers[idx], 
               s=70, edgecolor='black', linewidth=1, zorder=5)

    res = linregress(y_mm, x_gap)
    old_slope = res.slope
    old_intercept = res.intercept
    
    # Fit line
    y_fit = np.linspace(y_mm.min()-0.1, y_mm.max()+0.1, 300)
    x_fit = old_intercept + old_slope * y_fit
    ax.plot(y_fit, x_fit, color=colors[idx], lw=3, linestyle=linestyles[idx])

    fit_results[label] = {
        'slope': old_slope,
        'intercept': old_intercept,
        'r2': res.rvalue**2,
        'n': len(y_mm)
    }

# Add the specific highlight markers
if 'Bottom left diagonal A-D' in highlight_points:
    y_A_D, x_A_D = highlight_points['Bottom left diagonal A-D']
    ax.plot(y_A_D, x_A_D, highlight_markers['Bottom left diagonal A-D'], 
            mec='red', mfc='none', mew=2.5, ms=15, zorder=10)
    
if 'Bottom right D-C' in highlight_points:
    y_D_C, x_D_C = highlight_points['Bottom right D-C']
    ax.plot(y_D_C, x_D_C, highlight_markers['Bottom right D-C'], 
            mec='red', mfc='none', mew=2.5, ms=15, zorder=10)

# -------------------------- Plot Styling and Result Box --------------------------
ax.set_xlabel(r'Y Position [mm]', fontsize=14)
ax.set_ylabel(r'Gap X Position [mm]', fontsize=14)
ax.set_title(r'\textbf{QPD Gap Mapping}', fontsize=16, pad=10, fontweight='bold')
ax.grid(True, ls='--', alpha=0.6)
ax.tick_params(axis='both', which='major', labelsize=12, length=6, width=1.5, direction='in')

# --- Slope/Intercept Result Box (Primary Legend) ---
result_handles = []
result_labels  = []

for label in label_order:
    if label not in fit_results:
        continue
    
    idx = [p[2] for p in pairs].index(label)
    res = fit_results[label]
    new_title = result_box_titles[label]

    handle = Line2D([0], [0],
                    marker=markers[idx],
                    markeredgecolor='k',
                    markerfacecolor=colors[idx],
                    color=colors[idx],
                    linestyle=linestyles[idx],
                    markersize=10,
                    linewidth=3)

    text = (rf'\textbf{{{new_title}}}' + '\n' +
            fr'  Slope: ${res["slope"]:+.2f}$' + '\n' +
            fr'  Intercept: ${res["intercept"]:+.2f}\,\mathrm{{mm}}$')

    result_handles.append(handle)
    result_labels.append(text)

# Position the Slope/Intercept box on the left
inset1 = ax.inset_axes([0.02, 0.12, 0.30, 0.56]) 
inset1.axis('off')
inset1.legend(handles=result_handles, labels=result_labels,
             loc='upper left', fontsize=13.5, frameon=True, # <-- FONT SIZE INCREASED
             handlelength=2.2, handletextpad=0.8,
             labelspacing=1.1,
             borderpad=0.8)

# --- Highlighted Point Result Box (Second Legend) ---
highlight_handles = []
highlight_labels = []

if highlight_points:
    for label in ['Bottom left diagonal A-D', 'Bottom right D-C']:
        if label in highlight_points:
            x_gap = highlight_points[label][1]
            marker = highlight_markers[label]
            
            # Simplified Gap X Position label
            tag = label.split(' ')[-1]
            new_label = rf'Gap X ({tag}) = ${x_gap:+.2f}\,\mathrm{{mm}}$'
            
            # Handle using the red marker style
            highlight_handles.append(Line2D([0], [0], 
                                          marker=marker, color='w', markeredgecolor='red',
                                          markerfacecolor='none', markersize=10, markeredgewidth=2.0))
            highlight_labels.append(new_label)

    # Position the Highlight box near center top: [0.35, 0.70, 0.30, 0.25]
    inset2 = ax.inset_axes([0.32, 0.70, 0.30, 0.25]) 
    inset2.axis('off')
    # Title removed as requested
    inset2.legend(handles=highlight_handles, labels=highlight_labels, loc='upper left', 
                 fontsize=16, frameon=True, borderpad=0.6, labelspacing=1.0) # <-- FONT SIZE INCREASED

# ax.legend (primary legend) is removed as requested

save_path = os.path.join(base_dir, "QPD_Gap_vs_Y_FINAL_ENHANCED_V3.png")
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)

# -------------------------- Console Output (Simplified) --------------------------
print("\n" + "="*80)
print("DIAGONAL AVERAGES")
print("="*80)
slopes = []

if 'Bottom left diagonal A-D' in fit_results and 'Top right diagonal B-C' in fit_results:
    s1 = fit_results['Bottom left diagonal A-D']['slope']
    s2 = fit_results['Top right diagonal B-C']['slope']
    avg = (s1 + s2) / 2
    slopes.append(avg)
    print(f"Main diagonal (A-D & B-C) → Avg slope: {avg:+.7f} → Angle with Y: {np.degrees(np.arctan(avg)):+.4f}°")

if 'Top left diagonal A-B' in fit_results and 'Bottom right D-C' in fit_results:
    s1 = fit_results['Top left diagonal A-B']['slope']
    s2 = fit_results['Bottom right D-C']['slope']
    avg = (s1 + s2) / 2
    slopes.append(avg)
    print(f"Secondary diagonal (A-B & D-C) → Avg slope: {avg:+.7f} → Angle with Y: {np.degrees(np.arctan(avg)):+.4f}°")

if len(slopes) == 2:
    overall = np.mean(slopes)
    print(f"\n>>> OVERALL AVERAGE (both diagonals) <<<")
    print(f"    Slope = {overall:+.7f}")
    print(f"    Angle with Y-axis = {np.degrees(np.arctan(overall)):+.4f}°")

print("="*80)