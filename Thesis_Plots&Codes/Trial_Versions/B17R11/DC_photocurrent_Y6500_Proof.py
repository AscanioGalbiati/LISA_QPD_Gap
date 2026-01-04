# -*- coding: utf-8 -*-
'''
FINAL REVISION — PROOF-OF-CONCEPT DC PHOTOCURRENT PLOT (SINGLE Y-SCAN)
→ Processes ONLY the Y6500um folder.
→ Plots only quadA, quadC, quadD.
→ Y-axis converted to mA and formatted: '.2f' everywhere except '0' at zero.
→ Forces Y-axis start limit to zero.
→ Marks A-D (triangle) and D-C (square) gap crossings with custom red markers.
→ Includes an inset result box with gap X positions.
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

# ------------------------------ Setup ------------------------------
# Set up font and LaTeX rendering
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
cmu_serif = fm.FontProperties(fname=font_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

# **MODIFICATION: Set the specific folder path to process**
base_dir = "/Users/asca/Documents/University/Master Thesis/code/Data/all quadrants/20251121/VIGO_FPW01_QPD_1500_20_B17R11_251121_objLens_quadABCD_manual_setup_Z12.3mm_YScan_thresholdX_NDfilter_LOAD_GapIdentification"
TARGET_Y_FOLDER = os.path.join(base_dir, 'Y6500um')
TARGET_Y_UM = 6500
CURRENT_THRESHOLD = 0.000015  # 15 µA (Used for gap finding logic)

# Custom formatter for the Y-axis: .2f everywhere except '0' at zero
def current_ma_formatter(val, pos):
    if val == 0:
        return '0'
    return f'{val:.2f}'

# Function to load data from a .pkl file
def load_data(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

# Function to build X array with adaptive steps (copied from first script)
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

# Diagonals to find gap for
pairs = [
    ('quadA', 'quadD', 'A-D'),
    ('quadB', 'quadC', 'B-C'),
    ('quadA', 'quadB', 'A-B'),
    ('quadD', 'quadC', 'D-C')
]

# -------------------------- Data Processing & Plotting --------------------------
pkl_files = glob.glob(os.path.join(TARGET_Y_FOLDER, "*.pkl"))

if pkl_files:
    pkl_file = pkl_files[0]
    data = load_data(pkl_file)
    
    try: x_array = build_x_array(data['global_params'])
    except KeyError: x_array = None

    if x_array is not None:
        # 1. Calculate Mean Currents (converting to mA for plotting)
        means = {q: data['rawdata'][q]['dmm00_curr_amp'].mean(axis=1) * 1000 if q in data['rawdata'] else None 
                 for q in ['quadA','quadB','quadC','quadD']} # <-- Multiplied by 1000 for mA
        
        # Original current values (in A) needed for the threshold logic
        raw_means = {q: data['rawdata'][q]['dmm00_curr_amp'].mean(axis=1) if q in data['rawdata'] else None 
                     for q in ['quadA','quadB','quadC','quadD']}
        
        # 2. Find Gap X Positions (using the first script's logic, referencing raw_means)
        gap_x_positions = {}
        for quad1, quad2, label in pairs:
            if raw_means[quad1] is None or raw_means[quad2] is None: continue
            
            diff = raw_means[quad1] - raw_means[quad2] # Use raw A for crossing logic
            crossings = np.where(np.diff(np.sign(diff)) != 0)[0]
            if len(crossings) == 0: continue
            
            valid_crossings = []
            for i in crossings:
                x1, x2 = x_array[i], x_array[i+1]
                d1, d2 = diff[i], diff[i+1]
                x_cross = x1 if abs(d2-d1) < 1e-12 else x1 - d1*(x2-x1)/(d2-d1)
                
                sl = slice(max(0,i-5), min(len(x_array),i+6))
                total_curr = raw_means[quad1][sl] + raw_means[quad2][sl] # Use raw A for threshold check
                
                if np.max(total_curr) > CURRENT_THRESHOLD:
                    valid_crossings.append((x_cross, np.max(total_curr)))
            
            if not valid_crossings: continue

            # Crossings selection logic from your original code
            if label == 'A-B': x_cross, _ = valid_crossings[1]
            elif label == 'B-C': x_cross, _ = valid_crossings[0]
            else: x_cross, _ = valid_crossings[0]

            gap_x_positions[label] = x_cross
            
        # 3. Create plot
        fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
        
        # Quadrants to plot (excluding B)
        quadrants_to_plot = ['quadA', 'quadC', 'quadD']
        
        for q in quadrants_to_plot:
            if q in means and means[q] is not None:
                ax.plot(x_array, means[q], label=q)
        
        # 4. Mark the Gap X Positions (A-D and D-C only)
        highlight_info = []
        
        for label, x_gap in gap_x_positions.items():
            if label in ['A-D', 'D-C']:
                # Find the nearest index to approximate the current for the marker y-position
                nearest_idx = np.argmin(np.abs(x_array - x_gap))
                
                quad1_name, quad2_name, _ = next(p for p in pairs if p[2] == label)
                
                if means[quad1_name] is not None and means[quad2_name] is not None:
                    # Current is in mA
                    current_at_gap = (means[quad1_name][nearest_idx] + means[quad2_name][nearest_idx]) / 2
                    
                    if label == 'A-D':
                        marker = '^' # Changed to triangle
                    elif label == 'D-C':
                        marker = 's' # Changed to square
                    
                    # Plot the marker (red, no fill, thick edge)
                    ax.plot(x_gap, current_at_gap, 
                            marker=marker, mec='red', mfc='none', mew=2.5, ms=12,
                            linestyle='', zorder=10)
                    
                    highlight_info.append((label, x_gap, marker))

        # 5. Final Plot Styling
        ax.set_title(rf'\textbf{{DC Photocurrent (Y={TARGET_Y_UM}\textmu{{}}m) - Gap Crossings}}', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(r'X Position [mm]', fontsize=14)
        ax.set_ylabel(r'Photocurrent $[\mathrm{mA}]$', fontsize=14) # <-- Units changed to mA
        ax.set_xlim(8.75,12.75)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5, direction='in')
        
        # Apply Y-axis formatting and limits
        ax.yaxis.set_major_formatter(FuncFormatter(current_ma_formatter))
        ax.set_ylim(bottom=0) # <-- Set Y-axis to start from zero
        
        # 6. Create the Inset Result Box
        if highlight_info:
            inset = ax.inset_axes([0.65, 0.31, 0.35, 0.40])
            inset.axis('off')
            handles = []
            labels = []
            
            for tag, x_gap, marker in highlight_info:
                label = rf'Gap X ({tag}) = ${x_gap:+.2f}\,\mathrm{{mm}}$'
                handles.append(Line2D([0], [0], 
                                      marker=marker, color='w', markeredgecolor='red',
                                      markerfacecolor='none', markersize=10, markeredgewidth=2.0))
                labels.append(label)
            
            inset.legend(handles=handles, labels=labels, loc='upper left', 
                         fontsize=11.5, frameon=True, borderpad=0.6, labelspacing=1.0)
            '''inset.legend(handles=handles, labels=labels, bbox_to_anchor=(0.1, 1.0),
                         fontsize=14, frameon=True, borderpad=0.6, labelspacing=1.0)'''

        # Re-create the main legend for the quadrant lines
        #ax.legend(fontsize=12, loc='upper left')
        ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0.02, 1.0))

        # Create figure directory
        fig_dir = os.path.join(TARGET_Y_FOLDER, "fig")
        os.makedirs(fig_dir, exist_ok=True)
        
        # Save figure
        fig_name = f"DC_Photocurrent_Y{TARGET_Y_UM:04d}um_GapProof_Final.png"
        save_path = os.path.join(fig_dir, fig_name)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved: {save_path}")
        
        plt.close(fig)
    else:
        print(f"Could not build x_array for {pkl_file}")
else:
    print(f"Target folder not found or no .pkl files in: {TARGET_Y_FOLDER}")

print("Script finished.")