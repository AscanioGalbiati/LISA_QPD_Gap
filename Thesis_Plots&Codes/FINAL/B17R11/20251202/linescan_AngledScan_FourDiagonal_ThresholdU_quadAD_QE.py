''' 
Created on Dec 1 2025
@author: A_Galbiati
Diagonal linescan targeting quadrant pair A-D data analysis
Combined power from two 2025-11-30 measurements
CSV output with GS% for statistical study
''' 
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import re
import pandas as pd

# ====================== FONT & LATEX ======================
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
fm.FontProperties(fname=font_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

# ====================== USER-CHANGEABLE BASE DIRECTORY ======================
base_dir = "/Users/asca/Documents/University/Master Thesis/code/Data/20251215_final_data/20251202/VIGO_FPW01_QPD_1500_20_B17R11_251126_LB1471C_quadABCD_manual_setup_Z13.66mm_LINESCAN_Gap_AD"
# Target CSV output directory
STAT_DIR = '/Users/asca/Documents/University/Master Thesis/code/LISA_QPD_Gap/Thesis_Plots&Codes/FINAL/B17R11/statistical study'
# Target FIGURE output directory (New path)
FIG_DIR = '/Users/asca/Documents/University/Master Thesis/code/LISA_QPD_Gap/Thesis_Plots&Codes/FINAL/B17R11/fig'
os.makedirs(FIG_DIR, exist_ok=True) # Ensure the figure directory exists

# List to store the results for CSV output
overshoot_results = []
y_index = 0  # Counter for Y positions/scans

# ====================== DEFAULT PLOTTING/REGION PARAMETERS (Y_INDEX = 0) ======================
u_min_plot_default = 0.19
u_max_plot_default = 0.45

# Region definitions for the FIRST Y position
calc_min = 0.15
calc_max = 0.55
left_region_default  = (calc_min, 0.27)   # left region 
right_region_default = (0.37, calc_max)  # right region 
gap_region_default   = (0.27, 0.37)  # region to search 

# ====================== DYNAMIC PARAMETERS FOR SECOND Y POSITION (Y_INDEX = 1) ======================
# *** ADJUST THESE VALUES IF YOU PROCESS A SECOND SCAN AND NEED DIFFERENT PLOT/REGION RANGES ***
u_min_plot_y2 = 0.15
u_max_plot_y2 = 0.41
calc_min = 0.09
calc_max = 0.43
left_region_y2  = (calc_min, 0.23) 
right_region_y2 = (0.33, calc_max)
gap_region_y2   = (0.23, 0.33)


# ====================== LASER POWER (at least two measurements needed) ======================
P1 = 0.00089466 #0.0005389  # W 4.0948e-5
sigma1 = 4.2992e-7#9.3463e-7  # W5.0853e-7
P2 =  0.0009778  # W4.1115e-5
sigma2 = 4.7831e-6  # W 5.6474e-7 
w1 = 1 / (sigma1**2)
w2 = 1 / (sigma2**2)
P_combined = (P1 * w1 + P2 * w2) / (w1 + w2)

eta_100 = 1.0
eta_real = 0.8
I_eta100 = eta_100 * P_combined
I_eta080 = eta_real * P_combined
# QE reference lines (I = eta * P_combined)
eta_100 = 1.0
eta_real = 0.8
I_eta100 = eta_100 * P_combined
I_eta080 = eta_real * P_combined

# ====================== ZERO ALIGNMENT – FIXED & ROBUST ======================
def align_yaxis_zeros(ax1, ax2):
    """Force y=0 of ax1 (photocurrent) and ax2 (QE) to be at the same height."""
    ax1.figure.canvas.draw()
    y1_min, y1_max = ax1.get_ylim()
    y2_min, y2_max = ax2.get_ylim()
    if abs(y1_max - y1_min) < 1e-12:
        return
    zero_frac = (0 - y1_min) / (y1_max - y1_min)
    y2_range = y2_max - y2_min
    ax2.set_ylim(0 - zero_frac * y2_range, 0 - zero_frac * y2_range + y2_range)

# ====================== TICK FORMATTERS ======================
def curr_fmt(val, pos):
    return '0' if abs(val) < 1e-12 else f'{val:.5f}'
curr_formatter = FuncFormatter(curr_fmt)

def qe_fmt(val, pos):
    return '0' if abs(val) < 1e-12 else f'{val:.2f}'
qe_formatter = FuncFormatter(qe_fmt)

# ====================== HELPER: extract Y position if present in folder name ====================
def extract_y_um(folder_name):
    match = re.search(r'Along([\d.]+)um', folder_name)
    if match:
        return int(float(match.group(1)))
    return None

# ====================== GAP ANALYSIS FUNCTION (robust) ======================
def analyze_gap(u, I_sum, regions):
    """Return (gs_val, extremum_value, 'GS', u_ext, mean_plat)"""
    left_mask = (u >= regions['left'][0]) & (u <= regions['left'][1])
    right_mask = (u >= regions['right'][0]) & (u <= regions['right'][1])
    gap_mask = (u >= regions['gap'][0]) & (u <= regions['gap'][1])

    mean_left = np.mean(I_sum[left_mask]) if np.any(left_mask) else 0.0
    mean_right = np.mean(I_sum[right_mask]) if np.any(right_mask) else 0.0
    mean_plat = (mean_left + mean_right) / 2.0

    if not np.any(gap_mask) or mean_plat <= 0:
        return 0.0, 0.0, 'none', 0.0, mean_plat

    max_gap = np.max(I_sum[gap_mask])
    min_gap = np.min(I_sum[gap_mask])
    
    # Calculate percentages relative to the region mean
    overshoot_percent  = (max_gap / mean_plat - 1) * 100
    undershoot_percent = (min_gap / mean_plat - 1) * 100
    
    # Gap Sensitivity (GS) is the *largest magnitude* deviation (positive or negative)
    if abs(overshoot_percent) >= abs(undershoot_percent):
        # Overshoot is the extremum (positive GS)
        u_ext = u[gap_mask][np.argmax(I_sum[gap_mask])]
        val_ext = max_gap
        gs_val = overshoot_percent
    else:
        # Undershoot is the extremum (negative GS)
        u_ext = u[gap_mask][np.argmin(I_sum[gap_mask])]
        val_ext = min_gap
        gs_val = undershoot_percent

    return gs_val, val_ext, 'GS', u_ext, mean_plat


# ====================== PROCESS EACH ALONG FOLDER (single or multiple) ======================

# Check if base_dir is a single Along*um folder or its parent
if os.path.basename(base_dir).startswith("Along"):
    along_folders = [base_dir]
    y_um_base = extract_y_um(os.path.basename(base_dir))
else:
    along_folders = glob.glob(os.path.join(base_dir, "Along*um"))
    along_folders.sort()
    y_um_base = None # No single base Y

if not along_folders:
    raise FileNotFoundError(f"No Along*um folders found under {base_dir}")

# Define the preferred legend order for AD
preferred_order = [
    'quadA', 'quadD', 'quadA+quadD',
    r'Left Region', r'Right Region', r'Gap Region', r'Region Mean $I_{\rm{Plat}}$',
    r'$\eta = 1\,$A/W', r'$\eta = 0.8\,$A/W', r'$\eta_{\rm{LOAD}}$ [A/W]', r'$\eta_{\rm{GND}}$ [A/W]'
]

for along_folder in along_folders:
    along_name = os.path.basename(along_folder)
    y_um = extract_y_um(os.path.basename(along_folder)) or y_um_base
    if y_um is None:
        print(f"Skipping {os.path.basename(along_folder)} – no Y position found")
        continue

    # --- DYNAMIC PARAMETER SELECTION ---
    if y_index == 0:
        u_min_plot = u_min_plot_default
        u_max_plot = u_max_plot_default
        current_regions = {
            'left': left_region_default,
            'right': right_region_default,
            'gap': gap_region_default
        }
    elif y_index == 1:
        print(f"!!! Applying custom parameters for the second Y scan (Y={y_um} um) !!!")
        u_min_plot = u_min_plot_y2
        u_max_plot = u_max_plot_y2
        current_regions = {
            'left': left_region_y2,
            'right': right_region_y2,
            'gap': gap_region_y2
        }
    else:
        u_min_plot = u_min_plot_default
        u_max_plot = u_max_plot_default
        current_regions = {
            'left': left_region_default,
            'right': right_region_default,
            'gap': gap_region_default
        }
    y_index += 1
    
    print(f"\n=== Processing AD – Along{y_um}.0um (Y={y_um} um) ===")

    # fig_dir = os.path.join(along_folder, "fig_B17R11") # Original Line
    # os.makedirs(fig_dir, exist_ok=True) # Original Line
    fig_dir = FIG_DIR # Use the globally defined figure directory

    load_candidates = glob.glob(os.path.join(along_folder, "*_load2.pkl")) or glob.glob(os.path.join(along_folder, "*_load.pkl"))
    gnd_candidates = glob.glob(os.path.join(along_folder, "*_gnd.pkl"))

    # Initialize results with all required CSV fields
    current_results = {
        'Y_position_um': y_um, 
        'GS_LOAD_%': 0.0, 
        'GS_GND_%': 0.0,
        'Max_in_gap_LOAD_µA': 0.0,
        'Max_in_gap_GND_µA': 0.0,
        'Mean_region_LOAD_µA': 0.0,
        'Mean_region_GND_µA': 0.0
    }
    
    # Placeholder variables for console output
    gs_load, gs_gnd = 0.0, 0.0
    
    # --- Function to process a single file (LOAD or GND) ---
    def process_file(pkl_path, kind_label, current_regions, u_min_plot, u_max_plot, current_results):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        u_array = np.array(data['rawdata']['u_position'])
        mask_plot = (u_array >= u_min_plot) & (u_array <= u_max_plot)
        u_plot = u_array[mask_plot]

        # quadA and quadD for AD gap
        quadA = np.mean(data['rawdata']['quadA']['dmm00_curr_amp'], axis=1)[mask_plot]
        quadD = np.mean(data['rawdata']['quadD']['dmm00_curr_amp'], axis=1)[mask_plot]
        I_sum = quadA + quadD

        gs_val, val_ext, which_kind, u_ext, mean_plat = analyze_gap(u_plot, I_sum, current_regions)
        
        # Store results for CSV
        current_results[f'GS_{kind_label}_%'] = gs_val
        current_results[f'Max_in_gap_{kind_label}_µA'] = val_ext * 1e6
        current_results[f'Mean_region_{kind_label}_µA'] = mean_plat * 1e6
        
        # primary figure
        fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')

        # --- Region Highlighting ---
        ax.axvspan(current_regions['left'][0], current_regions['left'][1], alpha=0.12, color='tab:blue',   label=r'Left Region')
        ax.axvspan(current_regions['right'][0], current_regions['right'][1], alpha=0.12, color='tab:red', label=r'Right Region')
        ax.axvspan(current_regions['gap'][0], current_regions['gap'][1], alpha=0.22, color='gray',       label=r'Gap Region')

        # --- Plotting Data ---
        # Increased line width: 2.2
        ax.plot(u_plot, quadA, label='quadA', color='tab:blue', lw=2.2)
        ax.plot(u_plot, quadD, label='quadD', color='tab:red', lw=2.2) # quadD is traditionally red
        # Increased line width: 3.0
        ax.plot(u_plot, I_sum, '--', color="#992f7f", lw=3.0, alpha=0.7, label='quadA+quadD')

        # --- Reference Lines ---
        #ax.axhline(mean_plat, color='gray', ls=':', lw=1.5, label=r'Region Mean $I_{\rm{Plat}}$')
        # Increased line width: 2.2
        ax.axhline(I_eta100, color="#c39d7a", linestyle='--', linewidth=2.2, alpha=0.95, label=r'$\eta = 1\,$A/W')
        ax.axhline(I_eta080, color='#8b5a2b', linestyle='--', linewidth=2.2, alpha=0.95, label=r'$\eta = 0.8\,$A/W')

        # --- QE Axis ---
        QE = I_sum / P_combined
        ax_qe = ax.twinx()
        qe_color = "#00ff88" if kind_label == "LOAD" else "#44fd00"
        # Increased line width: 2.8
        ax_qe.plot(u_plot, QE, color=qe_color, linewidth=2.8, alpha=0.22, label=rf'$\eta_{{{kind_label}}}$ [A/W]')
        # Increased font size: 16
        ax_qe.set_ylabel(r'Responsivity $\eta$ [A/W]', color='#8b5a2b', fontsize=16)
        # Increased tick label size: 13
        ax_qe.tick_params(axis='y', colors='#8b5a2b', length=8, width=1.5, direction='in', labelsize=13)
        ax_qe.yaxis.set_major_formatter(qe_formatter)
        ax_qe.set_ylim(0, 1.05)

        # --- Extremum Marker (Red/Orange for positive GS, Blue for negative GS/undershoot) ---
        marker_color = '#1f77b4' # Default to blue
        mec_color = 'darkblue'   # Default to darkblue
        if which_kind != 'none':
            if kind_label == 'LOAD':
                marker_color = '#d62728' if gs_val >= 0 else '#1f77b4' 
                mec_color = 'darkred' if gs_val >= 0 else 'darkblue'
            else: # GND
                marker_color = '#ff7f0e' if gs_val >= 0 else '#1f77b4' 
                mec_color = '#cc5e00' if gs_val >= 0 else 'darkblue'
            
            # Increased marker edge width: 2.2; Increased marker size: 12
            ax.plot(u_ext, val_ext, 'o', color=marker_color,
                    mec=mec_color, mew=2.2, ms=12, zorder=10)

        # --- Inset with GS Terminology ---
        # Adjusted position: [0.09, 0.50, 0.38, 0.16]
        inset = ax.inset_axes([0.09, 0.50, 0.38, 0.16], transform=ax.transAxes)
        inset.axis('off')
        # Increased line width: 3.0; Increased marker edge width: 2.2; Increased marker size: 12
        handle = Line2D([0], [0], color="#992f7f", lw=3.0, ls='--',
                        marker='o', mfc=marker_color, mec=mec_color,
                        mew=2.2, ms=12)
        
        # Increased font size: 20
        inset.legend([handle], [rf'GS ({kind_label}): \textbf{{{gs_val:+.2f}\%}}'],
                     loc='center', fontsize=20, frameon=False, handletextpad=0.8, handlelength=1.8)

        # --- Final Plot Styling ---
        # Increased font size: 16
        ax.set_title(rf'\textbf{{DC Photocurrent {kind_label} – Gap AD (Y={y_um}\,\textmu{{m}})}}', fontsize=16, fontweight='bold', pad=10)
        # Increased font size: 16
        ax.set_xlabel(r'Perpendicular offset $u$ [mm]', fontsize=16)
        ax.set_ylabel(r'Photocurrent [A]', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.6)
        # Increased tick label size: 13
        ax.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in')
        ax.yaxis.set_major_formatter(curr_formatter)
        ax.set_xlim(u_min_plot+0.015, u_max_plot-0.015)
        ax.set_ylim(bottom=0)

        plt.draw()
        align_yaxis_zeros(ax, ax_qe)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_qe.get_legend_handles_labels()
        
        # --- Legend Filtering and Ordering (Single Column) ---
        combined_handles = lines1 + lines2
        combined_labels = labels1 + labels2
        
        # Adjust preferred_order to only include the QE line for the current kind_label
        ordered_legend_labels = []
        for label in preferred_order:
             if label == r'$\eta_{\rm{LOAD}}$ [A/W]' and kind_label != 'LOAD': continue
             if label == r'$\eta_{\rm{GND}}$ [A/W]' and kind_label != 'GND': continue
             ordered_legend_labels.append(label)

        label_to_handle = {}
        for h, l in zip(combined_handles, combined_labels):
            if l not in label_to_handle:
                label_to_handle[l] = h

        final_handles = []
        final_labels = []
        for label in ordered_legend_labels: 
            if label in label_to_handle:
                final_handles.append(label_to_handle[label])
                final_labels.append(label)
        
        # Increased font size
        ax.legend(final_handles, final_labels, 
                  fontsize=14, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                  frameon=True, edgecolor='black', ncol=1) 

        # Save to the new figure directory
        base_name = os.path.splitext(os.path.basename(pkl_path))[0]
        save_path = os.path.join(fig_dir, f"DC_Photocurrent_Diagonal_AD_Sum_{along_name}_{kind_label}.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)

        return gs_val

    # --- Execute processing ---
    for p in load_candidates:
        gs_load = process_file(p, "LOAD", current_regions, u_min_plot, u_max_plot, current_results)

    for p in gnd_candidates:
        gs_gnd = process_file(p, "GND", current_regions, u_min_plot, u_max_plot, current_results)

    # Append results only if data was processed
    if load_candidates or gnd_candidates:
        # Reorder keys to match the desired CSV output format
        ordered_results = {
            'Y_position_um': current_results['Y_position_um'],
            'GS_LOAD_%': current_results['GS_LOAD_%'],
            'GS_GND_%': current_results['GS_GND_%'],
            'Max_in_gap_LOAD_µA': current_results['Max_in_gap_LOAD_µA'],
            'Max_in_gap_GND_µA': current_results['Max_in_gap_GND_µA'],
            'Mean_region_LOAD_µA': current_results['Mean_region_LOAD_µA'],
            'Mean_region_GND_µA': current_results['Mean_region_GND_µA'],
        }
        overshoot_results.append(ordered_results)
    
    # Console summary
    print(f" → LOAD GS ({'Overshoot' if gs_load >= 0 else 'Undershoot' if gs_load != 0.0 else 'N/A'}) = {gs_load:+.2f}%")
    print(f" → GND GS ({'Overshoot' if gs_gnd >= 0 else 'Undershoot' if gs_gnd != 0.0 else 'N/A'}) = {gs_gnd:+.2f}%")


# ====================== CSV SUMMARY OUTPUT ======================

if overshoot_results:
    # 1. Define the exact folder path
    os.makedirs(STAT_DIR, exist_ok=True)
    
    # 2. Create the DataFrame and save
    df_results = pd.DataFrame(overshoot_results)
    
    # Rename columns to exactly match the requested format
    df_results.rename(columns={
        'Y_position_um': 'X_position_um',
        'GS_LOAD_%': 'Overshoot_LOAD_%',
        'GS_GND_%': 'Overshoot_GND_%',
        'Mean_region_LOAD_µA': 'Mean_plateau_LOAD_µA',
        'Mean_region_GND_µA': 'Mean_plateau_GND_µA'
    }, inplace=True)
    
    # Ensure correct column order
    column_order = [
        'X_position_um',
        'Overshoot_LOAD_%',
        'Overshoot_GND_%',
        'Max_in_gap_LOAD_µA',
        'Max_in_gap_GND_µA',
        'Mean_plateau_LOAD_µA',
        'Mean_plateau_GND_µA'
    ]
    df_results = df_results[column_order]
    
    csv_filename = "DiagonalScan_GapSensitivity_Results_AD_Y.csv"
    csv_path = os.path.join(STAT_DIR, csv_filename)
    df_results.to_csv(csv_path, index=False, float_format='%.3f') # Use 3 decimal places for µA values
    
    print("\n--- CSV SUMMARY OUTPUT ---")
    print(f"Gap Sensitivity results (Diagonal AD) successfully saved to:")
    print(f"    {csv_path}")
    print("\nHead of the results table:")
    print(df_results.head().to_string(index=False))