''' 
Created on Dec 1 2025
@author: A_Galbiati
Diagonal linescan targeting quadrant pair D-C data analysis
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

# ====================== BASE DIRECTORY & OUTPUT LIST ======================
base_dir = "/Users/asca/Documents/University/Master Thesis/code/Data/all quadrants/20251202/VIGO_FPW01_QPD_1500_20_B17R11_251126_LB1471C_quadABCD_manual_setup_Z13.66mm_LINESCAN_Gap_DC"
# Target CSV output directory
STAT_DIR = '/Users/asca/Documents/University/Master Thesis/code/B17R11_1/statistical study'
# Target FIGURE output directory (New path)
FIG_DIR = '/Users/asca/Documents/University/Master Thesis/code/B17R11_1/fig'
os.makedirs(FIG_DIR, exist_ok=True) # Ensure the figure directory exists

# List to store the results for CSV output
overshoot_results = []
y_index = 0 # Counter for Y positions/scans

# ====================== DEFAULT PLOTTING/REGION PARAMETERS (Y_INDEX = 0) ======================
# U plotting range – adjust only if your DC data is very shifted
u_min_plot_default = -0.44
u_max_plot_default = -0.18

# Region definitions for the FIRST Y position
calc_min = -0.55
calc_max = -0.13
left_region_default  = (calc_min, -0.36)   # left region
right_region_default = (-0.26, calc_max)   # right region
gap_region_default   = (-0.36, -0.26)   # gap region

# ====================== DYNAMIC PARAMETERS FOR SECOND Y POSITION (Y_INDEX = 1) ======================
# *** ADJUST THESE VALUES IF YOU PROCESS A SECOND SCAN AND NEED DIFFERENT PLOT/REGION RANGES ***
u_min_plot_y2 = -0.38
u_max_plot_y2 = -0.12
calc_min = -0.4
calc_max = -0.13
left_region_y2  = (calc_min, -0.30)
right_region_y2 = (-0.20, calc_max)
gap_region_y2   = (-0.30, -0.20)

# ====================== LASER POWER – IDENTICAL TO ALL SCANS ======================
P1 = 0.00089466 
sigma1 = 4.2992e-7
P2 =  0.0009778  
sigma2 = 4.7831e-6  
w1 = 1 / (sigma1**2)
w2 = 1 / (sigma2**2)
P_combined = (P1 * w1 + P2 * w2) / (w1 + w2)


eta_100 = 1.0
eta_real = 0.8
I_eta100 = eta_100 * P_combined
I_eta080 = eta_real * P_combined

# ====================== ZERO ALIGNMENT ======================
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
def curr_fmt(val, pos): return '0' if abs(val) < 1e-12 else f'{val:.5f}'
def qe_fmt(val, pos):   return '0' if abs(val) < 1e-12 else f'{val:.2f}'
curr_formatter = FuncFormatter(curr_fmt)
qe_formatter   = FuncFormatter(qe_fmt)

# ====================== EXTRACT Y POSITION ======================
def extract_y_um(folder_name):
    match = re.search(r'Along([\d.]+)um', folder_name)
    return int(float(match.group(1))) if match else None

# ====================== GAP ANALYSIS ======================
def analyze_gap(u, I_sum, regions):
    """Return (gs_val, extremum_value, 'GS', u_ext, mean_plat)"""
    left_mask  = (u >= regions['left'][0])  & (u <= regions['left'][1])
    right_mask = (u >= regions['right'][0]) & (u <= regions['right'][1])
    gap_mask   = (u >= regions['gap'][0])   & (u <= regions['gap'][1])

    mean_left  = np.mean(I_sum[left_mask])  if np.any(left_mask)  else 0
    mean_right = np.mean(I_sum[right_mask]) if np.any(right_mask) else 0
    mean_plat  = (mean_left + mean_right) / 2.0

    if not np.any(gap_mask) or mean_plat <= 0:
        return 0.0, 0.0, 'none', 0.0, mean_plat

    max_gap = np.max(I_sum[gap_mask])
    min_gap = np.min(I_sum[gap_mask])

    # Calculate percentages relative to the region mean
    overshoot_percent  = (max_gap / mean_plat - 1) * 100
    undershoot_percent = (min_gap / mean_plat - 1) * 100

    # Gap Sensitivity (GS) is the *largest magnitude* deviation
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

# ====================== PROCESS EACH ALONG FOLDER ======================
along_folders = glob.glob(os.path.join(base_dir, "Along*um"))
along_folders.sort()

# Define the preferred legend order for DC
preferred_order_dc = [
    'quadD', 'quadC', 'quadD+quadC',
    r'Left Region', r'Right Region', r'Gap Region', r'Region Mean $I_{\rm{Plat}}$',
    r'$\eta = 1\,$A/W', r'$\eta = 0.8\,$A/W'
]

for along_folder in along_folders:
    along_name = os.path.basename(along_folder)
    y_um = extract_y_um(along_name)
    if y_um is None:
        print(f"Could not extract Y from {along_name} → skipping")
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

    print(f"\n=== Processing DC – {along_name} (Y={y_um} um) ===")
    
    # fig_dir = os.path.join(along_folder, "fig_B17R11") # Original Line
    # os.makedirs(fig_dir, exist_ok=True) # Original Line
    fig_dir = FIG_DIR # Use the globally defined figure directory

    load_files = glob.glob(os.path.join(along_folder, "*_load2.pkl")) or glob.glob(os.path.join(along_folder, "*_load.pkl"))
    gnd_files  = glob.glob(os.path.join(along_folder, "*_gnd.pkl"))

    if not load_files and not gnd_files:
        print(" → No data files found")
        continue

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
    val_load, val_gnd = 0.0, 0.0
    mean_plat_load, mean_plat_gnd = 0.0, 0.0

    # --------------------- Function to process a single file (LOAD or GND) ---------------------
    def process_file(pkl_path, kind_label, current_regions, u_min_plot, u_max_plot):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        u_array = data['rawdata']['u_position']
        mask_plot = (u_array >= u_min_plot) & (u_array <= u_max_plot)
        u_plot = u_array[mask_plot]

        # DC specific quadrants
        quadD = np.mean(data['rawdata']['quadD']['dmm00_curr_amp'], axis=1)[mask_plot]
        quadC = np.mean(data['rawdata']['quadC']['dmm00_curr_amp'], axis=1)[mask_plot]
        I_sum = quadD + quadC

        gs_val, val_ext, kind_ext, u_ext, mean_plat = analyze_gap(u_plot, I_sum, current_regions)

        fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')

        # --- Region Highlighting ---
        # Red/Green/Gray for DC
        ax.axvspan(current_regions['left'][0], current_regions['left'][1], alpha=0.12, color='tab:red',   label=r'Left Region')
        ax.axvspan(current_regions['right'][0], current_regions['right'][1], alpha=0.12, color='tab:green', label=r'Right Region')
        ax.axvspan(current_regions['gap'][0], current_regions['gap'][1], alpha=0.22, color='gray',       label=r'Gap Region')

        # Quadrant plots (Increased line width: 2.2)
        ax.plot(u_plot, quadD, label='quadD', color='tab:red',   lw=2.2) 
        ax.plot(u_plot, quadC, label='quadC', color='tab:green', lw=2.2)
        # Increased line width: 3.0
        ax.plot(u_plot, I_sum, '--', color="#992f7f", alpha=0.7, lw=3.0, label='quadD+quadC')

        # --- Reference Lines (Increased line width: 2.2) ---
        #ax.axhline(mean_plat, color='gray', ls=':', lw=1.5, label=r'Region Mean $I_{\rm{Plat}}$')
        ax.axhline(I_eta100, color="#c39d7a", ls='--', lw=2.2, alpha=0.95, label=r'$\eta = 1\,$A/W')
        ax.axhline(I_eta080, color='#8b5a2b', ls='--', lw=2.2, alpha=0.95, label=r'$\eta = 0.8\,$A/W')

        # QE Axis
        QE = I_sum / P_combined
        ax_qe = ax.twinx()
        qe_color = "#00ff88" if kind_label == "LOAD" else "#44fd00"
        # Increased line width: 2.8
        ax_qe.plot(u_plot, QE, color=qe_color, lw=2.8, alpha=0.22, label=rf'$\eta_{{{kind_label}}}$ [A/W]')
        # Increased font size: 16
        ax_qe.set_ylabel(r'Quantum Efficiency $\eta$ [A/W]', color='#8b5a2b', fontsize=16)
        # Increased tick label size: 13
        ax_qe.tick_params(axis='y', colors='#8b5a2b', length=8, width=1.5, direction='in', labelsize=13)
        ax_qe.yaxis.set_major_formatter(qe_formatter)
        ax_qe.set_ylim(0, 1.05)

        # --- Extremum Marker ---
        if kind_ext != 'none':
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
        ax.set_title(rf'\textbf{{DC Photocurrent {kind_label} – Gap DC (Y={y_um}\,\textmu{{m}})}}', fontsize=16, fontweight='bold', pad=10)
        # Increased font size: 16
        ax.set_xlabel(r'Perpendicular offset $u$ [mm]', fontsize=16)
        ax.set_ylabel(r'Photocurrent [A]', fontsize=16)
        ax.grid(True, ls='--', alpha=0.6)
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

        # Filter preferred order based on whether it's LOAD or GND
        current_preferred_order = preferred_order_dc + [rf'$\eta_{{{kind_label}}}$ [A/W]']
        
        label_to_handle = {}
        for h, l in zip(combined_handles, combined_labels):
            if l not in label_to_handle:
                label_to_handle[l] = h

        final_handles = []
        final_labels = []
        for label in current_preferred_order:
            if label in label_to_handle:
                final_handles.append(label_to_handle[label])
                final_labels.append(label)

        # Increased font size: 14
        ax.legend(final_handles, final_labels,
                  fontsize=14, loc='upper right', bbox_to_anchor=(0.98, 0.98),
                  frameon=True, edgecolor='black', ncol=1)

        save_path = os.path.join(fig_dir, f"DC_Photocurrent_Diagonal_DC_Sum_{along_name}_{kind_label}.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)

        return gs_val, val_ext, mean_plat

    # ============================= LOAD =============================
    if load_files:
        gs_load, val_load, mean_plat_load = process_file(load_files[0], "LOAD", current_regions, u_min_plot, u_max_plot)

        # Store results for CSV
        current_results['GS_LOAD_%'] = gs_load
        current_results['Max_in_gap_LOAD_µA'] = val_load * 1e6
        current_results['Mean_region_LOAD_µA'] = mean_plat_load * 1e6


    # ============================= GND =============================
    if gnd_files:
        gs_gnd, val_gnd, mean_plat_gnd = process_file(gnd_files[0], "GND", current_regions, u_min_plot, u_max_plot)

        # Store results for CSV
        current_results['GS_GND_%'] = gs_gnd
        current_results['Max_in_gap_GND_µA'] = val_gnd * 1e6
        current_results['Mean_region_GND_µA'] = mean_plat_gnd * 1e6


    # Append results only if data was processed
    if load_files or gnd_files:
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

    csv_filename = "DiagonalScan_GapSensitivity_Results_DC_Y.csv"
    csv_path = os.path.join(STAT_DIR, csv_filename)
    df_results.to_csv(csv_path, index=False, float_format='%.3f') # Use 3 decimal places for µA values

    print("\n--- CSV SUMMARY OUTPUT ---")
    print(f"Gap Sensitivity results (Diagonal DC) successfully saved to:")
    print(f"    {csv_path}")
    print("\nHead of the results table:")
    print(df_results.head().to_string(index=False))

print("\nAll DC diagonal scans processed")
