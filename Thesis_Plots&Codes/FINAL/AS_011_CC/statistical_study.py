'''
Statistical comparison of Gap Sensitivity (Overshoot %) across QPD devices
Author: Ascanio Galbiati 
Date: Dec 04 2025 
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch

# === PLOTTING CONFIGURATION === #
PLOT_B17R11_LOAD = True
# Adjusted label font sizes (Slightly decreased for the final aesthetic)
VALUE_LABEL_FONTSIZE = 17
DEVICE_LABEL_FONTSIZE = 18

# === AXES CONFIGURATION === #
Y_LIM_BOTTOM = -9.3  # Final Requested Y limit bottom
Y_LIM_TOP = 20     # Final Requested Y limit top
DEVICE_GROUP_SPACING = 0.75 
bar_width = 0.50 # Slightly increased width

# === YOUR STYLE — SACRED AND UNTOUCHED (FONT SIZES IMPLEMENTED) === #
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
cmu_serif = fm.FontProperties(fname=font_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']
# FONT SIZE IMPLEMENTATION 
plt.rcParams['axes.labelsize'] = 18 
plt.rcParams['axes.titlesize'] = 19 
plt.rcParams['xtick.labelsize'] = 16 
plt.rcParams['ytick.labelsize'] = 16 
plt.rcParams['legend.fontsize'] = 16 

# --- NEW COLOR DEFINITIONS (FLIPPED: GND is now opaque, LOAD is transparent) ---
# Run 1 (GND: Red, LOAD: Transparent Red)
RUN1_COLOR_GND = "#d62728" 
RUN1_COLOR_LOAD = "#d6272870" 
# Run 2 (GND: Blue, LOAD: Transparent Blue)
RUN2_COLOR_GND = "#1f77b4" 
RUN2_COLOR_LOAD = "#1f77b470" 
# Fiber Laser (Orange)
FL_COLOR = "#ff7f0e"

# === DEVICES (Color/Label logic updated) === #
devices = {
    # Existing devices
    "AS_011_CC": {
        "path": "/Users/asca/Documents/University/Master Thesis/code/LISA_QPD_Gap/Thesis_Plots&Codes/FINAL/AS_011_CC/statistical study",
        "color": RUN1_COLOR_GND, # Base Color (GND)
        "label": r'\textbf{AS\_011\_CC}', 
        "type": "Run1"
    },
    "AS_011_CC_FL": {
        "path": "/Users/asca/Documents/University/Master Thesis/code/LISA_QPD_Gap/Thesis_Plots&Codes/FINAL/AS_011_CC/Fiber Laser/statistical study",
        "color": FL_COLOR, # Fiber Laser Color
        "label": r'\textbf{AS\_011\_CC}', 
        "type": "Fiber Laser"
    },
    "B17R11": {
        "path": "/Users/asca/Documents/University/Master Thesis/code/LISA_QPD_Gap/Thesis_Plots&Codes/FINAL/B17R11/statistical study",
        "color": RUN2_COLOR_GND, # Base Color (GND)
        "label": r'\textbf{B17R11}',
        "type": "Run2"
    },
    "NS089008": {
        "path": "/Users/asca/Documents/University/Master Thesis/code/NS089008/statistical study",
        "color": RUN1_COLOR_GND, # Base Color (GND)
        "label": r'\textbf{NS089008}',
        "type": "Run1"
    },
    "AK_004": {
        "path": "/Users/asca/Documents/University/Master Thesis/code/LISA_QPD_Gap/Thesis_Plots&Codes/FINAL/AK_004/statistical study",
        "color": RUN1_COLOR_GND, # Base Color (GND)
        "label": r'\textbf{AK\_004}', 
        "type": "Run1"
    },
    "AS_004": {
        "path": "/Users/asca/Documents/University/Master Thesis/code/LISA_QPD_Gap/Thesis_Plots&Codes/FINAL/AS_004/20251104/statistical study",
        "path_ND": "/Users/asca/Documents/University/Master Thesis/code/LISA_QPD_Gap/Thesis_Plots&Codes/FINAL/AS_004/20251110/statistical study",
        "color": RUN1_COLOR_GND, # Base Color (GND)
        "label": r'\textbf{AS\_004}', 
        "type": "Run1"
    }
}

# --- Reordering devices for plotting ---
plot_order = [
    "AS_011_CC", "AS_011_CC_FL", 
    "AS_004", 
    "AK_004", 
    "B17R11", 
    "NS089008"
]

# Keep only existing folders and items in plot_order
devices = {k: v for k, v in devices.items() if os.path.exists(v["path"])}
if "AS_011_CC_FL" in devices and not os.path.exists(devices["AS_011_CC_FL"]["path"]):
    del devices["AS_011_CC_FL"]

plot_order = [dev for dev in plot_order if dev in devices]

# --- File name mapping (unchanged) ---
B17R11_FILE_MAP = {
    "AB": "DiagonalScan_GapSensitivity_Results_AB_Y.csv",
    "AD": "DiagonalScan_GapSensitivity_Results_AD_Y.csv",
    "BC": "DiagonalScan_GapSensitivity_Results_BC_Y.csv",
    "CD": "DiagonalScan_GapSensitivity_Results_DC_Y.csv",
}
AS011CC_FL_FILE_MAP = {
    "AD": "HorizontalScan_GapSensitivity_Results_AD.csv",
    "BC": "HorizontalScan_GapSensitivity_Results_BC.csv",
    "AB": "VerticalScan_GapSensitivity_Results_AB.csv",
    "CD": "VerticalScan_GapSensitivity_Results_CD.csv",
}
AK004_FILE_MAP = {
    "AB_X": "Overshoot_Results_AB_X_up_to_7300um.csv",
    "AD_Y": "Overshoot_Results_AD_Y_up_to_2200um.csv",
    "BC_Y": "Overshoot_Results_BC_Y_from_2300um.csv",
    "CD_X": "Overshoot_Results_CD_X_from_7300um.csv",
}
# AS004 File Maps - CD_X is missing on purpose for the base path
AS004_FILE_MAP = {
    "AB_X": "Overshoot_Results_AB_X_from6650um.csv",
    "AD_Y": "Overshoot_Results_AD_Y_up_to_2400um.csv",
    "BC_Y": "Overshoot_Results_BC_Y_from_2300um.csv",
}
AS004_ND_FILE_MAP = {
    "AD_Y": "Overshoot_Results_AD_Y_up_to_2400um.csv",
}


# === COLLECT DATA + SAFETY CHECKS === #
results = {}
print("\n" + "="*80)
print("DATA COLLECTION & SAFETY CHECKS")
print("="*80)

for dev_name in plot_order:
    info = devices[dev_name]

    is_B17R11 = (dev_name == "B17R11")
    is_FL = (dev_name == "AS_011_CC_FL")
    is_AK004 = (dev_name == "AK_004")
    is_AS004 = (dev_name == "AS_004")
    
    # Select the correct file map (Logic remains the same)
    if is_B17R11:
        csv_files_config = B17R11_FILE_MAP.items()
    elif is_FL:
        csv_files_config = AS011CC_FL_FILE_MAP.items()
    elif is_AK004:
        csv_files_config = AK004_FILE_MAP.items()
    elif is_AS004:
        csv_files_config = AS004_FILE_MAP.items()
    else: 
        # Default for other devices (NS089008, AS_011_CC)
        csv_files_config = [
            ("AB_X", "Overshoot_Results_AB_X_up_to_7200um.csv"),
            ("BC_Y", "Overshoot_Results_BC_Y_from_2300um.csv"),
            ("AD_Y", "Overshoot_Results_AD_Y_up_to_2100um.csv"),
            ("CD_X", "Overshoot_Results_CD_X_from_7300um.csv"),
        ]
    
    overshoot_load = []
    overshoot_gnd  = []
    position_counts = {}
    
    clean_label = info['label'].replace(r'\textbf{', '').replace('}', '').replace('_', r'\_')
    print(f"\n→ Processing device: {clean_label} ({dev_name})")

    # Define all paths and configs to check
    file_map_list = [(info["path"], csv_files_config, "BASE")]

    if is_AS004 and "path_ND" in info and os.path.exists(info["path_ND"]):
        # Note: For AS_004, we check all BASE files, then check ND files (only AD_Y)
        file_map_list.append((info["path_ND"], AS004_ND_FILE_MAP.items(), "ND"))


    for base_path, files_config, data_source in file_map_list:
        for pair_name, csv_file in files_config:
            csv_path = os.path.join(base_path, csv_file.strip())

            # --- CRITICAL FIX: Removed the skip logic for BASE AD_Y data ---
            # if is_AS004 and data_source == "BASE" and pair_name == "AD_Y":
            #      if "path_ND" in info and os.path.exists(os.path.join(info["path_ND"], AS004_ND_FILE_MAP["AD_Y"])):
            #         continue
            # --- END CRITICAL FIX ---

            if not os.path.exists(csv_path):
                if data_source == "BASE":
                     print(f"   [MISSING] {csv_file}")
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"   [ERROR] Could not read {csv_file}: {e}")
                continue

            pos_cols = ['X_position_um', 'Y_position_um']
            pos_col = next((col for col in pos_cols if col in df.columns), None)
            positions = df[pos_col].dropna().astype(int).tolist() if pos_col else []
            pos_type = pos_col[0] if pos_col else '?'
            
            n_pos = len(positions)
            position_counts.setdefault(pair_name, []).extend(positions)

            # --- Data Extraction Logic ---
            if is_FL:
                gs_vals = pd.to_numeric(df['GS_%'], errors='coerce').dropna()
                overshoot_gnd.extend(gs_vals)
                print(f"   [OK] {csv_file:45} → {n_pos:2d} {pos_type}-positions (pair {pair_name}). **Fiber Laser GS (GND).**")
            else:
                load_vals = pd.to_numeric(df.get('Overshoot_LOAD_%', pd.Series()), errors='coerce').dropna()
                gnd_vals  = pd.to_numeric(df.get('Overshoot_GND_%', pd.Series()), errors='coerce').dropna()
                overshoot_load.extend(load_vals)
                overshoot_gnd.extend(gnd_vals)
                
                msg_suffix = "**No sign flip.**" if is_B17R11 else "**ND Filter Data (AD).**" if is_AS004 and data_source == "ND" else ""
                print(f"   [OK] {csv_file:45} → {n_pos:2d} {pos_type}-positions (pair {pair_name}) {msg_suffix}")
            # --- End Data Extraction Logic ---


    if len(overshoot_load) == 0 and len(overshoot_gnd) == 0:
        print(f"   [WARNING] No valid data found for {dev_name} → skipping")
        continue

    n_total_load = len(overshoot_load)
    n_total_gnd  = len(overshoot_gnd)
    n_total_measurements = max(n_total_load, n_total_gnd)


    mean_load = np.mean(overshoot_load) if n_total_load > 0 and not is_FL else np.nan
    std_load  = np.std(overshoot_load, ddof=1) if n_total_load > 0 and not is_FL else np.nan
    mean_gnd = np.mean(overshoot_gnd) if n_total_gnd > 0 else np.nan
    std_gnd  = np.std(overshoot_gnd, ddof=1) if n_total_gnd > 0 else np.nan


    results[dev_name] = {
        "mean_load": mean_load,
        "std_load": std_load,
        "mean_gnd": mean_gnd,
        "std_gnd": std_gnd,
        "n_total_load": n_total_load,
        "n_total_gnd": n_total_gnd,
        "n_total": n_total_measurements,
        "color": info["color"], # Base Color (GND)
        "type": info["type"],
        "label": info["label"],
        "is_FL": is_FL,
        "is_B17R11": is_B17R11,
    }

    print(f"   → Total measurements used: {n_total_measurements} (LOAD: {n_total_load}, GND: {n_total_gnd})")


# --- Helper function for drawing horizontal mean label (Mean .2f) ---
def draw_mean_label_horizontal(ax, x_pos, mean, std, color):
    
    # Create the text string (using LaTeX bf for text and .2f format)
    text_label = rf'$\mathbf{{{mean:+.2f}\%}}$' 
    
    # Calculate error bar limits
    yerr_top = mean + std
    yerr_bottom = mean - std
    
    # Determine the vertical placement position (outside the error bar)
    if mean >= 0:
        # Place above the top cap
        y_pos = yerr_top + 0.3 # Reduced offset slightly for the smaller Y-limits
        va = 'bottom'
    else:
        # Place below the bottom cap
        y_pos = yerr_bottom - 0.3 # Reduced offset slightly for the smaller Y-limits
        va = 'top'

    ax.text(x_pos, y_pos,
            text_label,
            ha='center', va=va, fontsize=VALUE_LABEL_FONTSIZE, 
            fontweight='bold', color=color, rotation=0, zorder=5)


# === PLOTTING (FINAL REVISION) === #
fig, ax = plt.subplots(figsize=(11, 7.5), layout='constrained')


x_pos_map = {}
device_group_centers = {} 
current_x_index = 0

# 1. Determine X-Positions and Group Centers (Logic unchanged)
i = 0
while i < len(plot_order):
    dev_name = plot_order[i]
    
    # --- AS_011_CC Group (LOAD, GND, FL) ---
    if dev_name == "AS_011_CC" and "AS_011_CC_FL" in plot_order:
        # 1. AS_011_CC LOAD
        x_pos_map["AS_011_CC_LOAD"] = current_x_index - bar_width
        # 2. AS_011_CC GND
        x_pos_map["AS_011_CC_GND"] = current_x_index 
        
        # 3. AS_011_CC_FL (Fiber Laser) - Place immediately next to GND bar
        fl_x_pos = current_x_index + bar_width
        x_pos_map["AS_011_CC_FL"] = fl_x_pos
        
        # Group Center: Halfway between LOAD center and FL center
        group_center = (x_pos_map["AS_011_CC_LOAD"] + fl_x_pos) / 2
        device_group_centers[group_center] = devices["AS_011_CC"]['label']
        
        current_x_index = fl_x_pos + bar_width/2 + DEVICE_GROUP_SPACING 
        i += 2 
        
    # --- Standard 2-Bar Devices (LOAD, GND) ---
    else:
        # 1. LOAD Bar
        x_pos_map[dev_name + "_LOAD"] = current_x_index - bar_width / 2
        # 2. GND Bar
        x_pos_map[dev_name + "_GND"] = current_x_index + bar_width / 2
        
        # Group Center is exactly at current_x_index
        device_group_centers[current_x_index] = devices[dev_name]['label']
        
        current_x_index += bar_width + DEVICE_GROUP_SPACING 
        i += 1

# Extract the final x-tick positions and labels for the axis
x_tick_positions = sorted(device_group_centers.keys())
x_tick_labels = [device_group_centers[x] for x in x_tick_positions]

# Set limits and draw zero line
ax.set_ylim(Y_LIM_BOTTOM, Y_LIM_TOP)
ax.axhline(0, color='black', linestyle=':', linewidth=2.5, zorder=4)

# 2. Plot bars and labels
for dev_name in results:
    r = results[dev_name]
    
    plot_load = (not r["is_FL"] and not np.isnan(r["mean_load"]))
    
    # Determine colors for this device based on the final FLIP
    if r["type"] == "Run1":
        bar_color_gnd = RUN1_COLOR_GND
        bar_color_load = RUN1_COLOR_LOAD
    elif r["type"] == "Run2":
        bar_color_gnd = RUN2_COLOR_GND
        bar_color_load = RUN2_COLOR_LOAD
    else: # Fiber Laser
        bar_color_gnd = FL_COLOR
        bar_color_load = FL_COLOR # Not used, but set for consistency

    # 1. LOAD PLOTTING (Transparent Bar, Transparent Text)
    if plot_load:
        load_map_key = dev_name + "_LOAD"
        bar_x_load = x_pos_map.get(load_map_key, np.nan)
        
        if not np.isnan(bar_x_load):
            ax.bar(bar_x_load, r["mean_load"], bar_width,
                   label='_nolegend_',
                   color=bar_color_load, edgecolor='black', linewidth=1.4,
                   yerr=r["std_load"], capsize=9, error_kw={'elinewidth': 2.2, 'capthick': 2.2},
                   zorder=3)
            
            # Text color matches the transparent bar color
            draw_mean_label_horizontal(ax, bar_x_load, r["mean_load"], r["std_load"], bar_color_load)

    # 2. GND / FIBER LASER PLOTTING (Opaque Bar, Opaque Text)
    if not np.isnan(r["mean_gnd"]):
        
        if r["is_FL"]:
            gnd_map_key = "AS_011_CC_FL"
        else:
            gnd_map_key = dev_name + "_GND"

        bar_x_gnd = x_pos_map.get(gnd_map_key, np.nan)

        if not np.isnan(bar_x_gnd):
            ax.bar(bar_x_gnd, r["mean_gnd"], bar_width,
                   label='_nolegend_',
                   color=bar_color_gnd,
                   edgecolor='black', linewidth=1.4,
                   yerr=r["std_gnd"], capsize=9, error_kw={'elinewidth': 2.2, 'capthick': 2.2},
                   zorder=3)
            
            # Text color matches the opaque bar color
            draw_mean_label_horizontal(ax, bar_x_gnd, r["mean_gnd"], r["std_gnd"], bar_color_gnd)


# 3. Final Formatting and Labels
ax.set_xlabel(r'QPD', fontsize=plt.rcParams['axes.labelsize'])
ax.set_ylabel(r'GS [\%]', fontsize=plt.rcParams['axes.labelsize'])
ax.set_title(r'\textbf{Gap Sensitivity Comparison Across QPDs}', fontsize=plt.rcParams['axes.titlesize'], pad=10)

ax.set_xticks(x_tick_positions)
ax.set_xticklabels([]) 

ax.grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)
ax.set_axisbelow(True)

ax.tick_params(axis='x', which='both', length=0)

# Y-Axis Ticks 
ax.tick_params(axis='y', which='major', labelsize=plt.rcParams['ytick.labelsize'], length=8, width=1.5, direction='in')
ax.tick_params(axis='y', which='minor', labelsize=plt.rcParams['ytick.labelsize'], length=5, width=1.0, direction='in')

ax.yaxis.set_major_locator(MultipleLocator(5)) # Adjusted to new Y-limits
ax.yaxis.set_minor_locator(MultipleLocator(1))

# Draw X-Axis Labels manually beneath the bars, centered on the group center
label_offset_y = 0.5 
for x_pos_val, label_text in zip(x_tick_positions, x_tick_labels):
    
    # Determine vertical placement based on B17R11 or others (as requested)
    is_B17R11_label = r'\textbf{B17R11}' in label_text
    
    if is_B17R11_label:
        # Place B17R11 label above the zero line
        label_y = 0 + label_offset_y
        va = 'bottom'
    else:
        # Place all other labels below the zero line
        label_y = 0 - label_offset_y
        va = 'top'

    ax.text(x_pos_val, label_y, label_text,
            ha='center', va=va, fontsize=DEVICE_LABEL_FONTSIZE,
            fontweight='bold', color='black', zorder=5)


# Legend entries (Updated for the color flip)
legend_elements = [
    Patch(facecolor=RUN1_COLOR_GND, edgecolor='black', label=r'Run1 (GND)'), # Opaque color for GND
    Patch(facecolor=RUN1_COLOR_LOAD, edgecolor='black', label=r'Run1 (LOAD)'), # Transparent color for LOAD
    Patch(facecolor=RUN2_COLOR_GND, edgecolor='black', label=r'Run2 (GND)'), # Opaque color for GND
    Patch(facecolor=RUN2_COLOR_LOAD, edgecolor='black', label=r'Run2 (LOAD)'), # Transparent color for LOAD
    Patch(facecolor=FL_COLOR, edgecolor='black', label=r'Fiber laser'),
]

# Legend positioning and Font Size
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.98), frameon=True,
          fancybox=False, edgecolor='black', fontsize=plt.rcParams['legend.fontsize'])

# Save
output_dir = "/Users/asca/Documents/University/Master Thesis/code/AS_011_CC/statistical study/fig"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "Gap_Sensitivity_Comparison_Across_Devices_FINAL_RESULT.png")
fig.savefig(output_path, dpi=400, bbox_inches='tight')
print(f"\n[OK] Final plot saved → {output_path}")

plt.show()
plt.close(fig)

# === FINAL SUMMARY TABLE (UNCHANGED) === #
print("\n" + "="*90)
print("FINAL GAP SENSITIVITY SUMMARY (Mean ± Std [%])")
print("="*90)
print(f"{'Device':<25} {'LOAD':>20} {'GND':>20} {'N meas':>8}")
print("-"*90)
for dev in results:
    r = results[dev]

    if np.isnan(r['mean_load']):
        load_str = "      N/A        "
    else:
        load_str = f"{r['mean_load']:+6.2f} ± {r['std_load']:.2f} %"

    if np.isnan(r['mean_gnd']):
        gnd_str = "      N/A        "
    else:
        gnd_str = f"{r['mean_gnd']:+6.2f} ± {r['std_gnd']:.2f} %"


    simple_label = dev.replace('_', '\\_')

    print(f"{simple_label:<25} "
          f"{load_str:20}    "
          f"{gnd_str:20}    "
          f"{r['n_total']:4}")
print("="*90)
print(f"All values computed from multiple quadrant pairs (AB, AD, BC, CD/DC) and scan directions. (PLOT_B17R11_LOAD = {PLOT_B17R11_LOAD})")
print("B17R11 values are processed as is (negative sensitivity retained).")
print("AS_004 AD pair now includes both standard and ND filter runs for maximum statistical points.")
print("AS_011_CC_FL shows a single GS bar (no LOAD/GND difference).")