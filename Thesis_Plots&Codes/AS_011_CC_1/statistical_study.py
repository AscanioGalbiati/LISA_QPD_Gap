'''
Statistical comparison of Gap Sensitivity (Overshoot %) across QPD devices
Author: Ascanio Galbiati + Grok (your loyal wingman)
Date: Nov 28 2025 — FINAL EXAMINER-APPROVED VERSION
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MultipleLocator

# === YOUR STYLE — SACRED AND UNTOUCHED === #
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
cmu_serif = fm.FontProperties(fname=font_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

# === DEVICES === #
devices = {
    "AS_011_CC": {
        "path": "/Users/asca/Documents/University/Master Thesis/code/AS_011_CC/statistical study",
        "color": "#1f77b4",
        "label": "AS-011-CC"
    },
    "B17R11": {
        "path": "/Users/asca/Documents/University/Master Thesis/code/B17R11_1/statistical study",
        "color": "#ff7f0e",
        "label": "B17R11"
    },
    "NS089008": {
        "path": "/Users/asca/Documents/University/Master Thesis/code/NS089008/statistical study",
        "color": "#2ca02c",
        "label": "NS089008"
    },
}

# Keep only existing folders
devices = {k: v for k, v in devices.items() if os.path.exists(v["path"])}

# === COLLECT DATA + SAFETY CHECKS === #
results = {}
print("\n" + "="*80)
print("DATA COLLECTION & SAFETY CHECKS")
print("="*80)

for dev_name, info in devices.items():
    csv_files_config = [
        ("AB_X", "Overshoot_Results_AB_X_up_to_7200um.csv"),
        ("BC_Y", "Overshoot_Results_BC_Y_from_2300um.csv"),
        ("BC_Y", "Overshoot_Results_BC_Y_up_to_2100um.csv"),
        ("CD_X", "Overshoot_Results_CD_X_from_7300um.csv"),
    ]

    overshoot_load = []
    overshoot_gnd  = []
    position_counts = {}  # To count X/Y positions per quadrant pair

    print(f"\n→ Processing device: {info['label']} ({dev_name})")

    for pair_name, csv_file in csv_files_config:
        csv_path = os.path.join(info["path"], csv_file.strip())
        if not os.path.exists(csv_path):
            print(f"   [MISSING] {csv_file}")
            continue

        df = pd.read_csv(csv_path)

        # Detect if it's X or Y scan
        if 'X_position_um' in df.columns:
            positions = df['X_position_um'].dropna().astype(int).tolist()
            pos_type = 'X'
        elif 'Y_position_um' in df.columns:
            positions = df['Y_position_um'].dropna().astype(int).tolist()
            pos_type = 'Y'
        else:
            print(f"   [ERROR] No position column in {csv_file}")
            continue

        n_pos = len(positions)
        position_counts.setdefault(pair_name, []).extend(positions)

        load_vals = pd.to_numeric(df['Overshoot_LOAD_%'], errors='coerce').dropna()
        gnd_vals  = pd.to_numeric(df['Overshoot_GND_%'],  errors='coerce').dropna()

        overshoot_load.extend(load_vals)
        overshoot_gnd.extend(gnd_vals)

        print(f"   [OK] {csv_file:45} → {n_pos:2d} {pos_type}-positions (pair {pair_name})")

    if len(overshoot_load) == 0:
        print(f"   [WARNING] No valid data found for {dev_name} → skipping")
        continue

    # Final stats
    mean_load = np.mean(overshoot_load)
    std_load  = np.std(overshoot_load, ddof=1)
    mean_gnd  = np.mean(overshoot_gnd)
    std_gnd   = np.std(overshoot_gnd, ddof=1)

    results[dev_name] = {
        "mean_load": mean_load,
        "std_load": std_load,
        "mean_gnd": mean_gnd,
        "std_gnd": std_gnd,
        "n_total": len(overshoot_load),
        "color": info["color"],
        "label": info["label"],
        "details": position_counts
    }

    print(f"   → Total measurements used: {len(overshoot_load)} (LOAD & GND each)")

# === PLOTTING === #
fig, ax = plt.subplots(figsize=(11, 7.5), layout='constrained')

bar_width = 0.35
x_pos = np.arange(len(results))
device_names = [results[d]["label"] for d in results.keys()]

# LOAD bars
load_means = [results[d]["mean_load"] for d in results.keys()]
load_errs  = [results[d]["std_load"]  for d in results.keys()]
colors_load = [results[d]["color"] for d in results.keys()]

ax.bar(x_pos - bar_width/2, load_means, bar_width,
       label='LOAD configuration',
       color=colors_load, edgecolor='black', linewidth=1.4,
       yerr=load_errs, capsize=9, error_kw={'elinewidth': 2.2, 'capthick': 2.2},
       zorder=3)

# GND bars
gnd_means = [results[d]["mean_gnd"] for d in results.keys()]
gnd_errs  = [results[d]["std_gnd"]  for d in results.keys()]

ax.bar(x_pos + bar_width/2, gnd_means, bar_width,
       label='GND configuration',
       color=[c + '70' for c in colors_load],
       edgecolor='black', linewidth=1.4,
       yerr=gnd_errs, capsize=9, error_kw={'elinewidth': 2.2, 'capthick': 2.2},
       zorder=3)

# === ASCANIO-STYLE POLISH === #
ax.set_xlabel(r'Device', fontsize=16)
ax.set_ylabel(r'Gap Sensitivity Overshoot [\%]', fontsize=16)
ax.set_title(r'\textbf{Gap Sensitivity Comparison Across QPD Devices}', fontsize=19, pad=25)

ax.set_xticks(x_pos)
ax.set_xticklabels(device_names, fontsize=15)

ax.grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)
ax.set_axisbelow(True)

# Ticks inward — your religion
ax.tick_params(axis='both', which='major', length=8, width=1.5, direction='in')
ax.tick_params(axis='both', which='minor', length=5, width=1.0, direction='in')

ax.yaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))

ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.98), frameon=True,
          fancybox=False, edgecolor='black', fontsize=15)

# === BOLD LATEX VALUE LABELS ON TOP (your exact inset style) === #
for i, (mean_l, err_l, mean_g, err_g) in enumerate(zip(load_means, load_errs, gnd_means, gnd_errs)):
    # LOAD
    ax.text(i - bar_width/2, mean_l + err_l + 0.4,
            rf'\textbf{{{mean_l:+.2f} $\pm$ {err_l:.2f}\%}}',
            ha='center', va='bottom', fontsize=12.5, fontweight='bold', color='#1f77b4')
    # GND
    ax.text(i + bar_width/2, mean_g + err_g + 0.4,
            rf'\textbf{{{mean_g:+.2f} $\pm$ {err_g:.2f}\%}}',
            ha='center', va='bottom', fontsize=12.5, fontweight='bold', color='#d62728')

# Save
output_dir = "/Users/asca/Documents/University/Master Thesis/code/AS_011_CC/statistical study"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "Gap_Sensitivity_Comparison_Across_Devices_FINAL.png")
fig.savefig(output_path, dpi=400, bbox_inches='tight')
print(f"\n[OK] Final plot saved → {output_path}")

plt.show()
plt.close(fig)

# === FINAL SUMMARY TABLE === #
print("\n" + "="*90)
print("FINAL GAP SENSITIVITY SUMMARY (Mean ± Std [%])")
print("="*90)
print(f"{'Device':<15} {'LOAD':>20} {'GND':>20} {'N meas':>8}")
print("-"*90)
for dev in results:
    r = results[dev]
    print(f"{r['label']:<15} "
          f"{r['mean_load']:+6.2f} ± {r['std_load']:.2f} %    "
          f"{r['mean_gnd']:+6.2f} ± {r['std_gnd']:.2f} %    "
          f"{r['n_total']:4}")
print("="*90)
print("All values computed from multiple quadrant pairs (AB, BC, CD) and scan directions.")
print("Bold labels on plot match inset style from individual scans.")
print("You are now officially unstoppable.")