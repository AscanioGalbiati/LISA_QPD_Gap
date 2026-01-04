'''
This code processes data from a series of .pkl files to track sign differences and find the gap positions for multiple quadrant pairs of a QPD.
It computes zero-crossings for specified pairs which are processed individually: (A-D, B-C; A-B, D-C), 
fits a linear model to the gap positions as a function of Y position for each pair, and plots the results.

Added support for different Y ranges (in µm) per pair via the 'pair_ranges' dictionary—adjust as needed.
Added a section to compute averaged slopes for the two main diagonals (A-D with B-C, and A-B with D-C), including average angles.

Compatible data: 
/Users/asca/Documents/University/Master Thesis/code/Data/final_data/20251002/VIGO17_333-1_QPD_0750_20_AW_011_CC_251001_LB1471C_quadABCD_manual_setup_Z50mm_Yscan_gap_identification_precise
'''

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
from scipy.stats import linregress

# Set up font and LaTeX rendering
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
cmu_serif = fm.FontProperties(fname=font_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

# Base directory containing Y-position folders
base_dir = "/Users/asca/Documents/University/Master Thesis/code/Data/all quadrants/20251002/VIGO17_333-1_QPD_0750_20_AW_011_CC_251001_LB1471C_quadABCD_manual_setup_Z50mm_Yscan_gap_identification_precise"

# Function to load data from a .pkl file
def load_data(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

# Function to extract Y position from folder name
def extract_y_position(folder_name):
    match = re.search(r'Y(\d+)um', folder_name)
    if match:
        return int(match.group(1))
    return None

# Define the quadrant pairs to analyze (first - second = 0 crossing)
pairs = [
    ('quadA', 'quadD', 'Bottom left diagonal A-D'),
    ('quadB', 'quadC', 'Top right diagonal B-C'),
    ('quadA', 'quadB', 'Top left diagonal A-B'),
    ('quadD', 'quadC', 'Bottom right D-C')
]

# Define Y ranges (in µm) for each pair: (min_y_um, max_y_um). Set to (None, None) to use all available Y positions
pair_ranges = {
    'Bottom left diagonal A-D': (1950, 2500),
    'Top right diagonal B-C': (2500, 3850),  # Example: Different range for B-C
    'Top left diagonal A-B': (2500, 3850),  # Example: Different range for A-B
    'Bottom right D-C': (1950, 2500)   # Example: Different range for D-C
}

# Find all Y-position folders (no global range filter—ranges are per pair)
y_folders = glob.glob(os.path.join(base_dir, "Y*um"))
y_positions = []
for folder in y_folders:
    y_um = extract_y_position(os.path.basename(folder))
    if y_um is not None:
        y_positions.append((y_um, folder))
    else:
        print(f"Could not extract Y position from {folder}")
y_positions.sort()  # Sort by Y position
print(f"All available Y positions: {[y for y, _ in y_positions]}")

# Dictionary to store gap positions for each pair
gap_data = {pair[2]: {'y_mm': [], 'x_gap': []} for pair in pairs}

# Process each Y-position folder
for y_um, y_folder in y_positions:
    # Find the first .pkl file in the folder
    pkl_files = glob.glob(os.path.join(y_folder, "*.pkl"))
    if not pkl_files:
        print(f"No .pkl files found in {y_folder}")
        continue
    pkl_file = pkl_files[0]  # Take the first file
    
    # Load data
    data = load_data(pkl_file)
    
    # Extract X position array from config
    xstart = data['global_params']['xstart_mm']
    xstop = data['global_params']['xstop_mm']
    xstep = data['global_params']['xstep_mm']
    steps = int((xstop - xstart) / xstep) + 1
    x_array = np.linspace(xstart, xstop, steps)
    
    # Get means for all quadrants
    means = {}
    for quad in ['quadA', 'quadB', 'quadC', 'quadD']:
        if quad in data['rawdata']:
            means[quad] = data['rawdata'][quad]['dmm00_curr_amp'].mean(axis=1)
        else:
            print(f"Warning: {quad} data missing in {pkl_file}")
            means[quad] = None
    
    # Process each pair, but only if Y is in the range for that pair
    for quad1, quad2, label in pairs:
        min_y, max_y = pair_ranges[label]
        if min_y is not None and (y_um < min_y or y_um > max_y):
            continue  # Skip this Y for this pair if out of range
        print(f"Processing {label} at Y={y_um}um (range: {min_y}-{max_y}um)")
        
        if means[quad1] is not None and means[quad2] is not None:
            diff = means[quad1] - means[quad2]
            
            # Find sign changes
            sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
            
            # Debug print
            print(f"{label} at Y={y_um}um: diff min={np.min(diff):.4e}, max={np.max(diff):.4e}, num_sign_changes={len(sign_changes)}")
            
            if len(sign_changes) > 0:
                i = sign_changes[0]  # Assume first (and ideally only) crossing
                x1 = x_array[i]
                x2 = x_array[i + 1]
                d1 = diff[i]
                d2 = diff[i + 1]
                x_cross = x1 - d1 * (x2 - x1) / (d2 - d1)
                
                # Append to lists (Y in mm)
                gap_data[label]['y_mm'].append(y_um / 1000.0)
                gap_data[label]['x_gap'].append(x_cross)
                print(f"{label} gap at Y={y_um}um: X={x_cross:.4f} mm")
            else:
                print(f"No intersection found for {label} at Y={y_um}um")
                # Debug plot for no crossing
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(x_array, means[quad1], label=f'{quad1}')
                ax.plot(x_array, means[quad2], label=f'{quad2}')
                ax.plot(x_array, diff, label='Difference', linestyle='--')
                ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)
                ax.legend()
                ax.set_title(f'No crossing for {label} at Y={y_um}um')
                ax.set_xlabel('X Position [mm]')
                ax.set_ylabel('Photocurrent [A]')
                debug_path = os.path.join(y_folder, f"debug_no_crossing_{label.replace(' ', '_').replace('-', '')}_Y{y_um}um.png")
                fig.savefig(debug_path, dpi=150)
                print(f"Debug plot saved: {debug_path}")
                plt.close(fig)
        else:
            print(f"Warning: Missing data for {label} at Y={y_um}um")

# Dictionary to store fit results for each pair
fit_results = {}

# Combined plot for all pairs (even if some have no data)
fig, ax = plt.subplots(figsize=(10, 8), layout='constrained')
colors = ['blue', 'green', 'red', 'orange']
for idx, (quad1, quad2, label) in enumerate(pairs):
    if gap_data[label]['y_mm']:
        y_array = np.array(gap_data[label]['y_mm'])
        x_array = np.array(gap_data[label]['x_gap'])
        ax.scatter(y_array, x_array, label=f'{label} Data', color=colors[idx], s=20)
        
        # Linear regression
        res = linregress(y_array, x_array)
        slope = res.slope
        intercept = res.intercept
        r2 = res.rvalue ** 2
        angle_deg = np.arctan(slope) * 180 / np.pi
        
        # Store fit results
        fit_results[label] = {
            'slope': slope,
            'intercept': intercept,
            'r2': r2,
            'angle_deg': angle_deg
        }
        
        # Fit line
        y_fit = np.linspace(min(y_array), max(y_array), 100)
        x_fit = intercept + slope * y_fit
        ax.plot(y_fit, x_fit, color=colors[idx], linestyle='-', linewidth=2, label=f'{label} Fit')
        
        print(f"\nFit results for {label}:")
        print(f"Slope: {slope:.4f} mm/mm")
        print(f"Intercept: {intercept:.4f} mm")
        print(f"Angle: {angle_deg:.2f} degrees")
        print(f"R²: {r2:.4f}")
    else:
        print(f"No valid data points found for fitting {label}.")

ax.legend(fontsize=10)
ax.set_title(r'\textbf{QPD Gap Positions vs Height - All Pairs}', fontsize=14, fontweight='bold', pad=10)
ax.set_xlabel(r'Y Position [mm]', fontsize=14)
ax.set_ylabel(r'Gap Position X [mm]', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)
ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5, direction='in')

# Save combined figure
fig_name = "QPD_Gap_vs_Y_All_Pairs"
fig_path = os.path.join(base_dir, f"{fig_name}.png")
fig.savefig(fig_path, dpi=300)
print(f"Combined figure saved: {fig_path}")
plt.close(fig)

# Individual plots for each pair (only if data exists)
for label, data_dict in gap_data.items():
    if data_dict['y_mm']:
        y_array = np.array(data_dict['y_mm'])
        x_array = np.array(data_dict['x_gap'])
        
        # Linear regression (reuse from above if needed)
        res = linregress(y_array, x_array)
        slope = res.slope
        intercept = res.intercept
        r2 = res.rvalue ** 2
        angle_deg = np.arctan(slope) * 180 / np.pi
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
        ax.plot(y_array, x_array, 'o', label='Data points')
        
        # Fit line
        y_fit = np.linspace(min(y_array), max(y_array), 100)
        x_fit = intercept + slope * y_fit
        ax.plot(y_fit, x_fit, '-', label='Linear fit')
        
        ax.legend(fontsize=12)
        ax.set_title(rf'\textbf{{QPD Gap Position vs Height - {label}}}', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(r'Y Position [mm]', fontsize=14)
        ax.set_ylabel(r'Gap Position X [mm]', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5, direction='in')
        
        # Add text box with fit info
        text = f'Slope: {slope:.4f} mm/mm\nAngle: {angle_deg:.2f}$^\circ$\nR$^2$: {r2:.4f}'
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save figure
        fig_name = f"QPD_Gap_vs_Y_{label.replace(' ', '_').replace('-', '')}"
        fig_path = os.path.join(base_dir, f"{fig_name}.png")
        fig.savefig(fig_path, dpi=300)
        print(f"Individual figure saved: {fig_path}")
        
        plt.close(fig)

# Compute averaged slopes for the two diagonals
if 'Bottom left diagonal A-D' in fit_results and 'Top right diagonal B-C' in fit_results:
    slope_ad = fit_results['Bottom left diagonal A-D']['slope']
    slope_bc = fit_results['Top right diagonal B-C']['slope']
    avg_slope_diag1 = (slope_ad + slope_bc) / 2
    avg_angle_diag1 = np.arctan(avg_slope_diag1) * 180 / np.pi
    print("\nAveraged results for Diagonal 1 (A-D and B-C):")
    print(f"Average Slope: {avg_slope_diag1:.4f} mm/mm")
    print(f"Average Angle: {avg_angle_diag1:.2f} degrees")

if 'Top left diagonal A-B' in fit_results and 'Bottom right D-C' in fit_results:
    slope_ab = fit_results['Top left diagonal A-B']['slope']
    slope_dc = fit_results['Bottom right D-C']['slope']
    avg_slope_diag2 = (slope_ab + slope_dc) / 2
    avg_angle_diag2 = np.arctan(avg_slope_diag2) * 180 / np.pi
    print("\nAveraged results for Diagonal 2 (A-B and D-C):")
    print(f"Average Slope: {avg_slope_diag2:.4f} mm/mm")
    print(f"Average Angle: {avg_angle_diag2:.2f} degrees")

print("Processing complete.")