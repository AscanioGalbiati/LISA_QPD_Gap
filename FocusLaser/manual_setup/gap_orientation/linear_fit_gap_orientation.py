'''
This code processes data from a series of .pkl files to track sign differences and find the gap position between Quadrant A and Quadrant D of a QPD, 
it then fits a linear model to the gap positions as a function of Y position and plots the results.

Compatible data: 
/Users/asca/Documents/University/Master Thesis/code/Data/final_data/20250924/quadA&D/VIGO17_333-1_QPD_0750_20_AW_011_CC_250924_LB1471C_quadA&D_manual_setup_Z50mm_Yscan_gap_identification_precise

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
base_dir = "/Users/asca/Documents/University/Master Thesis/code/Data/final_data/20250924/quadA&D/VIGO17_333-1_QPD_0750_20_AW_011_CC_250924_LB1471C_quadA&D_manual_setup_Z50mm_Yscan_gap_identification_precise"

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

# Find all Y-position folders within the range 1500um to 3750um
y_folders = glob.glob(os.path.join(base_dir, "Y*um"))
y_positions = []
for folder in y_folders:
    y_um = extract_y_position(os.path.basename(folder))
    if y_um is not None and 2100 <= y_um <= 3750:
        y_positions.append((y_um, folder))
y_positions.sort()  # Sort by Y position

# Lists to store gap positions
y_mm_list = []
x_gap_list = []

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
    
    # Get means
    if 'quadA' in data['rawdata'] and 'quadD' in data['rawdata']:
        quadA_mean = data['rawdata']['quadA']['dmm00_curr_amp'].mean(axis=1)
        quadC_mean = data['rawdata']['quadD']['dmm00_curr_amp'].mean(axis=1)
        
        # Compute difference
        diff = quadA_mean - quadC_mean
        
        # Find sign changes
        sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
        
        if len(sign_changes) > 0:
            i = sign_changes[0]  # Assume first (and ideally only) crossing
            x1 = x_array[i]
            x2 = x_array[i + 1]
            d1 = diff[i]
            d2 = diff[i + 1]
            x_cross = x1 - d1 * (x2 - x1) / (d2 - d1)
            
            # Append to lists (Y in mm)
            y_mm_list.append(y_um / 1000.0)
            x_gap_list.append(x_cross)
            print(f"Gap at Y={y_um}um: X={x_cross:.4f} mm")
        else:
            print(f"No intersection found for Y={y_um}um")
    else:
        print(f"Warning: quadA or quadC data missing in {pkl_file}")

# After processing all Y positions, perform linear fit and plot
if y_mm_list:
    y_array = np.array(y_mm_list)
    x_array = np.array(x_gap_list)
    
    # Linear regression
    res = linregress(y_array, x_array)
    slope = res.slope
    intercept = res.intercept
    r2 = res.rvalue ** 2
    angle_deg = np.arctan(slope) * 180 / np.pi
    
    # Print results
    print(f"\nFit results:")
    print(f"Slope: {slope:.4f} mm/mm")
    print(f"Intercept: {intercept:.4f} mm")
    print(f"Angle: {angle_deg:.2f} degrees")
    print(f"R²: {r2:.4f}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    ax.plot(y_array, x_array, 'o', label='Data points')
    
    # Fit line
    y_fit = np.linspace(min(y_array), max(y_array), 100)
    x_fit = intercept + slope * y_fit
    ax.plot(y_fit, x_fit, '-', label='Linear fit')
    
    ax.legend(fontsize=12)
    ax.set_title(r'\textbf{QPD Gap Position vs Height}', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel(r'Y Position [mm]', fontsize=14)
    ax.set_ylabel(r'Gap Position X [mm]', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5, direction='in')
    
    # Add text box with fit info
    text = f'Slope: {slope:.4f} mm/mm\nAngle: {angle_deg:.2f}$^\circ$\nR$^2$: {r2:.4f}'
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save figure
    fig_name = "QPD_Gap_vs_Y"
    fig_path = os.path.join(base_dir, f"{fig_name}.png")
    fig.savefig(fig_path, dpi=300)
    print(f"Figure saved: {fig_path}")
    
    plt.show()
else:
    print("No valid data points found for fitting.")

print("Processing complete.")