'''
This code processes data from a series of .pkl files to plot the DC photocurrents of all the quadrants of a QPD,
the scans are performed along a direction that is angled perpendicular to the gap between the quadrants (we use the negative inverse of the slope found in linear_fit_gap_orientation_AllQuadrants_variable_range.py).

Compatible data: 
/Users/asca/Documents/University/Master Thesis/code/Data/final_data/20251002/DualDiagonalScan

Folder structure: 
final_data/QPDspecs_date_YScan_lens_quadA&D_manual_setup_Z50mm_Yscan_gap_identification_precise
'''

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
from scipy.special import erf  # Kept for potential future use

# Set up font and LaTeX rendering
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
cmu_serif = fm.FontProperties(fname=font_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

# Base directory containing DualDiagonalScan folder
base_dir = "/Users/asca/Documents/University/Master Thesis/code/Data/all quadrants/20251002/DualDiagonalScan"

# Function to load data from a .pkl file
def load_data(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

# Function to extract along position from folder name
def extract_along_position(folder_name):
    match = re.search(r'Along(\d+)um', folder_name)
    if match:
        return int(match.group(1))
    return None

# Find all diagonal folders (Diagonal_1_AD_BC and Diagonal_2_AB_DC)
diagonal_folders = glob.glob(os.path.join(base_dir, "*_Diagonal_*"))
for diag_folder in diagonal_folders:
    diag_name = os.path.basename(diag_folder).split('_')[-1]  # Extract Diagonal_1_AD_BC or Diagonal_2_AB_DC
    
    # Find all Along-position folders within the diagonal folder
    along_folders = glob.glob(os.path.join(diag_folder, "Along*um"))
    along_positions = []
    for folder in along_folders:
        along_um = extract_along_position(os.path.basename(folder))
        if along_um is not None:
            along_positions.append((along_um, folder))
    along_positions.sort()  # Sort by along position

    # Process each Along-position folder
    for along_um, along_folder in along_positions:
        # Create figure directory
        fig_dir = os.path.join(along_folder, "fig")
        os.makedirs(fig_dir, exist_ok=True)
        
        # Find the first .pkl file in the folder (assuming one file per Along position)
        pkl_files = glob.glob(os.path.join(along_folder, "*.pkl"))
        if not pkl_files:
            print(f"No .pkl files found in {along_folder}")
            continue
        pkl_file = pkl_files[0]  # Take the first file
        
        # Load data
        data = load_data(pkl_file)
        
        # Extract u position array from data
        u_array = data['rawdata']['u_position']
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
        
        # Plot all quadrants
        for quad in ['quadA', 'quadB', 'quadC', 'quadD']:
            if quad in data['rawdata']:
                quad_mean = data['rawdata'][quad]['dmm00_curr_amp'].mean(axis=1)
                ax.plot(u_array, quad_mean, label=quad)
            else:
                print(f"Warning: {quad} data missing in {pkl_file}")
        
        ax.legend(fontsize=12)
        ax.set_title(rf'\textbf{{DC Photocurrent Comparison (Along={along_um}\textmu{{}}m, {diag_name})}}', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(r'u Position (Perpendicular to Gap) [mm]', fontsize=14)
        ax.set_ylabel(r'Photocurrent [A]', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5, direction='in')
        
        # Save figure
        fig_name = f"DC_Photocurrent_Along{along_um:04d}um_{diag_name}"
        fig_path = os.path.join(fig_dir, f"{fig_name}.png")
        fig.savefig(fig_path, dpi=300)
        print(f"Figure saved: {fig_path}")
        
        plt.close(fig)  # Close figure to free memory

print("All plots generated.")
plt.show()