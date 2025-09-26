''' 
This code processes data from a series of .pkl files, extracting and plotting the DC photocurrent for each Y position.

Compatible data: 
/Users/asca/Documents/University/Master Thesis/code/Data/final_data/20250910/VIGO17_333-2_QPD_0750_20_AP_011_CC_250910_LB1471C_quadA_manual_setup_fixedY_Z50mm (this is data from quadA but labelled as quadC)
/Users/asca/Documents/University/Master Thesis/code/Data/final_data/20250910/VIGO17_333-2_QPD_0750_20_AP_011_CC_250910_LB1471C_quadA_manual_setup_fixedY_Z50mm



Folder structure: 
final_data/QPDspecs_date_YScan_lens_quad_manual_setup
'''

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re

# Set up font and LaTeX rendering
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
cmu_serif = fm.FontProperties(fname=font_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

# Base directory containing Y-position folders
base_dir = "/Users/asca/Documents/University/Master Thesis/code/Data/manual setup/20250910/VIGO17_333-2_QPD_0750_20_AP_011_CC_250910_LB1471C_quadC_manual_setup_fixedY_Z50mm" 

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

# Find all Y-position folders
y_folders = glob.glob(os.path.join(base_dir, "Y*um"))
y_positions = []
for folder in y_folders:
    y_um = extract_y_position(os.path.basename(folder))
    if y_um is not None:
        y_positions.append((y_um, folder))
y_positions.sort()  # Sort by Y position

# Process each Y-position folder
for y_um, y_folder in y_positions:
    # Create figure directory
    fig_dir = os.path.join(y_folder, "fig")
    os.makedirs(fig_dir, exist_ok=True)
    
    # Find the first .pkl file in the folder (assuming one file per Y position)
    pkl_files = glob.glob(os.path.join(y_folder, "*.pkl"))
    if not pkl_files:
        print(f"No .pkl files found in {y_folder}")
        continue
    pkl_file = pkl_files[0]  # Take the first file
    
    # Load data
    data = load_data(pkl_file)
    
    # Extract X position array
    xstart = data['global_params']['xstart_mm']
    xstop = data['global_params']['xstop_mm']
    xstep = data['global_params']['xstep_mm']
    steps = int((xstop - xstart) / xstep) + 1
    x_array = np.linspace(xstart, xstop, steps)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    ax.plot(x_array, data['rawdata']['quadC']['dmm00_curr_amp'].mean(axis=1), label='quadC') # change quad here if needed
    ax.legend(fontsize=12)
    ax.set_title(rf'\textbf{{DC Photocurrent at Y={y_um}\textmu{{}}m}}', 
             fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel(r'X Position [mm]', fontsize=14)
    ax.set_ylabel(r'Photocurrent [A]', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5, direction='in')
    
    # Save figure
    fig_name = f"DC_Photocurrent_Y{y_um:04d}um"
    fig.savefig(os.path.join(fig_dir, f"{fig_name}_good.png"), dpi=300)
    print(f"Figure saved: {os.path.join(fig_dir, f'{fig_name}_good.png')}")
    
    plt.close(fig)  # Close figure to free memory

print("All plots generated.")
plt.show()