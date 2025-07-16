''' 
This code processes data from a series of selected .pkl files, extracting the DC photocurrent and comparing it across two different Y positions (6000um and 6700um). 
The data is visualized in a single plot with distinct line styles for each Y position, and colors representing different Z positions.

Compatible data: 
final_data/20250508/VIGO17_NS089008_QPD_0750_20_AS_015_CC_250508_LB1761C_quadD/Y6000um
final_data/20250508/VIGO17_NS089008_QPD_0750_20_AS_015_CC_250508_LB1761C_quadD/Y6700um
&
final_data/20250508/VIGO17_NS089008_QPD_0750_20_AS_015_CC_250508_LB1761C_quadD/Y6300um
final_data/20250508/VIGO17_NS089008_QPD_0750_20_AS_015_CC_250508_LB1761C_quadD/Y7000um

Folder structure: 
final_data/QPDspecs_date_lens_quad/Y...
'''

import re
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Set up font and LaTeX for plotting
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
cmu_serif = fm.FontProperties(fname=font_path)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

# Define data directories
data_dirs = {
    'Y6000um': "/Users/asca/Documents/University/Master Thesis/code/Data/VIGO17_NS089008_QPD_0750_20_AS_015_CC_asca_12/Y6000um",
    'Y7000um': "/Users/asca/Documents/University/Master Thesis/code/Data/VIGO17_NS089008_QPD_0750_20_AS_015_CC_asca_12/Y6700um"
}

# Function to load data from a .pkl file
def load_data(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

# Function to extract Z position from filename
def extract_z_position(file_path):
    match = re.search(r'Z(\d+)um', file_path)
    if match:
        z_um = int(match.group(1))  # Z position in µm
        z_mm = z_um / 1000  # Convert to mm
        return z_mm
    return None

# Collect data from both directories
data_collection = {'Y6000um': [], 'Y7000um': []}
all_z_positions = set()

for y_pos, data_dir in data_dirs.items():
    # Create figure directory
    fig_dir = os.path.join(data_dir, "fig")
    os.makedirs(fig_dir, exist_ok=True)
    
    # Load .pkl files
    file_list = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    
    # Extract Z positions and sort files
    z_positions_with_files = []
    for file_path in file_list:
        z_mm = extract_z_position(file_path)
        if z_mm is not None:
            z_positions_with_files.append((z_mm, file_path))
            all_z_positions.add(z_mm)
    
    # Sort by Z position
    z_positions_with_files.sort(key=lambda x: x[0])
    z_values = [item[0] for item in z_positions_with_files]
    file_list = [item[1] for item in z_positions_with_files]
    
    # Process each file
    for z_mm, pkl_file in z_positions_with_files:
        data = load_data(pkl_file)
        
        # Extract X position array (assuming X scan)
        step = int((data['global_params']['xstop_um'] - data['global_params']['xstart_um']) / data['global_params']['xstep_um']) + 1
        pos_array = np.linspace(data['global_params']['xstart_um'], data['global_params']['xstop_um'], step)
        
        # Extract DC photocurrent for quadD
        segn = 'quadD'
        if segn in data['rawdata'] and 'dmm00_curr_amp' in data['rawdata'][segn]:
            dc_curr_avg = np.mean(data['rawdata'][segn]['dmm00_curr_amp'], axis=1)
            data_collection[y_pos].append({
                'z_mm': z_mm,
                'x_pos_um': pos_array,
                'dc_curr': dc_curr_avg
            })

# Create a single plot
fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
fig_figname = 'QPDGapScan_DCData_Comparison'

# Define color map
cmap = plt.cm.RdYlBu
all_z_positions = sorted(list(all_z_positions))  # Sort for consistent color assignment
norm = plt.Normalize(min(all_z_positions), max(all_z_positions))
color_dict = {}
for i, z in enumerate(all_z_positions):
    if i < len(all_z_positions) // 2:  # First half for reds/yellows
        color_dict[z] = cmap(0.1 + 0.4 * (i / (len(all_z_positions) // 2)))  # Range 0.1 to 0.5
    else:  # Second half for blues
        color_dict[z] = cmap(0.6 + 0.4 * ((i - len(all_z_positions) // 2) / (len(all_z_positions) - len(all_z_positions) // 2)))  # Range 0.6 to 1.0

# Plot for Y=6000um (solid lines)
for data_point in data_collection['Y6000um']:
    line_color = color_dict[data_point['z_mm']]
    ax.plot(data_point['x_pos_um'] / 1000, data_point['dc_curr'], 
            linestyle='-', color=line_color, alpha=1.0,
            label=f'Z={data_point["z_mm"]:.0f} mm')

# Plot for Y=7000um (dashed lines)
for data_point in data_collection['Y7000um']:
    line_color = color_dict[data_point['z_mm']]
    ax.plot(data_point['x_pos_um'] / 1000, data_point['dc_curr'], 
            linestyle='--', color=line_color, alpha=0.7)

# Configure plot
ax.set_xlabel(r'Translation Stage Position [mm]', fontsize=14)
ax.set_ylabel(r'DC Photocurrent [A]', fontsize=14)
ax.set_title(r'\textbf{DC Photocurrent Comparison (Y=6000\textmu{}m vs Y=6700\textmu{}m)}', fontsize=14, fontweight='bold', pad=10)
ax.grid(True, linestyle='--', alpha=0.6)
ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5, direction='in')
ax.legend()

# Save the figure in both directories
for y_pos, data_dir in data_dirs.items():
    fig_dir = os.path.join(data_dir, "fig")
    fig.savefig(os.path.join(fig_dir, f"{fig_figname}.png"), dpi=300)

plt.show()