'''
@author: A_Galbiati
Comparison of focusing results using the KE tecnhique and BP data fitted with a Gaussian model
Gaussian fit on far-field BP data and theoretical waist estimate
'''

import os
import glob
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import curve_fit
from scipy.special import erf
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# -------------------------
# USER SETTINGS (change if needed)
# -------------------------
# QPD directory (old pipeline)
QPD_DIR = "/Users/asca/Documents/University/Master Thesis/code/Data/20251215_final_data/20251022/Y2500um"
# BP (new) directory (20251016)
BP_DIR = "/Users/asca/Documents/University/Master Thesis/code/Data/20251215_final_data/20251016_BP"

# Z shifts (structure supports non-zero later) — set to 0 now as requested
SLIT_OFFSET_MM = 2.25   # apply to slit z positions (was +5.5 etc.)
QPD_OFFSET_MM = 4.5    # apply to QPD z positions (was +14.62 etc.)

# Slit Z-mask (a fter shift) — these are defaults; tweak as needed
SLIT_Z_MIN_MM = 8.5    # chosen to match 20251016 snippet mask
SLIT_Z_MAX_MM = 15

# Gaussian Fit Z-Exclusion (in mm)
# Points within this distance from the minimum-radius Z-position will be EXCLUDED
# from the Gaussian fit to only use 'far data of the arms'.
GAUSS_Z_WAIST_EXCLUSION_MM_X = 0.2 #1.0
GAUSS_Z_WAIST_EXCLUSION_MM_Y = 0.2 #3.0

# QPD file Z filter (only consider files with Z >= ... mm)
QPD_Z_MIN_MM = 3 # allow reasonable range
QPD_Z_MAX_MM = 15

# Warnings / thresholds
SPOT_SIZE_SANITY_UM = 20000.0   # skip absurdly large spots
SPOT_SIZE_KE_WARN_UM = 1000.0  # warn threshold for KE fits

# Plot output
OUT_DIR = os.path.join(BP_DIR, "fig")
os.makedirs(OUT_DIR, exist_ok=True)

# Try set serif font + LaTeX if available, otherwise continue
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
try:
    fm.FontProperties(fname=font_path)
    plt.rcParams.update({'text.usetex': True,
                         'font.family': 'serif',
                         'font.serif': ['Computer Modern']})
except Exception as e:
    print("Warning: couldn't set requested font/LaTeX; continuing with defaults.", e)

# -------------------------
# Models & helpers
# -------------------------
lambda_wl_mm = 0.001064  # wavelength in mm (1064 nm)
lambda_wl_um = lambda_wl_mm * 1000.0 # wavelength in µm

def erf_model(x, A, B, C, D):
    """A * erf(B*(x-C)) + D"""
    return A * erf(B * (x - C)) + D

def gaussian_beam_model(z, w0, z0, M2):
    """
    Gaussian beam propagation model for beam radius w(z).
    z and z0 in mm. w0 in µm. M2 is dimensionless.
    Output w(z) in µm.
    lambda_wl_um is the global wavelength in µm.
    """
    # Convert Rayleigh range z_R from mm to mm, but use w0 in µm (multiplied by 1e-3 to be in mm)
    # The M2 factor is applied to w0 to represent a real beam.
    # z_R (mm) = pi * w0^2 (mm^2) / lambda (mm)
    w0_mm = w0 * 1e-3 # Convert fit w0 (µm) to mm for z_R calc
    z_R = np.pi * w0_mm**2 / lambda_wl_mm
    
    # Calculate the beam radius w(z) in mm first, then convert to µm
    w_z_mm = w0_mm * np.sqrt(M2 + (z - z0)**2 / z_R**2)
    return w_z_mm * 1000.0 # Return in µm

def find_intersection(fit_left, fit_right):
    m1, b1 = fit_left
    m2, b2 = fit_right
    if abs(m1 - m2) < 1e-12:
        return np.nan, np.nan
    z_int = (b2 - b1) / (m1 - m2)
    w_int = m1 * z_int + b1
    return z_int, w_int

def safe_float(s, default=None):
    try:
        return float(s)
    except Exception:
        return default

# -------------------------
# Load BP (slit) dataset (new)
# -------------------------
print("\n--- Loading BP (slit) data from:", BP_DIR)
slit_glob = os.path.join(BP_DIR, "*[m|mm]_#001.txt")
slit_files = sorted(
    [f for f in glob.glob(slit_glob) if "before lens" not in os.path.basename(f).lower()],
    key=lambda f: safe_float(os.path.basename(f).split('m')[0]) if safe_float(os.path.basename(f).split('m')[0]) is not None else np.inf
)

slit_z_mm = []
slit_wx_um = []
slit_wy_um = []

for file in slit_files:
    base = os.path.basename(file)
    # try to extract z raw as float from start of filename like '8.50m_#001.txt' or '8.5mm_#001.txt'
    try:
        z_raw = float(base.split('m')[0])
    except Exception:
        print("Skipping (bad name):", base)
        continue

    wx_dia = wy_dia = None
    try:
        with open(file, 'r', encoding='latin1') as fh:
            lines = fh.readlines()
    except Exception as e:
        print("Skipping (can't read):", base, e)
        continue

    for line in lines:
        if line.strip() and line.split('\t')[0].isdigit():
            parts = line.strip().split('\t')
            # defensive: ensure parts long enough
            if len(parts) > 20:
                try:
                    wx_dia = float(parts[19])
                    wy_dia = float(parts[20])
                except Exception:
                    wx_dia = wy_dia = None
            break

    if wx_dia is None or wy_dia is None:
        # skip files without proper numeric rows
        continue

    # Original slit wx_dia, wy_dia units: the old code used µm diameter -> divided by 2 to radius in µm.
    # The new BP snippet sometimes converted to mm. We assume the raw values are µm (matching your earlier code).
    # To be robust, detect if the values are < 10 (then likely in mm); otherwise treat as µm.
    # If value < 20 it's suspicious — but to keep safe, we infer:
    # - if wx_dia < 20 treat as mm * 1000 -> convert to µm
    # - else treat as µm
    def dia_to_um(d):
        if abs(d) < 20:
            # interpret as mm -> µm
            return d * 1000.0
        return d

    wx_dia_um = dia_to_um(wx_dia)
    wy_dia_um = dia_to_um(wy_dia)

    z_shifted = z_raw + SLIT_OFFSET_MM
    slit_z_mm.append(z_shifted)
    slit_wx_um.append(wx_dia_um / 2.0)  # diameter -> radius in µm
    slit_wy_um.append(wy_dia_um / 2.0)

slit_z_mm = np.array(slit_z_mm)
slit_wx_um = np.array(slit_wx_um)
slit_wy_um = np.array(slit_wy_um)

# sort by z
if slit_z_mm.size:
    order = np.argsort(slit_z_mm)
    slit_z_mm = slit_z_mm[order]
    slit_wx_um = slit_wx_um[order]
    slit_wy_um = slit_wy_um[order]

# apply Z-mask (after shift)
mask = (slit_z_mm >= SLIT_Z_MIN_MM) & (slit_z_mm <= SLIT_Z_MAX_MM)
slit_z_mm_masked = slit_z_mm[mask]
slit_wx_um_masked = slit_wx_um[mask]
slit_wy_um_masked = slit_wy_um[mask]

print(f"Slit scanner: total files parsed {len(slit_files)}, kept after mask: {len(slit_z_mm_masked)} points (Z-mask [{SLIT_Z_MIN_MM}, {SLIT_Z_MAX_MM}] mm)")

# -------------------------
# Slit arm fits -> left/right -> intersection (KEPT for Gaussian p0 guess only)
# -------------------------
# X direction
z0_x = w0_x_um = np.nan
fit_left_x = fit_right_x = [np.nan, np.nan] # Init for plotting
if slit_z_mm_masked.size >= 3:
    min_idx_x = np.argmin(slit_wx_um_masked)
    z_left_x = slit_z_mm_masked[:min_idx_x]
    wx_left = slit_wx_um_masked[:min_idx_x]
    z_right_x = slit_z_mm_masked[min_idx_x:]
    wx_right = slit_wx_um_masked[min_idx_x:]
    fit_left_x = np.polyfit(z_left_x, wx_left, 1) if len(z_left_x) > 1 else [np.nan, np.nan]
    fit_right_x = np.polyfit(z_right_x, wx_right, 1) if len(z_right_x) > 1 else [np.nan, np.nan]
    theta_left_x = fit_left_x[0] if not np.isnan(fit_left_x[0]) else np.nan
    theta_right_x = fit_right_x[0] if not np.isnan(fit_right_x[0]) else np.nan
    # convert theta (slope in µm/mm) to radians? Note: you earlier used lambda/(pi*abs(theta)) with theta in rad/mm.
    # If slope is in µm/mm, convert to mm/mm by dividing µm -> mm
    def theta_to_w0_um(theta_slope):
        if np.isnan(theta_slope) or theta_slope == 0:
            return np.nan
        # slope: d(w [µm]) / d(z [mm]) -> convert numerator to mm for divergence in mm/mm
        theta_mm_per_mm = theta_slope / 1000.0
        # treat theta_mm_per_mm as small angle in rad, then w0 (mm) = lambda / (pi * theta (rad))
        w0_mm = (lambda_wl_mm / (np.pi * abs(theta_mm_per_mm)))
        return w0_mm * 1000.0  # µm

    w0_left_x_um = theta_to_w0_um(theta_left_x)
    w0_right_x_um = theta_to_w0_um(theta_right_x)
    if len(z_left_x) > 1 and len(z_right_x) > 1:
        z0_x, w0_x_um = find_intersection(fit_left_x, fit_right_x)

else:
    print("Not enough slit X points for arm fit/intersection.")

# Y direction
z0_y = w0_y_um = np.nan
fit_left_y = fit_right_y = [np.nan, np.nan] # Init for plotting
if slit_z_mm_masked.size >= 3:
    min_idx_y = np.argmin(slit_wy_um_masked)
    z_left_y = slit_z_mm_masked[:min_idx_y]
    wy_left = slit_wy_um_masked[:min_idx_y]
    z_right_y = slit_z_mm_masked[min_idx_y:]
    wy_right = slit_wy_um_masked[min_idx_y:]
    fit_left_y = np.polyfit(z_left_y, wy_left, 1) if len(z_left_y) > 1 else [np.nan, np.nan]
    fit_right_y = np.polyfit(z_right_y, wy_right, 1) if len(z_right_y) > 1 else [np.nan, np.nan]
    theta_left_y = fit_left_y[0] if not np.isnan(fit_left_y[0]) else np.nan
    theta_right_y = fit_right_y[0] if not np.isnan(fit_right_y[0]) else np.nan
    w0_left_y_um = theta_to_w0_um(theta_left_y)
    w0_right_y_um = theta_to_w0_um(theta_right_y)
    if len(z_left_y) > 1 and len(z_right_y) > 1:
        z0_y, w0_y_um = find_intersection(fit_left_y, fit_right_y)
else:
    print("Not enough slit Y points for arm fit/intersection.")

# Print slit fit summary (KEPT for debugging, but not used in plot/main result box)
print("\n=== SLIT WAIST (intersection, for p0 only) ===")
print(f"X-waist: Z = {z0_x:.3f} mm, w₀ = {w0_x_um:.1f} µm" if not np.isnan(z0_x) else "X-waist: not fitted")
print(f"Y-waist: Z = {z0_y:.3f} mm, w₀ = {w0_y_um:.1f} µm" if not np.isnan(z0_y) else "Y-waist: not fitted")

# -------------------------
# NEW: Gaussian Fit on BP (slit) data (far-field arms only)
# -------------------------
print("\n--- NEW: Gaussian Fit on far-field BP data ---")

# Initialize results
gauss_params_x = gauss_params_y = [np.nan, np.nan, np.nan]
w0_fit_x = z0_fit_x = M2_fit_x = w0_DL_x = np.nan
w0_fit_y = z0_fit_y = M2_fit_y = w0_DL_y = np.nan

if slit_z_mm_masked.size >= 6: # Need more points for 3 parameters
    # 1. Determine the Z range to exclude around the minimum
    min_z_x = slit_z_mm_masked[np.argmin(slit_wx_um_masked)]
    min_z_y = slit_z_mm_masked[np.argmin(slit_wy_um_masked)]

    # 2. Create the mask (True for points OUTSIDE the exclusion zone)
    mask_gauss_x = np.abs(slit_z_mm_masked - min_z_x) > GAUSS_Z_WAIST_EXCLUSION_MM_X # Correctly excludes around min_z_x
    mask_gauss_y = np.abs(slit_z_mm_masked - min_z_y) > GAUSS_Z_WAIST_EXCLUSION_MM_Y # Correctly excludes around min_z_y

    slit_z_mm_gauss_mask_x = slit_z_mm_masked[mask_gauss_x]
    slit_wx_um_gauss_mask_x = slit_wx_um_masked[mask_gauss_x]
    
    slit_z_mm_gauss_mask_y = slit_z_mm_masked[mask_gauss_y]
    slit_wy_um_gauss_mask_y = slit_wy_um_masked[mask_gauss_y]

    print(f"X-fit: {len(slit_z_mm_gauss_mask_x)} points used (excluded {GAUSS_Z_WAIST_EXCLUSION_MM_X} mm around Z={min_z_x:.3f} mm)")
    print(f"Y-fit: {len(slit_z_mm_gauss_mask_y)} points used (excluded {GAUSS_Z_WAIST_EXCLUSION_MM_Y} mm around Z={min_z_y:.3f} mm)")

    # 3. Perform the fit for X (w0, z0, M2)
    if len(slit_z_mm_gauss_mask_x) >= 3:
        try:
            # Determine empirical minimum for robust initial guess
            min_idx_emp_x = np.argmin(slit_wx_um_masked)
            w_min_emp_x = slit_wx_um_masked[min_idx_emp_x]
            z_min_emp_x = slit_z_mm_masked[min_idx_emp_x]
            
            p0_x = [w_min_emp_x,  # Use empirical minimum waist as w0 guess
                    z_min_emp_x,  # Use empirical minimum Z as z0 guess
                    1.5] # Start M2 with an expected value > 1
            
            # Bounds: w0 > 0 (µm), z0 (mm) within Z range, M2 >= 1.0 (real beam)
            bounds_x = ([0, SLIT_Z_MIN_MM, 1.0], [np.inf, SLIT_Z_MAX_MM, np.inf])

            gauss_params_x, pcov_x = curve_fit(gaussian_beam_model, 
                                            slit_z_mm_gauss_mask_x, 
                                            slit_wx_um_gauss_mask_x, 
                                            p0=p0_x, 
                                            bounds=bounds_x,
                                            maxfev=5000)
            
            w0_fit_x, z0_fit_x, M2_fit_x = gauss_params_x
            
            # Calculate theoretical diffraction limit
            w0_DL_x = w0_fit_x / M2_fit_x

        except Exception as e:
            print(f"Gaussian fit failed for X-direction: {e}")

    # 4. Perform the fit for Y (w0, z0, M2)
    if len(slit_z_mm_gauss_mask_y) >= 3:
        try:
            # Determine empirical minimum for robust initial guess
            min_idx_emp_y = np.argmin(slit_wy_um_masked)
            w_min_emp_y = slit_wy_um_masked[min_idx_emp_y]
            z_min_emp_y = slit_z_mm_masked[min_idx_emp_y]
            
            p0_y = [w_min_emp_y,  # Use empirical minimum waist as w0 guess
                    z_min_emp_y,  # Use empirical minimum Z as z0 guess
                    1.5]
            
            bounds_y = ([0, SLIT_Z_MIN_MM, 1.0], [np.inf, SLIT_Z_MAX_MM, np.inf])

            gauss_params_y, pcov_y = curve_fit(gaussian_beam_model, 
                                            slit_z_mm_gauss_mask_y, 
                                            slit_wy_um_gauss_mask_y, 
                                            p0=p0_y, 
                                            bounds=bounds_y,
                                            maxfev=5000)
            
            w0_fit_y, z0_fit_y, M2_fit_y = gauss_params_y

            # Calculate theoretical diffraction limit
            w0_DL_y = w0_fit_y / M2_fit_y

        except Exception as e:
            print(f"Gaussian fit failed for Y-direction: {e}")

    print("\n=== SLIT WAIST (Gaussian Fit) ===")
    print(f"X-fit: Z₀ = {z0_fit_x:.3f} mm, w₀ = {w0_fit_x:.1f} µm, M² = {M2_fit_x:.2f}" if not np.isnan(w0_fit_x) else "X-fit: not fitted")
    print(f"Y-fit: Z₀ = {z0_fit_y:.3f} mm, w₀ = {w0_fit_y:.1f} µm, M² = {M2_fit_y:.2f}" if not np.isnan(w0_fit_y) else "Y-fit: not fitted")
    print("\n=== THEORETICAL DIFFRACTION LIMIT (from Gaussian Fit) ===")
    print(f"X-Theoretical waist w₀,DL: {w0_DL_x:.1f} µm" if not np.isnan(w0_DL_x) else "X-Theoretical waist: not estimated")
    print(f"Y-Theoretical waist w₀,DL: {w0_DL_y:.1f} µm" if not np.isnan(w0_DL_y) else "Y-Theoretical waist: not estimated")

else:
    print("Not enough slit points for Gaussian fit.")


# -------------------------
# Load QPD .pkl data (remaining code for QPD loading, KE fitting, and plotting prep is unchanged)
# -------------------------
print("\n--- Loading QPD .pkl data from (quadA loader kept):", QPD_DIR)
# [ ... QPD loading and KE fitting code (unchanged) ... ]

def extract_z_from_name(fname):
    m = re.search(r'Z(\d+)um', fname)
    return int(m.group(1)) / 1000.0 if m else None

pkl_files = sorted(glob.glob(os.path.join(QPD_DIR, "*.pkl")),
                   key=lambda f: extract_z_from_name(os.path.basename(f)) if extract_z_from_name(os.path.basename(f)) is not None else np.inf)

# original quadA outputs (kept but not plotted)
qpd_z_raw = []
qpd_z_shifted = []
qpd_w_um = []

for pkl in pkl_files:
    base = os.path.basename(pkl)
    z_raw = extract_z_from_name(base)
    if z_raw is None:
        continue
    # filter QPD files to the intended z range
    if z_raw < QPD_Z_MIN_MM or z_raw > QPD_Z_MAX_MM:
        # skip outside relevant range
        continue
    try:
        data = pickle.load(open(pkl, "rb"))
    except Exception as e:
        print("Skipping pkl (can't load):", base, e)
        continue

    # Build x array robustly: try mm keys then um keys
    if 'global_params' not in data:
        print("Skipping pkl (missing global_params):", base)
        continue
    gp = data['global_params']
    # Prefer um keys if present
    if 'xstart_um' in gp:
        xstart = gp['xstart_um']; xstop = gp['xstop_um']
        xbig = gp.get('xstep_big_um', gp.get('xstep_big', 50))
        xfine = gp.get('xstep_fine_um', gp.get('xstep_fine', 1))
        xth_s = gp.get('x_threshold_start_um', gp.get('x_threshold_start', -1e9))
        xth_e = gp.get('x_threshold_stop_um', gp.get('x_threshold_stop', 1e9))
        # x_arr in µm
        cur = xstart
        x_arr = []
        while cur <= xstop + 1e-9:
            x_arr.append(cur)
            cur += xfine if (xth_s <= cur <= xth_e) else xbig
        x_arr = np.array(x_arr)
        x_arr_units = 'um'
    elif 'xstart_mm' in gp:
        xstart = gp['xstart_mm']; xstop = gp['xstop_mm']
        xbig = gp.get('xstep_big_mm', gp.get('xstep_big', 0.1))
        xfine = gp.get('xstep_fine_mm', gp.get('xstep_fine', 0.01))
        xth_s = gp.get('x_threshold_start_mm', -1e9)
        xth_e = gp.get('x_threshold_stop_mm', 1e9)
        cur = xstart
        x_arr_mm = []
        while cur <= xstop + 1e-9:
            x_arr_mm.append(cur)
            cur += xfine if (xth_s <= cur <= xth_e) else xbig
        x_arr = np.array(x_arr_mm) * 1000.0  # convert to µm
        x_arr_units = 'um'
    else:
        # fallback: try 'x_values' or similar
        print("Warning: unknown x steps in", base, " — skipping")
        # continue # original had a continue here, now continue with the next file
        continue

    # Ensure quadA present
    if 'rawdata' not in data or 'quadA' not in data['rawdata'] or 'dmm00_curr_amp' not in data['rawdata']['quadA']:
        print("Skipping pkl (no quadA data):", base)
        continue

    dc = np.mean(data['rawdata']['quadA']['dmm00_curr_amp'], axis=1)

    # ensure same length
    if len(x_arr) != len(dc):
        print(f"Length mismatch in {base} (x_arr {len(x_arr)} vs dc {len(dc)}) — skipping")
        continue

    # ERF fit: x_arr in µm, dc in A (or arbitrary units)
    try:
        initial_guess = [np.max(dc) - np.min(dc), 1.0 / (np.std(x_arr) + 1e-9), np.mean(x_arr), np.min(dc)]
        # bounds: B (scale) should be >0
        params, _ = curve_fit(erf_model, x_arr, dc, p0=initial_guess, bounds=([-np.inf, 0, -np.inf, -np.inf],[np.inf, np.inf, np.inf, np.inf]))
        A_fit, B_fit, x0_fit, D_fit = params
        # spot_size [µm] = 1 / (sqrt(2) * B)  (because erf argument B*(x-C))
        spot_size_um = 1.0 / (np.sqrt(2.0) * B_fit) if B_fit != 0 else np.nan

        if np.isnan(spot_size_um) or spot_size_um <= 0 or spot_size_um > SPOT_SIZE_SANITY_UM:
            print(f"Fit produced bad spot size {spot_size_um} µm for {base} — skipping")
            continue

        z_shifted = z_raw + QPD_OFFSET_MM
        qpd_z_raw.append(z_raw)
        qpd_z_shifted.append(z_shifted)
        qpd_w_um.append(spot_size_um)
        # log
        print(f"QPD (quadA loader kept) {base} → Z_raw {z_raw:.3f} mm, Z_shifted {z_shifted:.3f} mm, w = {spot_size_um:.2f} µm")

    except Exception as e:
        print("Fit failed for", base, e)
        continue

qpd_z_raw = np.array(qpd_z_raw)
qpd_z_shifted = np.array(qpd_z_shifted)
qpd_w_um = np.array(qpd_w_um)

print(f"QPD (quadA loader kept): loaded {len(qpd_w_um)} knife-edge measurements within Z range [{QPD_Z_MIN_MM}, {QPD_Z_MAX_MM}] mm")

# -------------------------
# Load QPD .pkl data (new) — quadB and quadC KE fits (these will be plotted)
# -------------------------
print("\n--- Loading QPD .pkl data (quadB & quadC):", QPD_DIR)

# reuse extract_z_from_name as defined above
pkl_files = sorted(
    glob.glob(os.path.join(QPD_DIR, "*.pkl")),
    key=lambda f: extract_z_from_name(os.path.basename(f)) if extract_z_from_name(os.path.basename(f)) is not None else np.inf
)

qpd_z_shifted_B = []
qpd_w_um_B = []
qpd_z_shifted_C = []
qpd_w_um_C = []

def load_quad_dc(data, quad):
    """Return DC photocurrent averaged array for given quad label, or None"""
    try:
        return np.mean(data['rawdata'][quad]['dmm00_curr_amp'], axis=1)
    except Exception:
        return None

for pkl in pkl_files:
    base = os.path.basename(pkl)
    z_raw = extract_z_from_name(base)
    if z_raw is None:
        continue
    if z_raw < QPD_Z_MIN_MM or z_raw > QPD_Z_MAX_MM:
        continue

    try:
        data = pickle.load(open(pkl, 'rb'))
    except Exception as e:
        print("Skipping pkl (can't load):", base, e)
        continue

    # Build x array robustly (same logic as above)
    if 'global_params' not in data:
        print("Skipping pkl (missing global_params):", base)
        continue
    gp = data['global_params']
    if 'xstart_um' in gp:
        cur = gp['xstart_um']
        step_big = gp.get('xstep_big_um', gp.get('xstep_big', 50))
        step_fine = gp.get('xstep_fine_um', gp.get('xstep_fine', 1))
        th_s = gp.get('x_threshold_start_um', gp.get('x_threshold_start', -1e9))
        th_e = gp.get('x_threshold_stop_um', gp.get('x_threshold_stop', 1e9))
        arr = []
        while cur <= gp['xstop_um'] + 1e-9:
            arr.append(cur)
            cur += step_fine if (th_s <= cur <= th_e) else step_big
        x_arr = np.array(arr)
    elif 'xstart_mm' in gp:
        cur = gp['xstart_mm']
        step_big = gp.get('xstep_big_mm', gp.get('xstep_big', 0.1))
        step_fine = gp.get('xstep_fine_mm', gp.get('xstep_fine', 0.01))
        th_s = gp.get('x_threshold_start_mm', -1e9)
        th_e = gp.get('x_threshold_stop_mm', gp.get('x_threshold_stop_mm', 1e9)) if gp.get('x_threshold_stop_mm', None) is not None else gp.get('x_threshold_stop_mm', 1e9)
        arr = []
        while cur <= gp['xstop_mm'] + 1e-9:
            arr.append(cur)
            cur += step_fine if (th_s <= cur <= th_e) else step_big
        x_arr = np.array(arr) * 1000.0  # convert mm→µm
    else:
        print("Cannot determine x array:", base)
        continue

    # Extract quadB and quadC
    dc_B = load_quad_dc(data, 'quadB')
    dc_C = load_quad_dc(data, 'quadC')

    if dc_B is None and dc_C is None:
        print("Skipping (no quadB/quadC):", base)
        continue

    def fit_erf_spot(dc, label, base, z_raw):
        if dc is None:
            print(f"No data for {label} in {base}")
            return None
        if len(dc) != len(x_arr):
            print(f"Length mismatch for {label} in {base}: dc {len(dc)} vs x_arr {len(x_arr)} — using shortest")
            min_len = min(len(dc), len(x_arr))
            dc = dc[:min_len]
            x_local = x_arr[:min_len]
        else:
            x_local = x_arr

        try:
            p0 = [np.ptp(dc), 1.0/(np.std(x_local)+1e-9), np.mean(x_local), np.min(dc)]
            params, _ = curve_fit(erf_model, x_local, dc, p0=p0,
                                bounds=([-np.inf, 1e-6, -np.inf, -np.inf],
                                        [np.inf, np.inf, np.inf, np.inf]), maxfev=5000)
            B_fit = params[1]
            spot = 1.0 / (np.sqrt(2.0) * B_fit)
            if spot > SPOT_SIZE_SANITY_UM:
                print(f"{label} in {base} (Z={z_raw:.1f}mm): huge waist {spot:.1f} µm — keeping anyway")
            return spot
        except Exception as e:
            print(f"Fit failed for {label} in {base} (Z={z_raw:.1f}mm): {e} — keeping NaN")
            return np.nan

    spotB = fit_erf_spot(dc_B, 'quadB', base, z_raw)
    spotC = fit_erf_spot(dc_C, 'quadC', base, z_raw)

    z_shift = z_raw + QPD_OFFSET_MM

    if spotB is not None:
        qpd_z_shifted_B.append(z_shift)
        qpd_w_um_B.append(spotB)
        print(f"quadB: Z={z_shift:.3f} mm, w={spotB:.2f} µm")

    if spotC is not None:
        qpd_z_shifted_C.append(z_shift)
        qpd_w_um_C.append(spotC)
        print(f"quadC: Z={z_shift:.3f} mm, w={spotC:.2f} µm")

# convert and sort by z
if len(qpd_z_shifted_B):
    order_B = np.argsort(qpd_z_shifted_B)
    qpd_z_shifted_B = np.array(qpd_z_shifted_B)[order_B]
    qpd_w_um_B = np.array(qpd_w_um_B)[order_B]
else:
    qpd_z_shifted_B = np.array([])
    qpd_w_um_B = np.array([])

if len(qpd_z_shifted_C):
    order_C = np.argsort(qpd_z_shifted_C)
    qpd_z_shifted_C = np.array(qpd_z_shifted_C)[order_C]
    qpd_w_um_C = np.array(qpd_w_um_C)[order_C]
else:
    qpd_z_shifted_C = np.array([])
    qpd_w_um_C = np.array([])

print(f"\nQPD quadB total: {len(qpd_w_um_B)}")
print(f"QPD quadC total: {len(qpd_w_um_C)}")

# -------------------------
# Prepare color mapping for QPD points (use combined B+C z's for colormap)
# -------------------------
combined_z = np.concatenate([
    qpd_z_shifted_B if qpd_z_shifted_B.size else np.array([]),
    qpd_z_shifted_C if qpd_z_shifted_C.size else np.array([])
])
if combined_z.size:
    unique_z = sorted(set(np.round(combined_z, 6)))
else:
    unique_z = []

cmap = plt.cm.RdYlBu
color_dict = {}
if unique_z:
    n_z = len(unique_z)
    for i, z in enumerate(unique_z):
        if n_z == 1:
            color_dict[z] = cmap(0.5)
        else:
            if i < n_z // 2:
                color_dict[z] = cmap(0.1 + 0.4 * (i / max(1, (n_z // 2 - 1))))
            else:
                color_dict[z] = cmap(0.6 + 0.4 * ((i - n_z // 2) / max(1, (n_z - n_z // 2 - 1))))

def color_for_z(z):
    if not unique_z:
        return 'C0'
    z_r = min(unique_z, key=lambda u: abs(u - z))
    return color_dict.get(z_r, 'C0')

# -------------------------
# Plot combined figure - STYLED
# -------------------------
print("\n--- Creating combined comparison figure ---")
fig, ax = plt.subplots(figsize=(12, 7), layout='constrained') # FONTSIZE 12x7

ax.set_xlabel(r'Z Position [mm]', fontsize=16) # FONTSIZE 16
ax.set_ylabel(r'Beam Radius $w$(z) [\textmu{}m]', fontsize=16) # FONTSIZE 16
# Removed bold from title as requested
ax.set_title(r'\textbf{Beam Profile: KE (quadB \& quadC) vs. BP Data (Gaussian Fit)', fontsize=18) # FONTSIZE 18
ax.grid(True, ls='--', alpha=0.6)
ax.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5, direction='in') # FONTSIZE 13

# Plot slit BP points (converted to µm)
if slit_z_mm_masked.size:
    ax.plot(slit_z_mm_masked, slit_wx_um_masked, 's', mfc='none', mec='black', ms=9, label=r'$w_{\rm{{x}}}$(z) (BP)') # MS=9
    ax.plot(slit_z_mm_masked, slit_wy_um_masked, '^', mfc='none', mec='black', ms=9, label=r'$w_{\rm{{y}}}$(z) (BP)') # MS=9

# --- NEW: Plot Gaussian fit curves (Changed colors as requested) ---
z_range_gauss = np.linspace(np.min(slit_z_mm_masked) if slit_z_mm_masked.size else 0, 
                            np.max(slit_z_mm_masked) if slit_z_mm_masked.size else 1, 200)

if not np.isnan(w0_fit_x):
    w_x_gauss = gaussian_beam_model(z_range_gauss, w0_fit_x, z0_fit_x, M2_fit_x)
    # Changed color to red
    ax.plot(z_range_gauss, w_x_gauss, '-', color='red', lw=2.5, label=r'Gaussian fit: $w_{\rm{{x}}}$(z) (BP)')
    # Plot the calculated beam waist (w0_fit_x)
    ax.plot(z0_fit_x, w0_fit_x, 's', mec='red', mfc='red', mew=2, ms=8) # label removed

if not np.isnan(w0_fit_y):
    w_y_gauss = gaussian_beam_model(z_range_gauss, w0_fit_y, z0_fit_y, M2_fit_y)
    # Changed color to blue
    ax.plot(z_range_gauss, w_y_gauss, '-', color='blue', lw=2.5, label=r'Gaussian fit: $w_{\rm{{y}}}$(z) (BP)')
    # Plot the calculated beam waist (w0_fit_y)
    ax.plot(z0_fit_y, w0_fit_y, '^', mec='blue', mfc='blue', mew=2, ms=8) # label removed

# Plot QPD quadB & quadC separately (no averaging)
if qpd_z_shifted_B.size:
    for z_s, w in zip(qpd_z_shifted_B, qpd_w_um_B):
        ax.plot(z_s, w, 'o', color=color_for_z(z_s), markersize=7)
    ax.plot(qpd_z_shifted_B, qpd_w_um_B, '--', color='orange', lw=2.2, alpha=0.8, label=r'$w$(z) quadB (KE)') # LW=2.2

if qpd_z_shifted_C.size:
    for z_s, w in zip(qpd_z_shifted_C, qpd_w_um_C):
        ax.plot(z_s, w, 'D', color=color_for_z(z_s), markersize=7) # fillstyle='none'
    ax.plot(qpd_z_shifted_C, qpd_w_um_C, '-.', color='green', lw=2.2, alpha=0.8, label=r'$w$(z) quadC (KE)') # LW=2.2

# Inset: minima from quadB, quadC, and BP intersections
min_info = []
# quadB min
if qpd_w_um_B.size:
    idx_min_B = int(np.argmin(qpd_w_um_B))
    z_min_B = float(qpd_z_shifted_B[idx_min_B])
    w_min_B = float(qpd_w_um_B[idx_min_B])
    ax.plot(z_min_B, w_min_B, 'o', mec='red', mfc='none', mew=2.5, ms=14) # MS=14
    min_info.append(("B", w_min_B, z_min_B))
# quadC min
if qpd_w_um_C.size:
    idx_min_C = int(np.argmin(qpd_w_um_C))
    z_min_C = float(qpd_z_shifted_C[idx_min_C])
    w_min_C = float(qpd_w_um_C[idx_min_C])
    ax.plot(z_min_C, w_min_C, 'D', mec='red', mfc='none', mew=2.5, ms=14) # MS=14
    min_info.append(("C", w_min_C, z_min_C))

# BP minima info (Gaussian Fit)
if not np.isnan(w0_fit_x):
    min_info.append(("BP_x_Gauss", w0_fit_x, z0_fit_x))
if not np.isnan(w0_fit_y):
    min_info.append(("BP_y_Gauss", w0_fit_y, z0_fit_y))
# BP Theoretical Limit (Gauss Fit) - KEPT in min_info for text output ONLY
if not np.isnan(w0_DL_x):
    min_info.append(("BP_x_DL", w0_DL_x, z0_fit_x))
if not np.isnan(w0_DL_y):
    min_info.append(("BP_y_DL", w0_DL_y, z0_fit_y))


# Inset box with values - STYLED (Modified to show KE and BP Gaussian w0 values)
if min_info:
    # Separate inset for results
    inset = ax.inset_axes([0.48, 0.49, 0.45, 0.4]) # Adjusted position
    inset.axis('off')
    handles_inset = []
    labels_inset = []
    
    # --- 1. KE Results (quadB and quadC) ---
    ke_info = [item for item in min_info if item[0] in ("B", "C")]

    for tag, wv, zv in ke_info:
        if tag == "B":
            marker = 'o'; mcol = 'red'; label = rf'$w_{{0,\min}}$(KE, quadB) = ${wv:.2f}\,\mu\mathrm{{m}}$'
        elif tag == "C":
            marker = 'D'; mcol = 'red'; label = rf'$w_{{0,\min}}$(KE, quadC) = ${wv:.2f}\,\mu\mathrm{{m}}$'
        
        handles_inset.append(Line2D([0], [0], marker=marker, color='w', markeredgecolor=mcol,
                              markerfacecolor='none', markersize=10, markeredgewidth=2.0))
        labels_inset.append(label)

    # --- 2. BP Gaussian Fit Results ---
    gauss_fit_info = [item for item in min_info if "Gauss" in item[0] and "DL" not in item[0]]
    
    # Add a visual separator (dummy item)
    '''if ke_info and gauss_fit_info:
        handles_inset.append(Line2D([0], [0], marker='', linestyle='None'))
        labels_inset.append(r'')'''
    # Add a visual separator (dummy item)
    if ke_info and gauss_fit_info:
        handles_inset.append(Line2D([0], [0], marker='', linestyle='None'))
        labels_inset.append(' ') # Use a single space character instead of r''


    for tag, wv, zv in gauss_fit_info:
        if tag == "BP_x_Gauss":
            marker = 's'; mcol = 'red'; mfc = 'red'; label = rf'$w_{{0,\rm{{x}}}}$(Gauss) = ${wv:.2f}\,\mu\mathrm{{m}}$'
        elif tag == "BP_y_Gauss":
            marker = '^'; mcol = 'blue'; mfc = 'blue'; label = rf'$w_{{0,\rm{{y}}}}$(Gauss) = ${wv:.2f}\,\mu\mathrm{{m}}$'
        else:
            continue
            
        handles_inset.append(Line2D([0], [0], marker=marker, color='w', markeredgecolor=mcol,
                              markerfacecolor=mfc, markersize=10, markeredgewidth=2.0))
        labels_inset.append(label)
    
    # --- 3. Plot the inset (Removed bold title, increased fontsize) ---
    if handles_inset:
        try:
            inset.legend(handles=handles_inset, labels=labels_inset, 
                         loc='upper left', fontsize=16.8, frameon=False) # Increased fontsize to 16.8
        except Exception:
            txt = "\n".join(labels_inset)
            inset.text(0, 0.9, txt, va='top', fontsize=10)

# Legend (de-duplicate, single column, larger fontsize)
handles, labels = ax.get_legend_handles_labels()
# Filter out markers for w0_fit_x and w0_fit_y (since they no longer have labels)
# Removed 'Linear fit: w_x(z) (BP)' and 'Linear fit: w_y(z) (BP)'
labels_to_keep = [r'$w_{\rm{{x}}}$(z) (BP)', r'$w_{\rm{{y}}}$(z) (BP)', 
                  r'Gaussian fit: $w_{\rm{{x}}}$(z) (BP)', r'Gaussian fit: $w_{\rm{{y}}}$(z) (BP)', 
                  r'$w$(z) quadB (KE)', r'$w$(z) quadC (KE)']
by_label = {l: h for l, h in zip(labels, handles) if l in labels_to_keep}

# Single column, larger font, adjusted position for better fit
ax.legend(by_label.values(), by_label.keys(), fontsize=14, loc='upper left', frameon=True, bbox_to_anchor=(0.02, 0.98), ncols=1) 

# Axis limits: make sure data isn't outside and hide empty-plot risk
all_z_values = np.concatenate([
    slit_z_mm_masked if slit_z_mm_masked.size else np.array([]),
    qpd_z_shifted_B if qpd_z_shifted_B.size else np.array([]),
    qpd_z_shifted_C if qpd_z_shifted_C.size else np.array([])
])
if all_z_values.size:
    zmin = np.min(all_z_values) - 0.5
    zmax = np.max(all_z_values) + 0.5
    ax.set_xlim(8.2, 15.2) #zmin, zmax
    #ax.set_xlim(10.45, 13.55) #zmin, zmax
    #ax.set_xlim(6, 16)   # safe range covering raw and shifted data
else:
    ax.set_xlim(SLIT_Z_MIN_MM, SLIT_Z_MAX_MM)

all_w_values = np.concatenate([
    slit_wx_um_masked if slit_wx_um_masked.size else np.array([]),
    slit_wy_um_masked if slit_wy_um_masked.size else np.array([]),
    qpd_w_um_B if qpd_w_um_B.size else np.array([]),
    qpd_w_um_C if qpd_w_um_C.size else np.array([]),
    np.array([w0_fit_x, w0_fit_y]) # Excluded DL values here
])
if all_w_values.size and not np.all(np.isnan(all_w_values)):
    wmin = max(np.nanmin(all_w_values) - 20.0, 0.0)
    wmax = np.nanmax(all_w_values) + 50.0
    ax.set_ylim(bottom=-10.0, top=1800) #395
else:
    ax.set_ylim(bottom=-10, top=600)

# Save and show
out_path = os.path.join(OUT_DIR, "merged_QPD_vs_BP_comparison_styled_gauss.png")
fig.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\nComparison figure saved → {out_path}")

# Print final summary (text) - KEPT full detail here for logging/debugging
print("\n=== FINAL RESULTS SUMMARY (Log Detail) ===")
# Report quadB min
if qpd_w_um_B.size:
    i_min_B = int(np.argmin(qpd_w_um_B))
    print(f"QPD (KE quadB): w0_min = {qpd_w_um_B[i_min_B]:.2f} µm at Z = {qpd_z_shifted_B[i_min_B]:.3f} mm")
else:
    print("QPD (KE quadB): no valid fits")
# Report quadC min
if qpd_w_um_C.size:
    i_min_C = int(np.argmin(qpd_w_um_C))
    print(f"QPD (KE quadC): w0_min = {qpd_w_um_C[i_min_C]:.2f} µm at Z = {qpd_z_shifted_C[i_min_C]:.3f} mm")
else:
    print("QPD (KE quadC): no valid fits")

print("\n--- BP (Linear Fit) ---")
print(f"Slit X: {'not fitted' if np.isnan(z0_x) else f'w0 = {w0_x_um:.2f} µm at Z = {z0_x:.3f} mm'}")
print(f"Slit Y: {'not fitted' if np.isnan(z0_y) else f'w0 = {w0_y_um:.2f} µm at Z = {z0_y:.3f} mm'}")

print("\n--- BP (Gaussian Fit) ---")
print(f"Slit X: {'not fitted' if np.isnan(w0_fit_x) else f'w0,fit = {w0_fit_x:.2f} µm, Z0 = {z0_fit_x:.3f} mm, M² = {M2_fit_x:.2f}'}")
print(f"Slit Y: {'not fitted' if np.isnan(w0_fit_y) else f'w0,fit = {w0_fit_y:.2f} µm, Z0 = {z0_fit_y:.3f} mm, M² = {M2_fit_y:.2f}'}")

print("\n--- Theoretical Diffraction Limit ---")
print(f"Slit X: {'not estimated' if np.isnan(w0_DL_x) else f'w0,DL = {w0_DL_x:.2f} µm'}")
print(f"Slit Y: {'not estimated' if np.isnan(w0_DL_y) else f'w0,DL = {w0_DL_y:.2f} µm'}")

print("============================\n")

plt.show()