'''
@author: A_Galbiati

Comparisong between the results of obtained with the KE technique (quadB & quadC) and BP data for L4 of the TOS.
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

# -------------------------
# USER SETTINGS 
# -------------------------
# QPD directory (old pipeline)
QPD_DIR = "/Users/asca/Documents/University/Master Thesis/code/Data/all quadrants/20251022/Y2500um"
# BP directory (20251016)
BP_DIR = "/Users/asca/Documents/University/Master Thesis/code/Data/slit scanning beam profiler/20251016_telescoping"

# Z shifts (structure supports non-zero later) — set to 0 now as requested
SLIT_OFFSET_MM = 2.25   # apply to slit z positions 
QPD_OFFSET_MM = 4.5    # apply to QPD z positions

# Slit Z-mask (after shift) — these are defaults; tweak as needed
SLIT_Z_MIN_MM = 8.5    # chosen to match 20251016 snippet mask
SLIT_Z_MAX_MM = 15

# QPD file Z filter (only consider files with Z >= ... mm)
QPD_Z_MIN_MM = 5.85  
QPD_Z_MAX_MM = 15

# Warnings / thresholds
SPOT_SIZE_SANITY_UM = 2000.0   # skip large spots
SPOT_SIZE_KE_WARN_UM = 1000.0  # warn threshold for KE fits

# Plot output
OUT_DIR = os.path.join(BP_DIR, "fig")
os.makedirs(OUT_DIR, exist_ok=True)

# Font formatting
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
def erf_model(x, A, B, C, D):
    """A * erf(B*(x-C)) + D"""
    return A * erf(B * (x - C)) + D

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
# Load BP (slit) dataset
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
    # extract z
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
# Slit arm fits -> left/right -> intersection + divergence
# -------------------------
lambda_wl_mm = 0.001064  # wavelength in mm (1064 nm)

# X direction
z0_x = w0_x_um = np.nan
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
    # convert theta (slope in µm/mm) to radians. Earlier used lambda/(pi*abs(theta)) with theta in rad/mm.
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

# Print slit fit summary
print("\n=== SLIT WAIST (intersection) ===")
print(f"X-waist: Z = {z0_x:.3f} mm, w₀ = {w0_x_um:.1f} µm" if not np.isnan(z0_x) else "X-waist: not fitted")
print(f"Y-waist: Z = {z0_y:.3f} mm, w₀ = {w0_y_um:.1f} µm" if not np.isnan(z0_y) else "Y-waist: not fitted")
print("\n=== SLIT DIVERGENCE (from arm slopes) ===")
print(f"X-left w0 (from slope) : {w0_left_x_um:.1f} µm ; X-right w0: {w0_right_x_um:.1f} µm")
print(f"Y-left w0 (from slope) : {w0_left_y_um:.1f} µm ; Y-right w0: {w0_right_y_um:.1f} µm")

# -------------------------
# Load QPD .pkl data (old pipeline) — original quadA loader (kept for reference)
# -------------------------
print("\n--- Loading QPD .pkl data from (quadA loader kept):", QPD_DIR)
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

    # Build x array
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

    # ERF fit: x_arr in µm, dc in A 
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

    def fit_erf_spot(dc, label):
        if dc is None or len(dc) != len(x_arr):
            return None
        try:
            p0 = [np.max(dc) - np.min(dc), 1.0/(np.std(x_arr)+1e-9), np.mean(x_arr), np.min(dc)]
            params, _ = curve_fit(erf_model, x_arr, dc, p0=p0,
                                  bounds=([-np.inf, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]))
            B_fit = params[1]
            spot = 1.0 / (np.sqrt(2.0) * B_fit)
            if spot <= 0 or spot > SPOT_SIZE_SANITY_UM:
                print(f"Reject {label} spot {spot:.1f} µm in {base}")
                return None
            return spot
        except Exception as e:
            print(f"Fit failed for {label} in {base}: {e}")
            return None

    spotB = fit_erf_spot(dc_B, 'quadB')
    spotC = fit_erf_spot(dc_C, 'quadC')

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
# Plot combined figure
# -------------------------
print("\n--- Creating combined comparison figure ---")
fig, ax = plt.subplots(figsize=(11, 6), layout='constrained')

ax.set_xlabel(r'Z Position [mm]', fontsize=14)
ax.set_ylabel(r'Beam Radius $w$(z) [\textmu{}m]', fontsize=14)
ax.set_title(r'\textbf{Beam Profile: KE (quadB \& quadC) vs. BP Data}', fontsize=15, fontweight='bold')
ax.grid(True, ls='--', alpha=0.6)
ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.2, direction='in')

# Plot slit BP points (converted to µm)
if slit_z_mm_masked.size:
    ax.plot(slit_z_mm_masked, slit_wx_um_masked, 's', mfc='none', mec='black', ms=7, label=r'$w_{\rm{{x}}}$(z) (BP)')
    ax.plot(slit_z_mm_masked, slit_wy_um_masked, '^', mfc='none', mec='black', ms=7, label=r'$w_{\rm{{y}}}$(z) (BP)')

# Extend and plot slit arm fits to intersections
def extend_to_target(z_data, coeffs, z_target, npoints=60):
    if len(z_data) == 0 or np.isnan(z_target) or np.isnan(coeffs[0]) or np.isnan(coeffs[1]):
        return np.array([]), np.array([])
    # choose direction that goes from the segment towards the target
    z_start = z_data[0] if z_data[-1] < z_target else z_data[-1]
    z_ext = np.linspace(z_start, z_target, npoints)
    w_ext = np.polyval(coeffs, z_ext)
    return z_ext, w_ext

if slit_z_mm_masked.size:
    if 'fit_left_x' in locals() and not np.isnan(z0_x):
        zl, wl = extend_to_target(z_left_x, fit_left_x, z0_x)
        zr, wr = extend_to_target(z_right_x, fit_right_x, z0_x)
        if zl.size: ax.plot(zl, wl, '--', color='gray', lw=1.3)
        if zr.size: ax.plot(zr, wr, '--', color='gray', lw=1.3, label=r'Linear fit: $w_{\rm{{x}}}$(z) (BP)')
    if 'fit_left_y' in locals() and not np.isnan(z0_y):
        zl, wl = extend_to_target(z_left_y, fit_left_y, z0_y)
        zr, wr = extend_to_target(z_right_y, fit_right_y, z0_y)
        if zl.size: ax.plot(zl, wl, ':', color='dimgray', lw=1.3)
        if zr.size: ax.plot(zr, wr, ':', color='dimgray', lw=1.3, label=r'Linear fit: $w_{\rm{{y}}}$(z) (BP)')

# plot slit intersection waists
if not np.isnan(z0_x):
    ax.plot(z0_x, w0_x_um, 's', mec='red', mfc='none', mew=2, ms=10)
if not np.isnan(z0_y):
    ax.plot(z0_y, w0_y_um, '^', mec='red', mfc='none', mew=2, ms=10)

# Plot QPD quadB & quadC separately (no averaging)
if qpd_z_shifted_B.size:
    for z_s, w in zip(qpd_z_shifted_B, qpd_w_um_B):
        ax.plot(z_s, w, 'o', color=color_for_z(z_s), markersize=7)
    ax.plot(qpd_z_shifted_B, qpd_w_um_B, '--', color='orange', lw=1.6, alpha=0.8, label=r'$w$(z) quadB (KE)') # color='tab:orange'

if qpd_z_shifted_C.size:
    for z_s, w in zip(qpd_z_shifted_C, qpd_w_um_C):
        ax.plot(z_s, w, 'D', color=color_for_z(z_s), markersize=7) # fillstyle='none'
    ax.plot(qpd_z_shifted_C, qpd_w_um_C, '-.', color='green', lw=1.6, alpha=0.8, label=r'$w$(z) quadC (KE)')

# Inset: minima from quadB, quadC, and BP intersections
min_info = []
# quadB min
if qpd_w_um_B.size:
    idx_min_B = int(np.argmin(qpd_w_um_B))
    z_min_B = float(qpd_z_shifted_B[idx_min_B])
    w_min_B = float(qpd_w_um_B[idx_min_B])
    ax.plot(z_min_B, w_min_B, 'o', mec='red', mfc='none', mew=2.5, ms=12)
    min_info.append(("B", w_min_B, z_min_B))
# quadC min
if qpd_w_um_C.size:
    idx_min_C = int(np.argmin(qpd_w_um_C))
    z_min_C = float(qpd_z_shifted_C[idx_min_C])
    w_min_C = float(qpd_w_um_C[idx_min_C])
    ax.plot(z_min_C, w_min_C, 'D', mec='red', mfc='none', mew=2.5, ms=12)
    min_info.append(("C", w_min_C, z_min_C))

# BP minima info
if not np.isnan(w0_x_um):
    min_info.append(("BP_x", w0_x_um, z0_x))
if not np.isnan(w0_y_um):
    min_info.append(("BP_y", w0_y_um, z0_y))

# Inset box with values
if min_info:
    inset = ax.inset_axes([0.48, 0.45, 0.45, 0.4])
    inset.axis('off')
    handles = []
    labels = []
    for tag, wv, zv in min_info:
        if tag == "B":
            marker = 'o'; mcol = 'red'; label = rf'$w_{{0,\min}}$ (KE, quadB) = ${wv:.2f}\,\mu\mathrm{{m}}$'
        elif tag == "C":
            marker = 'D'; mcol = 'red'; label = rf'$w_{{0,\min}}$ (KE, quadC) = ${wv:.2f}\,\mu\mathrm{{m}}$'
        elif tag == "BP_x":
            marker = 's'; mcol = 'red'; label = rf'$w_{{0,\rm{{x}}}}$ (BP) = ${wv:.2f}\,\mu\mathrm{{m}}$'
        else:
            marker = '^'; mcol = 'red'; label = rf'$w_{{0,\rm{{y}}}}$ (BP) = ${wv:.2f}\,\mu\mathrm{{m}}$'
        handles.append(Line2D([0], [0], marker=marker, color='w', markeredgecolor=mcol,
                              markerfacecolor='none', markersize=10, markeredgewidth=2.0))
        labels.append(label)
    try:
        inset.legend(handles=handles, labels=labels, loc='upper left', fontsize=14, frameon=False)
    except Exception:
        txt = "\n".join(labels)
        inset.text(0, 0.9, txt, va='top', fontsize=10)

# Legend (de-duplicate)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=11, loc='upper left', frameon=True, bbox_to_anchor=(0.02, 0.98))

# Axis limits: make sure data isn't outside and hide empty-plot risk
all_z_values = np.concatenate([
    slit_z_mm_masked if slit_z_mm_masked.size else np.array([]),
    qpd_z_shifted_B if qpd_z_shifted_B.size else np.array([]),
    qpd_z_shifted_C if qpd_z_shifted_C.size else np.array([])
])
if all_z_values.size:
    zmin = np.min(all_z_values) - 0.5
    zmax = np.max(all_z_values) + 0.5
    ax.set_xlim(10.45, 13.55) #zmin, zmax
else:
    ax.set_xlim(SLIT_Z_MIN_MM, SLIT_Z_MAX_MM)

all_w_values = np.concatenate([
    slit_wx_um_masked if slit_wx_um_masked.size else np.array([]),
    slit_wy_um_masked if slit_wy_um_masked.size else np.array([]),
    qpd_w_um_B if qpd_w_um_B.size else np.array([]),
    qpd_w_um_C if qpd_w_um_C.size else np.array([])
])
if all_w_values.size:
    wmin = max(np.min(all_w_values) - 20.0, 0.0)
    wmax = np.max(all_w_values) + 50.0
    ax.set_ylim(bottom=-10.0, top=395)
else:
    ax.set_ylim(bottom=-10, top=600)

# Save and show
out_path = os.path.join(OUT_DIR, "merged_QPD_vs_BP_comparison.png")
fig.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\nComparison figure saved → {out_path}")

# Print final summary (text)
print("\n=== FINAL RESULTS SUMMARY ===")
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

print(f"Slit X: {'not fitted' if np.isnan(z0_x) else f'w0 = {w0_x_um:.2f} µm at Z = {z0_x:.3f} mm'}")
print(f"Slit Y: {'not fitted' if np.isnan(z0_y) else f'w0 = {w0_y_um:.2f} µm at Z = {z0_y:.3f} mm'}")
print("============================\n")

plt.show()
