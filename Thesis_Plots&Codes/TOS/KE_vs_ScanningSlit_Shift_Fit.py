"""
QPD (quad-A) + BP SLIT-SCANNER COMPARISON
→ QPD Z shifted by +5.5 mm 
Slit Z shifted by +14.62 mm
ERF-fit
Z-mask for slit data + full arm-fit (θ & w₀) reporting
"""
import re
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import curve_fit
from scipy.special import erf
from matplotlib.lines import Line2D  # for inset legend

# ----------------------------------------------------------------------
# USER SETTINGS 
# ----------------------------------------------------------------------
QPD_DIR = "/Users/asca/Documents/University/Master Thesis/code/Data/VIGO17_NS089008_QPD_0750_20_AS_015_CC_asca_250617_focus_laser_thres_LB1471C_quadA/Y6100um"
#SLIT_DIR = "/Users/asca/Documents/University/Master Thesis/code/Data/slit scanning beam profiler/251015_old_telescoping_LB1471-C"
SLIT_DIR = "/Users/asca/Documents/University/Master Thesis/code/Data/slit scanning beam profiler/251021_old_telescoping_LB1471-C"
SLIT_OFFSET_MM = +5.5# (+2.235) Shift slit forward
QPD_OFFSET_MM = +14.62 # (12.152) → to align the min
# ---- Slit Z-mask (after shift) ---------------------------------
SLIT_Z_MIN_MM = 46.5 # tweak these two values
SLIT_Z_MAX_MM = 52.8
# ----------------------------------------------------------------------

# --------------------------------------------------------------
# LaTeX / font
# --------------------------------------------------------------
font_path = "/Users/asca/Library/Fonts/cmunrm.ttf"
fm.FontProperties(fname=font_path)
plt.rcParams.update({
'text.usetex': True,
'font.family': 'serif',
'font.serif': ['Computer Modern'],
})

# --------------------------------------------------------------
# ERF model
# --------------------------------------------------------------
def erf_model(x, A, B, C, D):
    return A * erf(B * (x - C)) + D

# --------------------------------------------------------------
# Intersection of two linear fits
# --------------------------------------------------------------
def find_intersection(fit_left, fit_right):
    m1, b1 = fit_left
    m2, b2 = fit_right
    if abs(m1 - m2) < 1e-12:
        return np.nan, np.nan
    z_int = (b2 - b1) / (m1 - m2)
    w_int = m1 * z_int + b1
    return z_int, w_int

# --------------------------------------------------------------
# LOAD SLIT-SCANNER DATA + SHIFT + Z-MASK
# --------------------------------------------------------------
slit_files = sorted(
    [f for f in glob.glob(os.path.join(SLIT_DIR, "*[m|mm]_#001.txt"))
    if "before lens" not in os.path.basename(f).lower()],
    key=lambda f: float(os.path.basename(f).split('m')[0])
)

slit_z, slit_wx, slit_wy = [], [], []
for file in slit_files:
    z_raw = float(os.path.basename(file).split('m')[0])
    with open(file, 'r', encoding='latin1') as f:
        lines = f.readlines()
    wx_dia = wy_dia = None
    for line in lines:
        if line.strip() and line.split('\t')[0].isdigit():
            parts = line.strip().split('\t')
            try:
                wx_dia = float(parts[19])
                wy_dia = float(parts[20])
            except Exception:
                continue
            break
    if wx_dia is None or wy_dia is None:
        continue
    z_shifted = z_raw + SLIT_OFFSET_MM
    slit_z.append(z_shifted)
    slit_wx.append(wx_dia / 2)      # diameter → radius (µm)
    slit_wy.append(wy_dia / 2)

slit_z = np.array(slit_z)
slit_wx = np.array(slit_wx)
slit_wy = np.array(slit_wy)
order = np.argsort(slit_z)
slit_z, slit_wx, slit_wy = slit_z[order], slit_wx[order], slit_wy[order]
# ---- apply Z-mask ------------------------------------------------
mask = (slit_z >= SLIT_Z_MIN_MM) & (slit_z <= SLIT_Z_MAX_MM)
slit_z, slit_wx, slit_wy = slit_z[mask], slit_wx[mask], slit_wy[mask]

print(f"Slit scanner: {len(slit_z)} points (Z → Z + {SLIT_OFFSET_MM:.3f} mm, "f"masked to [{SLIT_Z_MIN_MM:.1f}, {SLIT_Z_MAX_MM:.1f}] mm)")

# --------------------------------------------------------------
# FIT LEFT / RIGHT ARMS → waist by intersection + divergence
# --------------------------------------------------------------
lambda_wl = 0.001064            # 1064 nm → mm
z0_x, w0_x = np.nan, np.nan
z0_y, w0_y = np.nan, np.nan

# ----- X -----
min_idx_x = np.argmin(slit_wx)
z_left_x, wx_left = slit_z[:min_idx_x], slit_wx[:min_idx_x]
z_right_x, wx_right = slit_z[min_idx_x:], slit_wx[min_idx_x:]

fit_left_x = np.polyfit(z_left_x, wx_left, 1) if len(z_left_x) > 1 else [0, 0]
fit_right_x = np.polyfit(z_right_x, wx_right, 1) if len(z_right_x) > 1 else [0, 0]

# Divergence & w₀ from each arm
theta_left_x = fit_left_x[0] if len(z_left_x) > 1 else np.nan
theta_right_x = fit_right_x[0] if len(z_right_x) > 1 else np.nan
w0_left_x_um = (lambda_wl / (np.pi * abs(theta_left_x))) * 1000 if not np.isnan(theta_left_x) else np.nan
w0_right_x_um = (lambda_wl / (np.pi * abs(theta_right_x))) * 1000 if not np.isnan(theta_right_x) else np.nan

if len(z_left_x) > 1 and len(z_right_x) > 1:
    z0_x, w0_x = find_intersection(fit_left_x, fit_right_x)

# ----- Y -----
min_idx_y = np.argmin(slit_wy)
z_left_y, wy_left = slit_z[:min_idx_y], slit_wy[:min_idx_y]
z_right_y, wy_right = slit_z[min_idx_y:], slit_wy[min_idx_y:]

fit_left_y = np.polyfit(z_left_y, wy_left, 1) if len(z_left_y) > 1 else [0, 0]
fit_right_y = np.polyfit(z_right_y, wy_right, 1) if len(z_right_y) > 1 else [0, 0]

theta_left_y = fit_left_y[0] if len(z_left_y) > 1 else np.nan
theta_right_y = fit_right_y[0] if len(z_right_y) > 1 else np.nan
w0_left_y_um = (lambda_wl / (np.pi * abs(theta_left_y))) * 1000 if not np.isnan(theta_left_y) else np.nan
w0_right_y_um = (lambda_wl / (np.pi * abs(theta_right_y))) * 1000 if not np.isnan(theta_right_y) else np.nan

if len(z_left_y) > 1 and len(z_right_y) > 1:
    z0_y, w0_y = find_intersection(fit_left_y, fit_right_y)

# ---- Print full arm-fit information ----------------------------
print("\n=== SLIT WAIST (intersection) ===")
print(f"X-waist: Z = {z0_x:.3f} mm, w₀ = {w0_x:.1f} µm"
      if not np.isnan(z0_x) else "X-waist: not enough arms")
print(f"Y-waist: Z = {z0_y:.3f} mm, w₀ = {w0_y:.1f} µm"
      if not np.isnan(z0_y) else "Y-waist: not enough arms")
print("\n=== SLIT SCANNER DIVERGENCE FIT (Shifted Reference) ===")
print("X-direction:")
print(f" Left arm : θ = {theta_left_x:.6f} rad/mm → w₀ = {w0_left_x_um:.1f} µm")
print(f" Right arm: θ = {theta_right_x:.6f} rad/mm → w₀ = {w0_right_x_um:.1f} µm")
print("Y-direction:")
print(f" Left arm : θ = {theta_left_y:.6f} rad/mm → w₀ = {w0_left_y_um:.1f} µm")
print(f" Right arm: θ = {theta_right_y:.6f} rad/mm → w₀ = {w0_right_y_um:.1f} µm")

# --------------------------------------------------------------
# LOAD OLD QPD DATA – EXACT OLD PIPELINE + QPD Z-SHIFT
# --------------------------------------------------------------
def extract_z_position(fname):
    m = re.search(r'Z(\d+)um', fname)
    return int(m.group(1)) / 1000 if m else None

pkl_files = sorted(
    [f for f in glob.glob(os.path.join(QPD_DIR, "*.pkl"))
    if (z_raw := extract_z_position(os.path.basename(f))) and z_raw >= 32],
    key=lambda f: extract_z_position(os.path.basename(f))
)

qpd_z_raw, qpd_z_shifted, qpd_w = [], [], []

for pkl in pkl_files:
    data = pickle.load(open(pkl, "rb"))
    z_raw = extract_z_position(os.path.basename(pkl))
    z_shifted = z_raw + QPD_OFFSET_MM

    # ---- build X array (µm) –----
    xstart = data['global_params']['xstart_um']
    xstop = data['global_params']['xstop_um']
    xbig = data['global_params']['xstep_big_um']
    xfine = data['global_params']['xstep_fine_um']
    xth_s = data['global_params']['x_threshold_start_um']
    xth_e = data['global_params']['x_threshold_stop_um']

    x_arr = []
    cur = xstart
    while cur <= xstop:
        x_arr.append(cur)
        cur += xfine if xth_s <= cur <= xth_e else xbig
    x_arr = np.array(x_arr)

    # ---- DC current (quadA) ----
    if 'quadA' not in data['rawdata'] or 'dmm00_curr_amp' not in data['rawdata']['quadA']:
        continue
    dc = np.mean(data['rawdata']['quadA']['dmm00_curr_amp'], axis=1)

    # ---- ERF fit ----
    if len(x_arr) != len(dc):
        print(f"Length mismatch in {os.path.basename(pkl)} – skipping")
        continue

    initial_guess_dc = [np.max(dc) / 2, 0.01, np.mean(x_arr), np.min(dc)]

    try:
        params_dc, _ = curve_fit(erf_model, x_arr, dc, p0=initial_guess_dc)
        A_fit_dc, B_fit_dc, x0_fit_dc, C_fit_dc = params_dc
        spot_size_dc = 1 / (np.sqrt(2) * B_fit_dc) # µm
        qpd_z_raw.append(z_raw)
        qpd_z_shifted.append(z_shifted)
        qpd_w.append(spot_size_dc)
        if 0 < spot_size_dc < 1000: # sanity
            qpd_z_raw.append(z_raw)
            qpd_z_shifted.append(z_shifted)
            qpd_w.append(spot_size_dc)
            print(f"QPD quadA – Z_raw={z_raw:.2f} → Z_shifted={z_shifted:.2f} mm → w={spot_size_dc:.2f} µm")
    except Exception as e:
        print(f"Fit failed for {os.path.basename(pkl)}: {e}")

qpd_z_raw = np.array(qpd_z_raw)
qpd_z_shifted = np.array(qpd_z_shifted)
qpd_w = np.array(qpd_w)

# --------------------------------------------------------------
# COLOR MAP (based on SHIFTED Z for consistency)
# --------------------------------------------------------------
cmap = plt.cm.RdYlBu
all_z = sorted(set(qpd_z_shifted))
n = len(all_z)
color_dict = {}
for i, z in enumerate(all_z):
    if i < n // 2:
        color_dict[z] = cmap(0.1 + 0.4 * (i / (n // 2)))
    else:
        color_dict[z] = cmap(0.6 + 0.4 * ((i - n // 2) / (n - n // 2)))

# --------------------------------------------------------------
# EXTEND ARM FITS TO INTERSECTION
# --------------------------------------------------------------
def extend_to_target(z_data, coeffs, z_target):
    if len(z_data) == 0 or np.isnan(z_target):
        return np.array([]), np.array([])
    z_start = z_data[0] if z_data[-1] < z_target else z_data[-1]
    if (z_data[-1] < z_target and z_start >= z_target) or \
       (z_data[-1] > z_target and z_start <= z_target):
        return np.array([]), np.array([])
    z_ext = np.linspace(z_start, z_target, 50)
    w_ext = np.polyval(coeffs, z_ext)
    return z_ext, w_ext

# --------------------------------------------------------------
# PLOT – ONE COMPARISON FIGURE (QPD SHIFTED!)
# --------------------------------------------------------------
fig_dir = os.path.join(QPD_DIR, "fig")
os.makedirs(fig_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
ax.set_xlabel(r'Z Position [mm]', fontsize=14)
ax.set_ylabel(r'Beam Radius $\rm{w}_{\rm{x,y}}$(z) \& Beam Waist $\rm{w}_0$(z) [\textmu{}m]', fontsize=14)
ax.set_title(r'\textbf{Beam Profile: KE (quadA) vs. BP Data}', fontsize=14, fontweight='bold')
ax.set_xlim(46.6, 52.8) # 42 , 56
ax.set_ylim(bottom=-20, top=650) # 42 , 56
ax.grid(True, ls='--', alpha=0.6)
ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5, direction='in')

# ---- QPD points (colour by SHIFTED Z) ----
for z_shifted, w in zip(qpd_z_shifted, qpd_w):
    ax.plot(z_shifted, w, 'o', color=color_dict[z_shifted], markersize=7)
qpd_line, = ax.plot(qpd_z_shifted, qpd_w, '-', color='blue', lw=1.5, alpha=0.5, label=r'$w_0$(z) quadA (KE)')

'''# create a combined handle (marker + line)
combined_handle = Line2D([], [], marker='o', color='w', markeredgecolor='C0',
                        markerfacecolor='none', markersize=7,
                        linestyle='-', linewidth=1.5, alpha=0.7,
                        label='QPD quad-A (connected)')'''
ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(0.02, 0.98))

# ---- Slit data (already shifted & masked) ----
ax.plot(slit_z, slit_wx, 's', mfc='none', mec='black', ms=8,
        label=r'$w_x$(z) (BP)')
ax.plot(slit_z, slit_wy, '^', mfc='none', mec='black', ms=8,
        label=r'$w_y$(z) (BP)')

# ---- Arm fits extended to intersection ----
if len(z_left_x) > 1 and not np.isnan(z0_x):
    zl, wl = extend_to_target(z_left_x, fit_left_x, z0_x)
    zr, wr = extend_to_target(z_right_x, fit_right_x, z0_x)
    ax.plot(zl, wl, '--', color='gray', lw=1.5)
    ax.plot(zr, wr, '--', color='gray', lw=1.5, label=r'Linear fit: $w_x$(z) (BP)')
if len(z_left_y) > 1 and not np.isnan(z0_y):
    zl, wl = extend_to_target(z_left_y, fit_left_y, z0_y)
    zr, wr = extend_to_target(z_right_y, fit_right_y, z0_y)
    ax.plot(zl, wl, ':', color='dimgray', lw=1.5)
    ax.plot(zr, wr, ':', color='dimgray', lw=1.5, label=r'Linear fit: $w_y$(z) (BP)')

# ---- Intersection waists (red markers) ----
if not np.isnan(z0_x):
    ax.plot(z0_x, w0_x, 's', mec='red', mfc='none', mew=2, ms=10)
if not np.isnan(z0_y):
    ax.plot(z0_y, w0_y, '^', mec='red', mfc='none', mew=2, ms=10)

'''# ---- QPD minimum waist: red circle + inset results box ----
if qpd_w.size > 0:
    idx_min_qpd = np.argmin(qpd_w)
    z_min_qpd = qpd_z_shifted[idx_min_qpd]
    w_min_qpd = qpd_w[idx_min_qpd]
    # Red open circle
    ax.plot(z_min_qpd, w_min_qpd, 'o', mec='red', mfc='none', mew=2.5, ms=12)
    
    # --- Inset results box ---
    tex_label = (r'$w_{{0,\min}} = {:.2f}\,\mu\mathrm{{m}}$'.format(w_min_qpd) + '\n' +
                 r'$Z = {:.2f}\,\mathrm{{mm}}$'.format(z_min_qpd))
    result_line = Line2D([0], [0], marker='o', color='w',
                        markeredgecolor='red', markerfacecolor='none',
                        markersize=10, markeredgewidth=2.5,
                        label=tex_label)
    inset = ax.inset_axes([0.58, 0.65, 0.38, 0.28])  # [x0, y0, width, height]
    inset.axis('off')
    inset.legend(handles=[result_line], loc='upper left', fontsize=13, frameon=False, handlelength=1.2)

# ---- Legend (remove duplicates) ----
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=11,
          loc='upper left', frameon=True, bbox_to_anchor=(0.02, 0.98))
# ---- Save ----
out_path = os.path.join(fig_dir, "QPD_quadA_oldPipeline_shifted_vs_SlitScanner.png")
fig.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\nComparison figure saved → {out_path}")'''

# ---- QPD minimum waist: red circle + inset results box ----
if qpd_w.size > 0:
    idx_min_qpd = np.argmin(qpd_w)
    z_min_qpd = qpd_z_shifted[idx_min_qpd]
    w_min_qpd = qpd_w[idx_min_qpd]
    # Red open circle
    ax.plot(z_min_qpd, w_min_qpd, 'o', mec='red', mfc='none', mew=2.5, ms=12)

    # --- Inset results box with QPD + Slit X + Slit Y ---
    inset = ax.inset_axes([0.38, 0.38, 0.38, 0.35])  # Adjusted for 3 lines
    inset.axis('off')
    handles_inset = []

    # QPD minimum (knife-edge)
    #from matplotlib.lines import Line2D
    handles_inset.append(Line2D([0], [0], marker='o', color='w',
                                markeredgecolor='red', markerfacecolor='none',
                                markersize=10, markeredgewidth=2.5,
                                label=rf'$w_{{0,\min}}$ (KE)= ${w_min_qpd:.2f}\,\mu\mathrm{{m}}$'))

    # Slit X waist
    if not np.isnan(z0_x):
        handles_inset.append(Line2D([0], [0], marker='s', color='w',
                                    markeredgecolor='red', markerfacecolor='none',
                                    markersize=8, markeredgewidth=2.0,
                                    label=rf'$w_{{0,x}}$ (BP) = ${w0_x:.2f}\,\mu\mathrm{{m}}$'))

    # Slit Y waist
    if not np.isnan(z0_y):
        handles_inset.append(Line2D([0], [0], marker='^', color='w',
                                    markeredgecolor='red', markerfacecolor='none',
                                    markersize=8, markeredgewidth=2.0,
                                    label=rf'$w_{{0,y}}$ (BP) = ${w0_y:.2f}\,\mu\mathrm{{m}}$'))

    inset.legend(handles=handles_inset, loc='upper left', fontsize=14, frameon=False,
                 handlelength=1.2, borderpad=0.8, labelspacing=0.8)

# ---- Legend (remove duplicates) ----
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=13,
          loc='upper left', frameon=True, bbox_to_anchor=(0.03, 0.98))
# ---- Save ----
out_path = os.path.join(fig_dir, "QPD_quadA_oldPipeline_shifted_vs_SlitScanner.png")
fig.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\nComparison figure saved → {out_path}")
    
# --------------------------------------------------------------
# MINIMUM WAIST REPORT (on shifted Z)
# --------------------------------------------------------------
if qpd_w.size:
    idx_min = np.argmin(qpd_w)
    print(f"\nQPD quad-A – minimum waist: {qpd_w[idx_min]:.2f} µm "
          f"(Z_raw = {qpd_z_raw[idx_min]:.2f} mm → Z_shifted = {qpd_z_shifted[idx_min]:.2f} mm)")

# --- Final print summary ---
print("\n=== FINAL RESULTS SUMMARY ===")
print(f"QPD (knife-edge):  w₀,min = {qpd_w[idx_min]:.2f} µm at Z = {qpd_z_shifted[idx_min]:.3f} mm")
print(f"Slit X:            w₀ = {w0_x:.2f} µm at Z = {z0_x:.3f} mm" if not np.isnan(z0_x) else "Slit X: not fitted")
print(f"Slit Y:            w₀ = {w0_y:.2f} µm at Z = {z0_y:.3f} mm" if not np.isnan(z0_y) else "Slit Y: not fitted")
print("============================\n")

plt.show()